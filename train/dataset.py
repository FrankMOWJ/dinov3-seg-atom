"""MoS2 images, point-to-circle labels, patch extraction, and augmentation.

Adapted from Model/scripts/train.py. Keep patch data on Dataset instances so
DataLoader workers created with spawn can read it without module globals.
"""

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


CLASS_NAMES = ('Background', 'Mo', 'S', 'V1S')
CLASS_MAP = {'Mo': 1, 'S': 2, 'V1S': 3, 'V2S': 3}
PATCH_SIZE = 256
ATOMCOUNT_SCALE_FACTORS = {
    '08': 2.028, '09': 2.025, '24': 1.049, '25': 1.042,
    '34': 1.038, '52': 1.036, '65': 1.038, '67': 1.033,
    'MOS2-0003': 0.974, 'MOS2-0004': 0.993, 'MOS2-0005': 0.989,
}


@dataclass
class Sample:
    image: str | np.ndarray
    mask: str | np.ndarray
    source: str
    frame_id: str
    sample_id: str

    def record(self):
        return dict(source=self.source, frame_id=self.frame_id, sample_id=self.sample_id)


def load_image(path):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        image = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as pil_image:
            image = np.array(pil_image)
    image = image.astype(np.float32)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=-1)
    if image.ndim != 2 or not image.size or not np.isfinite(image).all():
        raise ValueError(f'Expected a finite 2D image: {path}')
    return image


def robust_normalize(image):
    image = image.astype(np.float32)
    median = np.median(image)
    mad = np.median(np.abs(image - median))
    return np.clip((image - median) / (mad + 1e-8), -5, 5)


def create_mask_from_json(image_shape, json_path, radius=8, scale_factor=1.0):
    """Draw radius-8 circles at scaled points; V1S and V2S share label 3."""
    with open(json_path, encoding='utf-8') as stream:
        annotations = json.load(stream)
    mask = np.zeros(image_shape, dtype=np.uint8)
    for shape in annotations.get('shapes', []):
        if shape.get('shape_type') != 'point' or shape.get('label') not in CLASS_MAP:
            continue
        x, y = shape['points'][0]
        center = (int(x * scale_factor), int(y * scale_factor))
        cv2.circle(mask, center, radius, CLASS_MAP[shape['label']], -1)
    return mask


def extract_atomcount_patches(image_path, json_path, scale_factor, radius=8, rng=None):
    """Use the original scale table, ~50% sampling stride, and 77px spacing."""
    rng = rng or random.Random(42)
    if scale_factor <= 0:
        raise ValueError('Image scale factors must be positive')
    image = load_image(image_path)
    height, width = (int(d * scale_factor) for d in image.shape)
    if min(height, width) < PATCH_SIZE:
        raise ValueError(f'Scaled image is smaller than a 256px patch: {image_path}')
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = create_mask_from_json((height, width), json_path, radius, scale_factor)
    max_y, max_x = height - PATCH_SIZE, width - PATCH_SIZE
    count = (max_y // 128 + 1) * (max_x // 128 + 1)
    positions, patches = [], []
    for _ in range(count * 20):
        if len(patches) == count:
            break
        # Match the original RNG sequence: fixed axes consume no random draw.
        y = rng.randint(0, max_y) if max_y > 0 else 0
        x = rng.randint(0, max_x) if max_x > 0 else 0
        if any((y - old_y) ** 2 + (x - old_x) ** 2 < 77 ** 2
               for old_y, old_x in positions):
            continue
        positions.append((y, x))
        patches.append((image[y:y+PATCH_SIZE, x:x+PATCH_SIZE].copy(),
                        mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE].copy()))
    return patches


def extract_simulated_patches(image_path, mask_path, scale_factor=1.153):
    image = load_image(image_path)
    mask = np.load(mask_path, allow_pickle=False)
    if mask.shape != image.shape:
        raise ValueError(f'Simulation image/mask dimensions differ: {image_path}')
    if scale_factor <= 0:
        raise ValueError('Simulation scale factor must be positive')
    height, width = (int(d * scale_factor) for d in image.shape)
    if min(height, width) < PATCH_SIZE:
        raise ValueError(f'Scaled simulation is smaller than a patch: {image_path}')
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return [(image[y:y+PATCH_SIZE, x:x+PATCH_SIZE].copy(),
             mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE].copy())
            for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE)
            for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE)]


def apply_augmentation(image, mask):
    if random.random() < 0.5:
        image, mask = np.flip(image, 1).copy(), np.flip(mask, 1).copy()
    if random.random() < 0.5:
        image, mask = np.flip(image, 0).copy(), np.flip(mask, 0).copy()
    if random.random() < 0.5:
        k = random.randint(1, 3)
        image, mask = np.rot90(image, k).copy(), np.rot90(mask, k).copy()
    if random.random() < 0.3:
        image += np.random.normal(0, 0.1, image.shape).astype(np.float32)
    if random.random() < 0.3:
        low, high = image.min(), image.max()
        positive = np.clip((image - low) / (high - low + 1e-8) * 255, 0, 255)
        image = np.random.poisson(positive).astype(np.float32) / 255 * (high - low) + low
    if random.random() < 0.2:
        image = cv2.GaussianBlur(image, (0, 0), random.uniform(0.5, 1.5))
    if random.random() < 0.2:
        image = random.uniform(0.9, 1.1) * image + random.uniform(-0.1, 0.1)
    return image, mask


def real_frame_id(stem):
    # Join known MOS2 frames across the preprocessed and AtomCount sources.
    match = re.match(r'(MOS2-\d+)', stem, re.IGNORECASE)
    return 'real:' + (match.group(1).upper() if match else stem.split('_patch')[0])


def collect_samples(config):
    samples = []
    rng = random.Random(config['seed'])
    directories = config.get('part1_images_dirs') or []
    if directories and not config.get('part1_masks_dir'):
        raise ValueError('Set part1_masks_dir when part1_images_dirs is enabled')
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f'Part 1 images directory: {directory}')
        for path in sorted(directory.glob('*.npy')):
            mask_path = Path(config['part1_masks_dir']) / (path.stem + '_mask.npy')
            if not mask_path.is_file():
                raise FileNotFoundError(f'Missing Part 1 mask: {mask_path}')
            samples.append(Sample(str(path), str(mask_path), 'part1', real_frame_id(path.stem),
                                  f'part1:{path.resolve()}'))
    scales = {**ATOMCOUNT_SCALE_FACTORS, **config.get('atomcount_scale_factors', {})}
    #NOTE: Below is the data without annotation
    if config.get('part2_data_dir'):
        directory = Path(config['part2_data_dir'])
        if not directory.is_dir():
            raise FileNotFoundError(f'AtomCount directory: {directory}')
        for path in sorted(directory.glob('*.png')):
            json_path = path.with_suffix('.json')
            if not json_path.is_file():
                raise FileNotFoundError(f'Missing point annotations: {json_path}')
            patches = extract_atomcount_patches(path, json_path, scales.get(path.stem, 1.0),
                                                config['point_radius'], rng)
            for index, (image, mask) in enumerate(patches):
                samples.append(Sample(image, mask, 'part2', real_frame_id(path.stem),
                                      f'part2:{path.resolve()}:{index}'))
    if config.get('part3_data_dir'):
        directory = Path(config['part3_data_dir'])
        if not directory.is_dir():
            raise FileNotFoundError(f'Simulation directory: {directory}')
        paths = sorted(directory.glob('*_image.npy'))
        limit = config['num_simulated']
        if limit is not None:
            if limit < 0:
                raise ValueError('num_simulated must be nonnegative or null')
            paths = paths[:limit]
        for path in paths:
            mask_path = directory / 'masks' / (path.stem.removesuffix('_image') + '_mask.npy')
            if not mask_path.is_file():
                raise FileNotFoundError(f'Missing simulation mask: {mask_path}')
            patches = extract_simulated_patches(path, mask_path, config['simulated_scale_factor'])
            for index, (image, mask) in enumerate(patches):
                samples.append(Sample(image, mask, 'part3', 'sim:' + path.stem,
                                      f'part3:{path.resolve()}:{index}'))
    if not samples:
        raise ValueError('No image/mask pairs found; check the data paths in config.json')
    return samples


def grouped_split(samples, val_fraction=0.15, seed=42):
    """Stratify frame groups by source; keep shared real-frame IDs together."""
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be between 0 and 1')
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.frame_id].append(sample)
    if len(groups) < 2:
        raise ValueError('At least two independent original frames are needed for train/validation')
    strata = defaultdict(list)
    for frame, members in groups.items():
        strata[tuple(sorted({sample.source for sample in members}))].append(frame)
    rng = random.Random(seed)
    val_frames = set()
    for frames in strata.values():
        frames.sort()
        rng.shuffle(frames)
        count = min(len(frames) - 1, max(1, int(len(frames) * val_fraction))) if len(frames) > 1 else 0
        val_frames.update(frames[:count])
    if not val_frames:
        val_frames.add(rng.choice(sorted(groups)))
    training = [sample for sample in samples if sample.frame_id not in val_frames]
    validation = [sample for sample in samples if sample.frame_id in val_frames]
    rng.shuffle(training)
    return training, validation


class MoS2Dataset(Dataset):
    def __init__(self, samples, training=False, image_size=256):
        if image_size <= 0:
            raise ValueError('image_size must be positive')
        self.samples = samples
        self.training = training
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = load_image(sample.image) if isinstance(sample.image, str) else sample.image.copy()
        mask = (np.load(sample.mask, allow_pickle=False) if isinstance(sample.mask, str)
                else sample.mask.copy())
        if mask.ndim != 2 or not np.isin(mask, [0, 1, 2, 3, 255]).all():
            raise ValueError(f'Mask must contain labels 0..3 or ignore label 255: {sample.sample_id}')
        size = (self.image_size, self.image_size)
        if image.shape != size:
            image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        if mask.shape != size:
            mask = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_NEAREST)
        image = robust_normalize(image)
        if self.training:
            image, mask = apply_augmentation(image, mask)
        # PDZSeg repeats grayscale to three channels; do not apply ImageNet normalization again.
        return (torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(mask)).long())
