"""Train the current PDZSeg LoRA model on MoS2 data; edit config.json first."""

import argparse
import csv
import json
import logging
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arch.dino_seg import PDZSeg
from train.dataset import CLASS_NAMES, MoS2Dataset, collect_samples, grouped_split
from train.losses import DiceCELoss


def load_config(path):
    path = Path(path).resolve()
    defaults_path = Path(__file__).with_name('config.json')
    config = json.loads(defaults_path.read_text())
    config.update(json.loads(path.read_text()))
    def resolve(value):
        if not value or str(value).startswith(('https://', 'http://')):
            return value
        candidate = Path(value).expanduser()
        return str((path.parent / candidate).resolve())
    for name in ('part1_masks_dir', 'part2_data_dir', 'part3_data_dir',
                 'backbone_weights', 'output_dir'):
        config[name] = resolve(config.get(name))
    config['part1_images_dirs'] = [resolve(value) for value in config.get('part1_images_dirs', []) or []]
    if not str(config['backbone_repo']).startswith('facebookresearch/dinov3'):
        config['backbone_repo'] = resolve(config['backbone_repo'])
        if not Path(config['backbone_repo']).is_dir():
            raise FileNotFoundError(f"Local DINOv3 source directory: {config['backbone_repo']}")
    for name in ('epochs', 'batch_size', 'image_size', 'point_radius'):
        if not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if config['num_workers'] < 0 or config['warmup_epochs'] < 0 or config['patience'] < 0:
        raise ValueError('num_workers, warmup_epochs and patience must be nonnegative')
    if min(config['decoder_lr'], config['lora_lr']) <= 0 or config['grad_clip'] <= 0:
        raise ValueError('Learning rates and grad_clip must be positive')
    if not 0 <= config['min_lr_ratio'] <= 1:
        raise ValueError('min_lr_ratio must be between 0 and 1')
    if not config['output_dir']:
        raise ValueError('Set output_dir in config.json')
    if config['pretrained']:
        weights = config['backbone_weights']
        if not weights:
            raise ValueError('Set backbone_weights to your downloaded official DINOv3 checkpoint')
        if not weights.startswith(('https://', 'http://')) and not Path(weights).is_file():
            raise FileNotFoundError(f'DINOv3 pretrained checkpoint: {weights}')
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def build_optimizer(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        kind = 'decoder' if name.startswith('decode_head.') else 'lora'
        decay = config['weight_decay'] if parameter.ndim > 1 else 0.0
        key = (kind, decay)
        groups.setdefault(key, []).append(parameter)
    return torch.optim.AdamW([
        dict(params=parameters, lr=config[kind + '_lr'], weight_decay=decay, name=kind)
        for (kind, decay), parameters in groups.items()
    ])


def build_scheduler(optimizer, config, batches_per_epoch):
    total = config['epochs'] * batches_per_epoch
    warmup = min(config['warmup_epochs'] * batches_per_epoch, max(0, total - 1))
    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return config['min_lr_ratio'] + (1 - config['min_lr_ratio']) * (1 + math.cos(math.pi * progress)) / 2
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def metrics_from_confusion(confusion):
    matrix = confusion.double()
    intersection = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - intersection
    valid = union > 0
    iou = torch.where(valid, intersection / union.clamp_min(1), float('nan'))
    return dict(miou=iou[valid].mean().item(),
                accuracy=(intersection.sum() / matrix.sum().clamp_min(1)).item(),
                **{f'iou_{name}': value.item() for name, value in zip(CLASS_NAMES, iou)})


def run_epoch(model, loader, criterion, device, amp_dtype=None, optimizer=None,
              scaler=None, scheduler=None, grad_clip=1.0):
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((4, 4), dtype=torch.int64)
    total_loss, total_images = 0.0, 0
    progress = tqdm(loader, desc='Train' if training else 'Val', leave=False)
    with torch.set_grad_enabled(training):
        for images, targets in progress:
            images, targets = images.to(device), targets.to(device)
            valid = targets != 255
            if not valid.any():
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                # interface() is differentiable; the new trainer supplies Dice+CE externally.
                logits = model.interface(images)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; inspect image intensities and masks')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None and scaler.get_scale() >= previous_scale:
                    scheduler.step()
            total_loss += loss.detach().item() * images.shape[0]
            total_images += images.shape[0]
            predictions = logits.detach().argmax(1)
            indices = targets[valid] * 4 + predictions[valid]
            confusion += torch.bincount(indices, minlength=16).reshape(4, 4).cpu()
            progress.set_postfix(loss=f'{loss.item():.4f}')
    if not total_images:
        raise ValueError('The data loader has no samples with valid labels')
    return dict(loss=total_loss / total_images, **metrics_from_confusion(confusion))


def adapter_state(model):
    state = {key: value.detach().cpu() for key, value in model.decode_head.state_dict().items()}
    for prefix, layers in [('w_a', model.w_As), ('w_b', model.w_Bs)]:
        state.update({f'{prefix}_{i:03d}': layer.weight.detach().cpu()
                      for i, layer in enumerate(layers)})
    state['_metadata'] = model._checkpoint_metadata()
    return state


def save_checkpoint(state, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def random_state(generator):
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(),
                numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                torch=torch.get_rng_state(), loader=generator.get_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_random_state(state, generator):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def train(config, resume=None):
    seed_everything(config['seed'])
    output_dir = Path(config['output_dir'])
    if resume is None and (output_dir / 'last.pth').exists():
        raise FileExistsError(f'Run already exists: {output_dir}; use --resume or a new output_dir')
    if resume is not None and Path(resume).resolve().parent != output_dir.resolve():
        raise ValueError('For resume, output_dir must be the directory containing last.pth')
    samples = collect_samples(config)
    train_samples, val_samples = grouped_split(samples, config['val_fraction'], config['seed'])
    split = dict(train=[sample.record() for sample in train_samples],
                 val=[sample.record() for sample in val_samples])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('mos2_train')
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    for handler in [logging.StreamHandler(), logging.FileHandler(output_dir / 'train.log')]:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    for source in ('part1', 'part2', 'part3'):
        logger.info('%s: %d train, %d val patches', source,
                    sum(sample.source == source for sample in train_samples),
                    sum(sample.source == source for sample in val_samples))
    device_name = config['device']
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    generator = torch.Generator().manual_seed(config['seed'])
    loader_kwargs = dict(batch_size=config['batch_size'], num_workers=config['num_workers'],
                         pin_memory=device.type == 'cuda', worker_init_fn=seed_worker)
    train_loader = DataLoader(MoS2Dataset(train_samples, True, config['image_size']),
                              shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(MoS2Dataset(val_samples, False, config['image_size']),
                            shuffle=False, **loader_kwargs)
    model = PDZSeg(backbone_size=config['backbone_size'], r=config['lora_rank'],
                   image_shape=config['image_size'], num_classes=4,
                   decode_type=config['decode_type'], decoder_type=config['decoder_type'],
                   use_cls_token=config['use_cls_token'], decoder_channels=config['decoder_channels'],
                   backbone_repo=config['backbone_repo'], backbone_weights=config['backbone_weights'],
                   pretrained=config['pretrained']).to(device)
    criterion = DiceCELoss(config['class_weights'], config['ce_weight'], config['dice_weight'],
                           config['label_smoothing']).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    amp_dtype = None
    if config['amp'] and device.type == 'cuda':
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)
    logger.info('device=%s, AMP=%s, trainable parameters=%s', device, amp_dtype,
                f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    start_epoch, best_miou, best_epoch, stale_epochs = 0, -math.inf, 0, 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        state = checkpoint['_training_state']
        if state['split'] != split:
            raise ValueError('Data split changed; restore the original data paths and seed before resuming')
        for name in ('image_size', 'point_radius', 'simulated_scale_factor', 'atomcount_scale_factors',
                     'class_weights', 'ce_weight', 'dice_weight', 'label_smoothing',
                     'backbone_weights', 'pretrained'):
            if config[name] != state['config'][name]:
                raise ValueError(f'Resume config differs from the saved run: {name}')
        model.load_parameters(resume, device='cpu')
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start_epoch, best_miou = state['epoch'], state['best_miou']
        best_epoch, stale_epochs = state['best_epoch'], state['stale_epochs']
        restore_random_state(state['random'], generator)
        logger.info('Resuming after epoch %d', start_epoch)
    (output_dir / 'config.json').write_text(json.dumps(config, indent=2))
    (output_dir / 'split.json').write_text(json.dumps(split, indent=2))
    csv_path = output_dir / 'training_log.csv'
    columns = ['epoch', 'train_loss', 'val_loss', 'val_miou', 'val_accuracy']
    columns += ['val_iou_' + name for name in CLASS_NAMES]
    columns += ['decoder_lr', 'lora_lr', 'seconds']
    with csv_path.open('a' if resume else 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if stream.tell() == 0:
            writer.writeheader()
        for epoch in range(start_epoch, config['epochs']):
            started = time.perf_counter()
            training = run_epoch(model, train_loader, criterion, device, amp_dtype,
                                  optimizer, scaler, scheduler, config['grad_clip'])
            validation = run_epoch(model, val_loader, criterion, device, amp_dtype)
            row = dict(epoch=epoch + 1, train_loss=training['loss'], val_loss=validation['loss'],
                       val_miou=validation['miou'], val_accuracy=validation['accuracy'])
            row.update({'val_iou_' + name: validation['iou_' + name] for name in CLASS_NAMES})
            row.update({kind + '_lr': next(g['lr'] for g in optimizer.param_groups if g['name'] == kind)
                        for kind in ('decoder', 'lora')})
            row['seconds'] = time.perf_counter() - started
            writer.writerow(row)
            stream.flush()
            logger.info('Epoch %d/%d: train_loss=%.5f val_loss=%.5f mIoU=%.4f accuracy=%.4f',
                        epoch + 1, config['epochs'], training['loss'], validation['loss'],
                        validation['miou'], validation['accuracy'])
            state = adapter_state(model)
            if validation['miou'] > best_miou:
                best_miou, best_epoch, stale_epochs = validation['miou'], epoch + 1, 0
                save_checkpoint(state, output_dir / 'best.pth')
            else:
                stale_epochs += 1
            state['_training_state'] = dict(epoch=epoch + 1, best_miou=best_miou,
                best_epoch=best_epoch, stale_epochs=stale_epochs, config=config, split=split,
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(), random=random_state(generator))
            save_checkpoint(state, output_dir / 'last.pth')
            if config['patience'] and stale_epochs >= config['patience']:
                logger.info('Early stopping after %d epochs without improvement', stale_epochs)
                break
    summary = dict(best_epoch=best_epoch, best_miou=best_miou, output_dir=str(output_dir))
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    logger.info('Best mIoU=%.4f at epoch %d; weights: %s', best_miou, best_epoch, output_dir / 'best.pth')
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=Path(__file__).with_name('config.json'))
    parser.add_argument('--resume', default=None, help='Continue from this run\'s last.pth')
    args = parser.parse_args()
    train(load_config(args.config), args.resume)


if __name__ == '__main__':
    main()
