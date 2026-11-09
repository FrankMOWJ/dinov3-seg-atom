"""Load official DINOv3 without requiring its downstream evaluation stack."""

from pathlib import Path
import argparse
import sys
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError, URLError
import warnings

import torch
import os

CHECKPOINT_FILENAMES = {
    'dinov3_vits16': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
    'dinov3_vitb16': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    'dinov3_vitl16': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
}


def load_dinov3(repo, model_name, **kwargs):
    repo = str(repo)
    source = 'local' if Path(repo).is_dir() else 'github'
    try:
        return torch.hub.load(repo_or_dir=repo, model=model_name, source=source, **kwargs)
    except ModuleNotFoundError as exc:
        # Some official revisions import segmentation evaluation modules from
        # hubconf.py, even for a backbone-only request. The official backbone
        # factories are imported before those optional modules. Use that exact
        # factory rather than requiring torchmetrics/omegaconf/etc. for inference.
        factories = sys.modules.get('dinov3.hub.backbones')
        if (exc.name in ('torch', 'numpy', 'dinov3') or factories is None
                or not hasattr(factories, model_name)):
            raise
        warnings.warn(
            f'DINOv3 Hub imported optional dependency {exc.name!r}; '
            'loading the official backbone factory directly.',
            RuntimeWarning, stacklevel=2,
        )
        return getattr(factories, model_name)(**kwargs)


def download_dinov3(repo, model_name, save_dir, weights=None, overwrite=False):
    """Load pretrained weights and export a CPU state_dict for offline reuse.

    weights accepts an authorized official URL or an existing checkpoint path.
    With no override, the official factory attempts its default weight download.
    Preserve an explicit source filename, including the hash used by ViT-L.
    """
    if model_name not in CHECKPOINT_FILENAMES:
        raise ValueError(f'Unsupported model: {model_name}')
    filename = CHECKPOINT_FILENAMES[model_name]
    if weights is not None:
        source_name = Path(unquote(urlparse(str(weights)).path)).name
        if source_name.endswith('.pth'):
            filename = source_name
    destination = Path(save_dir).expanduser().resolve() / filename
    if destination.exists() and not overwrite:
        raise FileExistsError(f'{destination} already exists; use --overwrite to replace it')

    kwargs = {'pretrained': True}
    if weights is not None:
        kwargs['weights'] = str(weights)
    print(f'Loading pretrained {model_name}...', flush=True)
    model = load_dinov3(repo, model_name, **kwargs).cpu()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    print(f'Saved backbone weights: {destination}', flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description='Download an official DINOv3 backbone to a local directory.')
    parser.add_argument('--model_name', default='dinov3_vitb16', choices=CHECKPOINT_FILENAMES)
    parser.add_argument('--repo', default='facebookresearch/dinov3',
                        help='GitHub repository or local official DINOv3 clone.')
    parser.add_argument('--weights', default=None,
                        help='Authorized official checkpoint URL or an existing local .pth file.')
    parser.add_argument('--save_dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'checkpoints' / 'dinov3',
                        help='Directory for the exported backbone checkpoint.')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing output checkpoint.')
    args = parser.parse_args()
    try:
        download_dinov3(args.repo, args.model_name, args.save_dir,
                        weights=args.weights, overwrite=args.overwrite)
    except FileExistsError as exc:
        parser.exit(1, f'{exc}\n')
    except HTTPError as exc:
        parser.exit(1, f'Download failed (HTTP {exc.code}). '
                       'Check access to the official weights and pass an authorized URL with --weights.\n')
    except URLError:
        parser.exit(1, 'Download failed. Check the network connection and the --weights URL.\n')


if __name__ == '__main__':
    '''
    python arch/backbone.py \
        --model_name dinov3_vitb16 \
        --weights \
        --save_dir /home/cpsc4880_wm455/scratch_cpsc4880/cpsc4880_wm455/checkpoints/dinov3
    
    '''
    main()
