"""OpenOOD-compatible data and model helpers for standalone RAE."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from openood.evaluation_api.datasets import get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor

from .config import DEFAULT_CHECKPOINT, MODEL_ARCH, NUM_CLASSES, ROOT_DIR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_from_arg(device_arg: str | None = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_checkpoint(path: str | Path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def build_model(dataset: str, checkpoint: str | Path | None = None) -> torch.nn.Module:
    net = MODEL_ARCH[dataset](num_classes=NUM_CLASSES[dataset])
    checkpoint = checkpoint or DEFAULT_CHECKPOINT[dataset]
    state = load_checkpoint(checkpoint)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    net.load_state_dict(state)
    return net


def dataloader_kwargs(batch_size: int, num_workers: int) -> dict:
    kwargs = {
        'batch_size': int(batch_size),
        'shuffle': False,
        'num_workers': int(num_workers),
        'pin_memory': torch.cuda.is_available(),
    }
    if int(num_workers) > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return kwargs


def build_dataloaders(dataset: str,
                      *,
                      data_root: str | Path | None = None,
                      batch_size: int = 200,
                      num_workers: int = 4) -> Dict:
    data_root = Path(data_root) if data_root is not None else ROOT_DIR / 'data'
    preprocessor = get_default_preprocessor(dataset)
    return get_id_ood_dataloader(
        dataset,
        str(data_root),
        preprocessor,
        **dataloader_kwargs(batch_size, num_workers),
    )


def subset_loader(data_loader: DataLoader,
                  max_samples: int | None,
                  *,
                  batch_size: int | None = None,
                  num_workers: int | None = None) -> DataLoader:
    if not max_samples:
        return data_loader
    n = min(int(max_samples), len(data_loader.dataset))
    subset = Subset(data_loader.dataset, list(range(n)))
    return DataLoader(
        subset,
        **dataloader_kwargs(
            batch_size or data_loader.batch_size or n,
            num_workers if num_workers is not None else 0,
        ),
    )


def build_reference_loader(data_loader: DataLoader,
                           indices,
                           *,
                           batch_size: int,
                           num_workers: int) -> DataLoader:
    subset = Subset(data_loader.dataset, [int(idx) for idx in indices])
    return DataLoader(
        subset,
        **dataloader_kwargs(batch_size, num_workers),
    )


def split_dataloaders(dataset: str, dataloaders: Dict, scheme: str):
    splits = [('id', dataset, 'id', 'test', dataloaders['id']['test'])]
    if scheme == 'fsood':
        for name, loader in dataloaders['csid'].items():
            splits.append(('csid', name, 'csid', name, loader))
    for name, loader in dataloaders['ood']['near'].items():
        splits.append(('nearood', name, 'ood', name, loader))
    for name, loader in dataloaders['ood']['far'].items():
        splits.append(('farood', name, 'ood', name, loader))
    return splits
