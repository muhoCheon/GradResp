#!/usr/bin/env python
"""Save random-initialized classifier scores for ID-vs-dataset sanity checks."""

import argparse
import hashlib
import os
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn

from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32, ResNet18_224x224, ResNet50


SUPPORTED_DATASETS = ['cifar10', 'cifar100', 'imagenet', 'imagenet200']
SUPPORTED_METHODS = [
    'msp',
    'mls',
    'ebo',
    'odin',
    'iodin',
    'gradnorm',
    'mds',
    'rmds',
    'knn',
    'vim',
    'react',
    'ash',
    'dice',
    'gram',
    'klm',
    'she',
    'scale',
]

NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'imagenet': 1000,
    'imagenet200': 200,
}

MODEL_ARCH = {
    'cifar10': ResNet18_32x32,
    'cifar100': ResNet18_32x32,
    'imagenet': ResNet50,
    'imagenet200': ResNet18_224x224,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument('--method', default='msp', choices=SUPPORTED_METHODS)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--init', default='kaiming', choices=['kaiming', 'default'])
    parser.add_argument(
        '--output-root',
        default='results_test/random_sanity',
        help='Root directory for random sanity outputs and run artifacts.',
    )
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--no-progress', action='store_true')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def apply_kaiming_init(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_random_model(dataset, init_name):
    model = MODEL_ARCH[dataset](num_classes=NUM_CLASSES[dataset])
    if init_name == 'kaiming':
        model.apply(apply_kaiming_init)
    return model


def parameter_checksum(model):
    digest = hashlib.sha256()
    with torch.no_grad():
        for tensor in model.state_dict().values():
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_npz(path, pred, conf, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pred=as_numpy(pred).astype(np.int64),
        conf=as_numpy(conf).astype(np.float64),
        label=as_numpy(label).astype(np.int64),
    )


def save_scores(evaluator, dataset, output_dir):
    score_dir = output_dir / 'scores'
    id_pred, id_conf, id_gt = evaluator.scores['id']['test']
    save_npz(score_dir / f'{dataset}.npz', id_pred, id_conf, id_gt)

    for name, values in evaluator.scores['csid'].items():
        if values is None:
            continue
        pred, conf, label = values
        save_npz(score_dir / f'{name}.npz', pred, conf, label)

    for split in ['near', 'far']:
        for name, values in evaluator.scores['ood'][split].items():
            if values is None:
                continue
            pred, conf, label = values
            label = -1 * np.ones_like(as_numpy(label), dtype=np.int64)
            save_npz(score_dir / f'{name}.npz', pred, conf, label)


def write_run_info(path, args, checksum):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '\n'.join([
            '# Random Sanity Run',
            '',
            f'- dataset: {args.dataset}',
            f'- method: {args.method}',
            f'- seed: {args.seed}',
            f'- init: {args.init}',
            f'- parameter_checksum: {checksum}',
            f'- batch_size: {args.batch_size}',
            f'- num_workers: {args.num_workers}',
            '',
        ])
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = (
        Path(args.output_root)
        / 'outputs'
        / args.method
        / args.dataset
        / f'seed_{args.seed}'
    )

    model = build_random_model(args.dataset, args.init)
    checksum = parameter_checksum(model)
    write_run_info(output_dir / 'run_info.md', args, checksum)

    model.cuda()
    model.eval()

    evaluator = Evaluator(
        model,
        id_name=args.dataset,
        data_root=str(ROOT_DIR / 'data'),
        config_root=str(ROOT_DIR / 'configs'),
        preprocessor=None,
        postprocessor_name=args.method,
        postprocessor=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    progress = not args.no_progress

    # Use the FSOOD path only as a convenient way to run inference on
    # ID, csID, nearOOD, and farOOD loaders. OpenOOD's protocol metrics
    # are intentionally not saved for this random sanity check.
    evaluator.eval_ood(fsood=True, progress=progress)
    save_scores(evaluator, args.dataset, output_dir)

    print(f'output_dir: {output_dir}')
    print(f'parameter_checksum: {checksum}')


if __name__ == '__main__':
    main()
