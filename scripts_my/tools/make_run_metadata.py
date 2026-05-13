#!/usr/bin/env python
"""Create metadata.md for a results_test/runs/<method>/<dataset> run."""

import argparse
import csv
import re
from pathlib import Path


DATASET_LABELS = {
    'mnist': 'MNIST',
    'cifar10': 'CIFAR-10',
    'cifar100': 'CIFAR-100',
    'imagenet': 'ImageNet',
    'imagenet200': 'ImageNet-200',
}

OUTPUT_PREFIX = {
    'mnist': 'mnist_lenet',
    'cifar10': 'cifar10_resnet18_32x32',
    'cifar100': 'cifar100_resnet18_32x32',
    'imagenet': 'imagenet_resnet50',
    'imagenet200': 'imagenet200_resnet18_224x224',
}

METHOD_LABELS = {
    'msp': 'MSP',
    'mls': 'MLS',
    'ebo': 'EBO',
    'odin': 'ODIN',
    'iodin': 'IODIN',
    'gradnorm': 'GradNorm',
    'mds': 'MDS',
    'rmds': 'RMDS',
    'knn': 'KNN',
    'vim': 'ViM',
    'residual': 'Residual',
    'react': 'ReAct',
    'ash': 'ASH',
    'dice': 'DICE',
    'gram': 'Gram',
    'klm': 'KLM',
    'kl_matching': 'KLM',
    'she': 'SHE',
    'scale': 'SCALE',
    'adascale_a': 'AdaScale-A',
    'adascale_l': 'AdaScale-L',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--status', default='pass')
    parser.add_argument('--results-root', default='results_test')
    return parser.parse_args()


def read_runtime(log_path):
    pattern = re.compile(r'^RUNTIME=(.+)$')
    runtime = None
    with log_path.open() as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                runtime = match.group(1)
    if runtime is None:
        raise ValueError(f'RUNTIME line not found in {log_path}')
    return runtime


def read_auroc(csv_path, row_name):
    with csv_path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('dataset') == row_name:
                return row['AUROC']
    raise ValueError(f'{row_name} row not found in {csv_path}')


def output_paths(results_root, method, dataset):
    if dataset not in OUTPUT_PREFIX:
        raise ValueError(f'Unknown dataset: {dataset}')

    prefix = OUTPUT_PREFIX[dataset]
    outputs_root = Path(results_root) / 'outputs'
    ood_csv = outputs_root / f'{prefix}_test_ood_ood_{method}_0' / 's0' / 'ood' / 'ood.csv'
    fsood_csv = outputs_root / f'{prefix}_test_ood_fsood_{method}_0' / 'ood.csv'
    return ood_csv, fsood_csv


def main():
    args = parse_args()

    method = args.method.lower()
    dataset = args.dataset.lower()
    results_root = Path(args.results_root)
    run_dir = results_root / 'runs' / method / dataset
    command_path = run_dir / 'command.sh'
    log_path = run_dir / 'run.log'
    metadata_path = run_dir / 'metadata.md'

    if dataset not in DATASET_LABELS:
        raise ValueError(f'Unknown dataset: {dataset}')
    if not command_path.exists():
        raise FileNotFoundError(command_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    ood_csv, fsood_csv = output_paths(results_root, method, dataset)
    if not ood_csv.exists():
        raise FileNotFoundError(ood_csv)
    if not fsood_csv.exists():
        raise FileNotFoundError(fsood_csv)

    runtime = read_runtime(log_path)
    ood_near = read_auroc(ood_csv, 'nearood')
    ood_far = read_auroc(ood_csv, 'farood')
    fsood_near = read_auroc(fsood_csv, 'nearood')
    fsood_far = read_auroc(fsood_csv, 'farood')

    dataset_label = DATASET_LABELS[dataset]
    method_label = METHOD_LABELS.get(method, method.upper())
    eval_status = '-' if dataset == 'mnist' else args.status

    content = f"""# Run Metadata

- dataset: {dataset_label}
- method: {method_label}
- gpu: {args.gpu}
- status: {args.status}
- runtime: {runtime}
- command: `{command_path}`
- log: `{log_path}`
- main OOD: {args.status}
- main FSOOD: {args.status}
- eval OOD: {eval_status}
- eval FSOOD: {eval_status}
- OOD Near AUROC: {ood_near}
- OOD Far AUROC: {ood_far}
- FSOOD Near AUROC: {fsood_near}
- FSOOD Far AUROC: {fsood_far}
"""

    metadata_path.write_text(content)
    print(metadata_path)


if __name__ == '__main__':
    main()
