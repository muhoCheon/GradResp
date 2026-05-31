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

EVAL_API_ROOT = {
    'cifar10': 'cifar10_resnet18_32x32_base_e100_lr0.1_default',
    'cifar100': 'cifar100_resnet18_32x32_base_e100_lr0.1_default',
    'imagenet': 'imagenet_resnet50_tvsv1_base_default',
    'imagenet200': 'imagenet200_resnet18_224x224_base_e90_lr0.1_default',
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

SPLITS = ['OOD', 'FSOOD']
GROUPS = ['Near', 'Far']
METRICS = ['AUROC', 'FPR95']


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
    return read_metric_rows(csv_path)[row_name]['AUROC']['mean']


def parse_metric_value(value):
    value = str(value).strip()
    if not value or value == '-':
        return {'mean': '-', 'std': '-'}

    if '±' in value:
        mean, std = value.split('±', 1)
        return {'mean': mean.strip(), 'std': std.strip()}

    if '+/-' in value:
        mean, std = value.split('+/-', 1)
        return {'mean': mean.strip(), 'std': std.strip()}

    return {'mean': value, 'std': '-'}


def normalize_metric_name(name):
    name = name.strip()
    if name == 'FPR@95':
        return 'FPR95'
    return name


def row_dataset(row):
    for key in ('dataset', '', None):
        if key in row and row[key]:
            return row[key].strip()
    raise ValueError(f'dataset column not found in row: {row}')


def read_metric_rows(csv_path):
    rows = {}
    with csv_path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row_dataset(row)
            rows[name] = {}
            for key, value in row.items():
                if key in ('dataset', '', None):
                    continue
                metric = normalize_metric_name(key)
                rows[name][metric] = parse_metric_value(value)
    return rows


def metric_value(rows, group, metric, part='mean'):
    row_name = f'{group.lower()}ood'
    if row_name not in rows:
        raise ValueError(f'{row_name} row not found')
    if metric not in rows[row_name]:
        raise ValueError(f'{metric} column not found in {row_name} row')
    return rows[row_name][metric].get(part, '-')


def output_paths(results_root, method, dataset):
    if dataset not in OUTPUT_PREFIX:
        raise ValueError(f'Unknown dataset: {dataset}')

    prefix = OUTPUT_PREFIX[dataset]
    outputs_root = Path(results_root) / 'outputs'
    ood_csv = outputs_root / f'{prefix}_test_ood_ood_{method}_0' / 's0' / 'ood' / 'ood.csv'
    fsood_csv = outputs_root / f'{prefix}_test_ood_fsood_{method}_0' / 'ood.csv'
    return ood_csv, fsood_csv


def eval_api_paths(method, dataset):
    dataset_root = EVAL_API_ROOT.get(dataset)
    if dataset_root is None:
        return None, None

    root = Path('results') / dataset_root
    return root / 'ood' / f'{method}.csv', root / 'fsood' / f'{method}.csv'


def collect_source_metrics(source, ood_csv, fsood_csv, require=True):
    if ood_csv is None or fsood_csv is None:
        return {}
    if not ood_csv.exists():
        if require:
            raise FileNotFoundError(ood_csv)
        return {}
    if not fsood_csv.exists():
        if require:
            raise FileNotFoundError(fsood_csv)
        return {}

    split_rows = {
        'OOD': read_metric_rows(ood_csv),
        'FSOOD': read_metric_rows(fsood_csv),
    }
    metrics = {}
    for split in SPLITS:
        rows = split_rows[split]
        for group in GROUPS:
            for metric in METRICS:
                mean_key = f'{source} {split} {group} {metric}'
                metrics[mean_key] = metric_value(rows, group, metric, 'mean')
                std = metric_value(rows, group, metric, 'std')
                if std != '-':
                    metrics[f'{mean_key} std'] = std
    return metrics


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

    runtime = read_runtime(log_path)
    main_ood_csv, main_fsood_csv = output_paths(results_root, method, dataset)
    main_metrics = collect_source_metrics(
        'main', main_ood_csv, main_fsood_csv, require=True)
    eval_ood_csv, eval_fsood_csv = eval_api_paths(method, dataset)
    eval_metrics = collect_source_metrics(
        'eval_api', eval_ood_csv, eval_fsood_csv, require=dataset != 'mnist')

    dataset_label = DATASET_LABELS[dataset]
    method_label = METHOD_LABELS.get(method, method.upper())
    eval_api_status = '-' if dataset == 'mnist' else args.status
    metric_lines = []
    for metrics in (main_metrics, eval_metrics):
        for key in sorted(metrics):
            metric_lines.append(f'- {key}: {metrics[key]}')

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
- eval_api OOD: {eval_api_status}
- eval_api FSOOD: {eval_api_status}
{chr(10).join(metric_lines)}
"""

    metadata_path.write_text(content)
    print(metadata_path)


if __name__ == '__main__':
    main()
