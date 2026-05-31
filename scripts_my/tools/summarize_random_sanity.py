#!/usr/bin/env python
"""Summarize random sanity scores as ID-vs-dataset checks."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


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
SPLIT_ORDER = {'ID': 0, 'csID': 1, 'nearOOD': 2, 'farOOD': 3, 'unknown': 99}
DATASET_SPLITS = {
    'cifar10': {
        'ID': ['cifar10'],
        'csID': ['cifar10c'],
        'nearOOD': ['cifar100', 'tin'],
        'farOOD': ['mnist', 'svhn', 'texture', 'places365'],
    },
    'cifar100': {
        'ID': ['cifar100'],
        'csID': ['cifar100c'],
        'nearOOD': ['cifar10', 'tin'],
        'farOOD': ['mnist', 'svhn', 'texture', 'places365'],
    },
    'imagenet': {
        'ID': ['imagenet'],
        'csID': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
        'nearOOD': ['ssb_hard', 'ninco'],
        'farOOD': ['inaturalist', 'textures', 'openimage_o'],
    },
    'imagenet200': {
        'ID': ['imagenet200'],
        'csID': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
        'nearOOD': ['ssb_hard', 'ninco'],
        'farOOD': ['inaturalist', 'textures', 'openimage_o'],
    },
}
OLD_SUMMARY_GLOBS = [
    '*_detection_metrics_by_seed.csv',
    '*_detection_metrics_summary.csv',
    '*_score_distribution_by_seed.csv',
    '*_score_distribution_summary.csv',
    'random_sanity_*_per_seed.csv',
    'random_sanity_*_summary.csv',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', default='results_test/random_sanity')
    parser.add_argument('--methods', default='msp')
    parser.add_argument('--datasets', default='all')
    parser.add_argument('--seeds', default='0,1,2,3,4')
    return parser.parse_args()


def parse_csv_list(value):
    if value == 'all':
        return None
    return [item.strip() for item in value.split(',') if item.strip()]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path):
    with np.load(path) as data:
        return {
            'pred': np.asarray(data['pred'], dtype=np.int64),
            'conf': np.asarray(data['conf'], dtype=np.float64),
            'label': np.asarray(data['label'], dtype=np.int64),
        }


def split_name(dataset, score_dataset):
    for split, names in DATASET_SPLITS[dataset].items():
        if score_dataset in names:
            return split
    return 'unknown'


def dataset_rank(dataset, split, score_dataset):
    names = DATASET_SPLITS.get(dataset, {}).get(split, [])
    try:
        return names.index(score_dataset)
    except ValueError:
        return 9_999


def sort_key(row):
    return (
        row['dataset'],
        row['method'],
        int(row['seed']) if str(row['seed']).isdigit() else row['seed'],
        SPLIT_ORDER.get(row['split'], 99),
        dataset_rank(row['dataset'], row['split'], row['score_dataset']),
        row['score_dataset'],
    )


def summary_sort_key(item):
    (dataset, method, split, score_dataset), _ = item
    return (
        dataset,
        method,
        SPLIT_ORDER.get(split, 99),
        dataset_rank(dataset, split, score_dataset),
        score_dataset,
    )


def format_float(value, precision=6):
    if value is None or not np.isfinite(value):
        return '-'
    return f'{float(value):.{precision}f}'


def format_mean_std(values, precision=6):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return '-', '-'
    return (
        f'{float(np.mean(finite)):.{precision}f}',
        f'{float(np.std(finite)):.{precision}f}',
    )


def classification_acc(pred, label):
    if pred.size == 0 or label.size == 0:
        return np.nan
    return float(np.mean(pred == label) * 100)


def openood_auroc(id_conf, target_conf):
    """Return OpenOOD-style AUROC with target as OOD-positive using -conf."""
    id_conf = np.asarray(id_conf, dtype=np.float64)
    target_conf = np.asarray(target_conf, dtype=np.float64)
    id_conf = id_conf[np.isfinite(id_conf)]
    target_conf = target_conf[np.isfinite(target_conf)]
    n_id = id_conf.size
    n_target = target_conf.size
    if n_id == 0 or n_target == 0:
        return np.nan

    # OpenOOD treats OOD as the positive class and uses -conf because larger
    # conf means more ID-like.
    scores = -np.concatenate([id_conf, target_conf])
    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)

    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Average rank in 1-based ranks for tied scores.
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    target_rank_sum = np.sum(ranks[n_id:])
    auc = (
        target_rank_sum - n_target * (n_target + 1) / 2.0
    ) / (n_id * n_target)
    return float(auc * 100)


def score_dir_for_seed(base):
    flat_dir = base / 'scores'
    if any(flat_dir.glob('*.npz')):
        return flat_dir
    fsood_dir = base / 'scores' / 'fsood'
    if fsood_dir.exists():
        return fsood_dir
    ood_dir = base / 'scores' / 'ood'
    if ood_dir.exists():
        return ood_dir
    return None


def read_seed_rows(output_root, dataset, method, seed):
    base = output_root / 'outputs' / method / dataset / f'seed_{seed}'
    score_dir = score_dir_for_seed(base)
    if score_dir is None:
        return [], []

    id_path = score_dir / f'{dataset}.npz'
    if not id_path.exists():
        return [], []

    id_data = load_npz(id_path)
    acc_rows = [{
        'dataset': dataset,
        'method': method,
        'seed': str(seed),
        'split': 'ID',
        'score_dataset': dataset,
        'acc': format_float(classification_acc(id_data['pred'], id_data['label'])),
    }]
    auroc_rows = []

    for path in sorted(score_dir.glob('*.npz')):
        score_dataset = path.stem
        split = split_name(dataset, score_dataset)
        data = load_npz(path)

        if split == 'csID':
            acc_rows.append({
                'dataset': dataset,
                'method': method,
                'seed': str(seed),
                'split': split,
                'score_dataset': score_dataset,
                'acc': format_float(classification_acc(data['pred'], data['label'])),
            })

        if split == 'ID':
            continue

        auroc_rows.append({
            'dataset': dataset,
            'method': method,
            'seed': str(seed),
            'split': split,
            'score_dataset': score_dataset,
            'auroc': format_float(openood_auroc(id_data['conf'], data['conf'])),
        })

    return acc_rows, auroc_rows


def summarize_rows(rows, value_name, precision=6):
    grouped = defaultdict(list)
    for row in rows:
        key = (row['dataset'], row['method'], row['split'], row['score_dataset'])
        grouped[key].append(row)

    out_rows = []
    for key, group_rows in sorted(grouped.items(), key=summary_sort_key):
        dataset, method, split, score_dataset = key
        values = []
        for row in group_rows:
            value = row[value_name]
            values.append(np.nan if value == '-' else float(value))
        mean, std = format_mean_std(values, precision=precision)
        out_rows.append({
            'dataset': dataset,
            'method': method,
            'split': split,
            'score_dataset': score_dataset,
            'num_seeds': str(len(group_rows)),
            f'{value_name}_mean': mean,
            f'{value_name}_std': std,
        })
    return out_rows


def remove_old_summary_files(summary_dir, methods):
    if not summary_dir.exists():
        return
    patterns = []
    for method in methods:
        patterns.extend([
            f'{method}_classification_acc_by_seed.csv',
            f'{method}_classification_acc_summary.csv',
            f'{method}_id_vs_dataset_auroc_by_seed.csv',
            f'{method}_id_vs_dataset_auroc_summary.csv',
            f'{method}_detection_metrics_by_seed.csv',
            f'{method}_detection_metrics_summary.csv',
            f'{method}_score_distribution_by_seed.csv',
            f'{method}_score_distribution_summary.csv',
            f'random_sanity_{method}_per_seed.csv',
            f'random_sanity_{method}_summary.csv',
            f'random_sanity_{method}_score_per_seed.csv',
            f'random_sanity_{method}_score_summary.csv',
        ])
    for pattern in patterns:
        for path in summary_dir.glob(pattern):
            path.unlink()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    methods = parse_csv_list(args.methods) or SUPPORTED_METHODS
    datasets = parse_csv_list(args.datasets) or SUPPORTED_DATASETS
    seeds = [int(seed) for seed in (parse_csv_list(args.seeds) or [])]

    acc_rows = []
    auroc_rows = []
    for method in methods:
        for dataset in datasets:
            for seed in seeds:
                seed_acc_rows, seed_auroc_rows = read_seed_rows(
                    output_root, dataset, method, seed
                )
                acc_rows.extend(seed_acc_rows)
                auroc_rows.extend(seed_auroc_rows)

    summary_dir = output_root / 'summary'
    remove_old_summary_files(summary_dir, methods)

    for method in methods:
        method_acc_rows = sorted(
            [row for row in acc_rows if row['method'] == method],
            key=sort_key,
        )
        method_auroc_rows = sorted(
            [row for row in auroc_rows if row['method'] == method],
            key=sort_key,
        )

        write_csv(
            summary_dir / f'{method}_classification_acc_by_seed.csv',
            method_acc_rows,
            ['dataset', 'method', 'seed', 'split', 'score_dataset', 'acc'],
        )
        write_csv(
            summary_dir / f'{method}_classification_acc_summary.csv',
            summarize_rows(method_acc_rows, 'acc'),
            [
                'dataset',
                'method',
                'split',
                'score_dataset',
                'num_seeds',
                'acc_mean',
                'acc_std',
            ],
        )
        write_csv(
            summary_dir / f'{method}_id_vs_dataset_auroc_by_seed.csv',
            method_auroc_rows,
            ['dataset', 'method', 'seed', 'split', 'score_dataset', 'auroc'],
        )
        write_csv(
            summary_dir / f'{method}_id_vs_dataset_auroc_summary.csv',
            summarize_rows(method_auroc_rows, 'auroc'),
            [
                'dataset',
                'method',
                'split',
                'score_dataset',
                'num_seeds',
                'auroc_mean',
                'auroc_std',
            ],
        )

    print(f'acc_rows: {len(acc_rows)}')
    print(f'auroc_rows: {len(auroc_rows)}')
    print(f'summary_dir: {summary_dir}')


if __name__ == '__main__':
    main()
