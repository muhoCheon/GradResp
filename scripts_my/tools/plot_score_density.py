#!/usr/bin/env python
"""Plot saved OpenOOD score distributions as horizontal violin plots."""

import argparse
import csv
import os
import warnings
from pathlib import Path

if 'MPLCONFIGDIR' not in os.environ:
    mpl_cache = Path('/tmp/matplotlib-cache')
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(mpl_cache)
if 'XDG_CACHE_HOME' not in os.environ:
    xdg_cache = Path('/tmp/xdg-cache')
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ['XDG_CACHE_HOME'] = str(xdg_cache)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


DATASETS = ['mnist', 'cifar10', 'cifar100', 'imagenet', 'imagenet200']

METHODS = [
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
    'residual',
    'react',
    'ash',
    'dice',
    'gram',
    'klm',
    'she',
    'scale',
    'adascale_a',
    'adascale_l',
]

SCHEMES = ['ood', 'fsood']

OUTPUT_PREFIX = {
    'mnist': 'mnist_lenet',
    'cifar10': 'cifar10_resnet18_32x32',
    'cifar100': 'cifar100_resnet18_32x32',
    'imagenet': 'imagenet_resnet50',
    'imagenet200': 'imagenet200_resnet18_224x224',
}

ID_SCORE_NAME = {
    'mnist': 'mnist',
    'cifar10': 'cifar10',
    'cifar100': 'cifar100',
    'imagenet': 'imagenet',
    'imagenet200': 'imagenet200',
}

SPLIT_NAMES = {
    'mnist': {
        'csID': {'svhn'},
        'nearOOD': {'notmnist', 'fashionmnist'},
        'farOOD': {'texture', 'cifar10', 'tin', 'places365'},
    },
    'cifar10': {
        'csID': {'cinic10'},
        'nearOOD': {'cifar100', 'tin'},
        'farOOD': {'mnist', 'svhn', 'texture', 'place365'},
    },
    'cifar100': {
        'csID': {'cifar100c'},
        'nearOOD': {'cifar10', 'tin'},
        'farOOD': {'mnist', 'svhn', 'texture', 'places365'},
    },
    'imagenet': {
        'csID': {'imagenetv2', 'imagenetc', 'imagenetr'},
        'nearOOD': {'ssb_hard', 'ninco'},
        'farOOD': {'inaturalist', 'textures', 'openimageo'},
    },
    'imagenet200': {
        'csID': {'imagenetv2', 'imagenetc', 'imagenetr'},
        'nearOOD': {'ssb_hard', 'ninco'},
        'farOOD': {'inaturalist', 'textures', 'openimageo'},
    },
}

SPLIT_ORDER = ['ID', 'csID', 'nearOOD', 'farOOD', 'unknown']
LEGEND_ORDER = ['ID', 'ID+csID', 'csID', 'nearOOD', 'farOOD', 'OOD', 'unknown']
SPLIT_PALETTE = {
    'ID': '#1f77b4',
    'ID+csID': '#1f77b4',
    'OOD': '#d62728',
    'csID': '#2ca02c',
    'nearOOD': '#ff7f0e',
    'farOOD': '#d62728',
    'unknown': '#7f7f7f',
}

WARN_METHODS = {'dsvdd', 'rts', 'rts_var', 'rts-var'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=DATASETS + ['all'])
    parser.add_argument('--method', required=True, choices=METHODS + ['all'])
    parser.add_argument('--scheme', required=True, choices=SCHEMES + ['all'])
    parser.add_argument('--results-root', default='results_test')
    parser.add_argument(
        '--output-dir',
        default='results_test/plots/score_density',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=0,
        help='Per-distribution subsample size. 0 means use all scores.',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dpi', type=int, default=160)
    parser.add_argument('--no-box', action='store_true')
    return parser.parse_args()


def expand_choice(value, choices):
    return choices if value == 'all' else [value]


def score_dir(results_root, dataset, method, scheme):
    root = Path(results_root) / 'outputs'
    prefix = OUTPUT_PREFIX[dataset]
    if scheme == 'ood':
        return root / f'{prefix}_test_ood_ood_{method}_0' / 's0' / 'ood' / 'scores'
    return root / f'{prefix}_test_ood_fsood_{method}_0' / 'scores'


def metric_paths(results_root, dataset, method, scheme):
    root = Path(results_root) / 'outputs'
    prefix = OUTPUT_PREFIX[dataset]
    if scheme == 'ood':
        base = root / f'{prefix}_test_ood_ood_{method}_0' / 's0' / 'ood'
    else:
        base = root / f'{prefix}_test_ood_fsood_{method}_0'
    return base / 'ood.csv', base / 'csid.csv'


def read_metric_csv(path):
    if not path.exists():
        return {}
    metrics = {}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            name = row.get('dataset')
            if name:
                metrics[name] = row
    return metrics


def read_metrics(results_root, dataset, method, scheme):
    ood_csv, csid_csv = metric_paths(results_root, dataset, method, scheme)
    metrics = read_metric_csv(ood_csv)
    metrics.update(read_metric_csv(csid_csv))
    if metrics:
        for row in metrics.values():
            acc = row.get('ACC')
            if acc and acc != '-':
                metrics.setdefault('__summary__', {})['ACC'] = acc
                break
    return metrics


def metric_value(metrics, name, key):
    value = metrics.get(name, {}).get(key)
    if value is None or value == '-':
        return None
    return value


def metric_suffix(metrics, split, name):
    if split in {'nearOOD', 'farOOD', 'unknown'}:
        auroc = metric_value(metrics, name, 'AUROC')
        if auroc:
            return f', AUROC={auroc}'
    if split == 'csID':
        acc = metric_value(metrics, name, 'ACC')
        if acc:
            return f', ACC={acc}'
    if split == 'ID':
        acc = metric_value(metrics, '__summary__', 'ACC')
        if acc:
            return f', ACC={acc}'
    return ''


def summary_text(metrics):
    parts = []
    near = metric_value(metrics, 'nearood', 'AUROC')
    far = metric_value(metrics, 'farood', 'AUROC')
    acc = metric_value(metrics, '__summary__', 'ACC')
    if near:
        parts.append(f'Near AUROC={near}')
    if far:
        parts.append(f'Far AUROC={far}')
    if acc:
        parts.append(f'ACC={acc}')
    return ', '.join(parts)


def split_name(dataset, score_name):
    if score_name == ID_SCORE_NAME[dataset]:
        return 'ID'
    for split, names in SPLIT_NAMES[dataset].items():
        if score_name in names:
            return split
    return 'unknown'


def split_rank(split):
    try:
        return SPLIT_ORDER.index(split)
    except ValueError:
        return len(SPLIT_ORDER)


def load_ood_score(path):
    with np.load(path) as data:
        if 'conf' not in data:
            raise KeyError(f'{path} does not contain a conf array')
        conf = np.asarray(data['conf'], dtype=np.float64)
    score = -conf
    return score[np.isfinite(score)]


def maybe_subsample(scores, max_samples, rng):
    if max_samples <= 0 or scores.size <= max_samples:
        return scores
    indices = rng.choice(scores.size, size=max_samples, replace=False)
    return scores[indices]


def read_score_map(directory, args, rng):
    score_map = {}
    for path in sorted(directory.glob('*.npz')):
        scores = load_ood_score(path)
        scores = maybe_subsample(scores, args.max_samples, rng)
        if scores.size == 0:
            warnings.warn(f'Skipping empty score file: {path}')
            continue
        score_map[path.stem] = scores
    return score_map


def common_xlim(score_map):
    all_scores = np.concatenate(list(score_map.values()))
    return score_xlim(all_scores)


def score_xlim(all_scores):
    x_min = float(np.min(all_scores))
    x_max = float(np.max(all_scores))
    if x_min == x_max:
        pad = max(abs(x_min) * 0.05, 1.0)
    else:
        pad = (x_max - x_min) * 0.05
    return x_min - pad, x_max + pad


def shared_scheme_xlim(args, dataset, method):
    parts = []
    for scheme in SCHEMES:
        directory = score_dir(args.results_root, dataset, method, scheme)
        if not directory.exists():
            continue
        for path in sorted(directory.glob('*.npz')):
            try:
                scores = load_ood_score(path)
            except KeyError as exc:
                warnings.warn(str(exc))
                continue
            if scores.size:
                parts.append(scores)
    if not parts:
        return None
    return score_xlim(np.concatenate(parts))


def format_label(name, detail):
    return f'{name}\n({detail})'


def violinplot(records, output_path, title, subtitle, args, xlim):
    if not records:
        warnings.warn(f'No records to plot: {output_path}')
        return

    labels = [record['label'] for record in records]
    values = [record['scores'] for record in records]
    colors = [SPLIT_PALETTE.get(record['split'], SPLIT_PALETTE['unknown']) for record in records]

    row_count = len(records)
    height = max(3.0, 0.52 * row_count + 1.5)
    fig, ax = plt.subplots(figsize=(6.0, height))

    positions = np.arange(len(records))
    draw_half_violins(ax, values, positions, colors)

    if not args.no_box:
        box_positions = positions + 0.18
        box = ax.boxplot(
            values,
            positions=box_positions,
            vert=False,
            widths=0.13,
            patch_artist=True,
            showfliers=False,
            medianprops={'color': 'black', 'linewidth': 1.3},
            boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 0.8, 'alpha': 0.85},
            whiskerprops={'color': 'black', 'linewidth': 0.8},
            capprops={'color': 'black', 'linewidth': 0.8},
        )
        for patch in box['boxes']:
            patch.set_alpha(0.85)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(row_count - 0.45, -0.55)
    ax.set_xlim(xlim)
    ax.set_xlabel('ood_score = -conf')
    ax.xaxis.labelpad = 3
    ax.set_ylabel('dataset')
    full_title = title if not subtitle else f'{title}\n{subtitle}'
    ax.set_title(full_title, fontsize=11, linespacing=1.25)
    ax.grid(axis='x', alpha=0.25)

    legend_handles = []
    for split in LEGEND_ORDER:
        if split in {record['split'] for record in records}:
            handle = plt.Line2D(
                [0],
                [0],
                color=SPLIT_PALETTE.get(split, SPLIT_PALETTE['unknown']),
                lw=6,
                alpha=0.65,
                label=split,
            )
            legend_handles.append(handle)
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.015),
            ncol=len(legend_handles),
            fontsize=8,
            frameon=False,
        )

    bottom_margin = 0.12 if row_count <= 3 else 0.08
    fig.tight_layout(rect=(0, bottom_margin, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)


def draw_half_violins(ax, values, positions, colors):
    half_width = 0.38
    for scores, pos, color in zip(values, positions, colors):
        scores = np.asarray(scores)
        if scores.size == 0:
            continue

        x_min = float(np.min(scores))
        x_max = float(np.max(scores))
        if x_min == x_max:
            ax.vlines(
                x_min,
                pos - half_width,
                pos,
                color=color,
                alpha=0.7,
                linewidth=2.0,
            )
            continue

        x_pad = (x_max - x_min) * 0.03
        xs = np.linspace(x_min - x_pad, x_max + x_pad, 256)
        kde = gaussian_kde(scores)
        density = kde(xs)
        max_density = float(np.max(density))
        if max_density <= 0:
            continue

        ys = pos - (density / max_density) * half_width
        ax.fill_between(
            xs,
            pos,
            ys,
            facecolor=color,
            edgecolor='black',
            alpha=0.55,
            linewidth=0.5,
        )
        ax.plot(xs, ys, color='black', alpha=0.55, linewidth=0.45)
        ax.hlines(pos, xs[0], xs[-1], color='black', alpha=0.25, linewidth=0.4)


def per_dataset_records(dataset, score_map, metrics):
    records = []
    for name, scores in score_map.items():
        split = split_name(dataset, name)
        suffix = metric_suffix(metrics, split, name)
        records.append({
            'name': name,
            'split': split,
            'label': format_label(f'{split} / {name}', f'n={scores.size}{suffix}'),
            'scores': scores,
        })
    return sorted(records, key=lambda r: (split_rank(r['split']), r['name']))


def combined_eval_group_records(dataset, score_map):
    id_parts = []
    near_parts = []
    far_parts = []
    for name, scores in score_map.items():
        split = split_name(dataset, name)
        if split in {'ID', 'csID'}:
            id_parts.append(scores)
        elif split == 'nearOOD':
            near_parts.append(scores)
        elif split == 'farOOD':
            far_parts.append(scores)

    records = []
    if id_parts:
        scores = np.concatenate(id_parts)
        label = 'ID+csID' if any(split_name(dataset, name) == 'csID' for name in score_map) else 'ID'
        records.append({'split': 'ID+csID', 'label': format_label(label, f'n={scores.size}'), 'scores': scores})
    if near_parts:
        scores = np.concatenate(near_parts)
        records.append({'split': 'nearOOD', 'label': format_label('nearOOD', f'n={scores.size}'), 'scores': scores})
    if far_parts:
        scores = np.concatenate(far_parts)
        records.append({'split': 'farOOD', 'label': format_label('farOOD', f'n={scores.size}'), 'scores': scores})
    return records


def combined_id_group_records(dataset, score_map):
    grouped = {split: [] for split in SPLIT_ORDER}
    for name, scores in score_map.items():
        grouped.setdefault(split_name(dataset, name), []).append(scores)

    records = []
    for split in SPLIT_ORDER:
        parts = grouped.get(split, [])
        if not parts:
            continue
        scores = np.concatenate(parts)
        records.append({'split': split, 'label': format_label(split, f'n={scores.size}'), 'scores': scores})
    return records


def plot_one(args, dataset, method, scheme, xlim=None):
    directory = score_dir(args.results_root, dataset, method, scheme)
    if not directory.exists():
        warnings.warn(f'Score directory not found, skipping: {directory}')
        return False

    if method in WARN_METHODS:
        warnings.warn(
            f'{method} may not follow the standard convention that larger conf means ID. '
            'Still plotting ood_score = -conf.'
        )

    rng = np.random.default_rng(args.seed)
    score_map = read_score_map(directory, args, rng)
    metrics = read_metrics(args.results_root, dataset, method, scheme)
    id_name = ID_SCORE_NAME[dataset]
    if id_name not in score_map:
        warnings.warn(f'ID score file not found, skipping: {directory / (id_name + ".npz")}')
        return False

    output_dir = Path(args.output_dir) / dataset / method / scheme
    title_prefix = f'{dataset} / {method} / {scheme}'
    summary = summary_text(metrics)
    xlim = xlim or common_xlim(score_map)

    outputs = [
        (
            per_dataset_records(dataset, score_map, metrics),
            output_dir / 'per_dataset.png',
            f'{title_prefix}: per dataset',
            summary,
        ),
        (
            combined_eval_group_records(dataset, score_map),
            output_dir / 'combined_eval_groups.png',
            f'{title_prefix}: ID+csID vs nearOOD vs farOOD',
            summary,
        ),
        (
            combined_id_group_records(dataset, score_map),
            output_dir / 'combined_id_groups.png',
            f'{title_prefix}: ID vs csID vs nearOOD vs farOOD',
            summary,
        ),
    ]
    for records, output_path, title, subtitle in outputs:
        violinplot(records, output_path, title, subtitle, args, xlim)
        print(output_path)
    return True


def main():
    args = parse_args()
    plt.style.use('default')

    datasets = expand_choice(args.dataset, DATASETS)
    methods = expand_choice(args.method, METHODS)
    schemes = expand_choice(args.scheme, SCHEMES)

    plotted = 0
    for dataset in datasets:
        for method in methods:
            xlim = shared_scheme_xlim(args, dataset, method)
            for scheme in schemes:
                if plot_one(args, dataset, method, scheme, xlim=xlim):
                    plotted += 1
    print(f'plotted combinations: {plotted}')


if __name__ == '__main__':
    main()
