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

RANDOM_SANITY_METHODS = [
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

SCHEMES = ['ood', 'fsood']
SOURCES = ['openood', 'random_sanity']

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

RANDOM_SANITY_SPLIT_NAMES = {
    'cifar10': {
        'csID': {'cifar10c'},
        'nearOOD': {'cifar100', 'tin'},
        'farOOD': {'mnist', 'svhn', 'texture', 'places365'},
    },
    'cifar100': {
        'csID': {'cifar100c'},
        'nearOOD': {'cifar10', 'tin'},
        'farOOD': {'mnist', 'svhn', 'texture', 'places365'},
    },
    'imagenet': {
        'csID': {'imagenet_v2', 'imagenet_c', 'imagenet_r'},
        'nearOOD': {'ssb_hard', 'ninco'},
        'farOOD': {'inaturalist', 'textures', 'openimage_o'},
    },
    'imagenet200': {
        'csID': {'imagenet_v2', 'imagenet_c', 'imagenet_r'},
        'nearOOD': {'ssb_hard', 'ninco'},
        'farOOD': {'inaturalist', 'textures', 'openimage_o'},
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
CROSS_CHECK_TOL = 1e-4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='openood', choices=SOURCES)
    parser.add_argument('--dataset', required=True, choices=DATASETS + ['all'])
    parser.add_argument('--method', required=True, choices=METHODS + ['all'])
    parser.add_argument('--scheme', default='all', choices=SCHEMES + ['all'])
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
    parser.add_argument('--seed', default='all')
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


def random_sanity_split_name(dataset, score_name):
    if score_name == ID_SCORE_NAME[dataset]:
        return 'ID'
    for split, names in RANDOM_SANITY_SPLIT_NAMES.get(dataset, {}).items():
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


def load_random_sanity_npz(path):
    with np.load(path) as data:
        for key in ['pred', 'conf', 'label']:
            if key not in data:
                raise KeyError(f'{path} does not contain a {key} array')
        pred = np.asarray(data['pred'], dtype=np.int64)
        conf = np.asarray(data['conf'], dtype=np.float64)
        label = np.asarray(data['label'], dtype=np.int64)
    finite = np.isfinite(conf)
    return {
        'pred': pred[finite],
        'conf': conf[finite],
        'label': label[finite],
        'score': -conf[finite],
    }


def classification_acc(pred, label):
    if pred.size == 0 or label.size == 0:
        return None
    return float(np.mean(pred == label) * 100)


def openood_auroc(id_conf, target_conf):
    id_conf = np.asarray(id_conf, dtype=np.float64)
    target_conf = np.asarray(target_conf, dtype=np.float64)
    id_conf = id_conf[np.isfinite(id_conf)]
    target_conf = target_conf[np.isfinite(target_conf)]
    n_id = id_conf.size
    n_target = target_conf.size
    if n_id == 0 or n_target == 0:
        return None

    scores = -np.concatenate([id_conf, target_conf])
    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)

    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    target_rank_sum = np.sum(ranks[n_id:])
    auc = (
        target_rank_sum - n_target * (n_target + 1) / 2.0
    ) / (n_id * n_target)
    return float(auc * 100)


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


def random_sanity_base(results_root):
    return Path(results_root) / 'random_sanity'


def random_sanity_seed_dirs(results_root, dataset, method, seed):
    base = random_sanity_base(results_root) / 'outputs' / method / dataset
    if seed == 'all':
        dirs = [
            path for path in base.glob('seed_*')
            if path.is_dir() and (path / 'scores').exists()
        ]
        return sorted(dirs, key=seed_dir_sort_key)

    directory = base / f'seed_{seed}'
    return [directory] if (directory / 'scores').exists() else []


def seed_dir_sort_key(path):
    seed = path.name.replace('seed_', '', 1)
    try:
        return 0, int(seed)
    except ValueError:
        return 1, seed


def random_sanity_read_seed(seed_dir):
    scores_dir = seed_dir / 'scores'
    data = {}
    for path in sorted(scores_dir.glob('*.npz')):
        try:
            data[path.stem] = load_random_sanity_npz(path)
        except KeyError as exc:
            warnings.warn(str(exc))
    return data


def seed_name(seed_dir):
    return seed_dir.name.replace('seed_', '', 1)


def random_sanity_collect(results_root, dataset, method, seed):
    seed_dirs = random_sanity_seed_dirs(results_root, dataset, method, seed)
    if not seed_dirs:
        return {}

    collected = {}
    for seed_dir in seed_dirs:
        seed = seed_name(seed_dir)
        seed_data = random_sanity_read_seed(seed_dir)
        for name, arrays in seed_data.items():
            collected.setdefault(name, {})[seed] = arrays
    return collected


def random_sanity_scores_for_plot(collected, args, rng):
    score_map = {}
    for name, seed_map in collected.items():
        parts = [arrays['score'] for arrays in seed_map.values()]
        if not parts:
            continue
        scores = np.concatenate(parts)
        scores = maybe_subsample(scores, args.max_samples, rng)
        if scores.size:
            score_map[name] = scores
    return score_map


def random_sanity_metric_values(dataset, collected):
    id_name = ID_SCORE_NAME[dataset]
    if id_name not in collected:
        return {}, {}

    acc = {}
    auroc = {}
    id_by_seed = collected[id_name]
    for name, seed_map in collected.items():
        split = random_sanity_split_name(dataset, name)
        if split in {'ID', 'csID'}:
            acc_values = []
            for arrays in seed_map.values():
                value = classification_acc(arrays['pred'], arrays['label'])
                if value is not None:
                    acc_values.append(value)
            if acc_values:
                acc[name] = acc_values

        if split == 'ID':
            continue

        auroc_values = []
        for seed, arrays in seed_map.items():
            id_arrays = id_by_seed.get(seed)
            if id_arrays is None:
                continue
            value = openood_auroc(id_arrays['conf'], arrays['conf'])
            if value is not None:
                auroc_values.append(value)
        if auroc_values:
            auroc[name] = auroc_values
    return acc, auroc


def format_metric_values(values, precision=2):
    if not values:
        return None
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return f'{float(values[0]):.{precision}f}'
    return f'{float(np.mean(values)):.{precision}f}+/-{float(np.std(values)):.{precision}f}'


def read_csv_rows(path):
    if not path.exists():
        warnings.warn(f'Summary CSV not found for cross-check: {path}')
        return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def csv_float(value):
    if value in {None, '', '-'}:
        return None
    return float(value)


def random_sanity_cross_check(args, dataset, method, seed, acc_values, auroc_values):
    summary_dir = random_sanity_base(args.results_root) / 'summary'
    if seed == 'all':
        acc_path = summary_dir / f'{method}_classification_acc_summary.csv'
        auroc_path = summary_dir / f'{method}_id_vs_dataset_auroc_summary.csv'
        check_summary_csv(acc_path, dataset, method, acc_values, 'acc')
        check_summary_csv(auroc_path, dataset, method, auroc_values, 'auroc')
    else:
        acc_path = summary_dir / f'{method}_classification_acc_by_seed.csv'
        auroc_path = summary_dir / f'{method}_id_vs_dataset_auroc_by_seed.csv'
        check_by_seed_csv(acc_path, dataset, method, seed, acc_values, 'acc')
        check_by_seed_csv(auroc_path, dataset, method, seed, auroc_values, 'auroc')


def check_by_seed_csv(path, dataset, method, seed, values_by_name, value_key):
    rows = read_csv_rows(path)
    if not rows:
        return
    expected = {
        row['score_dataset']: csv_float(row.get(value_key))
        for row in rows
        if row.get('dataset') == dataset
        and row.get('method') == method
        and row.get('seed') == str(seed)
    }
    for name, values in values_by_name.items():
        if not values:
            continue
        actual = float(values[0])
        compare_value(path, name, actual, expected.get(name), value_key)


def check_summary_csv(path, dataset, method, values_by_name, value_key):
    rows = read_csv_rows(path)
    if not rows:
        return
    mean_key = f'{value_key}_mean'
    std_key = f'{value_key}_std'
    expected = {
        row['score_dataset']: (csv_float(row.get(mean_key)), csv_float(row.get(std_key)))
        for row in rows
        if row.get('dataset') == dataset and row.get('method') == method
    }
    for name, values in values_by_name.items():
        if not values:
            continue
        values = np.asarray(values, dtype=np.float64)
        actual_mean = float(np.mean(values))
        actual_std = float(np.std(values))
        expected_mean, expected_std = expected.get(name, (None, None))
        compare_value(path, name, actual_mean, expected_mean, mean_key)
        compare_value(path, name, actual_std, expected_std, std_key)


def compare_value(path, name, actual, expected, label):
    if expected is None:
        warnings.warn(f'Cross-check value missing in {path}: {name} {label}')
        return
    if abs(actual - expected) > CROSS_CHECK_TOL:
        warnings.warn(
            f'Cross-check mismatch in {path}: {name} {label} '
            f'plot={actual:.6f}, summary={expected:.6f}'
        )


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
    fig, ax = plt.subplots(figsize=(7.5, height))

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
    ax.set_xlabel('ood_score = -conf (larger = more OOD-like)')
    ax.xaxis.labelpad = 3
    ax.set_ylabel('')
    full_title = title if not subtitle else f'{title}\n{subtitle}'
    title_pad = 16 if (
        getattr(args, 'source', None) == 'openood'
        and output_path.name in {'combined_eval_groups.png', 'combined_id_groups.png'}
    ) else None
    if title_pad is None:
        ax.set_title(full_title, fontsize=11, linespacing=1.25)
    else:
        ax.set_title(full_title, fontsize=11, linespacing=1.25, pad=title_pad)
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

    if getattr(args, 'source', None) == 'random_sanity':
        if output_path.name == 'combined_id_groups.png':
            fig.subplots_adjust(left=0.375, right=0.665, top=0.88, bottom=0.24)
        else:
            fig.subplots_adjust(left=0.62, right=0.91, top=0.88, bottom=0.15)
    else:
        bottom_margin = 0.18 if output_path.name == 'per_dataset.png' else 0.26
        top_margin = 0.76 if output_path.name in {
            'combined_eval_groups.png',
            'combined_id_groups.png',
        } else 0.88
        fig.subplots_adjust(left=0.335, right=0.665, top=top_margin, bottom=bottom_margin)
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
            'label': format_label(f'{name} ({split})', f'n={scores.size}{suffix}'),
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


def random_sanity_label_detail(name, split, score_map, collected, acc_values, auroc_values):
    seed_count = len(collected.get(name, {}))
    n_total = sum(arrays['score'].size for arrays in collected.get(name, {}).values())
    parts = [f'n_total={n_total}', f'seeds={seed_count}']
    acc = format_metric_values(acc_values.get(name, []))
    auroc = format_metric_values(auroc_values.get(name, []))
    if acc is not None and split in {'ID', 'csID'}:
        parts.append(f'ACC={acc}')
    if auroc is not None and split != 'ID':
        parts.append(f'AUROC={auroc}')
    return ', '.join(parts)


def random_sanity_per_dataset_records(dataset, score_map, collected, acc_values, auroc_values):
    records = []
    for name, scores in score_map.items():
        split = random_sanity_split_name(dataset, name)
        detail = random_sanity_label_detail(
            name, split, score_map, collected, acc_values, auroc_values
        )
        records.append({
            'name': name,
            'split': split,
            'label': format_label(f'{name} ({split})', detail),
            'scores': scores,
        })
    return sorted(records, key=lambda r: (split_rank(r['split']), r['name']))


def random_sanity_combined_id_group_records(dataset, score_map):
    grouped = {split: [] for split in SPLIT_ORDER}
    for name, scores in score_map.items():
        grouped.setdefault(random_sanity_split_name(dataset, name), []).append(scores)

    records = []
    for split in SPLIT_ORDER:
        parts = grouped.get(split, [])
        if not parts:
            continue
        scores = np.concatenate(parts)
        records.append({
            'split': split,
            'label': format_label(split, f'n_total={scores.size}'),
            'scores': scores,
        })
    return records


def rng_seed(seed):
    try:
        return int(seed)
    except (TypeError, ValueError):
        return 0


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

    rng = np.random.default_rng(rng_seed(args.seed))
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


def plot_random_sanity_one(args, dataset, method):
    if method in WARN_METHODS:
        warnings.warn(
            f'{method} may not follow the standard convention that larger conf means ID. '
            'Still plotting ood_score = -conf.'
        )

    collected = random_sanity_collect(args.results_root, dataset, method, args.seed)
    if not collected:
        warnings.warn(
            'Random sanity score directory not found, skipping: '
            f'{random_sanity_base(args.results_root) / "outputs" / method / dataset}'
        )
        return False

    id_name = ID_SCORE_NAME[dataset]
    if id_name not in collected:
        warnings.warn(f'ID score file not found for random sanity: {dataset}/{method}')
        return False

    rng = np.random.default_rng(rng_seed(args.seed))
    score_map = random_sanity_scores_for_plot(collected, args, rng)
    if id_name not in score_map:
        warnings.warn(f'ID score is empty for random sanity: {dataset}/{method}')
        return False

    acc_values, auroc_values = random_sanity_metric_values(dataset, collected)
    random_sanity_cross_check(
        args, dataset, method, args.seed, acc_values, auroc_values
    )

    seed_label = f'seed_{args.seed}' if args.seed != 'all' else 'seed_all'
    output_dir = (
        random_sanity_base(args.results_root)
        / 'plots'
        / 'score_density'
        / dataset
        / method
        / seed_label
    )
    xlim = common_xlim(score_map)
    title_prefix = f'{dataset} / {method} / random_sanity / {seed_label}'
    subtitle = ''

    outputs = [
        (
            random_sanity_per_dataset_records(
                dataset, score_map, collected, acc_values, auroc_values
            ),
            output_dir / 'per_dataset.png',
            f'{title_prefix}\nper dataset',
            subtitle,
        ),
        (
            random_sanity_combined_id_group_records(dataset, score_map),
            output_dir / 'combined_id_groups.png',
            f'{title_prefix}\nID vs csID vs nearOOD vs farOOD',
            subtitle,
        ),
    ]
    for records, output_path, title, subtitle_text in outputs:
        violinplot(records, output_path, title, subtitle_text, args, xlim)
        print(output_path)
    return True


def main():
    args = parse_args()
    plt.style.use('default')

    datasets = expand_choice(args.dataset, DATASETS)
    method_choices = RANDOM_SANITY_METHODS if args.source == 'random_sanity' else METHODS
    methods = expand_choice(args.method, method_choices)

    plotted = 0
    if args.source == 'openood':
        schemes = expand_choice(args.scheme, SCHEMES)
        for dataset in datasets:
            for method in methods:
                xlim = shared_scheme_xlim(args, dataset, method)
                for scheme in schemes:
                    if plot_one(args, dataset, method, scheme, xlim=xlim):
                        plotted += 1
    else:
        if args.scheme != 'all':
            warnings.warn('--scheme is ignored when --source random_sanity')
        for dataset in datasets:
            if dataset == 'mnist':
                warnings.warn('random_sanity v1 does not support mnist, skipping')
                continue
            for method in methods:
                if plot_random_sanity_one(args, dataset, method):
                    plotted += 1


if __name__ == '__main__':
    main()
