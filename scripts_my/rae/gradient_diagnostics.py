"""Gradient-space diagnostics before RAE validation scoring."""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from tqdm import tqdm

from .artifacts import ensure_dir, ref_grad_bank_dir, write_csv, write_json
from .config import (
    CANDIDATE_MODES,
    DEFAULT_CHECKPOINT,
    DEFAULT_REFERENCE_PER_CLASS_GRID_ARG,
    GRADIENT_SPACES,
    NUMERIC_EPS,
    REFERENCE_FILTERS,
    ROOT_DIR,
    SUPPORTED_DATASETS,
    SUPPORTED_SCHEMES,
    ReferenceConfig,
    parse_csv_values,
)
from .data import (
    build_dataloaders,
    build_model,
    device_from_arg,
    set_seed,
    split_dataloaders,
    subset_loader,
)
from .eval import (
    build_ref_grad_bank,
    build_reference,
    reference_config_from_args,
    resolved_checkpoint,
)
from .gradient_space import select_gradient_parameters
from .gradients import (
    classifier_has_bias,
    dense_candidate_directions_and_norms,
    forward_logits_features,
)
from .score import (
    batch_validation_scores_from_k,
    candidate_classes_from_probs,
    classifier_pairwise_k,
    prepare_ref_grad_bank_for_scoring,
)


def diagnostic_id(args) -> str:
    ref = ReferenceConfig(
        dataset=args.dataset,
        per_class=args.reference_per_class,
        filter_name=args.reference_filter,
        min_confidence=args.reference_min_confidence,
        seed=args.reference_seed,
    )
    subset = f'_subset{args.max_target_samples}' if args.max_target_samples else ''
    return (
        f'{args.experiment_id}_graddiag_{args.gradient_space}_{ref.id}_'
        f'{args.candidate_mode}_refseed{ref.seed}{subset}'
    )


def direction_diagnostic_dir(args) -> Path:
    return (
        Path(args.output_root) / 'gradient_diagnostics' / args.dataset /
        diagnostic_id(args)
    )


def _append_metric(store: Dict[str, list], name: str, values) -> None:
    array = values.detach().cpu().numpy() if torch.is_tensor(values) else values
    array = np.asarray(array)
    if array.size:
        store[name].append(array.reshape(-1))


def _summary(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        return {
            f'{prefix}_mean': float('nan'),
            f'{prefix}_std': float('nan'),
            f'{prefix}_p10': float('nan'),
            f'{prefix}_p50': float('nan'),
            f'{prefix}_p90': float('nan'),
        }
    return {
        f'{prefix}_mean': float(np.mean(values)),
        f'{prefix}_std': float(np.std(values)),
        f'{prefix}_p10': float(np.percentile(values, 10)),
        f'{prefix}_p50': float(np.percentile(values, 50)),
        f'{prefix}_p90': float(np.percentile(values, 90)),
    }


def _finalize_split_summary(store: Dict[str, list], context: Dict) -> Dict:
    row = dict(context)
    for name, parts in sorted(store.items()):
        values = np.concatenate(parts) if parts else np.asarray([])
        row.update(_summary(values.astype(np.float64), name))
    return row


def _tensor_to_numpy(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


def _candidate_mean(values: torch.Tensor) -> torch.Tensor:
    return values if values.dim() == 1 else values.mean(dim=1)


def _candidate_pred(values: torch.Tensor, pred_mask: torch.Tensor) -> torch.Tensor:
    if values.dim() == 1:
        return values
    return (values * pred_mask.to(values.dtype)).sum(dim=1)


def _append_sample_records(records: list[Dict],
                           *,
                           split_kind: str,
                           split_name: str,
                           labels: torch.Tensor,
                           pred: torch.Tensor,
                           candidates: torch.Tensor,
                           model_stats: Dict[str, torch.Tensor],
                           candidate_stats: Dict[str, torch.Tensor],
                           direction_stats: Dict[str, torch.Tensor],
                           k_stats: Dict[str, torch.Tensor]) -> None:
    pred_mask = candidates.eq(pred[:, None])
    per_sample = {
        'label': labels,
        'pred': pred,
        'is_correct': pred.eq(labels).to(torch.float64),
        **model_stats,
    }
    for name, values in {**candidate_stats, **direction_stats, **k_stats}.items():
        per_sample[f'{name}_mean'] = _candidate_mean(values)
        per_sample[f'{name}_pred'] = _candidate_pred(values, pred_mask)

    numpy_values = {
        name: _tensor_to_numpy(values).reshape(-1)
        for name, values in per_sample.items()
    }
    batch_size = int(labels.shape[0])
    for i in range(batch_size):
        row = {'split': split_name, 'group': split_kind}
        for name, values in numpy_values.items():
            value = values[i]
            if name in {'label', 'pred'}:
                row[name] = int(value)
            else:
                row[name] = float(value)
        records.append(row)


def _split_file_stem(split_kind: str, dataset_name: str) -> str:
    if split_kind == 'id':
        return 'id'
    return f'{split_kind}_{str(dataset_name).replace("/", "_")}'


def _sample_level_model_stats(logits: torch.Tensor,
                              probs: torch.Tensor) -> Dict[str, torch.Tensor]:
    top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=1).values
    if top2.shape[1] == 1:
        prob_margin = top2[:, 0]
    else:
        prob_margin = top2[:, 0] - top2[:, 1]
    entropy = -(probs * probs.clamp_min(NUMERIC_EPS).log()).sum(dim=1)
    logit_top2 = torch.topk(logits, k=min(2, logits.shape[1]), dim=1).values
    if logit_top2.shape[1] == 1:
        logit_margin = logit_top2[:, 0]
    else:
        logit_margin = logit_top2[:, 0] - logit_top2[:, 1]
    return {
        'pred_conf': probs.max(dim=1).values,
        'prob_margin': prob_margin,
        'entropy': entropy,
        'logit_margin': logit_margin,
    }


def _classifier_direction_stats(features: torch.Tensor,
                                probs: torch.Tensor,
                                candidates: torch.Tensor,
                                *,
                                include_bias: bool) -> Dict[str, torch.Tensor]:
    target_res = probs[:, None, :].expand(-1, candidates.shape[1], -1).clone()
    target_res.scatter_add_(
        2,
        candidates[:, :, None],
        -torch.ones(
            (*candidates.shape, 1),
            dtype=target_res.dtype,
            device=target_res.device,
        ),
    )
    residual_norm = torch.linalg.norm(target_res, dim=2).clamp_min(NUMERIC_EPS)
    feature_norm_sq = (features * features).sum(dim=1)
    feature_total = torch.sqrt(feature_norm_sq + (1.0 if include_bias else 0.0))
    raw_norm = residual_norm * feature_total[:, None]

    pred = probs.argmax(dim=1)
    pred_res = probs.clone()
    pred_res[torch.arange(probs.shape[0], device=probs.device), pred] -= 1.0
    pred_res_norm = torch.linalg.norm(pred_res, dim=1).clamp_min(NUMERIC_EPS)
    cos_to_pred = (
        torch.einsum('bkc,bc->bk', target_res, pred_res) /
        (residual_norm * pred_res_norm[:, None])
    )
    return {
        'candidate_raw_grad_norm': raw_norm,
        'candidate_residual_norm': residual_norm,
        'candidate_feature_norm': torch.sqrt(feature_norm_sq)[:, None].expand_as(
            raw_norm),
        'candidate_direction_cos_to_pred': cos_to_pred,
    }


def _dense_direction_stats(net: torch.nn.Module,
                           data: torch.Tensor,
                           candidates: torch.Tensor,
                           selected_params) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    directions, raw_norms = dense_candidate_directions_and_norms(
        net, data, candidates, selected_params)
    pred_dirs = directions[:, :1, :]
    cos_to_pred = (directions * pred_dirs).sum(dim=2)
    abs_dirs = directions.abs()
    l1 = abs_dirs.sum(dim=2)
    linf = abs_dirs.max(dim=2).values
    effective_dim = 1.0 / directions.pow(4).sum(dim=2).clamp_min(NUMERIC_EPS)
    return directions, {
        'candidate_raw_grad_norm': raw_norms,
        'candidate_direction_cos_to_pred': cos_to_pred,
        'candidate_direction_l1': l1,
        'candidate_direction_linf': linf,
        'candidate_effective_dim': effective_dim,
    }


def _k_stat_tensors(k_batch: torch.Tensor,
                    ref_labels: torch.Tensor,
                    candidates: torch.Tensor) -> Dict[str, torch.Tensor]:
    same_mask = ref_labels[None, None, :].eq(candidates[:, :, None])
    other_mask = ~same_mask
    same_count = same_mask.sum(dim=2).clamp_min(1)
    other_count = other_mask.sum(dim=2).clamp_min(1)
    same_float = same_mask.to(torch.float64)
    other_float = other_mask.to(torch.float64)
    k_float = k_batch.to(torch.float64)

    same_mean = (k_float * same_float).sum(dim=2) / same_count
    other_mean = (k_float * other_float).sum(dim=2) / other_count
    same_pos = ((k_batch > 0) & same_mask).sum(dim=2).to(torch.float64) / same_count
    other_pos = ((k_batch > 0) & other_mask).sum(dim=2).to(torch.float64) / other_count
    pairwise_rank, rank_only, _ = batch_validation_scores_from_k(
        k_batch, ref_labels, candidates)
    return {
        'k_same_mean': same_mean,
        'k_other_mean': other_mean,
        'k_mean_margin': same_mean - other_mean,
        'k_same_positive_rate': same_pos,
        'k_other_positive_rate': other_pos,
        'k_pairwise_rank_positive': pairwise_rank,
        'k_pairwise_rank_only': rank_only,
    }


def _append_k_stats(store: Dict[str, list],
                    k_stats: Dict[str, torch.Tensor]) -> None:
    for name, values in k_stats.items():
        _append_metric(store, name, values)


def summarize_split(args,
                    net: torch.nn.Module,
                    loader,
                    bank: Dict,
                    *,
                    split_kind: str,
                    split_name: str,
                    device: torch.device,
                    selected_params=None,
                    sample_records: list[Dict] | None = None) -> Dict:
    loader = subset_loader(
        loader,
        args.max_target_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    ref_labels = bank['_labels_tensor']
    store: Dict[str, list] = defaultdict(list)
    sample_count = 0
    candidate_count = 0
    for batch in tqdm(loader, desc=f'RAE gradient diag {split_name}'):
        data = batch['data'].to(device)
        labels = batch['label'].to(device).long()
        if bank['bank_type'] == 'classifier_compact':
            with torch.no_grad():
                logits, features = forward_logits_features(net, data)
                probs = torch.softmax(logits, dim=1)
            candidates = candidate_classes_from_probs(probs, args.candidate_mode)
            direction_stats = _classifier_direction_stats(
                features,
                probs,
                candidates,
                include_bias=bool(bank['classifier_has_bias'].item()),
            )
            k_batch = classifier_pairwise_k(
                features,
                probs,
                candidates,
                bank['_features_tensor'],
                bank['_probs_tensor'],
                ref_labels,
                include_bias=bool(bank['classifier_has_bias'].item()),
            )
        else:
            logits = net(data)
            probs = torch.softmax(logits, dim=1)
            candidates = candidate_classes_from_probs(probs, args.candidate_mode)
            directions, direction_stats = _dense_direction_stats(
                net, data, candidates, selected_params)
            k_batch = torch.einsum('bkd,rd->bkr', directions, bank['_directions_tensor'])

        model_stats = _sample_level_model_stats(logits, probs)
        candidate_probs = probs.gather(1, candidates)
        pred = probs.argmax(dim=1)
        candidate_is_pred = candidates.eq(pred[:, None]).to(torch.float64)
        candidate_is_label = candidates.eq(labels[:, None]).to(torch.float64)
        candidate_stats = {
            'candidate_prob': candidate_probs,
            'candidate_is_pred': candidate_is_pred,
            'candidate_is_label': candidate_is_label,
        }
        k_stats = _k_stat_tensors(k_batch, ref_labels, candidates)

        for name, values in model_stats.items():
            _append_metric(store, name, values)
        for name, values in candidate_stats.items():
            _append_metric(store, name, values)
        for name, values in direction_stats.items():
            _append_metric(store, name, values)
        _append_k_stats(store, k_stats)
        if sample_records is not None:
            _append_sample_records(
                sample_records,
                split_kind=split_kind,
                split_name=split_name,
                labels=labels,
                pred=pred,
                candidates=candidates,
                model_stats=model_stats,
                candidate_stats=candidate_stats,
                direction_stats=direction_stats,
                k_stats=k_stats,
            )
        sample_count += int(data.shape[0])
        candidate_count += int(candidates.numel())

    return _finalize_split_summary(
        store,
        {
            'split': split_name,
            'group': split_kind,
            'target_samples': sample_count,
            'candidate_values': candidate_count,
        },
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float('nan')
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float('nan')
    return _pearson(_rankdata(x), _rankdata(y))


def _linear_residual(metric: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    valid = np.isfinite(metric) & np.isfinite(confidence)
    residual = np.full(metric.shape, np.nan, dtype=np.float64)
    if valid.sum() < 2 or np.std(confidence[valid]) == 0.0:
        residual[valid] = metric[valid] - np.mean(metric[valid])
        return residual
    slope, intercept = np.polyfit(confidence[valid], metric[valid], deg=1)
    residual[valid] = metric[valid] - (slope * confidence[valid] + intercept)
    return residual


def _eta2_by_group(values: np.ndarray, groups: np.ndarray) -> float:
    valid = np.isfinite(values)
    values = values[valid]
    groups = groups[valid]
    if values.size < 2:
        return float('nan')
    grand = np.mean(values)
    total = float(np.sum((values - grand)**2))
    if total <= 0.0:
        return float('nan')
    between = 0.0
    for group in sorted(set(groups.tolist())):
        part = values[groups == group]
        if part.size:
            between += float(part.size * (np.mean(part) - grand)**2)
    return between / total


def _id_side_auroc(values: np.ndarray, groups: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    valid = np.isfinite(values)
    values = values[valid]
    groups = groups[valid]
    positive = np.isin(groups, ['id', 'csid'])
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = _rankdata(values)
    rank_sum_pos = float(ranks[positive].sum())
    return (
        rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    ) / float(n_pos * n_neg)


def _read_csv_rows(path: Path) -> list[Dict[str, str]]:
    import csv

    with path.open() as f:
        return list(csv.DictReader(f))


def _metric_columns(rows: list[Dict]) -> list[str]:
    excluded = {'split', 'group', 'label', 'pred', 'is_correct'}
    excluded_prefixes = (
        'candidate_is_label',
        'candidate_is_pred',
        'candidate_prob',
    )
    columns = []
    for key in rows[0]:
        if (
                key in excluded or key == 'pred_conf' or
                key.startswith(excluded_prefixes)
        ):
            continue
        try:
            float(rows[0][key])
        except (TypeError, ValueError):
            continue
        if key.endswith('_mean') or key.endswith('_pred') or key in {
                'entropy', 'prob_margin', 'logit_margin'
        }:
            columns.append(key)
    return columns


def _confidence_independence_rows(sample_rows: list[Dict]) -> list[Dict]:
    if not sample_rows:
        return []
    confidence = np.asarray([float(row['pred_conf']) for row in sample_rows])
    groups = np.asarray([row['group'] for row in sample_rows])
    id_mask = groups == 'id'
    ood_mask = groups != 'id'
    rows = []
    for metric in _metric_columns(sample_rows):
        values = np.asarray([float(row[metric]) for row in sample_rows])
        residual = _linear_residual(values, confidence)
        row = {
            'metric': metric,
            'pearson_with_conf': _pearson(values, confidence),
            'spearman_with_conf': _spearman(values, confidence),
            'group_eta2_raw': _eta2_by_group(values, groups),
            'group_eta2_conf_residual': _eta2_by_group(residual, groups),
            'id_mean': float(np.nanmean(values[id_mask])) if id_mask.any() else float('nan'),
            'non_id_mean': float(np.nanmean(values[ood_mask])) if ood_mask.any() else float('nan'),
            'id_residual_mean': float(np.nanmean(residual[id_mask])) if id_mask.any() else float('nan'),
            'non_id_residual_mean': float(np.nanmean(residual[ood_mask])) if ood_mask.any() else float('nan'),
        }
        row['non_id_minus_id'] = row['non_id_mean'] - row['id_mean']
        row['non_id_minus_id_residual'] = (
            row['non_id_residual_mean'] - row['id_residual_mean'])
        rows.append(row)
    return rows


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _finite_values(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _axis_limits(values: np.ndarray) -> tuple[float, float]:
    finite = _finite_values(values)
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.percentile(finite, 1))
    high = float(np.percentile(finite, 99))
    if low == high:
        pad = max(1.0, abs(low) * 0.05)
        return low - pad, high + pad
    pad = 0.04 * (high - low)
    return low - pad, high + pad


def _draw_half_violin(ax,
                      data: list[np.ndarray],
                      positions: list[float],
                      *,
                      colors: list[str],
                      orientation: str) -> None:
    for values, position, color in zip(data, positions, colors):
        finite = _finite_values(values)
        if finite.size < 2 or float(np.std(finite)) == 0.0:
            continue
        parts = ax.violinplot(
            [finite],
            positions=[position],
            vert=(orientation == 'vertical'),
            widths=0.34,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts['bodies']:
            vertices = body.get_paths()[0].vertices
            if orientation == 'horizontal':
                vertices[:, 1] = np.maximum(vertices[:, 1], position)
            else:
                vertices[:, 0] = np.maximum(vertices[:, 0], position)
            body.set_facecolor(color)
            body.set_edgecolor('none')
            body.set_alpha(0.35)


def _density_curve(values: np.ndarray,
                   low: float,
                   high: float,
                   *,
                   points: int = 128) -> tuple[np.ndarray, np.ndarray]:
    finite = _finite_values(values)
    grid = np.linspace(low, high, points)
    if finite.size < 2 or low == high:
        return grid, np.zeros_like(grid)
    hist, edges = np.histogram(finite, bins=36, range=(low, high), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = np.interp(grid, centers, hist, left=0.0, right=0.0)
    if density.size >= 5:
        kernel = np.asarray([1, 4, 6, 4, 1], dtype=np.float64)
        kernel /= kernel.sum()
        density = np.convolve(density, kernel, mode='same')
    return grid, density


def _draw_overlay_density(top_ax,
                          right_ax,
                          *,
                          confidence_by_group: list[np.ndarray],
                          metric_by_group: list[np.ndarray],
                          xlim: tuple[float, float],
                          ylim: tuple[float, float],
                          colors: list[str]) -> tuple[float, float]:
    x_density_max = 0.0
    y_density_max = 0.0
    for x_values, y_values, color in zip(
            confidence_by_group, metric_by_group, colors):
        x_grid, x_density = _density_curve(x_values, xlim[0], xlim[1])
        y_grid, y_density = _density_curve(y_values, ylim[0], ylim[1])
        x_density_max = max(x_density_max, float(np.max(x_density)))
        y_density_max = max(y_density_max, float(np.max(y_density)))
        top_ax.fill_between(
            x_grid, 0.0, x_density, color=color, alpha=0.25, linewidth=0)
        top_ax.plot(x_grid, x_density, color=color, alpha=0.75, linewidth=0.9)
        right_ax.fill_betweenx(
            y_grid, 0.0, y_density, color=color, alpha=0.25, linewidth=0)
        right_ax.plot(y_density, y_grid, color=color, alpha=0.75, linewidth=0.9)
    return x_density_max, y_density_max


def _density_axis_limit(max_density: float) -> float:
    if max_density > 0.0 and np.isfinite(max_density):
        return max_density * 1.05
    return 1.0


def write_confidence_independence(sample_csv: Path, out_csv: Path) -> list[Dict]:
    sample_rows = _read_csv_rows(sample_csv)
    rows = _confidence_independence_rows(sample_rows)
    if rows:
        fieldnames = [
            'metric',
            'pearson_with_conf',
            'spearman_with_conf',
            'group_eta2_raw',
            'group_eta2_conf_residual',
            'id_mean',
            'non_id_mean',
            'non_id_minus_id',
            'id_residual_mean',
            'non_id_residual_mean',
            'non_id_minus_id_residual',
        ]
        write_csv(out_csv, rows, fieldnames)
    return rows


def _plot_metrics(sample_csv: Path,
                  independence_rows: list[Dict],
                  out_dir: Path) -> list[str]:
    sample_rows = _read_csv_rows(sample_csv)
    if not sample_rows:
        return []
    os.environ.setdefault('MPLCONFIGDIR', str(out_dir.parent / '.matplotlib'))
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    preferred = [
        'candidate_raw_grad_norm_pred',
        'candidate_raw_grad_norm_mean',
        'candidate_effective_dim_pred',
        'candidate_effective_dim_mean',
        'candidate_direction_cos_to_pred_mean',
        'candidate_direction_linf_mean',
        'k_mean_margin_pred',
        'k_mean_margin_mean',
        'k_same_positive_rate_mean',
        'k_other_positive_rate_mean',
    ]
    available = [metric for metric in preferred if metric in sample_rows[0]]
    if not available:
        return []
    group_order = ['id', 'csid', 'nearood', 'farood']
    colors = {
        'id': '#1f77b4',
        'csid': '#ff7f0e',
        'nearood': '#2ca02c',
        'farood': '#d62728',
    }
    confidence = np.asarray([_safe_float(row['pred_conf']) for row in sample_rows])
    groups = np.asarray([row['group'] for row in sample_rows])
    outputs = []

    metrics = available[:min(6, len(available))]
    ncols = 3
    nrows = math.ceil(len(metrics) / ncols)
    group_positions = (np.arange(len(group_order), dtype=np.float64) * 0.42).tolist()
    violin_limit = (-0.06, group_positions[-1] + 0.28)
    group_colors = [colors[group] for group in group_order]

    def draw_confidence_metric_grid(path: Path, *, overlay_density: bool) -> None:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.3 * ncols, 5.15 * nrows),
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.06,
            right=0.94,
            top=0.94,
            bottom=0.08,
            wspace=0.72,
            hspace=0.55,
        )
        axes = np.asarray(axes).reshape(-1)
        scatter_axes = []
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            scatter_axes.append(ax)
            values = np.asarray([_safe_float(row[metric]) for row in sample_rows])
            y_min, y_max = _axis_limits(values)
            confidence_by_group = []
            metric_by_group = []
            for group in group_order:
                mask = groups == group
                confidence_by_group.append(confidence[mask])
                metric_by_group.append(values[mask])
                if mask.any():
                    ax.scatter(
                        confidence[mask],
                        values[mask],
                        s=10,
                        alpha=0.45,
                        label=group,
                        color=colors[group],
                        edgecolors='none',
                    )
            ax.set_xlabel('pred_conf')
            ax.set_ylabel(metric)
            ax.text(
                0.5,
                -0.18,
                metric,
                transform=ax.transAxes,
                ha='center',
                va='top',
                fontsize=11,
            )
            x_auroc = _id_side_auroc(confidence, groups)
            y_auroc = _id_side_auroc(values, groups)
            ax.text(
                0.5,
                -0.255,
                f'ID-AUROC x={100 * x_auroc:.1f} y={100 * y_auroc:.1f}',
                transform=ax.transAxes,
                ha='center',
                va='top',
                fontsize=9.5,
            )
            ax.set_xlim(0.0, 1.01)
            ax.set_ylim(y_min, y_max)
            ax.set_box_aspect(1)

            top_ax = ax.inset_axes(
                [0.0, 1.03, 1.0, 0.24],
                transform=ax.transAxes,
                sharex=ax,
            )
            right_ax = ax.inset_axes(
                [1.03, 0.0, 0.24, 1.0],
                transform=ax.transAxes,
                sharey=ax,
            )
            if overlay_density:
                x_density_max, y_density_max = _draw_overlay_density(
                    top_ax,
                    right_ax,
                    confidence_by_group=confidence_by_group,
                    metric_by_group=metric_by_group,
                    xlim=ax.get_xlim(),
                    ylim=ax.get_ylim(),
                    colors=group_colors,
                )
                top_ax.set_ylim(0.0, _density_axis_limit(x_density_max))
                right_ax.set_xlim(0.0, _density_axis_limit(y_density_max))
            else:
                _draw_half_violin(
                    top_ax,
                    confidence_by_group,
                    group_positions,
                    colors=group_colors,
                    orientation='horizontal',
                )
                _draw_half_violin(
                    right_ax,
                    metric_by_group,
                    group_positions,
                    colors=group_colors,
                    orientation='vertical',
                )
                top_ax.set_ylim(*violin_limit)
                right_ax.set_xlim(*violin_limit)
            top_ax.set_yticks([])
            top_ax.tick_params(axis='x', labelbottom=False, length=2)
            top_ax.spines['right'].set_visible(False)
            top_ax.spines['top'].set_visible(False)
            right_ax.set_xticks([])
            right_ax.tick_params(axis='y', labelleft=False, length=2)
            right_ax.spines['right'].set_visible(False)
            right_ax.spines['top'].set_visible(False)
        for ax in axes[len(metrics):]:
            ax.axis('off')
        scatter_axes[0].legend(loc='best', fontsize=8)
        fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)

    path = out_dir / 'confidence_vs_gradient_metrics.png'
    draw_confidence_metric_grid(path, overlay_density=False)
    outputs.append(str(path))

    path = out_dir / 'confidence_vs_gradient_metrics_overlay.png'
    draw_confidence_metric_grid(path, overlay_density=True)
    outputs.append(str(path))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for ax, metric in zip(axes, metrics):
        values = np.asarray([_safe_float(row[metric]) for row in sample_rows])
        data = [values[groups == group] for group in group_order]
        ax.boxplot(data, labels=group_order, showfliers=False)
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=25)
    for ax in axes[len(metrics):]:
        ax.axis('off')
    fig.tight_layout()
    path = out_dir / 'group_boxplots.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(str(path))

    if independence_rows:
        ranked = sorted(
            independence_rows,
            key=lambda row: (
                abs(float(row['group_eta2_conf_residual']))
                if np.isfinite(float(row['group_eta2_conf_residual'])) else -1.0
            ),
            reverse=True,
        )[:12]
        labels = [row['metric'] for row in ranked]
        raw = [float(row['group_eta2_raw']) for row in ranked]
        residual = [float(row['group_eta2_conf_residual']) for row in ranked]
        y = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(labels))))
        ax.barh(y - 0.18, raw, height=0.35, label='raw group eta2')
        ax.barh(y + 0.18, residual, height=0.35, label='after confidence residual')
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel('group eta2')
        ax.legend()
        fig.tight_layout()
        path = out_dir / 'confidence_residual_group_signal.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(str(path))
    return outputs


def run_gradient_diagnostics(args) -> None:
    set_seed(args.seed)
    start = time.perf_counter()
    device = device_from_arg(args.device)
    checkpoint = resolved_checkpoint(args)
    net = build_model(args.dataset, checkpoint).to(device)
    net.eval()
    dataloaders = build_dataloaders(
        args.dataset,
        data_root=ROOT_DIR / 'data',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    ref_manifest = build_reference(args, net, dataloaders, device)
    ref_grad_bank_manifest, ref_grad_bank = build_ref_grad_bank(
        args, net, dataloaders, ref_manifest, device)
    bank = prepare_ref_grad_bank_for_scoring(ref_grad_bank, device)
    selected_params = None
    if bank['bank_type'] != 'classifier_compact':
        selected_params = [
            param for _, param in select_gradient_parameters(net, args.gradient_space)
        ]

    out_dir = direction_diagnostic_dir(args)
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f'Output directory already exists: {out_dir}')
    ensure_dir(out_dir)

    rows = []
    sample_records: list[Dict] = []
    for split_kind, dataset_name, _, _, loader in split_dataloaders(
            args.dataset, dataloaders, args.scheme):
        if args.split_groups != ['all'] and split_kind not in args.split_groups:
            continue
        split_name = 'id' if split_kind == 'id' else f'{split_kind}/{dataset_name}'
        row = summarize_split(
            args,
            net,
            loader,
            bank,
            split_kind=split_kind,
            split_name=split_name,
            device=device,
            selected_params=selected_params,
            sample_records=sample_records if args.sample_metrics else None,
        )
        rows.append(row)

    fieldnames = ['split', 'group', 'target_samples', 'candidate_values']
    metric_fields = sorted({key for row in rows for key in row if key not in fieldnames})
    write_csv(out_dir / 'split_summary.csv', rows, fieldnames + metric_fields)
    plot_paths = []
    if sample_records:
        sample_fieldnames = ['split', 'group', 'label', 'pred', 'is_correct']
        sample_metric_fields = sorted({
            key
            for row in sample_records
            for key in row
            if key not in sample_fieldnames
        })
        sample_csv = out_dir / 'sample_metrics.csv'
        write_csv(sample_csv, sample_records, sample_fieldnames + sample_metric_fields)
        independence_rows = write_confidence_independence(
            sample_csv, out_dir / 'confidence_independence.csv')
        if args.plots:
            plot_paths = _plot_metrics(
                sample_csv, independence_rows, out_dir / 'plots')
    write_json(out_dir / 'manifest.json', {
        'artifact': 'rae_gradient_diagnostics',
        'dataset': args.dataset,
        'scheme': args.scheme,
        'gradient_space': args.gradient_space,
        'candidate_mode': args.candidate_mode,
        'reference_manifest': ref_manifest,
        'gradient_manifest': ref_grad_bank_manifest,
        'checkpoint': checkpoint,
        'max_target_samples': args.max_target_samples,
        'batch_size': args.batch_size,
        'reference_batch_size': args.reference_batch_size,
        'split_groups': args.split_groups,
        'sample_metrics': bool(sample_records),
        'plots': plot_paths,
        'output_dir': str(out_dir),
        'elapsed_sec': time.perf_counter() - start,
    })


def _parse_split_groups(value: str) -> list[str]:
    groups = parse_csv_values(value)
    if not groups:
        raise ValueError('At least one split group must be selected')
    allowed = {'all', 'id', 'csid', 'nearood', 'farood'}
    unknown = sorted(set(groups) - allowed)
    if unknown:
        raise ValueError(f'Unknown split group(s): {unknown}')
    if 'all' in groups:
        return ['all']
    return groups


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description='RAE gradient diagnostics')
    parser.add_argument('--dataset', choices=SUPPORTED_DATASETS, default='cifar10')
    parser.add_argument('--scheme', choices=SUPPORTED_SCHEMES, default='fsood')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output-root', default='results_test/rae')
    parser.add_argument('--experiment-id', default='gradient_diagnostics')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--reference-batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--gradient-space', choices=GRADIENT_SPACES, default='classifier')
    parser.add_argument('--candidate-mode', choices=CANDIDATE_MODES, default='pred')
    parser.add_argument('--reference-per-class', type=int, default=16)
    parser.add_argument('--reference-filter', choices=REFERENCE_FILTERS, default='correct')
    parser.add_argument('--reference-min-confidence', type=float, default=0.9)
    parser.add_argument('--reference-seed', type=int, default=0)
    parser.add_argument('--rebuild-train-metadata', action='store_true')
    parser.add_argument('--rebuild-reference', action='store_true')
    parser.add_argument(
        '--rebuild-gradient-bank',
        dest='rebuild_ref_grad_bank',
        action='store_true',
    )
    parser.add_argument('--max-target-samples', type=int, default=512)
    parser.add_argument(
        '--no-sample-metrics',
        dest='sample_metrics',
        action='store_false',
        help='Skip sample-level metric export and confidence-separation summary.',
    )
    parser.set_defaults(sample_metrics=True)
    parser.add_argument(
        '--no-plots',
        dest='plots',
        action='store_false',
        help='Skip diagnostic plot generation.',
    )
    parser.set_defaults(plots=True)
    parser.add_argument(
        '--split-groups',
        default='all',
        help='Comma-separated subset of id,csid,nearood,farood or all.',
    )
    args = parser.parse_args(argv)
    args.split_groups = _parse_split_groups(args.split_groups)
    return args


def main(argv: Iterable[str] | None = None) -> None:
    run_gradient_diagnostics(parse_args(argv))


if __name__ == '__main__':
    main()
