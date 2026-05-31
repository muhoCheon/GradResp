#!/usr/bin/env python
"""TARR report commands for diagnostics, collection, and baseline comparison."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from scripts_my.tarr.protocol import (  # noqa: E402
    SCORE_RESULTS_DIR,
    TTA_RESPONSE_DIR,
    csid_datasets_from_run_manifest,
    display_label,
    expected_csid_datasets as protocol_expected_csid_datasets,
    far_dataset_names,
    near_dataset_names,
    ood_datasets_from_run_manifest,
    parse_dataset_list,
    supported_dataset_names,
)
try:  # noqa: E402
    from scripts_my.tarr import cache as tarr_cache
except Exception:  # pragma: no cover - diagnostics should still run standalone.
    tarr_cache = None
from scripts_my.tarr.scoring import (  # noqa: E402
    ACTIVE_SCORE_RULES,
    PERTURBATION_SCORE_RULE_CHOICES,
    SCORE_RULE_CHOICES,
    VECTOR_SCORE_RULE_CHOICES,
    fit_vector_score_reference,
    ood_score_from_cache,
    perturbation_ood_score_from_cache,
    selected_perturbation_score_rules,
    selected_score_rules,
    selected_vector_score_rules,
    vector_ood_score_from_cache,
)


METRIC_FIELDS = ['FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC']
SCORE_KIND_CHOICES = ['standard', 'vector', 'perturbation']
PERTURBATION_SCORE_NAME_KEYS = [
    'perturbation_score_rules',
    'perturbation_score_names',
    'perturbation_rule_names',
    'perturbation_score_rule_names',
]
PERTURBATION_SCORE_MATRIX_KEYS = [
    'perturbation_scores',
    'perturbation_ood_scores',
    'perturbation_score_matrix',
    'perturbation_rule_scores',
]
PERTURBATION_SCORE_FIELD_MARKERS = [
    'score',
    'ood',
    'separation',
    'drop',
    'gain',
    'delta',
    'response',
]
PERTURBATION_DIAGNOSTIC_FIELDS = [
    'perturbation_logit_l2',
    'perturbation_prob_l1',
    'perturbation_conf_delta',
    'perturbation_entropy_delta',
]
PERTURBATION_SUMMARY_FIELDS = [
    'row_type', 'score_family', 'score_rule', 'feature', 'source_field',
    'split', 'dataset', 'array_shape', 'cache_schema_version',
    'delta_definition', 'score_direction', 'n', 'mean', 'std', 'min',
    'max', 'q05', 'q25', 'median', 'q75', 'q90', 'q95', 'q99',
]
PERTURBATION_ALIGNMENT_FIELDS = [
    'row_type', 'score_family', 'score_rule', 'source_field', 'run_dir',
    'run_id', 'dataset', 'baseline_protocol', 'scheme',
    'reference_config_id', 'split', 'cache_dataset', 'cache_name',
    'cache_path', 'cache_num_shards', 'cache_paths', 'id_side', 'ood_split',
    'ood_dataset', 'n_clean', 'n_csid', 'n_id_side', 'n_ood', 'n_samples',
    'score_shape', 'expected_csid_datasets', 'resolved_csid_datasets',
    'csid_alignment_status', 'mean_gap', 'median_gap',
    'standardized_mean_gap', 'id_vs_csid_auroc', 'alignment_error',
    'csid_tail_at_clean_q95', 'csid_in_clean_5_95',
    'id_side_vs_ood_auroc', 'separation_error', 'ood_tail_at_id_q95',
    'ood_in_id_5_95', 'clean_mean', 'csid_mean', 'id_side_mean',
    'ood_mean', 'clean_median', 'csid_median', 'id_side_median',
    'ood_median', 'cache_schema_version', 'score_direction',
    'delta_definition',
]
AGGREGATE_TO_COLUMNS = {
    ('ood', 'nearood'): ('OOD Near AUROC', 'OOD Near FPR95'),
    ('ood', 'farood'): ('OOD Far AUROC', 'OOD Far FPR95'),
    ('fsood', 'nearood'): ('FSOOD Near AUROC', 'FSOOD Near FPR95'),
    ('fsood', 'farood'): ('FSOOD Far AUROC', 'FSOOD Far FPR95'),
}
def load_json(path):
    return json.loads(Path(path).read_text())


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return load_json(path)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def scalar_value(value):
    value = np.asarray(value)
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(-1)[0].item()
    return value.tolist()


def shape_text(value):
    return 'x'.join(str(dim) for dim in np.asarray(value).shape)


def first_dim(value):
    value = np.asarray(value)
    if value.shape == ():
        return ''
    return int(value.shape[0])


def manifest_csid_datasets(manifest):
    return parse_dataset_list(
        manifest.get('resolved_csid_datasets')
        or manifest.get('protocol_config', {}).get('resolved_csid_datasets'))


def _dataset_keys(value):
    if isinstance(value, dict):
        return [key for key in value if key not in {'datasets', 'names'}]
    return []


def manifest_split_datasets(manifest, split):
    if not isinstance(manifest, dict):
        return []
    for key in [f'{split}_datasets', f'{split}_dataset_names']:
        names = parse_dataset_list(manifest.get(key))
        if names:
            return names
    for key in ['protocol_config', 'score', 'score_config', SCORE_RESULTS_DIR]:
        names = manifest_split_datasets(manifest.get(key), split)
        if names:
            return names
    dataset_manifest = manifest.get('dataset_manifest', {})
    names = _dataset_keys(
        dataset_manifest.get('ood', {}).get(split, {})
        if isinstance(dataset_manifest, dict) else {})
    if names:
        return names
    ood_node = manifest.get('ood', {})
    if isinstance(ood_node, dict):
        names = parse_dataset_list(ood_node.get(split))
        if not names:
            names = _dataset_keys(ood_node.get(split, {}))
        if names:
            return names
    for key in ['splits', TTA_RESPONSE_DIR, 'tta_response_files']:
        node = manifest.get(key, {})
        if isinstance(node, dict):
            names = parse_dataset_list(node.get(split))
            if not names:
                names = _dataset_keys(node.get(split, {}))
            if names:
                return names
    return []


def default_split_datasets(dataset, split):
    try:
        if split == 'near':
            return near_dataset_names(dataset)
        if split == 'far':
            return far_dataset_names(dataset)
    except ValueError:
        return []
    return []


def run_split_datasets(run_dir, scheme, dataset, split, *manifests):
    for manifest in manifests:
        names = manifest_split_datasets(manifest, split)
        if names:
            return names
    names = parse_dataset_list(
        ood_datasets_from_run_manifest(run_dir, scheme, split))
    if names:
        return names
    return default_split_datasets(dataset, split)


def expected_csid_datasets(dataset, baseline_protocol):
    try:
        return protocol_expected_csid_datasets(dataset, baseline_protocol)
    except ValueError:
        return []


def csid_alignment_status(expected_csid, resolved_csid):
    if not expected_csid:
        return ''
    return 'aligned' if set(expected_csid) == set(resolved_csid) else 'mismatch'


def group1_dataset_label(dataset):
    return display_label(dataset)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def cache_path(run_dir, scheme, name, reference_config_id=None):
    scheme_dir = Path(run_dir) / scheme
    if reference_config_id:
        return (scheme_dir / 'references' / reference_config_id /
                TTA_RESPONSE_DIR / f'{name}.npz')
    return scheme_dir / TTA_RESPONSE_DIR / f'{name}.npz'


def cache_base_path(run_dir, scheme, name, reference_config_id=None):
    path = cache_path(run_dir, scheme, name, reference_config_id)
    return path.with_suffix('')


class ResponseCache(dict):
    @property
    def files(self):
        return list(self.keys())


def _load_npz_dict(path):
    with np.load(path, allow_pickle=False) as data:
        return ResponseCache({key: data[key] for key in data.files})


def _manifest_part_paths(cache_dir, manifest):
    for key in ['parts', 'part_paths', 'cache_paths', 'files', 'npz_files']:
        values = manifest.get(key)
        if not values:
            continue
        paths = []
        for value in values:
            if isinstance(value, dict):
                value = (
                    value.get('path')
                    or value.get('file')
                    or value.get('cache_path'))
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = cache_dir / path
            paths.append(path)
        if paths:
            return paths
    return sorted(cache_dir.glob('part_*.npz'))


def _concat_shard_values(values, sample_counts):
    arrays = [np.asarray(value) for value in values]
    first = arrays[0]
    if first.shape == ():
        return first
    sample_axis = (
        sample_counts
        and all(array.ndim > 0 and array.shape[0] == n
                for array, n in zip(arrays, sample_counts))
    )
    if sample_axis and all(array.shape[1:] == first.shape[1:] for array in arrays):
        return np.concatenate(arrays, axis=0)
    if all(array.shape == first.shape and np.array_equal(array, first)
           for array in arrays[1:]):
        return first
    return np.concatenate([array.reshape(-1) for array in arrays], axis=0)


def _merge_shard_dicts(parts):
    keys = set(parts[0])
    for part in parts[1:]:
        keys &= set(part)
    sample_counts = []
    if all('label' in part for part in parts):
        sample_counts = [int(np.asarray(part['label']).shape[0]) for part in parts]
    merged = ResponseCache()
    for key in sorted(keys):
        merged[key] = _concat_shard_values(
            [part[key] for part in parts], sample_counts)
    return merged


def _load_sharded_cache(cache_dir):
    manifest_path = cache_dir / 'manifest.json'
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    if isinstance(manifest, dict) and manifest.get('complete') is False:
        raise ValueError(f'{manifest_path} is incomplete')
    part_paths = _manifest_part_paths(cache_dir, manifest)
    if not part_paths:
        raise FileNotFoundError(f'No sharded TTA response parts found in {cache_dir}')
    missing = [path for path in part_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            'Sharded TTA response parts missing: ' +
            ', '.join(str(path) for path in missing))
    parts = [_load_npz_dict(path) for path in part_paths]
    return {
        'path': cache_dir,
        'cache': _merge_shard_dicts(parts),
        'cache_num_shards': len(part_paths),
        'cache_paths': part_paths,
        'manifest_path': manifest_path,
    }


def _load_cache_with_shared_logic(path, shard_dir):
    for name in ['load_tta_response', 'load_sharded_tta_response']:
        loader = getattr(tarr_cache, name, None) if tarr_cache is not None else None
        if loader is None:
            continue
        for arg in [path, shard_dir]:
            try:
                loaded = loader(arg)
            except (TypeError, FileNotFoundError):
                continue
            if isinstance(loaded, tuple) and len(loaded) >= 2:
                return {
                    'path': Path(loaded[0]),
                    'cache': loaded[1],
                    'cache_num_shards': 1,
                    'cache_paths': [Path(loaded[0])],
                }
            if isinstance(loaded, dict) and 'cache' in loaded:
                return loaded
            if loaded is not None:
                return {
                    'path': Path(arg),
                    'cache': loaded,
                    'cache_num_shards': 1,
                    'cache_paths': [Path(arg)],
                }
    return None


def load_cache(run_dir, scheme, name, reference_config_id=None):
    path = cache_path(run_dir, scheme, name, reference_config_id)
    shard_dir = cache_base_path(run_dir, scheme, name, reference_config_id)
    loaded = _load_cache_with_shared_logic(path, shard_dir)
    if loaded is not None:
        return loaded
    if path.exists():
        return {
            'path': path,
            'cache': _load_npz_dict(path),
            'cache_num_shards': 1,
            'cache_paths': [path],
        }
    if shard_dir.exists():
        return _load_sharded_cache(shard_dir)
    return None


def cache_paths_text(item):
    return ';'.join(str(path) for path in item.get('cache_paths', []))


def quantiles(values):
    if values.size == 0:
        return {}
    probs = [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    qs = np.quantile(values, probs)
    return {
        'q05': qs[0],
        'q25': qs[1],
        'median': qs[2],
        'q75': qs[3],
        'q90': qs[4],
        'q95': qs[5],
        'q99': qs[6],
    }


def summary_row(prefix, values):
    values = np.asarray(values, dtype=np.float64)
    row = dict(prefix)
    row.update({
        'n': int(values.size),
        'mean': float(np.mean(values)) if values.size else '',
        'std': float(np.std(values)) if values.size else '',
        'min': float(np.min(values)) if values.size else '',
        'max': float(np.max(values)) if values.size else '',
    })
    row.update({key: float(value) for key, value in quantiles(values).items()})
    return row


def cache_feature_summary(prefix, cache, feature, values):
    row = summary_row(prefix, values)
    row['feature'] = feature
    row['cache_schema_version'] = scalar_value(
        cache['cache_schema_version']) if 'cache_schema_version' in cache else ''
    row['delta_definition'] = scalar_value(
        cache['delta_definition']) if 'delta_definition' in cache else ''
    row['score_direction'] = scalar_value(
        cache['score_direction']) if 'score_direction' in cache else ''
    return row


def collect_caches(args):
    names = [(args.dataset, 'id', args.dataset)]
    run_dir = Path(args.run_dir)
    scheme_manifest = read_json(run_dir / args.scheme / 'scheme_manifest.json')
    run_manifest = read_json(run_dir / 'run_manifest.json')
    if args.scheme == 'fsood':
        csid_names = (
            manifest_csid_datasets(scheme_manifest)
            or manifest_csid_datasets(run_manifest)
            or csid_datasets_from_run_manifest(args.run_dir, args.scheme))
        if not csid_names:
            csid_names = parse_dataset_list(args.csid_datasets)
        if not csid_names:
            raise ValueError(
                'Unable to resolve FSOOD csID datasets from manifest. '
                'Pass --csid-datasets as a comma-separated fallback.')
        names.extend((name, 'csid', name) for name in csid_names)
    near_names = run_split_datasets(
        args.run_dir, args.scheme, args.dataset, 'near',
        scheme_manifest, run_manifest)
    far_names = run_split_datasets(
        args.run_dir, args.scheme, args.dataset, 'far',
        scheme_manifest, run_manifest)
    names.extend((name, 'near', name) for name in near_names)
    names.extend((name, 'far', name) for name in far_names)

    caches = []
    for name, split, dataset_name in names:
        loaded = load_cache(args.run_dir, args.scheme, name,
                            args.reference_config_id)
        if loaded is None:
            continue
        caches.append({
            'name': name,
            'split': split,
            'dataset': dataset_name,
            'path': loaded['path'],
            'cache': loaded['cache'],
            'cache_num_shards': loaded.get('cache_num_shards', 1),
            'cache_paths': loaded.get('cache_paths', [loaded['path']]),
        })
    return caches


def delta_features(cache):
    delta = cache['delta']
    row = np.arange(delta.shape[0])
    y_hat = cache['y_hat'].astype(np.int64)
    return {
        'predicted_class_loss_increase': delta[row, y_hat],
        'predicted_class_loss_decrease': -delta[row, y_hat],
        'mean_loss_increase': np.mean(delta, axis=1),
        'mean_loss_decrease': -np.mean(delta, axis=1),
        'positive_loss_increase_mean': np.mean(np.clip(delta, 0.0, None), axis=1),
        'positive_loss_decrease_mean': np.mean(np.clip(-delta, 0.0, None), axis=1),
        'classwise_max_delta': np.max(delta, axis=1),
        'classwise_min_delta': np.min(delta, axis=1),
    }


def build_score_summary(caches, score_rules):
    rows = []
    for item in caches:
        for score_rule in score_rules:
            rows.append(summary_row({
                'score_rule': score_rule,
                'split': item['split'],
                'dataset': item['dataset'],
            }, ood_score_from_cache(item['cache'], score_rule)))
    return rows


def build_delta_summary(caches):
    rows = []
    for item in caches:
        for feature, values in delta_features(item['cache']).items():
            rows.append(cache_feature_summary({
                'feature': feature,
                'split': item['split'],
                'dataset': item['dataset'],
            }, item['cache'], feature, values))
    return rows


def build_runtime_summary(caches):
    return [
        summary_row({'split': item['split'], 'dataset': item['dataset']},
                    item['cache']['runtime_per_sample'])
        for item in caches
    ]


def target_features(cache):
    probs = cache['target_probs']
    y_hat = cache['y_hat'].astype(np.int64)
    labels = cache['label'].astype(np.int64)
    rows = np.arange(probs.shape[0])
    sorted_probs = np.sort(probs, axis=1)
    top2 = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros(probs.shape[0])
    valid_label = (labels >= 0) & (labels < probs.shape[1])
    label_prob = np.full(probs.shape[0], np.nan, dtype=np.float64)
    label_prob[valid_label] = probs[rows[valid_label], labels[valid_label]]
    return {
        'target_conf': cache['target_conf'],
        'target_entropy': cache['target_entropy'],
        'target_margin': cache['target_margin'] if 'target_margin' in cache else (
            cache['target_conf'] - top2),
        'target_energy': cache['target_energy'] if 'target_energy' in cache else (
            np.full(probs.shape[0], np.nan)),
        'target_predicted_probability': probs[rows, y_hat],
        'target_probability_margin': cache['target_conf'] - top2,
        'target_label_probability': label_prob[~np.isnan(label_prob)],
        'target_probability_max': np.max(probs, axis=1),
        'target_probability_min': np.min(probs, axis=1),
    }


def build_target_summary(caches):
    rows = []
    for item in caches:
        features = target_features(item['cache'])
        for key in [
                'adapted_target_conf',
                'adapted_target_entropy',
                'adapted_target_margin',
                'adapted_target_energy',
                'target_conf_delta',
                'target_entropy_delta',
                'target_margin_delta',
                'target_energy_delta',
                'target_pred_changed',
        ]:
            if key in item['cache']:
                features[key] = item['cache'][key]
        for feature, values in features.items():
            rows.append(cache_feature_summary({
                'feature': feature,
                'split': item['split'],
                'dataset': item['dataset'],
            }, item['cache'], feature, values))
    return rows


def build_reference_summary(caches):
    rows = []
    for item in caches:
        cache = item['cache']
        for name in [
                'base_reference_loss',
                'adapted_reference_loss',
                'delta',
                'reference_conf_delta_by_class',
                'reference_entropy_delta_by_class',
                'reference_margin_delta_by_class',
                'reference_energy_delta_by_class',
                'reference_pred_changed_rate_by_class',
                'reference_correct_rate_before_by_class',
                'reference_correct_rate_after_by_class',
        ]:
            if name not in cache:
                continue
            values = cache[name]
            if values.ndim != 2:
                continue
            feature_values = {
                f'{name}_all_classes': values.reshape(-1),
                f'{name}_per_sample_mean': np.mean(values, axis=1),
                f'{name}_per_sample_std': np.std(values, axis=1),
                f'{name}_per_sample_range': (
                    np.max(values, axis=1) - np.min(values, axis=1)),
                f'{name}_per_class_mean': np.mean(values, axis=0),
                f'{name}_per_class_std': np.std(values, axis=0),
            }
            for feature, feature_data in feature_values.items():
                rows.append(cache_feature_summary({
                    'feature': feature,
                    'split': item['split'],
                    'dataset': item['dataset'],
                }, cache, feature, feature_data))
    return rows


def binary_auroc(negative_scores, positive_scores):
    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    scores = np.concatenate([negative_scores, positive_scores])
    labels = np.concatenate([
        np.zeros(negative_scores.size, dtype=np.int64),
        np.ones(positive_scores.size, dtype=np.int64),
    ])
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    pos_rank_sum = float(np.sum(ranks[labels == 1]))
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def score_kind_dir_name(kind):
    return '' if kind == 'standard' else kind


def score_root(run_dir, scheme, reference_config_id=None):
    scheme_dir = Path(run_dir) / scheme
    if reference_config_id:
        return scheme_dir / 'references' / reference_config_id / SCORE_RESULTS_DIR
    return scheme_dir / SCORE_RESULTS_DIR


def score_kind_root(run_dir, scheme, reference_config_id, kind):
    root = score_root(run_dir, scheme, reference_config_id)
    kind_dir = score_kind_dir_name(kind)
    return root / kind_dir if kind_dir else root


def resolve_score_result_dir(run_dir, scheme, reference_config_id, kind,
                             fsood_id_side):
    root = score_kind_root(run_dir, scheme, reference_config_id, kind)
    if kind == 'standard' and fsood_id_side == 'both':
        return root
    if fsood_id_side != 'auto':
        return root / f'id_side_{fsood_id_side}'
    if kind == 'standard' and any(
            (root / rule / 'scores').exists() for rule in SCORE_RULE_CHOICES):
        return root
    candidates = sorted(path for path in root.glob('id_side_*') if path.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f'No {SCORE_RESULTS_DIR} id_side_* directory found under {root}')
    choices = ', '.join(path.name for path in candidates)
    raise ValueError(
        f'Multiple {SCORE_RESULTS_DIR} id_side_* directories found under '
        f'{root}: {choices}. '
        'Pass --fsood-id-side explicitly.')


def read_score_manifest(score_dir):
    for name in ['score.json', 'vector_score.json', 'perturbation_score.json']:
        manifest = read_json(Path(score_dir) / name)
        if manifest:
            return manifest
    return {}


def score_result_rules(score_dir, score_rule):
    if score_rule != 'all':
        return [score_rule]
    manifest = read_score_manifest(score_dir)
    rules = [
        rule for rule in manifest.get('expanded_score_rules', [])
        if (Path(score_dir) / rule / 'scores').exists()
    ]
    if rules:
        return rules
    return [
        path.name for path in sorted(Path(score_dir).iterdir())
        if path.is_dir() and (path / 'scores').exists()
    ]


def score_npz_value(cache, manifest=None):
    if 'ood_score' in cache:
        return cache['ood_score']
    if 'score' in cache:
        return cache['score']
    if 'conf' not in cache:
        raise KeyError('score npz is missing ood_score, score, and conf')
    transform = str(manifest.get('conf_boundary_transform', '')) if manifest else ''
    if transform.strip() == 'conf = ood_score':
        return cache['conf']
    return -cache['conf']


def score_dataset_names(args, manifest):
    dataset = manifest.get('dataset', args.dataset)
    names = [(dataset, 'id', dataset)]
    run_dir = Path(args.run_dir)
    scheme_manifest = read_json(run_dir / args.scheme / 'scheme_manifest.json')
    run_manifest = read_json(run_dir / 'run_manifest.json')
    csid_names = parse_dataset_list(manifest.get('csid_datasets'))
    if args.scheme == 'fsood' and not csid_names:
        csid_names = (
            manifest_csid_datasets(scheme_manifest)
            or manifest_csid_datasets(run_manifest)
            or parse_dataset_list(args.csid_datasets)
            or csid_datasets_from_run_manifest(args.run_dir, args.scheme))
    names.extend((name, 'csid', name) for name in csid_names)

    near_names = parse_dataset_list(manifest.get('near_datasets'))
    if not near_names:
        near_names = run_split_datasets(
            args.run_dir, args.scheme, dataset, 'near',
            scheme_manifest, run_manifest)
    far_names = parse_dataset_list(manifest.get('far_datasets'))
    if not far_names:
        far_names = run_split_datasets(
            args.run_dir, args.scheme, dataset, 'far',
            scheme_manifest, run_manifest)
    names.extend((name, 'near', name) for name in near_names)
    names.extend((name, 'far', name) for name in far_names)
    return names


def collect_score_result_items(args):
    score_dir = resolve_score_result_dir(
        args.run_dir, args.scheme, args.reference_config_id,
        args.score_kind, args.fsood_id_side)
    manifest = read_score_manifest(score_dir)
    rules = score_result_rules(score_dir, args.score_rule)
    if not rules:
        raise FileNotFoundError(f'No score rules found under {score_dir}')

    items = []
    for score_rule in rules:
        scores_dir = score_dir / score_rule / 'scores'
        if not scores_dir.exists():
            continue
        for name, split, dataset_name in score_dataset_names(args, manifest):
            path = scores_dir / f'{name}.npz'
            if not path.exists():
                continue
            items.append({
                'name': name,
                'split': split,
                'dataset': dataset_name,
                'score_rule': score_rule,
                'path': path,
                'cache': np.load(path),
                'score_dir': score_dir,
                'manifest': manifest,
            })
    return score_dir, manifest, items


def build_score_result_summary(items):
    rows = []
    for item in items:
        cache = item['cache']
        row = summary_row({
            'score_rule': item['score_rule'],
            'split': item['split'],
            'dataset': item['dataset'],
            'score_path': str(item['path']),
        }, score_npz_value(cache, item['manifest']))
        for key in ['pred', 'conf', 'label', 'ood_score', 'score']:
            row[f'{key}_shape'] = shape_text(cache[key]) if key in cache else ''
        rows.append(row)
    return rows


SCORE_ALIGNMENT_FIELDS = [
    'row_type', 'score_rule', 'run_dir', 'run_id', 'dataset',
    'baseline_protocol', 'scheme', 'reference_config_id', 'score_kind',
    'score_result_dir', 'fsood_metric_id_side', 'split', 'cache_dataset',
    'cache_name', 'cache_path', 'n_clean', 'n_csid', 'n_samples',
    'pred_shape', 'conf_shape', 'label_shape', 'score_shape',
    'score_label_shape_aligned', 'expected_csid_datasets',
    'resolved_csid_datasets', 'csid_alignment_status', 'mean_gap',
    'median_gap', 'standardized_mean_gap', 'id_vs_csid_auroc',
    'alignment_error', 'csid_tail_at_clean_q95', 'csid_in_clean_5_95',
    'clean_mean', 'csid_mean', 'clean_median', 'csid_median',
    'score_direction', 'conf_boundary_transform',
]


def empty_score_alignment_row(run_dir, run_manifest, scheme, reference_config_id,
                                score_kind, score_dir, metric_id_side,
                                dataset, baseline_protocol, expected_csid,
                                resolved_csid, alignment_status):
    row = {key: '' for key in SCORE_ALIGNMENT_FIELDS}
    row.update({
        'run_dir': str(run_dir),
        'run_id': run_manifest.get('run_id', Path(run_dir).name),
        'dataset': dataset,
        'baseline_protocol': baseline_protocol,
        'scheme': scheme,
        'reference_config_id': reference_config_id or '',
        'score_kind': score_kind,
        'score_result_dir': str(score_dir),
        'fsood_metric_id_side': metric_id_side,
        'expected_csid_datasets': ','.join(expected_csid),
        'resolved_csid_datasets': ','.join(resolved_csid),
        'csid_alignment_status': alignment_status,
    })
    return row


def build_score_alignment_summary(run_dir, scheme, reference_config_id,
                                    score_kind, score_dir, manifest, items):
    run_dir = Path(run_dir)
    run_manifest = read_json(run_dir / 'run_manifest.json')
    scheme_manifest = read_json(run_dir / scheme / 'scheme_manifest.json')
    dataset = identity_value(
        manifest, scheme_manifest, run_manifest, key='dataset')
    baseline_protocol = identity_value(
        scheme_manifest, run_manifest, key='baseline_protocol')
    expected_csid = expected_csid_datasets(dataset, baseline_protocol)
    resolved_csid = (
        parse_dataset_list(manifest.get('csid_datasets'))
        or manifest_csid_datasets(scheme_manifest)
        or manifest_csid_datasets(run_manifest)
        or csid_datasets_from_run_manifest(run_dir, scheme))
    alignment_status = csid_alignment_status(expected_csid, resolved_csid)
    score_dir_name = Path(score_dir).name
    metric_id_side = manifest.get(
        'fsood_id_side',
        score_dir_name[len('id_side_'):] if score_dir_name.startswith(
            'id_side_') else score_dir_name)

    rows = []
    for item in items:
        cache = item['cache']
        score_values = score_npz_value(cache, manifest)
        label_n = first_dim(cache['label']) if 'label' in cache else ''
        row = empty_score_alignment_row(
            run_dir, run_manifest, scheme, reference_config_id, score_kind,
            score_dir, metric_id_side, dataset, baseline_protocol,
            expected_csid, resolved_csid, alignment_status)
        row.update({
            'row_type': 'score_shape',
            'score_rule': item['score_rule'],
            'split': item['split'],
            'cache_dataset': item['dataset'],
            'cache_name': item['name'],
            'cache_path': str(item['path']),
            'n_samples': first_dim(score_values),
            'pred_shape': shape_text(cache['pred']) if 'pred' in cache else '',
            'conf_shape': shape_text(cache['conf']) if 'conf' in cache else '',
            'label_shape': shape_text(cache['label']) if 'label' in cache else '',
            'score_shape': shape_text(score_values),
            'score_label_shape_aligned': (
                first_dim(score_values) == label_n if label_n != '' else ''),
            'score_direction': manifest.get('score_direction', ''),
            'conf_boundary_transform': manifest.get('conf_boundary_transform', ''),
        })
        rows.append(row)

    by_rule = {}
    for item in items:
        by_rule.setdefault(item['score_rule'], []).append(item)
    for score_rule, rule_items in sorted(by_rule.items()):
        id_items = [item for item in rule_items if item['split'] == 'id']
        csid_items = [item for item in rule_items if item['split'] == 'csid']
        if not id_items or not csid_items:
            continue
        clean_scores = score_npz_value(id_items[0]['cache'], manifest)
        csid_scores = np.concatenate([
            score_npz_value(item['cache'], manifest) for item in csid_items
        ])
        clean_q05, clean_q95 = np.quantile(clean_scores, [0.05, 0.95])
        pooled_std = float(np.std(np.concatenate([clean_scores, csid_scores])))
        mean_gap = float(np.mean(csid_scores) - np.mean(clean_scores))
        auroc = binary_auroc(clean_scores, csid_scores)
        row = empty_score_alignment_row(
            run_dir, run_manifest, scheme, reference_config_id, score_kind,
            score_dir, metric_id_side, dataset, baseline_protocol,
            expected_csid, resolved_csid, alignment_status)
        row.update({
            'row_type': 'clean_vs_csid_score',
            'score_rule': score_rule,
            'split': 'id_vs_csid',
            'cache_dataset': 'clean_vs_csid',
            'cache_name': 'clean_vs_csid',
            'n_clean': int(clean_scores.size),
            'n_csid': int(csid_scores.size),
            'mean_gap': mean_gap,
            'median_gap': float(np.median(csid_scores) - np.median(clean_scores)),
            'standardized_mean_gap': (
                mean_gap / pooled_std if pooled_std > 0 else np.nan),
            'id_vs_csid_auroc': auroc,
            'alignment_error': abs(auroc - 0.5) if np.isfinite(auroc) else np.nan,
            'csid_tail_at_clean_q95': float(np.mean(csid_scores > clean_q95)),
            'csid_in_clean_5_95': float(
                np.mean((csid_scores >= clean_q05) & (csid_scores <= clean_q95))),
            'clean_mean': float(np.mean(clean_scores)),
            'csid_mean': float(np.mean(csid_scores)),
            'clean_median': float(np.median(clean_scores)),
            'csid_median': float(np.median(csid_scores)),
            'score_direction': manifest.get('score_direction', ''),
            'conf_boundary_transform': manifest.get('conf_boundary_transform', ''),
        })
        rows.append(row)
    return rows


def build_alignment_summary(run_dir, scheme, reference_config_id, caches,
                            score_rules):
    run_manifest = read_json(Path(run_dir) / 'run_manifest.json')
    scheme_manifest = read_json(Path(run_dir) / scheme / 'scheme_manifest.json')
    dataset = identity_value(scheme_manifest, run_manifest, key='dataset')
    if not dataset:
        dataset = run_manifest.get('dataset', '')
    baseline_protocol = identity_value(
        scheme_manifest, run_manifest, key='baseline_protocol')
    if not baseline_protocol:
        baseline_protocol = run_manifest.get('baseline_protocol', '')
    expected_csid = expected_csid_datasets(dataset, baseline_protocol)
    resolved_csid = (
        manifest_csid_datasets(scheme_manifest)
        or manifest_csid_datasets(run_manifest)
        or csid_datasets_from_run_manifest(run_dir, scheme))
    alignment_status = csid_alignment_status(expected_csid, resolved_csid)

    id_items = [item for item in caches if item['split'] == 'id']
    csid_items = [item for item in caches if item['split'] == 'csid']
    rows = []
    for item in caches:
        cache = item['cache']
        target_probs = cache['target_probs']
        delta = cache['delta']
        base_loss = cache['base_reference_loss']
        adapted_loss = cache['adapted_reference_loss']
        rows.append({
            'row_type': 'shape_protocol',
            'score_rule': '',
            'run_dir': str(run_dir),
            'run_id': run_manifest.get('run_id', Path(run_dir).name),
            'dataset': dataset,
            'baseline_protocol': baseline_protocol,
            'scheme': scheme,
            'reference_config_id': reference_config_id or '',
            'split': item['split'],
            'cache_dataset': item['dataset'],
            'cache_name': item['name'],
            'cache_path': str(item['path']),
            'cache_num_shards': item.get('cache_num_shards', 1),
            'cache_paths': cache_paths_text(item),
            'n_clean': '',
            'n_csid': '',
            'n_samples': int(cache['label'].shape[0]),
            'num_classes': int(target_probs.shape[1]) if target_probs.ndim == 2 else '',
            'target_probs_shape': shape_text(target_probs),
            'delta_shape': shape_text(delta),
            'base_reference_loss_shape': shape_text(base_loss),
            'adapted_reference_loss_shape': shape_text(adapted_loss),
            'target_reference_shape_aligned': (
                target_probs.shape == delta.shape == base_loss.shape
                == adapted_loss.shape),
            'expected_csid_datasets': ','.join(expected_csid),
            'resolved_csid_datasets': ','.join(resolved_csid),
            'csid_alignment_status': alignment_status,
            'fsood_metric_id_side': scheme_manifest.get(
                'fsood_metric_id_side', ''),
            'mean_gap': '',
            'median_gap': '',
            'standardized_mean_gap': '',
            'id_vs_csid_auroc': '',
            'alignment_error': '',
            'csid_tail_at_clean_q95': '',
            'csid_in_clean_5_95': '',
            'clean_mean': '',
            'csid_mean': '',
            'clean_median': '',
            'csid_median': '',
            'cache_schema_version': scalar_value(
                cache['cache_schema_version']) if 'cache_schema_version' in cache else '',
            'score_direction': scalar_value(
                cache['score_direction']) if 'score_direction' in cache else '',
            'delta_definition': scalar_value(
                cache['delta_definition']) if 'delta_definition' in cache else '',
        })
    if id_items and csid_items:
        clean_cache = id_items[0]['cache']
        for score_rule in score_rules:
            clean_scores = ood_score_from_cache(clean_cache, score_rule)
            csid_scores = np.concatenate([
                ood_score_from_cache(item['cache'], score_rule)
                for item in csid_items
            ])
            clean_q05, clean_q95 = np.quantile(clean_scores, [0.05, 0.95])
            pooled_std = float(np.std(np.concatenate([clean_scores, csid_scores])))
            mean_gap = float(np.mean(csid_scores) - np.mean(clean_scores))
            auroc = binary_auroc(clean_scores, csid_scores)
            rows.append({
                'row_type': 'clean_vs_csid_score',
                'score_rule': score_rule,
                'run_dir': str(run_dir),
                'run_id': run_manifest.get('run_id', Path(run_dir).name),
                'dataset': dataset,
                'baseline_protocol': baseline_protocol,
                'scheme': scheme,
                'reference_config_id': reference_config_id or '',
                'split': 'id_vs_csid',
                'cache_dataset': 'clean_vs_csid',
                'cache_name': 'clean_vs_csid',
                'cache_path': '',
                'cache_num_shards': '',
                'cache_paths': '',
                'n_clean': int(clean_scores.size),
                'n_csid': int(csid_scores.size),
                'n_samples': '',
                'num_classes': '',
                'target_probs_shape': '',
                'delta_shape': '',
                'base_reference_loss_shape': '',
                'adapted_reference_loss_shape': '',
                'target_reference_shape_aligned': '',
                'expected_csid_datasets': ','.join(expected_csid),
                'resolved_csid_datasets': ','.join(resolved_csid),
                'csid_alignment_status': alignment_status,
                'fsood_metric_id_side': scheme_manifest.get(
                    'fsood_metric_id_side', ''),
                'mean_gap': mean_gap,
                'median_gap': float(np.median(csid_scores)
                                    - np.median(clean_scores)),
                'standardized_mean_gap': (
                    mean_gap / pooled_std if pooled_std > 0 else np.nan),
                'id_vs_csid_auroc': auroc,
                'alignment_error': abs(auroc - 0.5) if np.isfinite(auroc) else np.nan,
                'csid_tail_at_clean_q95': float(np.mean(csid_scores > clean_q95)),
                'csid_in_clean_5_95': float(
                    np.mean((csid_scores >= clean_q05) & (csid_scores <= clean_q95))),
                'clean_mean': float(np.mean(clean_scores)),
                'csid_mean': float(np.mean(csid_scores)),
                'clean_median': float(np.median(clean_scores)),
                'csid_median': float(np.median(csid_scores)),
                'cache_schema_version': scalar_value(
                    clean_cache['cache_schema_version'])
                if 'cache_schema_version' in clean_cache else '',
                'score_direction': scalar_value(
                    clean_cache['score_direction'])
                if 'score_direction' in clean_cache else '',
                'delta_definition': scalar_value(
                    clean_cache['delta_definition'])
                if 'delta_definition' in clean_cache else '',
            })
    return rows


def _vector_fit_from_caches(caches):
    id_items = [item for item in caches if item['split'] == 'id']
    if not id_items:
        raise ValueError('Vector diagnostics require a clean ID cache')
    return fit_vector_score_reference(id_items[0]['cache'])


def build_vector_summary(caches, vector_score_rules):
    rows = []
    vector_fit = _vector_fit_from_caches(caches)
    for item in caches:
        for score_rule in vector_score_rules:
            rows.append(summary_row({
                'score_family': 'vector',
                'score_rule': score_rule,
                'split': item['split'],
                'dataset': item['dataset'],
                'fit_source': 'clean_id_cache',
            }, vector_ood_score_from_cache(
                item['cache'], score_rule, vector_fit)))
    return rows


def build_vector_alignment_summary(run_dir, scheme, reference_config_id, caches,
                                   vector_score_rules):
    run_manifest = read_json(Path(run_dir) / 'run_manifest.json')
    scheme_manifest = read_json(Path(run_dir) / scheme / 'scheme_manifest.json')
    dataset = identity_value(scheme_manifest, run_manifest, key='dataset')
    if not dataset:
        dataset = run_manifest.get('dataset', '')
    baseline_protocol = identity_value(
        scheme_manifest, run_manifest, key='baseline_protocol')
    if not baseline_protocol:
        baseline_protocol = run_manifest.get('baseline_protocol', '')
    expected_csid = expected_csid_datasets(dataset, baseline_protocol)
    resolved_csid = (
        manifest_csid_datasets(scheme_manifest)
        or manifest_csid_datasets(run_manifest)
        or csid_datasets_from_run_manifest(run_dir, scheme))
    alignment_status = csid_alignment_status(expected_csid, resolved_csid)

    id_items = [item for item in caches if item['split'] == 'id']
    csid_items = [item for item in caches if item['split'] == 'csid']
    if not id_items or not csid_items:
        return []

    clean_cache = id_items[0]['cache']
    vector_fit = fit_vector_score_reference(clean_cache)
    rows = []
    for score_rule in vector_score_rules:
        clean_scores = vector_ood_score_from_cache(
            clean_cache, score_rule, vector_fit)
        csid_scores = np.concatenate([
            vector_ood_score_from_cache(item['cache'], score_rule, vector_fit)
            for item in csid_items
        ])
        clean_q05, clean_q95 = np.quantile(clean_scores, [0.05, 0.95])
        pooled_std = float(np.std(np.concatenate([clean_scores, csid_scores])))
        mean_gap = float(np.mean(csid_scores) - np.mean(clean_scores))
        auroc = binary_auroc(clean_scores, csid_scores)
        rows.append({
            'score_family': 'vector',
            'score_rule': score_rule,
            'run_dir': str(run_dir),
            'run_id': run_manifest.get('run_id', Path(run_dir).name),
            'dataset': dataset,
            'baseline_protocol': baseline_protocol,
            'scheme': scheme,
            'reference_config_id': reference_config_id or '',
            'fit_source': 'clean_id_cache',
            'expected_csid_datasets': ','.join(expected_csid),
            'resolved_csid_datasets': ','.join(resolved_csid),
            'csid_alignment_status': alignment_status,
            'n_clean': int(clean_scores.size),
            'n_csid': int(csid_scores.size),
            'mean_gap': mean_gap,
            'median_gap': float(np.median(csid_scores)
                                - np.median(clean_scores)),
            'standardized_mean_gap': (
                mean_gap / pooled_std if pooled_std > 0 else np.nan),
            'id_vs_csid_auroc': auroc,
            'alignment_error': abs(auroc - 0.5) if np.isfinite(auroc) else np.nan,
            'csid_tail_at_clean_q95': float(np.mean(csid_scores > clean_q95)),
            'csid_in_clean_5_95': float(
                np.mean((csid_scores >= clean_q05) & (csid_scores <= clean_q95))),
            'clean_mean': float(np.mean(clean_scores)),
            'csid_mean': float(np.mean(csid_scores)),
            'clean_median': float(np.median(clean_scores)),
            'csid_median': float(np.median(csid_scores)),
        })
    return rows


def perturbation_keys(cache):
    return [
        key for key in cache.files
        if key.startswith('perturbation_') or key.startswith('perturbed_')
    ]


def has_perturbation_fields(caches):
    return any(perturbation_keys(item['cache']) for item in caches)


def is_numeric_array(value):
    value = np.asarray(value)
    return (
        np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.bool_)
    )


def finite_array(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def decode_text_list(value):
    result = []
    for item in np.asarray(value).reshape(-1):
        if isinstance(item, bytes):
            text = item.decode('utf-8')
        else:
            text = str(item)
        text = text.strip()
        if text:
            result.append(text)
    return result


def perturbation_rule_names(cache):
    for key in PERTURBATION_SCORE_NAME_KEYS:
        if key in cache:
            names = decode_text_list(cache[key])
            if names:
                return names
    return []


def perturbation_distribution_row(prefix, cache, values):
    row = summary_row(prefix, values)
    row['array_shape'] = shape_text(values)
    row['cache_schema_version'] = scalar_value(
        cache['cache_schema_version']) if 'cache_schema_version' in cache else ''
    row['delta_definition'] = scalar_value(
        cache['delta_definition']) if 'delta_definition' in cache else ''
    row['score_direction'] = scalar_value(
        cache['score_direction']) if 'score_direction' in cache else ''
    return row


def iter_named_score_columns(cache, key, names, sample_count):
    values = np.asarray(cache[key])
    if not is_numeric_array(values):
        return
    if values.ndim == 1 and values.shape[0] == sample_count:
        rule = names[0] if len(names) == 1 else key
        yield rule, values.astype(np.float64), key
        return
    if values.ndim != 2:
        return
    matrix = values
    if matrix.shape[0] == sample_count:
        pass
    elif matrix.shape[1] == sample_count:
        matrix = matrix.T
    else:
        return
    for idx in range(matrix.shape[1]):
        rule = names[idx] if idx < len(names) else f'{key}_{idx}'
        yield rule, matrix[:, idx].astype(np.float64), key


def perturbation_score_arrays(cache):
    sample_count = first_dim(cache['label']) if 'label' in cache else ''
    if sample_count == '':
        return []
    arrays = []
    current_rules = selected_perturbation_score_rules('all')
    allowed_rules = set(current_rules)
    for rule in current_rules:
        try:
            values = perturbation_ood_score_from_cache(cache, rule)
        except Exception:
            continue
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != sample_count:
            continue
        arrays.append({
            'score_rule': rule,
            'values': values,
            'source_field': 'computed_from_perturbation_fields',
        })
    names = perturbation_rule_names(cache)
    seen = {(item['score_rule'], item['source_field']) for item in arrays}
    for key in PERTURBATION_SCORE_MATRIX_KEYS:
        if key not in cache:
            continue
        for rule, values, source_field in iter_named_score_columns(
                cache, key, names, sample_count):
            if rule not in allowed_rules:
                continue
            identity = (rule, source_field)
            if identity in seen:
                continue
            seen.add(identity)
            arrays.append({
                'score_rule': rule,
                'values': values,
                'source_field': source_field,
            })
    return arrays


def perturbation_feature_arrays(cache):
    sample_count = first_dim(cache['label']) if 'label' in cache else ''
    rows = []
    for key in PERTURBATION_DIAGNOSTIC_FIELDS:
        if key not in cache:
            continue
        if key in PERTURBATION_SCORE_NAME_KEYS:
            continue
        values = np.asarray(cache[key])
        if not is_numeric_array(values):
            continue
        if values.ndim == 0:
            rows.append((key, key, values.reshape(1)))
            continue
        if values.ndim == 1:
            rows.append((key, key, values))
            continue
        if sample_count != '' and values.shape[0] == sample_count:
            rows.extend([
                (f'{key}_all_values', key, values.reshape(-1)),
                (f'{key}_per_sample_mean', key, np.mean(values, axis=1)),
                (f'{key}_per_sample_std', key, np.std(values, axis=1)),
                (f'{key}_per_sample_range', key,
                 np.max(values, axis=1) - np.min(values, axis=1)),
                (f'{key}_per_column_mean', key, np.mean(values, axis=0)),
                (f'{key}_per_column_std', key, np.std(values, axis=0)),
            ])
        else:
            rows.append((f'{key}_all_values', key, values.reshape(-1)))
    return rows


def build_perturbation_summary(caches):
    rows = []
    for item in caches:
        cache = item['cache']
        for score in perturbation_score_arrays(cache):
            rows.append(perturbation_distribution_row({
                'row_type': 'perturbation_score',
                'score_family': 'perturbation',
                'score_rule': score['score_rule'],
                'feature': '',
                'source_field': score['source_field'],
                'split': item['split'],
                'dataset': item['dataset'],
            }, cache, score['values']))
        for feature, source_field, values in perturbation_feature_arrays(cache):
            rows.append(perturbation_distribution_row({
                'row_type': 'perturbation_field',
                'score_family': 'perturbation',
                'score_rule': '',
                'feature': feature,
                'source_field': source_field,
                'split': item['split'],
                'dataset': item['dataset'],
            }, cache, values))
    return rows


def perturbation_protocol_context(run_dir, scheme):
    run_manifest = read_json(Path(run_dir) / 'run_manifest.json')
    scheme_manifest = read_json(Path(run_dir) / scheme / 'scheme_manifest.json')
    dataset = identity_value(scheme_manifest, run_manifest, key='dataset')
    if not dataset:
        dataset = run_manifest.get('dataset', '')
    baseline_protocol = identity_value(
        scheme_manifest, run_manifest, key='baseline_protocol')
    if not baseline_protocol:
        baseline_protocol = run_manifest.get('baseline_protocol', '')
    expected_csid = expected_csid_datasets(dataset, baseline_protocol)
    resolved_csid = (
        manifest_csid_datasets(scheme_manifest)
        or manifest_csid_datasets(run_manifest)
        or csid_datasets_from_run_manifest(run_dir, scheme))
    alignment_status = csid_alignment_status(expected_csid, resolved_csid)
    return {
        'run_manifest': run_manifest,
        'scheme_manifest': scheme_manifest,
        'dataset': dataset,
        'baseline_protocol': baseline_protocol,
        'expected_csid': expected_csid,
        'resolved_csid': resolved_csid,
        'alignment_status': alignment_status,
    }


def empty_perturbation_alignment_row(run_dir, scheme, reference_config_id,
                                     context):
    row = {key: '' for key in PERTURBATION_ALIGNMENT_FIELDS}
    row.update({
        'score_family': 'perturbation',
        'run_dir': str(run_dir),
        'run_id': context['run_manifest'].get('run_id', Path(run_dir).name),
        'dataset': context['dataset'],
        'baseline_protocol': context['baseline_protocol'],
        'scheme': scheme,
        'reference_config_id': reference_config_id or '',
        'expected_csid_datasets': ','.join(context['expected_csid']),
        'resolved_csid_datasets': ','.join(context['resolved_csid']),
        'csid_alignment_status': context['alignment_status'],
    })
    return row


def perturbation_score_items(caches):
    rows = []
    for item in caches:
        for score in perturbation_score_arrays(item['cache']):
            rows.append({
                'split': item['split'],
                'dataset': item['dataset'],
                'name': item['name'],
                'path': item['path'],
                'cache_num_shards': item.get('cache_num_shards', 1),
                'cache_paths': item.get('cache_paths', [item['path']]),
                'cache': item['cache'],
                'score_rule': score['score_rule'],
                'source_field': score['source_field'],
                'values': finite_array(score['values']),
                'score_shape': shape_text(score['values']),
            })
    return rows


def alignment_metrics(negative_scores, positive_scores):
    negative_scores = finite_array(negative_scores)
    positive_scores = finite_array(positive_scores)
    if negative_scores.size == 0 or positive_scores.size == 0:
        return {}
    q05, q95 = np.quantile(negative_scores, [0.05, 0.95])
    pooled_std = float(np.std(np.concatenate([negative_scores, positive_scores])))
    mean_gap = float(np.mean(positive_scores) - np.mean(negative_scores))
    auroc = binary_auroc(negative_scores, positive_scores)
    return {
        'mean_gap': mean_gap,
        'median_gap': float(np.median(positive_scores)
                            - np.median(negative_scores)),
        'standardized_mean_gap': (
            mean_gap / pooled_std if pooled_std > 0 else np.nan),
        'auroc': auroc,
        'tail_at_negative_q95': float(np.mean(positive_scores > q95)),
        'positive_in_negative_5_95': float(
            np.mean((positive_scores >= q05) & (positive_scores <= q95))),
        'negative_mean': float(np.mean(negative_scores)),
        'positive_mean': float(np.mean(positive_scores)),
        'negative_median': float(np.median(negative_scores)),
        'positive_median': float(np.median(positive_scores)),
        'negative_n': int(negative_scores.size),
        'positive_n': int(positive_scores.size),
    }


def grouped_ood_items(rule_items):
    groups = []
    ood_items = [item for item in rule_items if item['split'] in {'near', 'far'}]
    groups.extend((item['split'], item['dataset'], [item]) for item in ood_items)
    for split in ['near', 'far']:
        split_items = [item for item in ood_items if item['split'] == split]
        if split_items:
            groups.append((split, f'{split}_all', split_items))
    if ood_items:
        groups.append(('ood', 'ood_all', ood_items))
    return groups


def build_perturbation_alignment_summary(run_dir, scheme, reference_config_id,
                                         caches):
    context = perturbation_protocol_context(run_dir, scheme)
    rows = []
    score_items = perturbation_score_items(caches)
    for item in score_items:
        cache = item['cache']
        row = empty_perturbation_alignment_row(
            run_dir, scheme, reference_config_id, context)
        row.update({
            'row_type': 'score_shape',
            'score_rule': item['score_rule'],
            'source_field': item['source_field'],
            'split': item['split'],
            'cache_dataset': item['dataset'],
            'cache_name': item['name'],
            'cache_path': str(item['path']),
            'cache_num_shards': item.get('cache_num_shards', 1),
            'cache_paths': cache_paths_text(item),
            'n_samples': int(item['values'].size),
            'score_shape': item['score_shape'],
            'cache_schema_version': scalar_value(
                cache['cache_schema_version'])
            if 'cache_schema_version' in cache else '',
            'score_direction': scalar_value(
                cache['score_direction']) if 'score_direction' in cache else '',
            'delta_definition': scalar_value(
                cache['delta_definition']) if 'delta_definition' in cache else '',
        })
        rows.append(row)

    by_rule = {}
    for item in score_items:
        by_rule.setdefault(
            (item['score_rule'], item['source_field']), []).append(item)

    for (score_rule, source_field), rule_items in sorted(by_rule.items()):
        id_items = [item for item in rule_items if item['split'] == 'id']
        csid_items = [item for item in rule_items if item['split'] == 'csid']
        if id_items and csid_items:
            clean_scores = id_items[0]['values']
            csid_scores = np.concatenate([item['values'] for item in csid_items])
            metrics = alignment_metrics(clean_scores, csid_scores)
            row = empty_perturbation_alignment_row(
                run_dir, scheme, reference_config_id, context)
            row.update({
                'row_type': 'clean_vs_csid_score',
                'score_rule': score_rule,
                'source_field': source_field,
                'split': 'id_vs_csid',
                'cache_dataset': 'clean_vs_csid',
                'cache_name': 'clean_vs_csid',
                'n_clean': metrics.get('negative_n', ''),
                'n_csid': metrics.get('positive_n', ''),
                'mean_gap': metrics.get('mean_gap', ''),
                'median_gap': metrics.get('median_gap', ''),
                'standardized_mean_gap': metrics.get(
                    'standardized_mean_gap', ''),
                'id_vs_csid_auroc': metrics.get('auroc', ''),
                'alignment_error': (
                    abs(metrics['auroc'] - 0.5)
                    if np.isfinite(metrics.get('auroc', np.nan)) else ''),
                'csid_tail_at_clean_q95': metrics.get(
                    'tail_at_negative_q95', ''),
                'csid_in_clean_5_95': metrics.get(
                    'positive_in_negative_5_95', ''),
                'clean_mean': metrics.get('negative_mean', ''),
                'csid_mean': metrics.get('positive_mean', ''),
                'clean_median': metrics.get('negative_median', ''),
                'csid_median': metrics.get('positive_median', ''),
            })
            rows.append(row)

        negative_sets = []
        if id_items:
            negative_sets.append(('clean', id_items[0]['values']))
        if csid_items:
            negative_sets.append((
                'csid', np.concatenate([item['values'] for item in csid_items])))
        if id_items and csid_items:
            negative_sets.append((
                'both',
                np.concatenate([
                    id_items[0]['values'],
                    *[item['values'] for item in csid_items],
                ]),
            ))
        for ood_split, ood_dataset, ood_group in grouped_ood_items(rule_items):
            ood_scores = np.concatenate([item['values'] for item in ood_group])
            for id_side, negative_scores in negative_sets:
                metrics = alignment_metrics(negative_scores, ood_scores)
                row = empty_perturbation_alignment_row(
                    run_dir, scheme, reference_config_id, context)
                row.update({
                    'row_type': 'ood_separation_score',
                    'score_rule': score_rule,
                    'source_field': source_field,
                    'id_side': id_side,
                    'ood_split': ood_split,
                    'ood_dataset': ood_dataset,
                    'n_id_side': metrics.get('negative_n', ''),
                    'n_ood': metrics.get('positive_n', ''),
                    'mean_gap': metrics.get('mean_gap', ''),
                    'median_gap': metrics.get('median_gap', ''),
                    'standardized_mean_gap': metrics.get(
                        'standardized_mean_gap', ''),
                    'id_side_vs_ood_auroc': metrics.get('auroc', ''),
                    'separation_error': (
                        1.0 - metrics['auroc']
                        if np.isfinite(metrics.get('auroc', np.nan)) else ''),
                    'ood_tail_at_id_q95': metrics.get(
                        'tail_at_negative_q95', ''),
                    'ood_in_id_5_95': metrics.get(
                        'positive_in_negative_5_95', ''),
                    'id_side_mean': metrics.get('negative_mean', ''),
                    'ood_mean': metrics.get('positive_mean', ''),
                    'id_side_median': metrics.get('negative_median', ''),
                    'ood_median': metrics.get('positive_median', ''),
                })
                rows.append(row)
    return rows


def build_score_direction(caches, score_rules):
    id_items = [item for item in caches if item['split'] == 'id']
    if not id_items:
        return []
    id_cache = id_items[0]['cache']
    rows = []
    for score_rule in score_rules:
        id_ood_score = ood_score_from_cache(id_cache, score_rule)
        id_mean = float(np.mean(id_ood_score))
        id_median = float(np.median(id_ood_score))
        for item in caches:
            if item['split'] == 'id':
                continue
            ood_score = ood_score_from_cache(item['cache'], score_rule)
            mean_delta = float(np.mean(ood_score) - id_mean)
            median_delta = float(np.median(ood_score) - id_median)
            rows.append({
                'score_rule': score_rule,
                'split': item['split'],
                'dataset': item['dataset'],
                'id_mean': id_mean,
                'other_mean': float(np.mean(ood_score)),
                'mean_ood_minus_id': mean_delta,
                'id_median': id_median,
                'other_median': float(np.median(ood_score)),
                'median_ood_minus_id': median_delta,
                'mean_sign_flip': mean_delta < 0,
                'median_sign_flip': median_delta < 0,
            })
    return rows


def build_failure_cases(caches, score_rules, top_k):
    rows = []
    for item in caches:
        cache = item['cache']
        features = delta_features(cache)
        for score_rule in score_rules:
            ood_score = ood_score_from_cache(cache, score_rule)
            order = np.argsort(ood_score)
            if item['split'] in {'id', 'csid'}:
                selected = order[-top_k:][::-1]
                case_type = 'id_like_split_ood_like_score'
            else:
                selected = order[:top_k]
                case_type = 'ood_split_id_like_score'
            for rank, idx in enumerate(selected):
                rows.append({
                    'case_type': case_type,
                    'rank': rank,
                    'score_rule': score_rule,
                    'split': item['split'],
                    'dataset': item['dataset'],
                    'cache_index': int(idx),
                    'ood_score': float(ood_score[idx]),
                    'pred': int(cache['pred'][idx]),
                    'label': int(cache['label'][idx]),
                    'y_hat': int(cache['y_hat'][idx]),
                    'target_conf': float(cache['target_conf'][idx]),
                    'target_entropy': float(cache['target_entropy'][idx]),
                    'runtime_per_sample': float(cache['runtime_per_sample'][idx]),
                    'predicted_class_loss_increase': float(
                        features['predicted_class_loss_increase'][idx]),
                    'mean_loss_increase': float(
                        features['mean_loss_increase'][idx]),
                    'positive_loss_increase_mean': float(
                        features['positive_loss_increase_mean'][idx]),
                    'positive_loss_decrease_mean': float(
                        features['positive_loss_decrease_mean'][idx]),
                    'classwise_max_delta': float(
                        features['classwise_max_delta'][idx]),
                })
    return rows


def run_diagnostics(args):
    run_dir = Path(args.run_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.reference_config_id:
        output_dir = (run_dir / args.scheme / 'references' /
                      args.reference_config_id / 'diagnostics')
    else:
        output_dir = run_dir / args.scheme / 'diagnostics'
    caches = collect_caches(args)
    if not caches:
        raise FileNotFoundError(
            f'No TTA response files found under {run_dir / args.scheme}')

    score_rules = selected_score_rules(args.score_rule)
    write_csv(output_dir / 'score_summary.csv',
              build_score_summary(caches, score_rules))
    write_csv(output_dir / 'target_summary.csv', build_target_summary(caches))
    write_csv(output_dir / 'delta_summary.csv', build_delta_summary(caches))
    write_csv(output_dir / 'runtime_summary.csv', build_runtime_summary(caches))
    write_csv(output_dir / 'reference_summary.csv',
              build_reference_summary(caches))
    write_csv(output_dir / 'alignment_summary.csv',
              build_alignment_summary(run_dir, args.scheme,
                                      args.reference_config_id, caches,
                                      score_rules))
    write_csv(output_dir / 'score_direction.csv',
              build_score_direction(caches, score_rules))
    write_csv(output_dir / 'failure_cases.csv',
              build_failure_cases(caches, score_rules, args.top_k))
    if has_perturbation_fields(caches):
        write_csv(output_dir / 'perturbation_summary.csv',
                  build_perturbation_summary(caches),
                  PERTURBATION_SUMMARY_FIELDS)
        write_csv(output_dir / 'perturbation_alignment_summary.csv',
                  build_perturbation_alignment_summary(
                      run_dir, args.scheme, args.reference_config_id, caches),
                  PERTURBATION_ALIGNMENT_FIELDS)
    if args.vector_score_rule:
        vector_score_rules = selected_vector_score_rules(args.vector_score_rule)
        write_csv(output_dir / 'vector_summary.csv',
                  build_vector_summary(caches, vector_score_rules))
        write_csv(output_dir / 'vector_alignment_summary.csv',
                  build_vector_alignment_summary(
                      run_dir, args.scheme, args.reference_config_id,
                      caches, vector_score_rules))
    print(f'output_dir: {output_dir}')


def run_score_diagnostics(args):
    score_dir, manifest, items = collect_score_result_items(args)
    if not items:
        raise FileNotFoundError(f'No score npz files found under {score_dir}')
    output_dir = Path(args.output_dir) if args.output_dir else score_dir / 'diagnostics'
    write_csv(output_dir / 'score_summary.csv',
              build_score_result_summary(items))
    write_csv(output_dir / 'alignment_summary.csv',
              build_score_alignment_summary(
                  args.run_dir, args.scheme, args.reference_config_id,
                  args.score_kind, score_dir, manifest, items),
              SCORE_ALIGNMENT_FIELDS)
    write_json(output_dir / 'diagnostics_manifest.json', {
        'source_score_dir': str(score_dir),
        'score_kind': args.score_kind,
        'scheme': args.scheme,
        'reference_config_id': args.reference_config_id or '',
        'score_rule_arg': args.score_rule,
    })
    print(f'output_dir: {output_dir}')


def iter_run_dirs(root, dataset=None, protocol=None):
    root = Path(root)
    if (root / 'run_manifest.json').exists():
        yield root
        return
    for path in sorted(root.glob('*/*/seed*/*')):
        if not path.is_dir():
            continue
        parts = path.relative_to(root).parts
        if len(parts) < 4:
            continue
        ds, proto = parts[0], parts[1]
        if dataset and ds != dataset:
            continue
        if protocol and proto != protocol:
            continue
        yield path


def read_metrics(path):
    with Path(path).open(newline='') as f:
        return {row['dataset']: row for row in csv.DictReader(f)}


def validate_run(run_dir, run_manifest, allow_smoke):
    if not allow_smoke and not run_manifest.get('is_full_run', False):
        raise ValueError('subset/smoke run')
    unknown = [
        rule for rule in run_manifest.get('expanded_score_rules', [])
        if rule not in ACTIVE_SCORE_RULES
    ]
    if unknown:
        raise ValueError(f'noncanonical score rules: {unknown}')
    if 'baseline_protocol' not in run_manifest:
        raise ValueError('missing baseline_protocol')
    if not (run_dir / 'run_info.md').exists():
        raise ValueError('missing run_info.md')


def identity_value(*manifests, key, default=''):
    for manifest in manifests:
        if key in manifest and manifest[key] not in {None, ''}:
            return manifest[key]
    return default


def reference_config_ids_from_manifest(scheme_manifest, run_manifest):
    for manifest in [scheme_manifest, run_manifest]:
        configs = manifest.get('reference_configs')
        if isinstance(configs, list):
            ids = [item.get('id') for item in configs if isinstance(item, dict)]
            return [item for item in ids if item]
        if isinstance(configs, dict):
            return list(configs.keys())
    return []


def reference_metric_paths(run_dir, scheme, scheme_manifest, run_manifest):
    scheme_dir = run_dir / scheme
    score_rules = scheme_manifest.get(
        'expanded_score_rules',
        run_manifest.get('expanded_score_rules', []),
    )
    refs_dir = scheme_dir / 'references'
    if not refs_dir.exists():
        raise ValueError(f'missing reference-specific metrics directory: {refs_dir}')
    ref_ids = reference_config_ids_from_manifest(scheme_manifest, run_manifest)
    if not ref_ids:
        ref_ids = [path.name for path in sorted(refs_dir.iterdir())
                   if path.is_dir()]
    if not ref_ids:
        raise ValueError(f'no reference configs found in {refs_dir}')
    for ref_id in ref_ids:
        ref_dir = refs_dir / ref_id
        missing = []
        for score_rule in score_rules:
            metric_path = (
                ref_dir / SCORE_RESULTS_DIR / score_rule / 'ood.csv')
            if metric_path.exists():
                yield ref_id, score_rule, metric_path
            else:
                missing.append(score_rule)
        if missing:
            raise ValueError(
                f'missing canonical {SCORE_RESULTS_DIR} metric files for '
                f'reference {ref_id}: {missing}')


def collect_run_rows(run_dir, allow_smoke=False):
    run_dir = Path(run_dir)
    run_manifest_path = run_dir / 'run_manifest.json'
    if not run_manifest_path.exists():
        raise ValueError('missing run_manifest.json')
    run_manifest = load_json(run_manifest_path)
    validate_run(run_dir, run_manifest, allow_smoke)

    rows = []
    for scheme in run_manifest.get('schemes', []):
        scheme_manifest_path = run_dir / scheme / 'scheme_manifest.json'
        if not scheme_manifest_path.exists():
            raise ValueError(f'missing {scheme}/scheme_manifest.json')
        scheme_manifest = load_json(scheme_manifest_path)
        if scheme == 'fsood' and scheme_manifest.get('fsood_metric_id_side') != 'both':
            raise ValueError('FSOOD metric ID side is not both')
        for ref_id, score_rule, metric_path in reference_metric_paths(
                run_dir, scheme, scheme_manifest, run_manifest):
            metrics = read_metrics(metric_path)
            for aggregate in ['nearood', 'farood']:
                if aggregate not in metrics:
                    raise ValueError(f'missing {aggregate} row in {metric_path}')
                row = metrics[aggregate]
                output = {
                    'run_dir': str(run_dir),
                    'run_id': run_manifest.get('run_id', run_dir.name),
                    'dataset': run_manifest.get('dataset', ''),
                    'baseline_protocol': run_manifest.get('baseline_protocol', ''),
                    'seed': run_manifest.get('seed', ''),
                    'scheme': scheme,
                    'reference_config_id': ref_id,
                    'tta_config_id': identity_value(
                        scheme_manifest, run_manifest, key='tta_config_id'),
                    'scoring_config_id': identity_value(
                        scheme_manifest, run_manifest,
                        key='scoring_config_id', default=score_rule),
                    'protocol_config_id': identity_value(
                        scheme_manifest, run_manifest, key='protocol_config_id'),
                    'cache_run_id': identity_value(
                        scheme_manifest, run_manifest,
                        key='cache_run_id', default=run_manifest.get('run_id', '')),
                    'score_result_id': metric_path.parent.name,
                    'score_rule': score_rule,
                    'aggregate': aggregate,
                    'is_full_run': run_manifest.get('is_full_run', ''),
                    'fsood_metric_id_side': scheme_manifest.get(
                        'fsood_metric_id_side', ''),
                }
                for field in METRIC_FIELDS:
                    output[field] = row.get(field, '')
                rows.append(output)
    return rows


def run_collect(args):
    output_rows = []
    skipped = []
    for run_dir in iter_run_dirs(args.runs_root, args.dataset,
                                 args.baseline_protocol):
        try:
            output_rows.extend(collect_run_rows(run_dir, args.allow_smoke))
        except ValueError as exc:
            skipped.append({'run_dir': str(run_dir), 'reason': str(exc)})

    fieldnames = [
        'run_dir',
        'run_id',
        'dataset',
        'baseline_protocol',
        'seed',
        'scheme',
        'reference_config_id',
        'tta_config_id',
        'scoring_config_id',
        'protocol_config_id',
        'cache_run_id',
        'score_result_id',
        'score_rule',
        'aggregate',
        'is_full_run',
        'fsood_metric_id_side',
    ] + METRIC_FIELDS
    write_csv(args.output_csv, output_rows, fieldnames)
    if skipped:
        skip_path = Path(args.output_csv).with_suffix('.skipped.csv')
        write_csv(skip_path, skipped, ['run_dir', 'reason'])
    print(f'rows: {len(output_rows)}')
    print(f'output_csv: {args.output_csv}')


def infer_score_context(metric_path):
    metric_path = Path(metric_path)
    parts = metric_path.parts
    try:
        score_results_index = parts.index(SCORE_RESULTS_DIR)
    except ValueError as exc:
        raise ValueError(
            f'Unexpected score result metric path: {metric_path}') from exc
    score_rule = metric_path.parent.name
    score_results_parts = parts[score_results_index + 1:-2]
    id_side_index = None
    for idx, name in enumerate(score_results_parts):
        if name.startswith('id_side_'):
            id_side_index = idx
            break
    if id_side_index is None:
        fsood_id_side = 'both'
        family_parts = score_results_parts
    else:
        fsood_id_side = score_results_parts[id_side_index].replace(
            'id_side_', '', 1)
        family_parts = score_results_parts[:id_side_index]
    if family_parts:
        score_family = family_parts[0]
    else:
        score_family = 'score'

    score_results_dir = Path(*parts[:score_results_index + 1])
    if (score_results_index >= 2
            and parts[score_results_index - 2] == 'references'):
        reference_config_id = parts[score_results_index - 1]
        scheme_dir = Path(*parts[:score_results_index - 2])
    else:
        reference_config_id = ''
        scheme_dir = Path(*parts[:score_results_index])
    run_dir = scheme_dir.parent
    return {
        'run_dir': run_dir,
        'scheme': scheme_dir.name,
        'reference_config_id': reference_config_id,
        'score_rule': score_rule,
        'score_family': score_family,
        'fsood_id_side': fsood_id_side,
        'score_results_dir': score_results_dir,
    }


def collect_score_rows(metric_path):
    context = infer_score_context(metric_path)
    run_manifest = read_json(context['run_dir'] / 'run_manifest.json')
    rows = []
    metrics = read_metrics(metric_path)
    for aggregate, row in metrics.items():
        output = {
            'run_dir': str(context['run_dir']),
            'run_id': run_manifest.get('run_id', context['run_dir'].name),
            'dataset': run_manifest.get('dataset', ''),
            'baseline_protocol': run_manifest.get('baseline_protocol', ''),
            'scheme': context['scheme'],
            'reference_config_id': context['reference_config_id'],
            'score_family': context['score_family'],
            'score_rule': context['score_rule'],
            'fsood_id_side': context['fsood_id_side'],
            'aggregate': aggregate,
            'diagnostic_only': (
                context['score_family'] != 'score'
                or context['fsood_id_side'] != 'both'),
        }
        for field in METRIC_FIELDS:
            output[field] = row.get(field, '')
        rows.append(output)
    return rows


def run_collect_score(args):
    root = Path(args.runs_root)
    rows = []
    skipped = []
    pattern = f'**/{SCORE_RESULTS_DIR}/**/ood.csv'
    for metric_path in sorted(root.glob(pattern)):
        try:
            context = infer_score_context(metric_path)
            run_manifest = read_json(context['run_dir'] / 'run_manifest.json')
            if args.dataset and run_manifest.get('dataset') != args.dataset:
                continue
            if (args.baseline_protocol
                    and run_manifest.get('baseline_protocol') != args.baseline_protocol):
                continue
            if args.score_family and context['score_family'] != args.score_family:
                continue
            rows.extend(collect_score_rows(metric_path))
        except Exception as exc:  # pragma: no cover - diagnostic collection path
            skipped.append({'metric_path': str(metric_path), 'reason': str(exc)})

    fieldnames = [
        'run_dir',
        'run_id',
        'dataset',
        'baseline_protocol',
        'scheme',
        'reference_config_id',
        'score_family',
        'score_rule',
        'fsood_id_side',
        'aggregate',
        'diagnostic_only',
    ] + METRIC_FIELDS
    write_csv(args.output_csv, rows, fieldnames)
    if skipped:
        write_csv(Path(args.output_csv).with_suffix('.skipped.csv'), skipped,
                  ['metric_path', 'reason'])
    print(f'rows: {len(rows)}')
    print(f'output_csv: {args.output_csv}')


def split_table_row(line):
    return [cell.strip() for cell in line.strip().strip('|').split('|')]


def marker_for_protocol(protocol):
    if protocol == 'main_py':
        return '<!-- GROUP1_MAIN_RESULTS:BEGIN -->', '<!-- GROUP1_MAIN_RESULTS:END -->'
    return '<!-- GROUP1_EVAL_API_RESULTS:BEGIN -->', '<!-- GROUP1_EVAL_API_RESULTS:END -->'


def read_group1_table(path, protocol):
    begin, end = marker_for_protocol(protocol)
    lines = Path(path).read_text().splitlines()
    start = lines.index(begin)
    stop = lines.index(end)
    header = None
    rows = {}
    for line in lines[start + 1:stop]:
        if not line.startswith('| '):
            continue
        cells = split_table_row(line)
        if cells[:2] == ['Dataset', 'Method']:
            header = cells
            continue
        if header is None or len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        rows[(row['Dataset'], row['Method'])] = row
    return rows


def read_tarr_rows(path, dataset, protocol, score_rule, reference_config_id):
    rows = []
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            if row['dataset'] != dataset:
                continue
            if row['baseline_protocol'] != protocol:
                continue
            if row['score_rule'] != score_rule:
                continue
            if reference_config_id and row.get('reference_config_id') != reference_config_id:
                continue
            is_full_run = str(row.get('is_full_run', 'true')).lower()
            if is_full_run not in {'true', '1'}:
                continue
            fsood_side = row.get('fsood_metric_id_side') or row.get('fsood_id_side')
            if row['scheme'] == 'fsood' and fsood_side != 'both':
                continue
            rows.append(row)
    return rows


def parse_number(value):
    value = str(value).strip()
    if value in {'', '-'}:
        return None
    if '±' in value:
        value = value.split('±', 1)[0].strip()
    return float(value)


def run_compare_group1(args):
    group_rows = read_group1_table(args.group1_md, args.baseline_protocol)
    tarr_rows = read_tarr_rows(args.tarr_summary, args.dataset,
                               args.baseline_protocol, args.score_rule,
                               args.reference_config_id)
    dataset_label = group1_dataset_label(args.dataset)
    baseline_rows = [
        row for (dataset, _), row in group_rows.items() if dataset == dataset_label
    ]
    output_rows = []
    for tarr in tarr_rows:
        key = (tarr['scheme'], tarr['aggregate'])
        if key not in AGGREGATE_TO_COLUMNS:
            continue
        auroc_col, fpr_col = AGGREGATE_TO_COLUMNS[key]
        tarr_auroc = parse_number(tarr['AUROC'])
        tarr_fpr = parse_number(tarr['FPR@95'])
        for baseline in baseline_rows:
            base_auroc = parse_number(baseline.get(auroc_col, '-'))
            base_fpr = parse_number(baseline.get(fpr_col, '-'))
            if base_auroc is None or base_fpr is None:
                continue
            output_rows.append({
                'dataset': args.dataset,
                'baseline_protocol': args.baseline_protocol,
                'scheme': tarr['scheme'],
                'aggregate': tarr['aggregate'],
                'reference_config_id': tarr.get('reference_config_id', ''),
                'score_rule': args.score_rule,
                'tarr_run_id': tarr['run_id'],
                'baseline_method': baseline['Method'],
                'tarr_AUROC': tarr_auroc,
                'baseline_AUROC': base_auroc,
                'AUROC_gap_tarr_minus_baseline': tarr_auroc - base_auroc,
                'tarr_FPR95': tarr_fpr,
                'baseline_FPR95': base_fpr,
                'FPR95_gap_tarr_minus_baseline': tarr_fpr - base_fpr,
            })

    fieldnames = [
        'dataset',
        'baseline_protocol',
        'scheme',
        'aggregate',
        'reference_config_id',
        'score_rule',
        'tarr_run_id',
        'baseline_method',
        'tarr_AUROC',
        'baseline_AUROC',
        'AUROC_gap_tarr_minus_baseline',
        'tarr_FPR95',
        'baseline_FPR95',
        'FPR95_gap_tarr_minus_baseline',
    ]
    write_csv(args.output_csv, output_rows, fieldnames)
    print(f'rows: {len(output_rows)}')
    print(f'output_csv: {args.output_csv}')


def add_score_diagnostics_args(parser):
    parser.add_argument('--dataset', required=True,
                        choices=supported_dataset_names())
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--scheme', default='ood', choices=['ood', 'fsood'])
    parser.add_argument('--reference-config-id')
    parser.add_argument('--score-kind', dest='score_kind', default='vector',
                        choices=SCORE_KIND_CHOICES)
    parser.add_argument('--fsood-id-side',
                        default='auto',
                        choices=['auto', 'both', 'clean', 'csid'])
    parser.add_argument('--score-rule', default='all')
    parser.add_argument('--output-dir')
    parser.add_argument('--csid-datasets')
    parser.set_defaults(func=run_score_diagnostics)


def add_collect_score_args(parser):
    parser.add_argument('--runs-root', default='results_test/tarr/outputs')
    parser.add_argument('--output-csv',
                        default='results_test/tarr/summary/score_runs.csv')
    parser.add_argument('--dataset')
    parser.add_argument('--baseline-protocol',
                        choices=['main_py', 'eval_api'])
    parser.add_argument('--score-family',
                        choices=[
                            'score',
                            'vector',
                            'perturbation',
                        ])
    parser.set_defaults(func=run_collect_score)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    diagnostics = subparsers.add_parser('diagnostics')
    diagnostics.add_argument('--dataset', required=True,
                             choices=supported_dataset_names())
    diagnostics.add_argument('--run-dir', required=True)
    diagnostics.add_argument('--scheme', default='ood', choices=['ood', 'fsood'])
    diagnostics.add_argument('--reference-config-id')
    diagnostics.add_argument('--output-dir')
    diagnostics.add_argument('--score-rule', default='all',
                             choices=SCORE_RULE_CHOICES)
    diagnostics.add_argument('--vector-score-rule',
                             choices=VECTOR_SCORE_RULE_CHOICES)
    diagnostics.add_argument('--top-k', type=int, default=50)
    diagnostics.add_argument('--csid-datasets')
    diagnostics.set_defaults(func=run_diagnostics)

    score_diagnostics = subparsers.add_parser('score-diagnostics')
    add_score_diagnostics_args(score_diagnostics)

    collect = subparsers.add_parser('collect')
    collect.add_argument('--runs-root', default='results_test/tarr/outputs')
    collect.add_argument('--output-csv',
                         default='results_test/tarr/summary/protocol_runs.csv')
    collect.add_argument('--dataset')
    collect.add_argument('--baseline-protocol', choices=['main_py', 'eval_api'])
    collect.add_argument('--allow-smoke', action='store_true')
    collect.set_defaults(func=run_collect)

    collect_score = subparsers.add_parser('collect-score')
    add_collect_score_args(collect_score)

    compare = subparsers.add_parser('compare-group1')
    compare.add_argument('--tarr-summary',
                         default='results_test/tarr/summary/score_runs.csv')
    compare.add_argument('--group1-md',
                         default='docs_my/experiments/group1_validation.md')
    compare.add_argument('--baseline-protocol', required=True,
                         choices=['main_py', 'eval_api'])
    compare.add_argument('--dataset', required=True)
    compare.add_argument('--score-rule', required=True)
    compare.add_argument('--reference-config-id')
    compare.add_argument('--output-csv',
                         default='results_test/tarr/summary/tarr_vs_group1.csv')
    compare.set_defaults(func=run_compare_group1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
