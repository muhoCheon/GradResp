#!/usr/bin/env python
"""Validate and score saved TARR TTA response artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from scripts_my.tarr.protocol import (
    SCORE_RESULTS_DIR,
    TTA_RESPONSE_DIR,
    cache_context,
    csid_datasets_from_cache_manifest,
    far_dataset_names,
    near_dataset_names,
    ood_datasets_from_cache_manifest,
    parse_dataset_list,
    resolve_cache_dir,
    supported_dataset_names,
)
from scripts_my.tarr.scoring import (
    BRANCH_BANK_METADATA_CACHE_KEYS,
    BRANCH_BANK_SAMPLE_CACHE_KEYS,
    CACHE_SCHEMA_VERSION,
    DELTA_DEFINITION,
    PERTURBATION_DEFINITION,
    PERTURBATION_SCORE_DIRECTION,
    PERTURBATION_SCORE_RULE_CHOICES,
    PROBE_FIELD_ALIASES,
    PROBE_METADATA_CACHE_KEYS,
    PROBE_SAMPLE_CACHE_KEYS,
    PROBE_SCORE_RULES,
    SCORE_DIRECTION,
    SCORE_RULE_CHOICES,
    VECTOR_SCORE_RULE_CHOICES,
    branch_bank_shape_errors,
    branch_ids_from_cache,
    branch_score_rule_roles,
    fit_vector_score_reference,
    has_branch_bank,
    materialize_branch_bank,
    ood_score_from_cache,
    perturbation_ood_score_from_cache,
    probe_ood_score_from_cache,
    probe_score_rule_has_required_fields,
    selected_perturbation_score_rules,
    selected_probe_score_rules,
    selected_score_rules,
    selected_vector_score_rules,
    select_response_step,
    response_steps_from_cache,
    validate_branch_bank_shapes,
    vector_ood_score_from_cache,
)

REQUIRED_CACHE_KEYS = [
    'pred',
    'label',
    'y_hat',
    'target_conf',
    'target_entropy',
    'target_probs',
    'target_margin',
    'target_energy',
    'perturbation_logit_l2',
    'perturbation_prob_l1',
    'perturbation_conf_delta',
    'perturbation_entropy_delta',
    'perturbation_response_code',
    'perturbation_kind_code',
    'perturbation_eps',
    'perturbation_repeats',
    'perturbation_seed',
    'perturbation_cache_policy_code',
    'perturbation_config_id',
    'perturbation_response',
    'perturbation_kind',
    'perturbation_cache_policy',
    'perturbation_eps_config',
    'perturbation_repeats_config',
    'perturbation_seed_config',
    'target_tta_loss_before',
    'target_tta_loss_after',
    'post_tta_pred',
    'post_tta_target_conf',
    'post_tta_target_entropy',
    'post_tta_target_probs',
    'post_tta_pseudo_label_prob',
    'adapted_target_pred',
    'adapted_target_conf',
    'adapted_target_entropy',
    'adapted_target_margin',
    'adapted_target_energy',
    'adapted_target_probs',
    'target_conf_delta',
    'target_entropy_delta',
    'target_margin_delta',
    'target_energy_delta',
    'target_pred_changed',
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
    'runtime_per_sample',
    'score_rules',
    'args_score_rule',
    'reference_config_id',
    'cache_schema_version',
    'score_direction',
    'delta_definition',
    'perturbation_score_direction',
    'perturbation_definition',
    'perturbation_score_rules',
]

SAMPLE_CACHE_KEYS = {
    'pred',
    'label',
    'y_hat',
    'target_conf',
    'target_entropy',
    'target_probs',
    'target_margin',
    'target_energy',
    'perturbation_logit_l2',
    'perturbation_prob_l1',
    'perturbation_conf_delta',
    'perturbation_entropy_delta',
    'perturbation_response_code',
    'perturbation_kind_code',
    'perturbation_eps',
    'perturbation_repeats',
    'perturbation_seed',
    'perturbation_cache_policy_code',
    'target_tta_loss_before',
    'target_tta_loss_after',
    'post_tta_pred',
    'post_tta_target_conf',
    'post_tta_target_entropy',
    'post_tta_target_probs',
    'post_tta_pseudo_label_prob',
    'adapted_target_pred',
    'adapted_target_conf',
    'adapted_target_entropy',
    'adapted_target_margin',
    'adapted_target_energy',
    'adapted_target_probs',
    'target_conf_delta',
    'target_entropy_delta',
    'target_margin_delta',
    'target_energy_delta',
    'target_pred_changed',
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
    'runtime_per_sample',
}
SAMPLE_CACHE_KEYS.update(PROBE_SAMPLE_CACHE_KEYS)
SAMPLE_CACHE_KEYS.update(BRANCH_BANK_SAMPLE_CACHE_KEYS)

METADATA_CACHE_KEYS = (
    set(REQUIRED_CACHE_KEYS) - SAMPLE_CACHE_KEYS
) | set(PROBE_METADATA_CACHE_KEYS) | set(BRANCH_BANK_METADATA_CACHE_KEYS) | {
    'response_steps',
    'response_step',
    'selected_accept_branch_id',
    'selected_reject_branch_id',
}


def selected_datasets(defaults, choice):
    if choice == 'all':
        return list(defaults)
    return [item.strip() for item in choice.split(',') if item.strip()]


def _read_json(path):
    try:
        with Path(path).open() as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _dataset_keys(value):
    if isinstance(value, dict):
        return [key for key in value if key not in {'datasets', 'names'}]
    return []


def _split_dataset_names_from_manifest(manifest, split):
    if not isinstance(manifest, dict):
        return []
    for key in [f'{split}_datasets', f'{split}_dataset_names']:
        names = parse_dataset_list(manifest.get(key))
        if names:
            return names
    for key in ['protocol_config', 'score', 'score_config', SCORE_RESULTS_DIR]:
        names = _split_dataset_names_from_manifest(manifest.get(key), split)
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


def _cache_manifest_paths(cache_dir):
    context = cache_context(cache_dir)
    paths = [
        context['owner_dir'] / 'manifest.json',
        context['cache_dir'] / 'manifest.json',
        context['scheme_dir'] / 'scheme_manifest.json',
        context['run_dir'] / 'scheme_manifest.json',
        context['run_dir'] / 'run_manifest.json',
    ]
    seen = set()
    result = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def split_datasets_from_cache_manifest(cache_dir, split):
    names = parse_dataset_list(
        ood_datasets_from_cache_manifest(cache_dir, split))
    if names:
        return names
    for path in _cache_manifest_paths(cache_dir):
        names = _split_dataset_names_from_manifest(_read_json(path), split)
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


def resolve_split_datasets(args, cache_dir, split):
    choice = getattr(args, f'{split}_datasets')
    if choice != 'all':
        return selected_datasets([], choice)
    names = split_datasets_from_cache_manifest(cache_dir, split)
    if names:
        return names
    names = default_split_datasets(args.dataset, split)
    if names:
        return names
    raise ValueError(
        f'Unable to resolve {split} OOD datasets from manifest. '
        f'Pass --{split}-datasets as a comma-separated fallback.')


def _load_npz_cache(path):
    with np.load(path) as cache:
        return {key: cache[key] for key in cache.files}


def _shard_path(dataset_dir, part):
    if isinstance(part, str):
        path = Path(part)
    elif isinstance(part, dict):
        value = (
            part.get('path')
            or part.get('file')
            or part.get('filename')
            or part.get('name')
        )
        if not value:
            raise ValueError(f'shard entry missing path: {part}')
        path = Path(value)
    else:
        raise ValueError(f'invalid shard entry: {part}')
    if path.is_absolute():
        return path
    return dataset_dir / path


def _sharded_part_paths(dataset_dir):
    manifest_path = dataset_dir / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f'{manifest_path} must contain a JSON object')
    if manifest.get('complete') is False:
        raise ValueError(f'{manifest_path} is incomplete')
    parts = (
        manifest.get('parts')
        or manifest.get('shards')
        or manifest.get('files')
        or manifest.get('part_files')
    )
    if isinstance(parts, dict):
        parts = list(parts.values())
    elif isinstance(parts, str):
        parts = [parts]
    if parts:
        paths = [_shard_path(dataset_dir, part) for part in parts]
    else:
        paths = sorted(dataset_dir.glob('part_*.npz'))
    if not paths:
        raise FileNotFoundError(f'No shard parts found in {dataset_dir}')
    return paths


def _arrays_equal(left, right):
    return np.array_equal(np.asarray(left), np.asarray(right))


def _is_sample_key(key, values, part_sizes):
    if key in SAMPLE_CACHE_KEYS:
        return True
    if key in METADATA_CACHE_KEYS:
        return False
    arrays = [np.asarray(value) for value in values]
    if any(array.ndim == 0 for array in arrays):
        return False
    if any(size is None for size in part_sizes):
        return False
    if any(array.shape[0] != size
           for array, size in zip(arrays, part_sizes)):
        return False
    return len({array.shape[1:] for array in arrays}) == 1


def _merge_sharded_cache(parts, paths):
    if not parts:
        raise ValueError('cannot merge an empty sharded cache')

    first_keys = set(parts[0].keys())
    for path, part in zip(paths[1:], parts[1:]):
        keys = set(part.keys())
        if keys != first_keys:
            missing = sorted(first_keys - keys)
            extra = sorted(keys - first_keys)
            detail = []
            if missing:
                detail.append('missing: ' + ', '.join(missing))
            if extra:
                detail.append('extra: ' + ', '.join(extra))
            raise ValueError(f'{path} shard keys differ ({"; ".join(detail)})')

    part_sizes = []
    for path, part in zip(paths, parts):
        pred = np.asarray(part.get('pred'))
        if pred.ndim != 1:
            raise ValueError(f'{path} pred must be a 1-D array')
        part_sizes.append(pred.shape[0])

    merged = {}
    for key in parts[0]:
        values = [part[key] for part in parts]
        arrays = [np.asarray(value) for value in values]
        if _is_sample_key(key, arrays, part_sizes):
            if any(array.ndim == 0 for array in arrays):
                raise ValueError(f'{key} is marked sample-wise but is scalar')
            trailing_shapes = {array.shape[1:] for array in arrays}
            if len(trailing_shapes) != 1:
                raise ValueError(f'{key} shard trailing shapes differ')
            for path, array, size in zip(paths, arrays, part_sizes):
                if array.shape[0] != size:
                    raise ValueError(
                        f'{path} {key} first dimension {array.shape[0]} != '
                        f'pred size {size}')
            merged[key] = np.concatenate(arrays, axis=0)
            continue

        first = arrays[0]
        for path, array in zip(paths[1:], arrays[1:]):
            if not _arrays_equal(first, array):
                raise ValueError(
                    f'{path} metadata key {key} differs across shards')
        merged[key] = first
    return merged


def load_logical_cache(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return _load_npz_cache(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    part_paths = _sharded_part_paths(path)
    parts = [_load_npz_cache(part_path) for part_path in part_paths]
    return _merge_sharded_cache(parts, part_paths)


def load_cache(cache_dir, name):
    cache_dir = Path(cache_dir)
    single_path = cache_dir / f'{name}.npz'
    if single_path.exists():
        return load_logical_cache(single_path)
    sharded_path = cache_dir / name
    if sharded_path.exists():
        return load_logical_cache(sharded_path)
    raise FileNotFoundError(
        f'{single_path} or {sharded_path / "manifest.json"}')


def resolve_csid_datasets(args, cache_dir):
    names = csid_datasets_from_cache_manifest(cache_dir)
    if names:
        return names
    names = parse_dataset_list(args.csid_datasets)
    if names:
        return names
    raise ValueError(
        'Unable to resolve FSOOD csID datasets from manifest. '
        'Pass --csid-datasets as a comma-separated fallback.')


def raw_ood_scores(cache, score_rule):
    return ood_score_from_cache(cache, score_rule).astype(np.float64)


def raw_vector_ood_scores(cache, score_rule, vector_fit):
    return vector_ood_score_from_cache(
        cache, score_rule, vector_fit).astype(np.float64)


def raw_perturbation_ood_scores(cache, score_rule):
    return perturbation_ood_score_from_cache(
        cache, score_rule).astype(np.float64)


def score_tuple_from_ood(cache, ood_score, label_override=None):
    label = cache['label'] if label_override is None else label_override
    conf = -ood_score
    return (
        cache['pred'].astype(np.int64),
        conf.astype(np.float64),
        label.astype(np.int64),
    )


def score_tuple(cache, score_rule, label_override=None):
    return score_tuple_from_ood(
        cache, raw_ood_scores(cache, score_rule),
        label_override)


def resolve_response_step_values(cache, response_step_arg):
    steps = response_steps_from_cache(cache)
    choice = '' if response_step_arg is None else str(response_step_arg).strip()
    if choice == 'all':
        if steps.size == 0:
            return [None]
        return [int(step) for step in steps.tolist()]
    if choice in {'', 'final'}:
        if steps.size == 0:
            return [None]
        return [int(steps[-1])]
    if steps.size == 0:
        raise ValueError(
            '--response-step was set, but this single-step cache has no '
            'response_steps metadata.')
    step = int(choice)
    if step not in set(int(value) for value in steps.tolist()):
        raise ValueError(
            f'--response-step {step} is not available; saved steps are '
            f'{steps.tolist()}')
    return [step]


def should_use_response_step_dir(cache, response_step_arg, response_step_values):
    steps = response_steps_from_cache(cache)
    choice = '' if response_step_arg is None else str(response_step_arg).strip()
    if steps.size == 0:
        return False
    return (
        choice not in {'', 'final'}
        or len(response_step_values) > 1
        or steps.size > 1
    )


def rule_output_dir(output_dir, score_rule, response_step, use_step_dir):
    path = Path(output_dir) / score_rule
    if use_step_dir:
        path = path / f'step_{int(response_step)}'
    return path


def _selector_is_explicit(selector):
    return str(selector).strip() not in {'', 'auto'}


def _safe_branch_tag(value):
    text = str(value)
    safe = []
    for char in text:
        if char.isalnum() or char in {'-', '_', '.'}:
            safe.append(char)
        else:
            safe.append('-')
    tag = ''.join(safe).strip('-._')
    return tag or 'branch'


def _branch_choice_label(choice):
    if choice is None:
        return 'primary'
    return choice['id']


def _resolve_branch_token(cache, role, token):
    ids = branch_ids_from_cache(cache, role)
    if not ids:
        raise ValueError(f'No {role} response-bank branches are available')
    try:
        index = int(token)
    except ValueError:
        matches = [idx for idx, value in enumerate(ids) if value == token]
        if not matches:
            raise ValueError(
                f'Unknown {role} response-bank branch {token!r}; '
                f'available branches: {ids}')
        index = matches[0]
    if index < 0 or index >= len(ids):
        raise ValueError(
            f'{role} response-bank branch index {index} out of range; '
            f'available branches: {ids}')
    return {'index': index, 'id': ids[index]}


def _all_branch_choices(cache, role):
    ids = branch_ids_from_cache(cache, role)
    if not ids:
        raise ValueError(f'No {role} response-bank branches are available')
    return [{'index': index, 'id': branch_id}
            for index, branch_id in enumerate(ids)]


def resolve_branch_choices(cache, role, selector, needed):
    choice = 'auto' if selector is None else str(selector).strip()
    if choice in {'', 'auto'}:
        if not needed:
            return [None]
        if has_branch_bank(cache, role):
            return _all_branch_choices(cache, role)
        return [None]
    if choice == 'legacy':
        raise ValueError(
            f'--{role}-branch legacy is no longer supported. Use auto for '
            'primary singleton fields, or all/a branch name/a branch index '
            'for response-bank scoring.')
    if not needed:
        raise ValueError(
            f'--{role}-branch was set, but the selected score rule does not '
            f'use a {role} response-bank branch')
    if choice == 'all':
        return _all_branch_choices(cache, role)
    tokens = [token.strip() for token in choice.split(',') if token.strip()]
    if not tokens:
        return [None]
    return [_resolve_branch_token(cache, role, token) for token in tokens]


def validate_branch_choice_compatibility(caches, role, choices):
    materialized = [choice for choice in choices if choice is not None]
    if not materialized:
        return
    expected_ids = branch_ids_from_cache(caches[0][1], role)
    for name, cache in caches[1:]:
        ids = branch_ids_from_cache(cache, role)
        if ids != expected_ids:
            raise ValueError(
                f'{name} {role} response-bank branch ids {ids} != clean ID '
                f'response-bank branch ids {expected_ids}')
    for choice in materialized:
        if choice['id'] != expected_ids[choice['index']]:
            raise ValueError(
                f'{role} response-bank branch choice {choice} does not match '
                f'branch ids {expected_ids}')


def branch_variants_for_rule(cache, caches_for_compat, score_rule, args):
    roles = set(branch_score_rule_roles(score_rule))
    if not roles:
        if (_selector_is_explicit(args.accept_branch)
                or _selector_is_explicit(args.reject_branch)):
            raise ValueError(
                f'Score rule {score_rule} does not use response-bank branch '
                'fields')
        return [(None, None)]

    accept_choices = resolve_branch_choices(
        cache, 'accept', args.accept_branch, 'accept' in roles)
    reject_choices = resolve_branch_choices(
        cache, 'reject', args.reject_branch, 'reject' in roles)
    validate_branch_choice_compatibility(
        caches_for_compat, 'accept', accept_choices)
    validate_branch_choice_compatibility(
        caches_for_compat, 'reject', reject_choices)

    if 'accept' in roles and 'reject' in roles:
        if args.branch_combine == 'zip':
            if len(accept_choices) != len(reject_choices):
                raise ValueError(
                    '--branch-combine zip requires the same number of accept '
                    'and reject response-bank branches')
            return list(zip(accept_choices, reject_choices))
        return [
            (accept_choice, reject_choice)
            for accept_choice in accept_choices
            for reject_choice in reject_choices
        ]
    if 'accept' in roles:
        return [(accept_choice, None) for accept_choice in accept_choices]
    return [(None, reject_choice) for reject_choice in reject_choices]


def branch_variant_dir(base_dir, accept_choice=None, reject_choice=None):
    parts = []
    if accept_choice is not None:
        parts.append(f'accept_{_safe_branch_tag(accept_choice["id"])}')
    if reject_choice is not None:
        parts.append(f'reject_{_safe_branch_tag(reject_choice["id"])}')
    if not parts:
        return base_dir
    return Path(base_dir) / '__'.join(parts)


def materialize_branch_variant(cache, accept_choice=None, reject_choice=None):
    accept_index = None if accept_choice is None else accept_choice['index']
    reject_index = None if reject_choice is None else reject_choice['index']
    if accept_index is None and reject_index is None:
        return cache
    return materialize_branch_bank(
        cache, accept_branch=accept_index, reject_branch=reject_index)


def concat_scores(parts):
    return (
        np.concatenate([part[0] for part in parts]),
        np.concatenate([part[1] for part in parts]),
        np.concatenate([part[2] for part in parts]),
    )


def metric_summary(id_scores, split_scores):
    from openood.evaluators.metrics import compute_all_metrics

    pred = np.concatenate([id_scores[0], split_scores[0]])
    conf = np.concatenate([id_scores[1], split_scores[1]])
    label = np.concatenate([
        id_scores[2],
        -1 * np.ones_like(split_scores[2], dtype=np.int64),
    ])
    return compute_all_metrics(conf, label, pred)


def format_metric_row(dataset_name, metrics):
    fpr, auroc, aupr_in, aupr_out, acc = metrics
    return {
        'dataset': dataset_name,
        'FPR@95': f'{100 * fpr:.2f}',
        'AUROC': f'{100 * auroc:.2f}',
        'AUPR_IN': f'{100 * aupr_in:.2f}',
        'AUPR_OUT': f'{100 * aupr_out:.2f}',
        'ACC': f'{100 * acc:.2f}',
    }


def save_npz(path, scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pred=scores[0], conf=scores[1], label=scores[2])


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')


def write_metrics_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['dataset', 'FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _scalar_string(cache, key):
    if key not in cache:
        return None
    value = np.asarray(cache[key])
    if value.shape != ():
        return None
    return str(value.item())


def _scalar_int(cache, key):
    if key not in cache:
        return None
    value = np.asarray(cache[key])
    if value.shape != ():
        return None
    return int(value.item())


def _validate_scalar_string(cache, key, expected, errors):
    value = np.asarray(cache[key])
    if value.shape != ():
        errors.append(f'{key} must be a scalar string')
        return
    actual = str(value.item())
    if actual != expected:
        errors.append(f'{key} {actual} != {expected}')


def _validate_scalar_int(cache, key, expected, errors):
    value = np.asarray(cache[key])
    if value.shape != ():
        errors.append(f'{key} must be a scalar int')
        return
    actual = int(value.item())
    if actual != expected:
        errors.append(f'{key} {actual} != {expected}')


def _validate_score_rules(cache, errors):
    values = np.asarray(cache['score_rules'])
    if values.ndim != 1:
        errors.append('score_rules must be a 1-D array')
        return
    allowed = (
        set(selected_score_rules('all'))
        | set(selected_score_rules('probe_all'))
    )
    for value in values:
        score_rule = str(value)
        if score_rule not in allowed:
            errors.append(f'score_rules contains unknown rule: {score_rule}')
    if 'args_score_rule' in cache:
        score_rule_arg_value = np.asarray(cache['args_score_rule'])
        if score_rule_arg_value.shape != ():
            return
        score_rule_arg = str(score_rule_arg_value.item())
        try:
            expected = selected_score_rules(score_rule_arg)
        except Exception as exc:
            errors.append(f'args_score_rule invalid: {exc}')
            return
        actual = [str(value) for value in values]
        if actual != expected:
            errors.append(
                f'score_rules {actual} != args_score_rule expansion {expected}')


def _validate_perturbation_score_rules(cache, errors):
    values = np.asarray(cache['perturbation_score_rules'])
    if values.ndim != 1:
        errors.append('perturbation_score_rules must be a 1-D array')
        return
    allowed = set(selected_perturbation_score_rules('all'))
    for value in values:
        score_rule = str(value)
        if score_rule not in allowed:
            errors.append(
                f'perturbation_score_rules contains unknown rule: {score_rule}')


def _validate_probe_score_rules(cache, errors):
    if 'probe_score_rules' not in cache:
        return
    values = np.asarray(cache['probe_score_rules'])
    if values.ndim != 1:
        errors.append('probe_score_rules must be a 1-D array')
        return
    allowed = set(selected_probe_score_rules('probe_all'))
    for value in values:
        score_rule = str(value)
        if score_rule not in allowed:
            errors.append(f'probe_score_rules contains unknown rule: {score_rule}')
    if 'probe_score_rule_arg' in cache:
        score_rule_arg_value = np.asarray(cache['probe_score_rule_arg'])
        if score_rule_arg_value.shape != ():
            errors.append('probe_score_rule_arg must be scalar config metadata')
            return
        score_rule_arg = str(score_rule_arg_value.item())
        try:
            expected = selected_probe_score_rules(score_rule_arg)
        except Exception as exc:
            errors.append(f'probe_score_rule_arg invalid: {exc}')
            return
        actual = [str(value) for value in values]
        if score_rule_arg == 'probe_all':
            missing_from_current = [
                score_rule for score_rule in actual
                if score_rule not in expected
            ]
            if missing_from_current:
                errors.append(
                    f'probe_score_rules {actual} contains rules not in '
                    f'current probe_all expansion {expected}')
        elif actual != expected:
            errors.append(
                f'probe_score_rules {actual} != probe_score_rule_arg '
                f'expansion {expected}')


def _numeric_optional_array(cache, key, errors):
    try:
        return np.asarray(cache[key], dtype=np.float64)
    except (TypeError, ValueError):
        errors.append(f'{key} must be numeric')
        return None


def _validate_optional_probe_fields(cache, n, num_classes, errors, num_steps=0):
    vector_groups = {
        'accept_target_objective_delta',
        'reject_target_objective_delta',
        'reject_target_entropy_delta',
    }
    matrix_groups = {
        'accept_ref_loss_delta',
        'reject_ref_loss_delta',
    }

    for group, aliases in PROBE_FIELD_ALIASES.items():
        for key in aliases:
            if key not in cache:
                continue
            values = _numeric_optional_array(cache, key, errors)
            if values is None:
                continue
            if group in vector_groups:
                if values.ndim not in {1, 2}:
                    errors.append(f'{key} must be a 1-D or 2-D probe array')
                    continue
                if values.shape[0] != n:
                    errors.append(f'{key} first dimension {values.shape[0]} != {n}')
                if values.ndim == 2 and num_steps and values.shape[1] != num_steps:
                    errors.append(
                        f'{key} step dimension {values.shape[1]} != '
                        f'{num_steps}')
            elif group in matrix_groups:
                if values.ndim not in {2, 3}:
                    errors.append(f'{key} must be a 2-D or 3-D probe array')
                    continue
                if values.shape[0] != n:
                    errors.append(f'{key} first dimension {values.shape[0]} != {n}')
                class_axis = 1 if values.ndim == 2 else 2
                if values.shape[class_axis] != num_classes:
                    errors.append(
                        f'{key} class dimension {values.shape[class_axis]} != '
                        f'{num_classes}')
                if values.ndim == 3 and num_steps and values.shape[1] != num_steps:
                    errors.append(
                        f'{key} step dimension {values.shape[1]} != '
                        f'{num_steps}')
            if not np.all(np.isfinite(values)):
                errors.append(f'{key} contains non-finite values')

    for key in [
            'probe_score_alpha',
            'probe_score_beta',
            'probe_score_gamma',
            'probe_score_temperature',
            'probe_score_eps',
    ]:
        if key not in cache:
            continue
        value = _numeric_optional_array(cache, key, errors)
        if value is None:
            continue
        if value.shape != ():
            errors.append(f'{key} must be scalar config metadata')
        elif not np.isfinite(value.item()):
            errors.append(f'{key} must be finite')
        elif key in {'probe_score_temperature', 'probe_score_eps'}:
            if float(value.item()) <= 0.0:
                errors.append(f'{key} must be positive')


def _declared_probe_score_rules(cache):
    rules = []
    for key in ['score_rules', 'probe_score_rules']:
        if key not in cache:
            continue
        values = np.asarray(cache[key])
        if values.ndim != 1:
            continue
        rules.extend(str(value) for value in values
                     if str(value) in PROBE_SCORE_RULES)
    for key, selector in [
            ('args_score_rule', selected_score_rules),
            ('probe_score_rule_arg', selected_probe_score_rules),
    ]:
        if key not in cache:
            continue
        value = np.asarray(cache[key])
        if value.shape != ():
            continue
        try:
            selected = selector(str(value.item()))
        except Exception:
            continue
        rules.extend(rule for rule in selected if rule in PROBE_SCORE_RULES)
    seen = set()
    result = []
    for rule in PROBE_SCORE_RULES:
        if rule in rules and rule not in seen:
            seen.add(rule)
            result.append(rule)
    return result


def _has_probe_fields(cache):
    return any(key in cache for key in PROBE_SAMPLE_CACHE_KEYS)


def _probe_rules_to_validate(cache):
    declared = _declared_probe_score_rules(cache)
    if declared:
        return declared
    if not _has_probe_fields(cache):
        return []
    rules = []
    for score_rule in PROBE_SCORE_RULES:
        try:
            has_fields = probe_score_rule_has_required_fields(cache, score_rule)
        except Exception:
            has_fields = False
        if has_fields:
            rules.append(score_rule)
    return rules


def _validate_scalar_config_metadata(cache, errors):
    for key in [
            'args_score_rule',
            'reference_config_id',
            'perturbation_config_id',
            'perturbation_response',
            'perturbation_kind',
            'perturbation_cache_policy',
            'perturbation_eps_config',
            'perturbation_repeats_config',
            'perturbation_seed_config',
    ]:
        value = np.asarray(cache[key])
        if value.shape != ():
            errors.append(f'{key} must be scalar config metadata')


def validate_cache(cache):
    errors = []
    missing = [key for key in REQUIRED_CACHE_KEYS if key not in cache]
    if missing:
        errors.append('missing keys: ' + ', '.join(missing))
        return errors

    pred = cache['pred']
    label = cache['label']
    y_hat = cache['y_hat']
    target_probs = cache['target_probs']
    adapted_target_probs = cache['adapted_target_probs']
    delta = cache['delta']
    base_loss = cache['base_reference_loss']
    adapted_loss = cache['adapted_reference_loss']
    runtime = cache['runtime_per_sample']
    response_steps = response_steps_from_cache(cache)
    num_steps = int(response_steps.size)

    if pred.ndim != 1 or label.ndim != 1 or y_hat.ndim != 1:
        errors.append('pred, label, and y_hat must be 1-D arrays')
    if target_probs.ndim != 2 or delta.ndim not in {2, 3}:
        errors.append('target_probs must be 2-D and delta must be 2-D or 3-D')
    if target_probs.ndim == 2 and delta.ndim in {2, 3}:
        num_classes = delta.shape[2] if delta.ndim == 3 else delta.shape[1]
        if target_probs.shape[1] != num_classes:
            errors.append(
                f'target_probs class dimension {target_probs.shape[1]} != '
                f'{num_classes}')
    else:
        num_classes = 0
    if delta.ndim == 3:
        if num_steps and delta.shape[1] != num_steps:
            errors.append(
                f'delta step dimension {delta.shape[1]} != {num_steps}')
        elif not num_steps:
            errors.append('step-wise delta requires response_steps metadata')
        expected_adapted_target_probs_shape = (
            target_probs.shape[0], delta.shape[1], delta.shape[2])
        expected_reference_shape = delta.shape
    elif delta.ndim == 2:
        expected_adapted_target_probs_shape = target_probs.shape
        expected_reference_shape = delta.shape
    else:
        expected_adapted_target_probs_shape = target_probs.shape
        expected_reference_shape = delta.shape
    if adapted_target_probs.shape != expected_adapted_target_probs_shape:
        errors.append(
            f'adapted_target_probs shape {adapted_target_probs.shape} != '
            f'{expected_adapted_target_probs_shape}')
    post_probs = cache['post_tta_target_probs']
    if post_probs.shape != expected_adapted_target_probs_shape:
        errors.append(
            f'post_tta_target_probs shape {post_probs.shape} != '
            f'{expected_adapted_target_probs_shape}')
    if adapted_loss.shape != expected_reference_shape:
        errors.append(
            f'adapted_reference_loss shape {adapted_loss.shape} != '
            f'{expected_reference_shape}')
    if base_loss.shape != target_probs.shape:
        errors.append(
            f'base_reference_loss shape {base_loss.shape} != '
            f'{target_probs.shape}')

    n = pred.shape[0]
    for key, value in [
            ('label', label),
            ('y_hat', y_hat),
            ('target_conf', cache['target_conf']),
            ('target_entropy', cache['target_entropy']),
            ('target_margin', cache['target_margin']),
            ('target_energy', cache['target_energy']),
            ('target_probs', target_probs),
            ('perturbation_logit_l2', cache['perturbation_logit_l2']),
            ('perturbation_prob_l1', cache['perturbation_prob_l1']),
            ('perturbation_conf_delta', cache['perturbation_conf_delta']),
            ('perturbation_entropy_delta',
             cache['perturbation_entropy_delta']),
            ('perturbation_response_code',
             cache['perturbation_response_code']),
            ('perturbation_kind_code', cache['perturbation_kind_code']),
            ('perturbation_eps', cache['perturbation_eps']),
            ('perturbation_repeats', cache['perturbation_repeats']),
            ('perturbation_seed', cache['perturbation_seed']),
            ('perturbation_cache_policy_code',
             cache['perturbation_cache_policy_code']),
            ('target_tta_loss_before', cache['target_tta_loss_before']),
            ('target_tta_loss_after', cache['target_tta_loss_after']),
            ('post_tta_pred', cache['post_tta_pred']),
            ('post_tta_target_conf', cache['post_tta_target_conf']),
            ('post_tta_target_entropy', cache['post_tta_target_entropy']),
            ('post_tta_target_probs', cache['post_tta_target_probs']),
            ('post_tta_pseudo_label_prob',
             cache['post_tta_pseudo_label_prob']),
            ('adapted_target_pred', cache['adapted_target_pred']),
            ('adapted_target_conf', cache['adapted_target_conf']),
            ('adapted_target_entropy', cache['adapted_target_entropy']),
            ('adapted_target_margin', cache['adapted_target_margin']),
            ('adapted_target_energy', cache['adapted_target_energy']),
            ('adapted_target_probs', adapted_target_probs),
            ('target_conf_delta', cache['target_conf_delta']),
            ('target_entropy_delta', cache['target_entropy_delta']),
            ('target_margin_delta', cache['target_margin_delta']),
            ('target_energy_delta', cache['target_energy_delta']),
            ('target_pred_changed', cache['target_pred_changed']),
            ('base_reference_loss', base_loss),
            ('adapted_reference_loss', adapted_loss),
            ('delta', delta),
            ('reference_conf_delta_by_class',
             cache['reference_conf_delta_by_class']),
            ('reference_entropy_delta_by_class',
             cache['reference_entropy_delta_by_class']),
            ('reference_margin_delta_by_class',
             cache['reference_margin_delta_by_class']),
            ('reference_energy_delta_by_class',
             cache['reference_energy_delta_by_class']),
            ('reference_pred_changed_rate_by_class',
             cache['reference_pred_changed_rate_by_class']),
            ('reference_correct_rate_before_by_class',
             cache['reference_correct_rate_before_by_class']),
            ('reference_correct_rate_after_by_class',
             cache['reference_correct_rate_after_by_class']),
            ('runtime_per_sample', runtime),
    ]:
        if value.shape[0] != n:
            errors.append(f'{key} first dimension {value.shape[0]} != {n}')

    by_class_keys = [
        'reference_conf_delta_by_class',
        'reference_entropy_delta_by_class',
        'reference_margin_delta_by_class',
        'reference_energy_delta_by_class',
        'reference_pred_changed_rate_by_class',
        'reference_correct_rate_after_by_class',
    ]
    for key in by_class_keys:
        if cache[key].shape != expected_reference_shape:
            errors.append(
                f'{key} shape {cache[key].shape} != '
                f'{expected_reference_shape}')
    if cache['reference_correct_rate_before_by_class'].shape != target_probs.shape:
        errors.append(
            'reference_correct_rate_before_by_class shape '
            f'{cache["reference_correct_rate_before_by_class"].shape} != '
            f'{target_probs.shape}')
    finite_keys = [
        'target_conf',
        'target_entropy',
        'target_margin',
        'target_energy',
        'target_probs',
        'perturbation_logit_l2',
        'perturbation_prob_l1',
        'perturbation_conf_delta',
        'perturbation_entropy_delta',
        'perturbation_response_code',
        'perturbation_kind_code',
        'perturbation_eps',
        'perturbation_repeats',
        'perturbation_seed',
        'perturbation_cache_policy_code',
        'target_tta_loss_before',
        'target_tta_loss_after',
        'post_tta_target_conf',
        'post_tta_target_entropy',
        'post_tta_target_probs',
        'post_tta_pseudo_label_prob',
        'adapted_target_conf',
        'adapted_target_entropy',
        'adapted_target_margin',
        'adapted_target_energy',
        'adapted_target_probs',
        'target_conf_delta',
        'target_entropy_delta',
        'target_margin_delta',
        'target_energy_delta',
        'base_reference_loss',
        'adapted_reference_loss',
        'delta',
    ] + by_class_keys
    for key in finite_keys:
        if not np.all(np.isfinite(cache[key])):
            errors.append(f'{key} contains non-finite values')
    _validate_optional_probe_fields(cache, n, num_classes, errors, num_steps)
    errors.extend(branch_bank_shape_errors(
        cache, n=n, num_steps=num_steps, num_classes=num_classes))

    _validate_scalar_int(
        cache, 'cache_schema_version', CACHE_SCHEMA_VERSION, errors)
    _validate_scalar_string(
        cache, 'score_direction', SCORE_DIRECTION, errors)
    _validate_scalar_string(
        cache, 'delta_definition', DELTA_DEFINITION, errors)
    _validate_scalar_string(
        cache, 'perturbation_score_direction',
        PERTURBATION_SCORE_DIRECTION, errors)
    _validate_scalar_string(
        cache, 'perturbation_definition', PERTURBATION_DEFINITION, errors)
    _validate_scalar_config_metadata(cache, errors)
    _validate_score_rules(cache, errors)
    _validate_perturbation_score_rules(cache, errors)
    _validate_probe_score_rules(cache, errors)

    for score_rule in selected_score_rules('all'):
        try:
            scores = ood_score_from_cache(cache, score_rule)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f'{score_rule} failed: {exc}')
            continue
        if scores.shape[0] != n:
            errors.append(f'{score_rule} returned {scores.shape[0]} scores')
    for score_rule in selected_perturbation_score_rules('all'):
        try:
            scores = perturbation_ood_score_from_cache(cache, score_rule)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f'{score_rule} failed: {exc}')
            continue
        if scores.shape[0] != n:
            errors.append(f'{score_rule} returned {scores.shape[0]} scores')
        if not np.all(np.isfinite(scores)):
            errors.append(f'{score_rule} contains non-finite values')
    for score_rule in _probe_rules_to_validate(cache):
        try:
            scores = probe_ood_score_from_cache(cache, score_rule)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f'{score_rule} failed: {exc}')
            continue
        if scores.shape[0] != n:
            errors.append(f'{score_rule} returned {scores.shape[0]} scores')
        if not np.all(np.isfinite(scores)):
            errors.append(f'{score_rule} contains non-finite values')
    return errors


def validate_cache_file(path):
    try:
        cache = load_logical_cache(path)
    except Exception as exc:
        return [f'load failed: {exc}']
    return validate_cache(cache)


def logical_cache_entries(cache_dir):
    cache_dir = Path(cache_dir)
    entries = []
    for path in sorted(cache_dir.glob('*.npz')):
        entries.append((path.stem, path))
    for path in sorted(cache_dir.iterdir() if cache_dir.exists() else []):
        if path.is_dir() and (path / 'manifest.json').exists():
            entries.append((path.name, path))
    return entries


def actual_cache_declared_paths(cache_dir):
    cache_dir = Path(cache_dir)
    paths = {path.resolve() for _, path in logical_cache_entries(cache_dir)}
    for name, path in logical_cache_entries(cache_dir):
        if path.is_dir():
            paths.add((path / 'manifest.json').resolve())
            paths.add((cache_dir / f'{name}.npz').resolve())
            paths.update(part.resolve() for part in sorted(path.glob('*.npz')))
            try:
                paths.update(part.resolve() for part in _sharded_part_paths(path))
            except Exception:
                pass
    return paths


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def file_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def reference_config_map(manifest):
    configs = manifest.get('reference_configs', [])
    if isinstance(configs, dict):
        return configs
    if isinstance(configs, list):
        return {
            item.get('id'): item
            for item in configs
            if isinstance(item, dict) and item.get('id')
        }
    return {}


def nested_dataset_entries(node):
    if isinstance(node, dict):
        if 'imglist_path' in node:
            yield node
            return
        for value in node.values():
            yield from nested_dataset_entries(value)
    elif isinstance(node, list):
        for value in node:
            yield from nested_dataset_entries(value)


def _nested_get(node, keys):
    current = node
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def train_dataset_entry(manifest):
    train_entry = _nested_get(manifest, ['dataset_manifest', 'train'])
    if isinstance(train_entry, dict) and 'imglist_path' in train_entry:
        return train_entry
    return {}


def train_metadata_manifest_refs(manifest):
    refs = []
    for key in ['train_candidate_metadata', 'train_candidate_metadata_path']:
        value = manifest.get(key)
        if isinstance(value, dict):
            refs.append(value)
        elif isinstance(value, str):
            refs.append({'manifest_path': value})
    path_value = manifest.get('train_candidate_metadata_manifest_path')
    if path_value:
        refs.append({'manifest_path': path_value})
    return refs


def train_metadata_identity(manifest):
    for key in ['train_candidate_metadata_identity']:
        value = manifest.get(key)
        if isinstance(value, dict):
            return value
    metadata_node = manifest.get('train_candidate_metadata')
    if isinstance(metadata_node, dict):
        value = metadata_node.get('identity')
        if isinstance(value, dict):
            return value
    artifact_identity = manifest.get('artifact_identity')
    if isinstance(artifact_identity, dict):
        value = artifact_identity.get('train_candidate_metadata_identity')
        if isinstance(value, dict):
            return value
    for ref in train_metadata_manifest_refs(manifest):
        identity = ref.get('identity')
        if isinstance(identity, dict):
            return identity
    return {}


def _train_metadata_identity_value(identity, *keys):
    for key in keys:
        value = identity.get(key)
        if value is not None and value != '':
            return value
    return None


def _train_metadata_schema_version(node):
    if not isinstance(node, dict):
        return None
    return _train_metadata_identity_value(
        node, 'train_candidate_metadata_schema_version', 'schema_version')


def _resolve_train_metadata_manifest_path(manifest_path, base_dir):
    def as_manifest_path(path):
        if path.is_dir():
            return path / 'manifest.json'
        if path.suffix != '.json':
            return path.with_name('manifest.json')
        return path

    path = Path(manifest_path).expanduser()
    direct_path = as_manifest_path(path)
    if path.is_absolute() or base_dir is None or direct_path.exists():
        return direct_path
    path = Path(base_dir) / path
    path = as_manifest_path(path)
    return path


def _validate_train_metadata_schema_metadata(identity, manifest, errors, source):
    identity_version = _train_metadata_schema_version(identity)
    manifest_version = _train_metadata_schema_version(manifest)
    if (identity_version is not None and manifest_version is not None
            and identity_version != manifest_version):
        errors.append(
            f'{source} train_candidate_metadata schema_version mismatch: '
            f'identity={identity_version} manifest={manifest_version}')

    manifest_identity = {}
    if isinstance(manifest, dict):
        manifest_identity = (
            manifest.get('identity')
            or train_metadata_identity(manifest)
        )
    manifest_identity_version = _train_metadata_schema_version(manifest_identity)
    if (identity_version is not None and manifest_identity_version is not None
            and identity_version != manifest_identity_version):
        errors.append(
            f'{source} train_candidate_metadata schema_version mismatch: '
            f'identity={identity_version} '
            f'manifest_identity={manifest_identity_version}')

    identity_preprocessor = _train_metadata_identity_value(
        identity, 'preprocessor_identity')
    manifest_preprocessor = _train_metadata_identity_value(
        manifest_identity, 'preprocessor_identity')
    if (identity_preprocessor is not None
            and manifest_preprocessor is not None
            and identity_preprocessor != manifest_preprocessor):
        errors.append(
            f'{source} preprocessor_identity mismatch: '
            f'identity={identity_preprocessor} '
            f'manifest_identity={manifest_preprocessor}')


def _validate_train_metadata_identity(identity, run_manifest, scheme_manifest,
                                 errors, source):
    if not identity:
        return
    train_entry = train_dataset_entry(scheme_manifest)
    checks = [
        ('dataset', _train_metadata_identity_value(identity, 'dataset'),
         run_manifest.get('dataset')),
        ('checkpoint_sha256',
         _train_metadata_identity_value(identity, 'checkpoint_sha256'),
         run_manifest.get('checkpoint_sha256')),
        ('model_arch', _train_metadata_identity_value(identity, 'model_arch'),
         run_manifest.get('model_arch')),
        ('num_classes', _train_metadata_identity_value(identity, 'num_classes'),
         run_manifest.get('num_classes')),
        ('preprocessor_identity',
         _train_metadata_identity_value(identity, 'preprocessor_identity'),
         _train_metadata_identity_value(
             train_metadata_identity(run_manifest), 'preprocessor_identity')),
        ('train_candidate_metadata_schema_version',
         _train_metadata_schema_version(identity),
         _train_metadata_schema_version(train_metadata_identity(run_manifest))),
        ('train_imglist_sha256',
         _train_metadata_identity_value(identity, 'train_imglist_sha256',
                                   'imglist_sha256'),
         train_entry.get('imglist_sha256')),
        ('train_imglist_path',
         _train_metadata_identity_value(identity, 'train_imglist_path',
                                   'imglist_path'),
         train_entry.get('imglist_path')),
    ]
    for name, actual, expected_value in checks:
        if actual is not None and expected_value is not None and actual != expected_value:
            errors.append(
                f'{source} {name} mismatch: identity={actual} '
                f'manifest={expected_value}')

    checkpoint_path = _train_metadata_identity_value(
        identity, 'checkpoint_path', 'checkpoint_resolved')
    checkpoint_hash = _train_metadata_identity_value(identity, 'checkpoint_sha256')
    if checkpoint_path and checkpoint_hash and Path(checkpoint_path).exists():
        actual_hash = file_sha256(checkpoint_path)
        if actual_hash != checkpoint_hash:
            errors.append(
                f'{source} checkpoint_sha256 mismatch: '
                f'identity={checkpoint_hash} current={actual_hash}')
    imglist_path = _train_metadata_identity_value(identity, 'train_imglist_path',
                                             'imglist_path')
    imglist_hash = _train_metadata_identity_value(identity, 'train_imglist_sha256',
                                             'imglist_sha256')
    if imglist_path and imglist_hash and Path(imglist_path).exists():
        actual_hash = file_sha256(imglist_path)
        if actual_hash != imglist_hash:
            errors.append(
                f'{source} train_imglist_sha256 mismatch: '
                f'identity={imglist_hash} current={actual_hash}')


def _validate_train_metadata_manifest_refs(manifest, run_manifest, scheme_manifest,
                                      errors, source, base_dir=None):
    for ref in train_metadata_manifest_refs(manifest):
        manifest_path = (
            ref.get('manifest_path')
            or ref.get('path')
            or ref.get('metadata_dir')
            or ref.get('metadata_path')
        )
        if not manifest_path:
            continue
        path = _resolve_train_metadata_manifest_path(manifest_path, base_dir)
        if not path.exists():
            errors.append(
                f'{source} train_candidate_metadata manifest missing: {path}')
            continue
        candidate_manifest = read_json(path)
        if not candidate_manifest:
            errors.append(
                f'{source} train_candidate_metadata manifest unreadable: {path}')
            continue
        embedded_identity = (
            ref.get('identity')
            or train_metadata_identity(manifest)
        )
        file_identity = (
            candidate_manifest.get('identity')
            or train_metadata_identity(candidate_manifest)
        )
        if embedded_identity and file_identity and embedded_identity != file_identity:
            errors.append(
                f'{source} train_candidate_metadata identity mismatch: '
                f'manifest={file_identity} embedded={embedded_identity}')
        if embedded_identity:
            _validate_train_metadata_schema_metadata(
                embedded_identity, candidate_manifest, errors,
                f'{source} train_candidate_metadata manifest')
        _validate_train_metadata_identity(
            file_identity, run_manifest, scheme_manifest, errors,
            f'{source} train_candidate_metadata manifest')
        _validate_train_metadata_schema_metadata(
            file_identity, candidate_manifest, errors,
            f'{source} train_candidate_metadata manifest')


def manifest_validation_errors(cache_dir, expected=None):
    context = cache_context(cache_dir)
    errors = []
    expected = expected or {}
    run_manifest = read_json(context['run_dir'] / 'run_manifest.json')
    scheme_manifest = read_json(context['scheme_dir'] / 'scheme_manifest.json')
    if not run_manifest:
        errors.append(f'missing run manifest: {context["run_dir"] / "run_manifest.json"}')
    if not scheme_manifest:
        errors.append(
            f'missing scheme manifest: {context["scheme_dir"] / "scheme_manifest.json"}')
    if not run_manifest or not scheme_manifest:
        return errors

    if run_manifest.get('cache_schema_version') != CACHE_SCHEMA_VERSION:
        errors.append('run_manifest cache_schema_version mismatch')
    if scheme_manifest.get('cache_schema_version') != CACHE_SCHEMA_VERSION:
        errors.append('scheme_manifest cache_schema_version mismatch')
    if run_manifest.get('score_direction') != SCORE_DIRECTION:
        errors.append('run_manifest score_direction mismatch')
    if scheme_manifest.get('score_direction') != SCORE_DIRECTION:
        errors.append('scheme_manifest score_direction mismatch')
    if run_manifest.get('delta_definition') != DELTA_DEFINITION:
        errors.append('run_manifest delta_definition mismatch')
    if scheme_manifest.get('delta_definition') != DELTA_DEFINITION:
        errors.append('scheme_manifest delta_definition mismatch')

    checkpoint_path = run_manifest.get('checkpoint_resolved')
    checkpoint_hash = run_manifest.get('checkpoint_sha256')
    if not checkpoint_hash:
        errors.append('missing checkpoint_sha256 in run manifest')
    elif checkpoint_path and Path(checkpoint_path).exists():
        actual_hash = file_sha256(checkpoint_path)
        if actual_hash != checkpoint_hash:
            errors.append(
                f'checkpoint_sha256 mismatch: manifest={checkpoint_hash} current={actual_hash}')

    dataset_manifest = scheme_manifest.get('dataset_manifest', {})
    for entry in nested_dataset_entries(dataset_manifest):
        imglist_path = entry.get('imglist_path')
        expected_hash = entry.get('imglist_sha256')
        if not expected_hash:
            errors.append(f'missing imglist_sha256 for {imglist_path}')
            continue
        if imglist_path and Path(imglist_path).exists():
            actual_hash = file_sha256(imglist_path)
            if actual_hash != expected_hash:
                errors.append(
                    f'imglist_sha256 mismatch for {imglist_path}: '
                    f'manifest={expected_hash} current={actual_hash}')

    run_train_metadata_identity = train_metadata_identity(run_manifest)
    scheme_train_metadata_identity = train_metadata_identity(scheme_manifest)
    if (run_train_metadata_identity and scheme_train_metadata_identity
            and run_train_metadata_identity != scheme_train_metadata_identity):
        errors.append(
            'train_candidate_metadata identity mismatch between run_manifest '
            'and scheme_manifest')
    if run_train_metadata_identity and scheme_train_metadata_identity:
        run_schema = _train_metadata_schema_version(run_train_metadata_identity)
        scheme_schema = _train_metadata_schema_version(scheme_train_metadata_identity)
        if (run_schema is not None and scheme_schema is not None
                and run_schema != scheme_schema):
            errors.append(
                'train_candidate_metadata schema_version '
                'mismatch between run_manifest and scheme_manifest')
        run_preprocessor = _train_metadata_identity_value(
            run_train_metadata_identity, 'preprocessor_identity')
        scheme_preprocessor = _train_metadata_identity_value(
            scheme_train_metadata_identity, 'preprocessor_identity')
        if (run_preprocessor is not None and scheme_preprocessor is not None
                and run_preprocessor != scheme_preprocessor):
            errors.append(
                'train_candidate_metadata preprocessor_identity mismatch '
                'between run_manifest and scheme_manifest')
    _validate_train_metadata_identity(
        run_train_metadata_identity, run_manifest, scheme_manifest, errors,
        'run_manifest train_candidate_metadata')
    _validate_train_metadata_identity(
        scheme_train_metadata_identity, run_manifest, scheme_manifest, errors,
        'scheme_manifest train_candidate_metadata')
    _validate_train_metadata_manifest_refs(
        run_manifest, run_manifest, scheme_manifest, errors, 'run_manifest',
        context['run_dir'])
    _validate_train_metadata_manifest_refs(
        scheme_manifest, run_manifest, scheme_manifest, errors,
        'scheme_manifest', context['scheme_dir'])

    reference_config_id = context.get('reference_config_id')
    if reference_config_id:
        reference_ids = scheme_manifest.get('reference_config_ids', [])
        if reference_ids and reference_config_id not in reference_ids:
            errors.append(
                f'reference_config_id {reference_config_id} missing from scheme manifest')
        ref_configs = reference_config_map(scheme_manifest)
        if not ref_configs:
            ref_configs = reference_config_map(run_manifest)
        ref_config = ref_configs.get(reference_config_id)
        if not ref_config:
            errors.append(
                f'reference_config_id {reference_config_id} missing from reference_configs')
        else:
            reference_checks = [
                ('reference_source', ref_config.get('source'),
                 expected.get('reference_source')),
                ('reference_per_class', ref_config.get('per_class'),
                 expected.get('reference_per_class')),
                ('reference_filter', ref_config.get('filter'),
                 expected.get('reference_filter')),
                ('reference_min_confidence', ref_config.get('min_confidence'),
                 expected.get('reference_min_confidence')),
                ('reference_seed', ref_config.get('seed'),
                 expected.get('reference_seed')),
            ]
            for name, actual, wanted in reference_checks:
                if wanted is not None and actual != wanted:
                    errors.append(
                        f'{name} mismatch: manifest={actual} expected={wanted}')
        reference_stats = scheme_manifest.get(
            'reference_stats', run_manifest.get('reference_stats', {}))
        ref_stats = reference_stats.get(reference_config_id, {})
        selected_hash = ref_stats.get('selected_reference_hash')
        if not selected_hash:
            errors.append(
                f'missing selected_reference_hash for {reference_config_id}')
        expected_reference_hash = expected.get('selected_reference_hash')
        if (expected_reference_hash is not None
                and selected_hash != expected_reference_hash):
            errors.append(
                f'selected_reference_hash mismatch: manifest={selected_hash} '
                f'expected={expected_reference_hash}')
        artifact_identity = scheme_manifest.get(
            'artifact_identity', run_manifest.get('artifact_identity', {}))
        identity_hash = artifact_identity.get('reference_hashes',
                                              {}).get(reference_config_id)
        if identity_hash and selected_hash and identity_hash != selected_hash:
            errors.append(
                f'artifact_identity reference hash mismatch for {reference_config_id}')
        tta_response_files = scheme_manifest.get('tta_response_files', {}).get(
            reference_config_id, {})
        declared = []

        def collect_declared(node):
            if isinstance(node, str):
                path = Path(node)
                candidates = [path.resolve()]
                if not path.is_absolute():
                    candidates.extend([
                        (Path(cache_dir) / path).resolve(),
                        (context['owner_dir'] / path).resolve(),
                        (context['scheme_dir'] / path).resolve(),
                    ])
                declared.append((path, candidates))
            elif isinstance(node, dict):
                for key in ['manifest', 'path', 'cache_path']:
                    value = node.get(key)
                    if value:
                        collect_declared(value)
                for key, value in node.items():
                    if key in {
                            'storage',
                            'manifest',
                            'path',
                            'cache_path',
                            'num_samples',
                            'num_shards',
                            'shard_size',
                    }:
                        continue
                    collect_declared(value)
            elif isinstance(node, list):
                for value in node:
                    collect_declared(value)

        collect_declared(tta_response_files)
        if declared:
            actual = actual_cache_declared_paths(cache_dir)
            missing = [
                path for path, candidates in declared
                if not any(candidate in actual for candidate in candidates)
            ]
            if missing:
                errors.append('manifest-declared cache files missing: ' +
                              ', '.join(str(path) for path in missing))
    tta_config = run_manifest.get('tta_config', {})
    protocol_config = run_manifest.get('protocol_config', {})
    checks = [
        ('dataset', run_manifest.get('dataset'), expected.get('dataset')),
        ('baseline_protocol', run_manifest.get('baseline_protocol'),
         expected.get('baseline_protocol')),
        ('scheme', scheme_manifest.get('scheme'), expected.get('scheme')),
        ('checkpoint_sha256', run_manifest.get('checkpoint_sha256'),
         expected.get('checkpoint_sha256')),
        ('model_arch', run_manifest.get('model_arch'),
         expected.get('model_arch')),
        ('num_classes', run_manifest.get('num_classes'),
         expected.get('num_classes')),
        ('classifier_layer', run_manifest.get('classifier_layer'),
         expected.get('classifier_layer')),
        ('objective', tta_config.get('objective'), expected.get('objective')),
        ('steps', tta_config.get('steps'), expected.get('steps')),
        ('lr', tta_config.get('lr'), expected.get('lr')),
        ('update_scope', tta_config.get('update_scope'),
         expected.get('update_scope')),
        ('runtime_mode', tta_config.get('runtime_mode'),
         expected.get('runtime_mode')),
        ('freeze_bn_stats', tta_config.get('freeze_bn_stats'),
         expected.get('freeze_bn_stats')),
    ]
    for name, actual, wanted in checks:
        if wanted is not None and actual != wanted:
            errors.append(f'{name} mismatch: manifest={actual} expected={wanted}')
    expected_csid = expected.get('csid_datasets')
    if expected_csid is not None:
        actual_csid = protocol_config.get(
            'resolved_csid_datasets',
            scheme_manifest.get('resolved_dataset_names', {}).get('csid', []),
        )
        if list(actual_csid) != list(expected_csid):
            errors.append(
                f'csid_datasets mismatch: manifest={actual_csid} expected={expected_csid}')
    return errors


def validate_tta_response(cache_dir, expected=None):
    cache_dir = Path(cache_dir)
    entries = logical_cache_entries(cache_dir)
    if not entries:
        raise FileNotFoundError(
            f'No TTA response datasets found in {cache_dir}')
    failures = []
    manifest_errors = manifest_validation_errors(cache_dir, expected)
    if manifest_errors:
        failures.append((cache_dir, manifest_errors))
    for _, path in entries:
        errors = validate_cache_file(path)
        if errors:
            failures.append((path, errors))
    return failures


def validate_command(args):
    cache_dir = resolve_cache_dir(
        cache_dir=args.cache_dir,
        run_dir=args.run_dir,
        scheme=args.scheme,
        reference_config_id=args.reference_config_id,
    )
    if not cache_dir.exists():
        raise FileNotFoundError(cache_dir)

    entries = logical_cache_entries(cache_dir)
    expected = {
        'dataset': args.expect_dataset,
        'baseline_protocol': args.expect_baseline_protocol,
        'scheme': args.expect_scheme,
        'checkpoint_sha256': args.expect_checkpoint_sha256,
        'model_arch': args.expect_model_arch,
        'num_classes': args.expect_num_classes,
        'classifier_layer': args.expect_classifier_layer,
        'objective': args.expect_objective,
        'steps': args.expect_steps,
        'lr': args.expect_lr,
        'update_scope': args.expect_update_scope,
        'runtime_mode': args.expect_runtime_mode,
        'freeze_bn_stats': args.expect_freeze_bn_stats,
        'csid_datasets': parse_dataset_list(args.expect_csid_datasets)
        if args.expect_csid_datasets else None,
        'reference_source': args.expect_reference_source,
        'reference_per_class': args.expect_reference_per_class,
        'reference_filter': args.expect_reference_filter,
        'reference_min_confidence': args.expect_reference_min_confidence,
        'reference_seed': args.expect_reference_seed,
        'selected_reference_hash': args.expect_selected_reference_hash,
    }
    failures = validate_tta_response(cache_dir, expected)

    if failures:
        for path, errors in failures:
            print(f'FAIL {path}')
            for error in errors:
                print(f'  - {error}')
        raise SystemExit(1)

    print(f'OK {len(entries)} TTA response dataset(s): {cache_dir}')


def score_command(args):
    cache_dir = resolve_cache_dir(
        cache_dir=args.cache_dir,
        run_dir=args.run_dir,
        scheme=args.scheme,
        reference_config_id=args.reference_config_id,
    )
    diagnostic_modes = [
        bool(args.vector_score_rule),
        bool(args.perturbation_score_rule),
    ]
    if sum(diagnostic_modes) > 1:
        raise ValueError(
            '--vector-score-rule and --perturbation-score-rule are separate '
            'diagnostic paths and cannot be combined.')

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.vector_score_rule:
        output_dir = (
            cache_dir.parent / SCORE_RESULTS_DIR / 'vector' /
            f'id_side_{args.fsood_id_side}')
    elif args.perturbation_score_rule:
        output_dir = (
            cache_dir.parent / SCORE_RESULTS_DIR / 'perturbation' /
            f'id_side_{args.fsood_id_side}')
    elif args.fsood_id_side == 'both':
        output_dir = cache_dir.parent / SCORE_RESULTS_DIR
    else:
        output_dir = (
            cache_dir.parent / SCORE_RESULTS_DIR /
            f'id_side_{args.fsood_id_side}')
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f'Output directory already exists: {output_dir}. '
            'Use --output-dir with a new path or pass --overwrite.')

    base_score_rules = selected_score_rules(args.score_rule)
    vector_fit = None
    if args.vector_score_rule:
        score_rules = selected_vector_score_rules(args.vector_score_rule)
    elif args.perturbation_score_rule:
        score_rules = selected_perturbation_score_rules(
            args.perturbation_score_rule)
    else:
        score_rules = base_score_rules
    id_cache = load_cache(cache_dir, args.dataset)

    csid_caches = []
    if args.scheme == 'fsood':
        csid_names = resolve_csid_datasets(args, cache_dir)
        csid_caches = [(name, load_cache(cache_dir, name)) for name in csid_names]
    if args.fsood_id_side == 'csid' and not csid_caches:
        raise ValueError('--fsood-id-side csid requires --scheme fsood '
                         'with resolvable csID caches')

    near_names = resolve_split_datasets(args, cache_dir, 'near')
    far_names = resolve_split_datasets(args, cache_dir, 'far')
    near_caches = [(name, load_cache(cache_dir, name)) for name in near_names]
    far_caches = [(name, load_cache(cache_dir, name)) for name in far_names]
    score_manifest = {
        'cache_schema_version': CACHE_SCHEMA_VERSION,
        'diagnostic_only': bool(args.vector_score_rule
                                or args.perturbation_score_rule),
        'score_direction': SCORE_DIRECTION,
        'conf_boundary_transform': 'conf = -ood_score',
        'perturbation_score_direction': PERTURBATION_SCORE_DIRECTION,
        'perturbation_definition': PERTURBATION_DEFINITION,
        'source_tta_response_dir': str(cache_dir),
        'dataset': args.dataset,
        'scheme': args.scheme,
        'fsood_id_side': args.fsood_id_side,
        'score_rule_arg': args.score_rule,
        'base_score_rules': base_score_rules,
        'expanded_score_rules': score_rules,
        'vector_score_rule_arg': args.vector_score_rule or '',
        'perturbation_score_rule_arg': args.perturbation_score_rule or '',
        'score_family': 'vector' if args.vector_score_rule else (
            'perturbation' if args.perturbation_score_rule else 'score'),
        'near_datasets': near_names,
        'far_datasets': far_names,
        'csid_datasets': [name for name, _ in csid_caches],
    }
    response_step_values = resolve_response_step_values(
        id_cache, args.response_step)
    use_step_dir = should_use_response_step_dir(
        id_cache, args.response_step, response_step_values)
    manifest_data = dict(score_manifest)
    manifest_data.update({
        'response_step_arg': args.response_step,
        'response_steps': response_steps_from_cache(id_cache).tolist(),
        'response_step_values': response_step_values,
    })

    for response_step in response_step_values:
        id_cache_step = select_response_step(id_cache, response_step)
        csid_caches_step = [
            (name, select_response_step(cache, response_step))
            for name, cache in csid_caches
        ]
        near_caches_step = [
            (name, select_response_step(cache, response_step))
            for name, cache in near_caches
        ]
        far_caches_step = [
            (name, select_response_step(cache, response_step))
            for name, cache in far_caches
        ]
        step_manifest = dict(score_manifest)
        step_manifest.update({
            'response_step_arg': args.response_step,
            'response_steps': response_steps_from_cache(id_cache).tolist(),
            'response_step': response_step,
        })
        vector_manifest = None
        if args.vector_score_rule:
            vector_fit = fit_vector_score_reference(id_cache_step)
            vector_manifest = dict(step_manifest)
            vector_manifest.update({
                'diagnostic_only': True,
                'fit_source': 'clean_id_cache',
                'vector_fit': vector_fit,
            })
            if not use_step_dir:
                write_json(output_dir / 'vector_fit.json', vector_fit)

        for score_rule in score_rules:
            compat_caches = (
                [('clean_id', id_cache_step)]
                + [(name, cache) for name, cache in csid_caches_step]
                + [(name, cache) for name, cache in near_caches_step]
                + [(name, cache) for name, cache in far_caches_step]
            )
            if args.vector_score_rule or args.perturbation_score_rule:
                branch_variants = [(None, None)]
            else:
                branch_variants = branch_variants_for_rule(
                    id_cache_step, compat_caches, score_rule, args)

            for accept_choice, reject_choice in branch_variants:
                current_rule_dir = rule_output_dir(
                    output_dir, score_rule, response_step, use_step_dir)
                current_rule_dir = branch_variant_dir(
                    current_rule_dir, accept_choice, reject_choice)
                id_cache_variant = materialize_branch_variant(
                    id_cache_step, accept_choice, reject_choice)
                if args.vector_score_rule:
                    id_ood = raw_vector_ood_scores(
                        id_cache_variant, score_rule, vector_fit)
                    id_scores = score_tuple_from_ood(id_cache_variant, id_ood)
                elif args.perturbation_score_rule:
                    id_ood = raw_perturbation_ood_scores(
                        id_cache_variant, score_rule)
                    id_scores = score_tuple_from_ood(id_cache_variant, id_ood)
                else:
                    id_scores = score_tuple(id_cache_variant, score_rule)
                metric_id_scores = id_scores
                save_npz(current_rule_dir / 'scores' / f'{args.dataset}.npz',
                         id_scores)

                if csid_caches_step:
                    csid_score_parts = []
                    for csid_name, csid_cache in csid_caches_step:
                        csid_cache_variant = materialize_branch_variant(
                            csid_cache, accept_choice, reject_choice)
                        if args.vector_score_rule:
                            csid_ood = raw_vector_ood_scores(
                                csid_cache_variant, score_rule, vector_fit)
                            csid_scores = score_tuple_from_ood(
                                csid_cache_variant, csid_ood)
                        elif args.perturbation_score_rule:
                            csid_ood = raw_perturbation_ood_scores(
                                csid_cache_variant, score_rule)
                            csid_scores = score_tuple_from_ood(
                                csid_cache_variant, csid_ood)
                        else:
                            csid_scores = score_tuple(
                                csid_cache_variant, score_rule)
                        save_npz(
                            current_rule_dir / 'scores' / f'{csid_name}.npz',
                            csid_scores)
                        csid_score_parts.append(csid_scores)
                    if args.fsood_id_side == 'both':
                        metric_id_scores = concat_scores(
                            [id_scores] + csid_score_parts)
                    elif args.fsood_id_side == 'csid':
                        metric_id_scores = concat_scores(csid_score_parts)

                rows = []
                for split_name, split_caches in [
                        ('near', near_caches_step),
                        ('far', far_caches_step),
                ]:
                    split_metrics = []
                    for name, split_cache in split_caches:
                        split_cache_variant = materialize_branch_variant(
                            split_cache, accept_choice, reject_choice)
                        ood_label = -1 * np.ones_like(
                            split_cache_variant['label'], dtype=np.int64)
                        if args.vector_score_rule:
                            split_ood = raw_vector_ood_scores(
                                split_cache_variant, score_rule, vector_fit)
                            split_scores = score_tuple_from_ood(
                                split_cache_variant, split_ood, ood_label)
                        elif args.perturbation_score_rule:
                            split_ood = raw_perturbation_ood_scores(
                                split_cache_variant, score_rule)
                            split_scores = score_tuple_from_ood(
                                split_cache_variant, split_ood, ood_label)
                        else:
                            split_scores = score_tuple(
                                split_cache_variant, score_rule, ood_label)
                        save_npz(
                            current_rule_dir / 'scores' / f'{name}.npz',
                            split_scores)
                        metrics = metric_summary(metric_id_scores, split_scores)
                        split_metrics.append(metrics)
                        rows.append(format_metric_row(name, metrics))
                    if split_metrics:
                        rows.append(
                            format_metric_row(
                                f'{split_name}ood',
                                np.mean(np.asarray(split_metrics), axis=0)))
                write_metrics_csv(current_rule_dir / 'ood.csv', rows)
                rule_manifest = (
                    vector_manifest if vector_manifest is not None
                    else dict(step_manifest))
                if vector_manifest is not None:
                    write_json(current_rule_dir / 'vector_fit.json',
                               vector_manifest)
                if accept_choice is not None:
                    rule_manifest['selected_accept_branch_id'] = (
                        accept_choice['id'])
                if reject_choice is not None:
                    rule_manifest['selected_reject_branch_id'] = (
                        reject_choice['id'])
                rule_manifest['accept_branch_arg'] = args.accept_branch
                rule_manifest['reject_branch_arg'] = args.reject_branch
                rule_manifest['branch_combine'] = args.branch_combine
                write_json(current_rule_dir / 'score_manifest.json',
                           rule_manifest)

    if args.vector_score_rule:
        manifest_name = 'vector_score.json'
    elif args.perturbation_score_rule:
        manifest_name = 'perturbation_score.json'
    else:
        manifest_name = 'score.json'
    write_json(output_dir / manifest_name,
               manifest_data)
    print(f'output_dir: {output_dir}')


def add_cache_location_args(parser):
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument('--tta-response-dir', dest='cache_dir')
    location.add_argument('--run-dir')
    parser.add_argument('--scheme', default='ood', choices=['ood', 'fsood'])
    parser.add_argument(
        '--reference-config-id',
        help=('Use <run>/<scheme>/references/<reference_config_id>/'
              f'{TTA_RESPONSE_DIR} when resolving from --run-dir.'),
    )


def add_expected_validation_args(parser):
    parser.add_argument('--expect-dataset')
    parser.add_argument('--expect-baseline-protocol')
    parser.add_argument('--expect-scheme')
    parser.add_argument('--expect-checkpoint-sha256')
    parser.add_argument('--expect-model-arch')
    parser.add_argument('--expect-num-classes', type=int)
    parser.add_argument('--expect-classifier-layer')
    parser.add_argument('--expect-objective')
    parser.add_argument('--expect-steps', type=int)
    parser.add_argument('--expect-lr', type=float)
    parser.add_argument('--expect-update-scope')
    parser.add_argument('--expect-runtime-mode')
    freeze = parser.add_mutually_exclusive_group()
    freeze.add_argument('--expect-freeze-bn-stats',
                        dest='expect_freeze_bn_stats',
                        action='store_true')
    freeze.add_argument('--expect-no-freeze-bn-stats',
                        dest='expect_freeze_bn_stats',
                        action='store_false')
    parser.set_defaults(expect_freeze_bn_stats=None)
    parser.add_argument('--expect-csid-datasets')
    parser.add_argument('--expect-reference-source')
    parser.add_argument('--expect-reference-per-class', type=int)
    parser.add_argument('--expect-reference-filter')
    parser.add_argument('--expect-reference-min-confidence', type=float)
    parser.add_argument('--expect-reference-seed', type=int)
    parser.add_argument('--expect-selected-reference-hash')


def add_score_command_args(parser):
    add_cache_location_args(parser)
    parser.add_argument('--dataset',
                        required=True,
                        choices=supported_dataset_names())
    parser.add_argument('--output-dir')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Allow writing into an existing output directory.',
    )
    parser.add_argument(
        '--fsood-id-side',
        default='both',
        choices=['both', 'clean', 'csid'],
        help=('Metric aggregation side for saved FSOOD TTA responses. This '
              'selects which ID-side responses are used when writing '
              'score_result artifacts.'),
    )
    parser.add_argument('--score-rule',
                        default='all',
                        choices=SCORE_RULE_CHOICES)
    parser.add_argument(
        '--response-step',
        default='final',
        help=('Saved response-bank step to score, or "all" to write one '
              'score result per saved step. Defaults to the final saved '
              'response step.'),
    )
    parser.add_argument(
        '--accept-branch',
        default='auto',
        help=('Acceptance response-bank branch selector for branch scoring: '
              'auto, all, branch name, branch index, or comma list. Primary '
              'singleton fields are used automatically when no response bank '
              'is present.'),
    )
    parser.add_argument(
        '--reject-branch',
        default='auto',
        help=('Rejection response-bank branch selector for branch scoring: '
              'auto, all, branch name, branch index, or comma list. Primary '
              'singleton fields are used automatically when no response bank '
              'is present.'),
    )
    parser.add_argument(
        '--branch-combine',
        default='cross',
        choices=['cross', 'zip'],
        help=('How to combine selected accept/reject response-bank branches '
              'for paired branch scores.'),
    )
    parser.add_argument(
        '--vector-score-rule',
        choices=VECTOR_SCORE_RULE_CHOICES,
        help=('Write diagnostic-only direction/vector-aware score candidates '
              f'under {SCORE_RESULTS_DIR}/vector/.'),
    )
    parser.add_argument(
        '--perturbation-score-rule',
        choices=PERTURBATION_SCORE_RULE_CHOICES,
        help=('Write diagnostic-only target perturbation-response score '
              f'candidates under {SCORE_RESULTS_DIR}/perturbation/.'),
    )
    parser.add_argument('--near-datasets', default='all')
    parser.add_argument('--far-datasets', default='all')
    parser.add_argument(
        '--csid-datasets',
        help=('Comma-separated fallback csID dataset names when no TARR '
              'manifest is available. FSOOD defaults read scheme/run manifest '
              'metadata.'),
    )
    parser.set_defaults(func=score_command)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    validate = subparsers.add_parser('validate')
    add_cache_location_args(validate)
    add_expected_validation_args(validate)
    validate.set_defaults(func=validate_command)

    score = subparsers.add_parser('score')
    add_score_command_args(score)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
