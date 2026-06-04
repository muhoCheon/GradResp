"""Shared TARR score-rule definitions."""

from __future__ import annotations

import numpy as np

ACTIVE_SCORE_RULES = [
    'predicted_class_loss_increase',
    'predicted_class_loss_decrease',
    'target_weighted_loss_increase',
    'target_weighted_loss_decrease',
    'mean_loss_increase',
    'mean_loss_decrease',
    'positive_loss_increase_mean',
    'positive_loss_decrease_mean',
]
PROBE_SCORE_RULES = [
    'accept_efficiency',
    'reject_efficiency',
    'log_reject_efficiency',
    'ar_efficiency_contrast',
    'ref_loss_delta_contrast',
    'accept_target_objective_delta',
    'reject_target_objective_delta',
    'target_objective_delta_contrast',
    'accept_pos_ref_loss_delta_mean',
    'reject_pos_ref_loss_delta_mean',
    'reject_pos_ref_loss_delta_ood',
    'accept_signed_ref_loss_delta_mean',
    'reject_signed_ref_loss_delta_mean',
    'signed_ref_loss_delta_contrast',
    'accept_abs_ref_loss_delta_mean',
    'reject_abs_ref_loss_delta_mean',
    'abs_ref_loss_delta_contrast',
    'accept_pred_ref_loss_delta',
    'reject_pred_ref_loss_delta',
    'pred_ref_loss_delta_contrast',
    'accept_target_weighted_ref_loss_delta',
    'reject_target_weighted_ref_loss_delta',
    'target_weighted_ref_loss_delta_contrast',
    'accept_abs_ref_efficiency',
    'reject_abs_ref_efficiency',
    'log_reject_abs_ref_efficiency',
    'ar_abs_ref_efficiency_contrast',
    'accept_pred_ref_efficiency',
    'reject_pred_ref_efficiency',
    'log_reject_pred_ref_efficiency',
    'ar_pred_ref_efficiency_contrast',
    'accept_target_weighted_ref_efficiency',
    'reject_target_weighted_ref_efficiency',
    'log_reject_target_weighted_ref_efficiency',
    'ar_target_weighted_ref_efficiency_contrast',
]
SCORE_RULE_CHOICES = (
    ACTIVE_SCORE_RULES + PROBE_SCORE_RULES + [
    'all',
    'probe_all',
])
VECTOR_SCORE_RULES = [
    'rest_positive_loss_increase_mean',
    'pred_vs_rest_loss_increase_gap',
    'target_centered_loss_increase',
    'delta_l2_norm',
    'clean_delta_z_l2',
    'clean_delta_cosine_distance',
]
VECTOR_SCORE_RULE_CHOICES = VECTOR_SCORE_RULES + ['all']
PERTURBATION_SCORE_RULES = [
    'logit_l2',
    'prob_l1',
    'conf_drop',
    'entropy_increase',
]
PERTURBATION_SCORE_RULE_CHOICES = PERTURBATION_SCORE_RULES + ['all']
PROBE_SCORE_RULE_CHOICES = PROBE_SCORE_RULES + ['probe_all']

PROBE_FIELD_ALIASES = {
    'accept_ref_loss_delta': (
        'accept_ref_loss_delta',
    ),
    'reject_ref_loss_delta': (
        'reject_ref_loss_delta',
    ),
    'accept_target_objective_delta': (
        'accept_target_objective_delta',
    ),
    'reject_target_objective_delta': (
        'reject_target_objective_delta',
    ),
    'reject_target_entropy_delta': (
        'reject_target_entropy_delta',
    ),
}
PROBE_SAMPLE_CACHE_KEYS = sorted({
    key
    for aliases in PROBE_FIELD_ALIASES.values()
    for key in aliases
})
PROBE_METADATA_CACHE_KEYS = [
    'probe_score_rules',
    'probe_score_rule_arg',
    'probe_score_alpha',
    'probe_score_beta',
    'probe_score_gamma',
    'probe_score_temperature',
    'probe_score_eps',
]

ACCEPT_BRANCH_BANK_KEYS = [
    'accept_ref_loss_delta_bank',
    'accept_target_objective_delta_bank',
]
REJECT_BRANCH_BANK_KEYS = [
    'reject_ref_loss_delta_bank',
    'reject_target_objective_delta_bank',
    'reject_target_entropy_delta_bank',
]
BRANCH_BANK_SAMPLE_CACHE_KEYS = (
    ACCEPT_BRANCH_BANK_KEYS + REJECT_BRANCH_BANK_KEYS
)
BRANCH_BANK_METADATA_CACHE_KEYS = [
    'accept_branch_ids',
    'accept_branch_probe_types',
    'reject_branch_ids',
    'reject_branch_probe_types',
    'primary_accept_branch_id',
    'primary_reject_branch_id',
    'response_bank_schema_version',
]

CACHE_SCHEMA_VERSION = 5
SCORE_DIRECTION = 'higher_is_ood'
DELTA_DEFINITION = 'adapted_minus_base'
PERTURBATION_DEFINITION = 'perturbed_minus_clean'
PERTURBATION_SCORE_DIRECTION = SCORE_DIRECTION
NUMERIC_EPS = 1e-12
VECTOR_FIT_METHOD = 'clean_id_delta_vector'

STEP_INDEXED_CACHE_KEYS = {
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
    'adapted_reference_loss',
    'delta',
    'reference_conf_delta_by_class',
    'reference_entropy_delta_by_class',
    'reference_margin_delta_by_class',
    'reference_energy_delta_by_class',
    'reference_pred_changed_rate_by_class',
    'reference_correct_rate_after_by_class',
    'adapted_reference_loss_mean',
    'adapted_reference_loss_std',
    'adapted_reference_loss_min',
    'adapted_reference_loss_max',
    'reference_delta_mean',
    'reference_delta_std',
    'reference_delta_min',
    'reference_delta_max',
    'reference_delta_positive_mean',
    'accept_ref_loss_delta',
    'reject_ref_loss_delta',
    'accept_target_objective_delta',
    'reject_target_objective_delta',
    'reject_target_entropy_delta',
    'accept_ref_loss_delta_bank',
    'reject_ref_loss_delta_bank',
    'accept_target_objective_delta_bank',
    'reject_target_objective_delta_bank',
    'reject_target_entropy_delta_bank',
}

BRANCH_BANK_ROLE_FIELDS = {
    'accept': {
        'ref_loss_delta': (
            'accept_ref_loss_delta_bank',
            'accept_ref_loss_delta',
        ),
        'target_objective_delta': (
            'accept_target_objective_delta_bank',
            'accept_target_objective_delta',
        ),
    },
    'reject': {
        'ref_loss_delta': (
            'reject_ref_loss_delta_bank',
            'reject_ref_loss_delta',
        ),
        'target_objective_delta': (
            'reject_target_objective_delta_bank',
            'reject_target_objective_delta',
        ),
        'target_entropy_delta': (
            'reject_target_entropy_delta_bank',
            'reject_target_entropy_delta',
        ),
    },
}


def response_steps_from_cache(cache):
    if 'response_steps' not in cache:
        return np.asarray([], dtype=np.int64)
    steps = np.asarray(cache['response_steps'], dtype=np.int64)
    if steps.ndim == 0:
        steps = steps.reshape(1)
    return steps.reshape(-1)


def response_step_index(cache, response_step=None):
    steps = response_steps_from_cache(cache)
    if steps.size == 0:
        return None
    if response_step is None or str(response_step).strip() in {'', 'final'}:
        return int(steps.size - 1)
    step = int(response_step)
    matches = np.flatnonzero(steps == step)
    if matches.size == 0:
        raise ValueError(
            f'Response step {step} is not saved; available steps: '
            f'{steps.tolist()}')
    return int(matches[0])


def select_response_step(cache, response_step=None):
    default_step = (
        response_step is None or str(response_step).strip() in {'', 'final'}
    )
    if default_step and 'response_step' in cache:
        return dict(cache)
    steps = response_steps_from_cache(cache)
    step_index = response_step_index(cache, response_step)
    if step_index is None:
        return dict(cache)
    selected = {}
    for key, value in cache.items():
        array = np.asarray(value)
        if (key in STEP_INDEXED_CACHE_KEYS
                and array.ndim >= 2
                and array.shape[1] == steps.size):
            selected[key] = array[:, step_index, ...]
        else:
            selected[key] = value
    selected['response_step'] = np.asarray(int(steps[step_index]),
                                           dtype=np.int64)
    return selected


def _metadata_string(value):
    value = np.asarray(value)
    if value.shape != ():
        raise ValueError(f'branch metadata value must be scalar, got {value.shape}')
    item = value.item()
    if isinstance(item, bytes):
        return item.decode('utf-8')
    return str(item)


def _metadata_string_list(value):
    values = np.asarray(value)
    if values.ndim != 1:
        raise ValueError(
            f'branch metadata must be 1-D, got shape {values.shape}')
    result = []
    for item in values.tolist():
        if isinstance(item, bytes):
            item = item.decode('utf-8')
        result.append(str(item))
    return result


def has_branch_bank(cache, role=None):
    if role is None:
        return any(key in cache for key in BRANCH_BANK_SAMPLE_CACHE_KEYS)
    return any(
        bank_key in cache
        for bank_key, _ in BRANCH_BANK_ROLE_FIELDS[role].values()
    )


def branch_bank_role_count(cache, role):
    for bank_key, _ in BRANCH_BANK_ROLE_FIELDS[role].values():
        if bank_key not in cache:
            continue
        values = np.asarray(cache[bank_key])
        if values.ndim in {2, 3}:
            return int(
                values.shape[-2 if _is_ref_loss_delta_bank(bank_key) else -1]
            )
        if values.ndim == 4:
            return int(values.shape[2])
    return 0


def branch_ids_from_cache(cache, role):
    count = branch_bank_role_count(cache, role)
    for key in [f'{role}_branch_ids', f'{role}_branch_probe_types']:
        if key not in cache:
            continue
        values = _metadata_string_list(cache[key])
        if count and len(values) != count:
            raise ValueError(
                f'{key} length {len(values)} != {role} bank branch count '
                f'{count}')
        return values
    return [str(index) for index in range(count)]


def _is_probe_contrast_rule(score_rule):
    return (
        score_rule.startswith('ar_')
        or score_rule.endswith('_contrast')
    )


def branch_score_rule_roles(score_rule):
    if score_rule not in PROBE_SCORE_RULES:
        return ()
    if _is_probe_contrast_rule(score_rule):
        return ('accept', 'reject')
    if score_rule.startswith('accept_'):
        return ('accept',)
    if score_rule.startswith('reject_') or score_rule.startswith('log_reject_'):
        return ('reject',)
    return ()


def _is_ref_loss_delta_bank(key):
    return key in {
        'accept_ref_loss_delta_bank',
        'reject_ref_loss_delta_bank',
    }


def _branch_bank_error_prefix(key):
    return f'{key} response-bank branch shape'


def _branch_bank_array_errors(cache, key, role, n, num_steps, num_classes):
    if key not in cache:
        return []
    values = np.asarray(cache[key])
    errors = []
    prefix = _branch_bank_error_prefix(key)
    is_delta = _is_ref_loss_delta_bank(key)
    if is_delta:
        allowed_ndim = {3, 4}
    else:
        allowed_ndim = {2, 3}
    if values.ndim not in allowed_ndim:
        errors.append(
            f'{prefix} must be '
            f'[N,S,{"A" if role == "accept" else "R"},C] or '
            f'[N,{"A" if role == "accept" else "R"},C]'
            if is_delta else
            f'{prefix} must be [N,S,'
            f'{"A" if role == "accept" else "R"}] or '
            f'[N,{"A" if role == "accept" else "R"}]')
        return errors
    if values.shape[0] != n:
        errors.append(f'{key} first dimension {values.shape[0]} != {n}')
    if is_delta:
        if values.ndim == 4:
            if num_steps and values.shape[1] != num_steps:
                errors.append(
                    f'{key} step dimension {values.shape[1]} != {num_steps}')
            if values.shape[3] != num_classes:
                errors.append(
                    f'{key} class dimension {values.shape[3]} != '
                    f'{num_classes}')
        elif values.ndim == 3:
            if values.shape[2] != num_classes:
                errors.append(
                    f'{key} class dimension {values.shape[2]} != '
                    f'{num_classes}')
    else:
        if values.ndim == 3 and num_steps and values.shape[1] != num_steps:
            errors.append(
                f'{key} step dimension {values.shape[1]} != {num_steps}')
    if values.size and not np.all(np.isfinite(values.astype(np.float64))):
        errors.append(f'{key} contains non-finite values')
    return errors


def branch_bank_shape_errors(cache, n=None, num_steps=None, num_classes=None):
    if n is None:
        n = int(np.asarray(cache['pred']).shape[0])
    if num_steps is None:
        num_steps = int(response_steps_from_cache(cache).size)
    if num_classes is None:
        target_probs = np.asarray(cache['target_probs'])
        num_classes = int(target_probs.shape[1]) if target_probs.ndim == 2 else 0

    errors = []
    role_counts = {}
    for role, fields in BRANCH_BANK_ROLE_FIELDS.items():
        counts = []
        for bank_key, _ in fields.values():
            errors.extend(
                _branch_bank_array_errors(
                    cache, bank_key, role, n, num_steps, num_classes))
            if bank_key not in cache:
                continue
            values = np.asarray(cache[bank_key])
            if values.ndim == 4:
                counts.append(values.shape[2])
            elif values.ndim == 3:
                counts.append(
                    values.shape[1]
                    if _is_ref_loss_delta_bank(bank_key)
                    else values.shape[2])
            elif values.ndim == 2:
                counts.append(values.shape[1])
        if counts and len(set(counts)) != 1:
            errors.append(
                f'{role} response-bank branch counts differ: {counts}')
        if counts:
            role_counts[role] = counts[0]

    for role, count in role_counts.items():
        for key in [f'{role}_branch_ids', f'{role}_branch_probe_types']:
            if key not in cache:
                continue
            try:
                values = _metadata_string_list(cache[key])
            except ValueError as exc:
                errors.append(f'{key}: {exc}')
                continue
            if len(values) != count:
                errors.append(f'{key} length {len(values)} != {count}')
        primary_key = f'primary_{role}_branch_id'
        if primary_key in cache:
            try:
                _metadata_string(cache[primary_key])
            except ValueError as exc:
                errors.append(f'{primary_key}: {exc}')
    return errors


def validate_branch_bank_shapes(cache):
    errors = branch_bank_shape_errors(cache)
    if errors:
        raise ValueError('; '.join(errors))


def _selected_branch_array(values, index, key, role):
    values = np.asarray(values)
    if _is_ref_loss_delta_bank(key):
        if values.ndim != 3:
            raise ValueError(
                f'{key} must be [N,{"A" if role == "accept" else "R"},C] '
                f'after response-step selection, got {values.shape}')
        return values[:, index, :]
    if values.ndim != 2:
        raise ValueError(
            f'{key} must be [N,{"A" if role == "accept" else "R"}] after '
            f'response-step selection, got {values.shape}')
    return values[:, index]


def materialize_branch_bank(cache, accept_branch=None, reject_branch=None):
    cache = select_response_step(cache)
    selected = dict(cache)
    for role, index in [('accept', accept_branch), ('reject', reject_branch)]:
        if index is None:
            continue
        ids = branch_ids_from_cache(cache, role)
        if index < 0 or index >= len(ids):
            raise ValueError(
                f'{role} response-bank branch index {index} out of range '
                f'for {ids}')
        for bank_key, primary_key in BRANCH_BANK_ROLE_FIELDS[role].values():
            if bank_key not in cache:
                continue
            selected[primary_key] = _selected_branch_array(
                cache[bank_key], int(index), bank_key, role)
        selected[f'selected_{role}_branch_id'] = np.asarray(ids[int(index)])
    return selected


def selected_score_rules(score_rule):
    if score_rule == 'all':
        return list(ACTIVE_SCORE_RULES)
    if score_rule == 'probe_all':
        return list(PROBE_SCORE_RULES)
    if (score_rule not in ACTIVE_SCORE_RULES
            and score_rule not in PROBE_SCORE_RULES):
        raise ValueError(f'Unknown score rule: {score_rule}')
    return [score_rule]


def selected_probe_score_rules(score_rule):
    if score_rule == 'probe_all':
        return list(PROBE_SCORE_RULES)
    if score_rule not in PROBE_SCORE_RULES:
        raise ValueError(f'Unknown probe score rule: {score_rule}')
    return [score_rule]


def selected_vector_score_rules(score_rule):
    if score_rule == 'all':
        return list(VECTOR_SCORE_RULES)
    if score_rule not in VECTOR_SCORE_RULES:
        raise ValueError(f'Unknown vector score rule: {score_rule}')
    return [score_rule]


def selected_perturbation_score_rules(score_rule):
    if score_rule == 'all':
        return list(PERTURBATION_SCORE_RULES)
    if score_rule not in PERTURBATION_SCORE_RULES:
        raise ValueError(f'Unknown perturbation score rule: {score_rule}')
    return [score_rule]


def _target_probs(probs):
    if getattr(probs, 'ndim', 0) == 2:
        return probs[0]
    return probs


def score_from_delta(delta, probs, y_hat, score_rule):
    probs = _target_probs(probs)
    if score_rule == 'predicted_class_loss_increase':
        return delta[y_hat]
    if score_rule == 'predicted_class_loss_decrease':
        return -delta[y_hat]
    if score_rule == 'target_weighted_loss_increase':
        return (probs * delta).sum()
    if score_rule == 'target_weighted_loss_decrease':
        return -(probs * delta).sum()
    if score_rule == 'mean_loss_increase':
        return delta.mean()
    if score_rule == 'mean_loss_decrease':
        return -delta.mean()
    if score_rule == 'positive_loss_increase_mean':
        if hasattr(delta, 'clamp'):
            return delta.clamp(min=0.0).mean()
        return np.clip(delta, 0.0, None).mean()
    if score_rule == 'positive_loss_decrease_mean':
        if hasattr(delta, 'clamp'):
            return (-delta).clamp(min=0.0).mean()
        return np.clip(-delta, 0.0, None).mean()
    if score_rule in PROBE_SCORE_RULES:
        raise ValueError(
            f'Probe score rule {score_rule} requires cached probe fields in '
            'tta_response; use ood_score_from_cache(cache, score_rule).')
    raise ValueError(f'Unknown score rule: {score_rule}')


def ood_score_from_cache(cache, score_rule, vector_fit=None):
    cache = select_response_step(cache)
    if score_rule in VECTOR_SCORE_RULES:
        return vector_ood_score_from_cache(cache, score_rule, vector_fit)
    if score_rule in PROBE_SCORE_RULES:
        return probe_ood_score_from_cache(cache, score_rule)

    delta = cache['delta']
    probs = cache['target_probs']
    y_hat = cache['y_hat'].astype(np.int64)
    row = np.arange(delta.shape[0])

    if score_rule == 'predicted_class_loss_increase':
        return delta[row, y_hat]
    if score_rule == 'predicted_class_loss_decrease':
        return -delta[row, y_hat]
    if score_rule == 'target_weighted_loss_increase':
        return np.sum(probs * delta, axis=1)
    if score_rule == 'target_weighted_loss_decrease':
        return -np.sum(probs * delta, axis=1)
    if score_rule == 'mean_loss_increase':
        return np.mean(delta, axis=1)
    if score_rule == 'mean_loss_decrease':
        return -np.mean(delta, axis=1)
    if score_rule == 'positive_loss_increase_mean':
        return np.mean(np.clip(delta, 0.0, None), axis=1)
    if score_rule == 'positive_loss_decrease_mean':
        return np.mean(np.clip(-delta, 0.0, None), axis=1)
    raise ValueError(f'Unknown score rule: {score_rule}')


def _probe_field_label(group):
    return ' or '.join(PROBE_FIELD_ALIASES[group])


def _probe_source_label(groups):
    return ' or '.join(_probe_field_label(group) for group in groups)


def _find_probe_field(cache, group):
    for key in PROBE_FIELD_ALIASES[group]:
        if key in cache:
            return key
    return None


def _has_probe_field(cache, group):
    return _find_probe_field(cache, group) is not None


def _required_probe_groups(score_rule):
    roles = branch_score_rule_roles(score_rule)
    groups = []
    if 'accept' in roles and (
            score_rule.startswith('accept_target_objective_delta')
            or score_rule == 'target_objective_delta_contrast'
            or 'efficiency' in score_rule):
        groups.append('accept_target_objective_delta')
    if 'reject' in roles and (
            score_rule.startswith('reject_target_objective_delta')
            or score_rule == 'target_objective_delta_contrast'
            or 'efficiency' in score_rule):
        groups.append('reject_target_objective_delta')
    target_only = score_rule in {
        'accept_target_objective_delta',
        'reject_target_objective_delta',
        'target_objective_delta_contrast',
    }
    if 'accept' in roles and not target_only:
        groups.append('accept_ref_loss_delta')
    if 'reject' in roles and not target_only:
        groups.append('reject_ref_loss_delta')
    return groups


def probe_score_rule_missing_fields(cache, score_rule):
    if score_rule not in PROBE_SCORE_RULES:
        raise ValueError(f'Unknown probe score rule: {score_rule}')

    missing = []
    for group in _required_probe_groups(score_rule):
        if not _has_probe_field(cache, group):
            missing.append(_probe_source_label([group]))
    return missing


def probe_score_rule_has_required_fields(cache, score_rule):
    return not probe_score_rule_missing_fields(cache, score_rule)


def _raise_missing_probe_fields(cache, score_rule):
    missing = probe_score_rule_missing_fields(cache, score_rule)
    if missing:
        raise ValueError(
            f'Probe score rule {score_rule} requires probe fields in '
            f'tta_response: {", ".join(missing)}')


def _probe_scalar(cache, name, default, score_rule):
    key = f'probe_score_{name}'
    if key not in cache:
        return default
    value = np.asarray(cache[key])
    if value.shape != ():
        raise ValueError(
            f'{key} must be scalar for probe score rule {score_rule}')
    value = float(value.item())
    if not np.isfinite(value):
        raise ValueError(
            f'{key} must be finite for probe score rule {score_rule}')
    if name in {'eps', 'temperature'} and value <= 0.0:
        raise ValueError(
            f'{key} must be positive for probe score rule {score_rule}')
    return value


def _probe_array(cache, group, score_rule, ndim):
    key = _find_probe_field(cache, group)
    if key is None:
        _raise_missing_probe_fields(cache, score_rule)
    values = np.asarray(cache[key], dtype=np.float64)
    if values.ndim != ndim:
        raise ValueError(
            f'Probe score rule {score_rule} requires {key} to be {ndim}-D, '
            f'got shape {values.shape}')
    return values, key


def _probe_vector(cache, group, score_rule):
    return _probe_array(cache, group, score_rule, 1)


def _probe_matrix(cache, group, score_rule):
    return _probe_array(cache, group, score_rule, 2)


def _validate_probe_length(values, key, expected_n, score_rule):
    if expected_n is not None and values.shape[0] != expected_n:
        raise ValueError(
            f'Probe score rule {score_rule} requires {key} length '
            f'{values.shape[0]} to match {expected_n}')


def _finite_matrix(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f'{name} must be 2-D, got shape {values.shape}')
    finite_rows = np.all(np.isfinite(values), axis=1)
    finite = values[finite_rows]
    if finite.shape[0] == 0:
        raise ValueError(f'{name} has no fully finite rows')
    return values, finite


def _robust_vector_fit(values, name):
    values, finite = _finite_matrix(values, name)
    q25, q50, q75 = np.quantile(finite, [0.25, 0.5, 0.75], axis=0)
    iqr = q75 - q25
    std = np.std(finite, axis=0)
    scale = np.where(iqr > NUMERIC_EPS, iqr / 1.349, std)
    scale = np.where(scale > NUMERIC_EPS, scale, 1.0)
    return {
        'center': q50.tolist(),
        'scale': scale.tolist(),
        'q25': q25.tolist(),
        'q75': q75.tolist(),
        'n': int(finite.shape[0]),
        'dim': int(values.shape[1]),
        'nonfinite_count': int(values.shape[0] - finite.shape[0]),
    }


def _robust_vector_z(values, fit, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f'{name} must be 2-D, got shape {values.shape}')
    center = np.asarray(fit['center'], dtype=np.float64)
    scale = np.asarray(fit['scale'], dtype=np.float64)
    if center.ndim != 1 or scale.ndim != 1:
        raise ValueError(f'{name} clean calibration must be 1-D vectors')
    if values.shape[1] != center.shape[0] or center.shape != scale.shape:
        raise ValueError(
            f'{name} clean calibration dimension mismatch: values '
            f'{values.shape}, center {center.shape}, scale {scale.shape}')
    if np.any(scale <= NUMERIC_EPS):
        raise ValueError(f'{name} clean calibration scales must be positive')
    return (values - center[None, :]) / scale[None, :]


def _probe_ref_loss_delta(cache, role, score_rule, expected_n=None):
    delta, key = _probe_matrix(cache, f'{role}_ref_loss_delta', score_rule)
    _validate_probe_length(delta, key, expected_n, score_rule)
    return delta


def _probe_target_objective_delta(cache, role, score_rule, expected_n=None):
    values, key = _probe_vector(cache, f'{role}_target_objective_delta', score_rule)
    _validate_probe_length(values, key, expected_n, score_rule)
    return values


def _probe_y_hat(cache, expected_n, expected_classes, score_rule):
    y_hat = np.asarray(cache['y_hat'], dtype=np.int64)
    if y_hat.ndim != 1:
        raise ValueError(
            f'Probe score rule {score_rule} requires y_hat to be 1-D, '
            f'got shape {y_hat.shape}')
    _validate_probe_length(y_hat, 'y_hat', expected_n, score_rule)
    if np.any((y_hat < 0) | (y_hat >= expected_classes)):
        raise ValueError(
            f'Probe score rule {score_rule} requires y_hat in '
            f'[0, {expected_classes})')
    return y_hat


def _probe_target_probs(cache, expected_n, expected_classes, score_rule):
    probs = np.asarray(cache['target_probs'], dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(
            f'Probe score rule {score_rule} requires target_probs to be 2-D, '
            f'got shape {probs.shape}')
    _validate_probe_length(probs, 'target_probs', expected_n, score_rule)
    if probs.shape[1] != expected_classes:
        raise ValueError(
            f'Probe score rule {score_rule} requires target_probs class '
            f'dimension {probs.shape[1]} to match {expected_classes}')
    return probs


def _ref_loss_delta_scalar(cache, role, score_rule, mode, expected_n=None):
    delta = _probe_ref_loss_delta(cache, role, score_rule, expected_n)
    if mode == 'pos_mean':
        return np.mean(np.clip(delta, 0.0, None), axis=1)
    if mode == 'signed_mean':
        return np.mean(delta, axis=1)
    if mode == 'abs_mean':
        return np.mean(np.abs(delta), axis=1)
    if mode == 'pred':
        y_hat = _probe_y_hat(
            cache, delta.shape[0], delta.shape[1], score_rule)
        return np.clip(delta[_row_indices(delta), y_hat], 0.0, None)
    if mode == 'target_weighted':
        probs = _probe_target_probs(
            cache, delta.shape[0], delta.shape[1], score_rule)
        return np.sum(probs * np.clip(delta, 0.0, None), axis=1)
    raise ValueError(f'Unknown ref loss delta scalar mode: {mode}')


def _efficiency(cache, role, score_rule, penalty_mode, expected_n=None):
    penalty = _ref_loss_delta_scalar(
        cache, role, score_rule, penalty_mode, expected_n=expected_n)
    target_delta = _probe_target_objective_delta(
        cache, role, score_rule, expected_n=penalty.shape[0])
    eps = _probe_scalar(cache, 'eps', NUMERIC_EPS, score_rule)
    return -target_delta / (eps + penalty)


def _log_reject_efficiency(cache, score_rule, penalty_mode):
    penalty = _ref_loss_delta_scalar(cache, 'reject', score_rule, penalty_mode)
    target_delta = _probe_target_objective_delta(
        cache, 'reject', score_rule, expected_n=penalty.shape[0])
    return (
        np.log1p(np.clip(-target_delta, 0.0, None))
        - np.log1p(np.clip(penalty, 0.0, None))
    )


def _efficiency_penalty_mode(score_rule):
    if '_abs_ref_efficiency' in score_rule:
        return 'abs_mean'
    if '_pred_ref_efficiency' in score_rule:
        return 'pred'
    if '_target_weighted_ref_efficiency' in score_rule:
        return 'target_weighted'
    return 'pos_mean'


def _ref_scalar_mode(score_rule):
    if '_signed_ref_loss_delta' in score_rule:
        return 'signed_mean'
    if '_abs_ref_loss_delta' in score_rule:
        return 'abs_mean'
    if '_pred_ref_loss_delta' in score_rule:
        return 'pred'
    if '_target_weighted_ref_loss_delta' in score_rule:
        return 'target_weighted'
    return 'pos_mean'


def probe_ood_score_from_cache(cache, score_rule):
    cache = select_response_step(cache)
    if score_rule not in PROBE_SCORE_RULES:
        raise ValueError(f'Unknown probe score rule: {score_rule}')
    _raise_missing_probe_fields(cache, score_rule)

    if score_rule in {
            'accept_efficiency',
            'reject_efficiency',
            'log_reject_efficiency',
            'ar_efficiency_contrast',
            'accept_abs_ref_efficiency',
            'reject_abs_ref_efficiency',
            'log_reject_abs_ref_efficiency',
            'ar_abs_ref_efficiency_contrast',
            'accept_pred_ref_efficiency',
            'reject_pred_ref_efficiency',
            'log_reject_pred_ref_efficiency',
            'ar_pred_ref_efficiency_contrast',
            'accept_target_weighted_ref_efficiency',
            'reject_target_weighted_ref_efficiency',
            'log_reject_target_weighted_ref_efficiency',
            'ar_target_weighted_ref_efficiency_contrast',
    }:
        mode = _efficiency_penalty_mode(score_rule)
        if score_rule.startswith('accept_'):
            return _efficiency(cache, 'accept', score_rule, mode)
        if score_rule.startswith('reject_'):
            return _efficiency(cache, 'reject', score_rule, mode)
        if score_rule.startswith('log_reject_'):
            return _log_reject_efficiency(cache, score_rule, mode)
        accept_efficiency = _efficiency(cache, 'accept', score_rule, mode)
        reject_efficiency = _efficiency(
            cache, 'reject', score_rule,
            mode, expected_n=accept_efficiency.shape[0])
        return reject_efficiency - accept_efficiency

    if score_rule == 'accept_target_objective_delta':
        return _probe_target_objective_delta(cache, 'accept', score_rule)
    if score_rule == 'reject_target_objective_delta':
        return _probe_target_objective_delta(cache, 'reject', score_rule)
    if score_rule == 'target_objective_delta_contrast':
        accept_target_delta = _probe_target_objective_delta(
            cache, 'accept', score_rule)
        reject_target_delta = _probe_target_objective_delta(
            cache, 'reject', score_rule,
            expected_n=accept_target_delta.shape[0])
        return accept_target_delta - reject_target_delta

    if score_rule == 'reject_pos_ref_loss_delta_ood':
        return -_ref_loss_delta_scalar(cache, 'reject', score_rule, 'pos_mean')

    if score_rule == 'ref_loss_delta_contrast':
        mode = 'pos_mean'
    else:
        mode = _ref_scalar_mode(score_rule)
    if score_rule.startswith('accept_'):
        return _ref_loss_delta_scalar(cache, 'accept', score_rule, mode)
    if score_rule.startswith('reject_'):
        return _ref_loss_delta_scalar(cache, 'reject', score_rule, mode)
    if score_rule.endswith('_contrast'):
        accept_penalty = _ref_loss_delta_scalar(cache, 'accept', score_rule, mode)
        reject_penalty = _ref_loss_delta_scalar(
            cache, 'reject', score_rule, mode,
            expected_n=accept_penalty.shape[0])
        return accept_penalty - reject_penalty

    raise ValueError(f'Unknown probe score rule: {score_rule}')


def _row_indices(delta):
    return np.arange(delta.shape[0])


def _rest_mean(values, y_hat):
    mask = np.ones_like(values, dtype=bool)
    mask[_row_indices(values), y_hat] = False
    denom = max(values.shape[1] - 1, 1)
    return np.sum(np.where(mask, values, 0.0), axis=1) / denom


def fit_vector_score_reference(cache):
    cache = select_response_step(cache)
    delta = np.asarray(cache['delta'], dtype=np.float64)
    finite = np.all(np.isfinite(delta), axis=1)
    if not np.any(finite):
        raise ValueError('Cannot fit vector scores: no finite clean ID deltas')
    finite_delta = delta[finite]
    mean = np.mean(finite_delta, axis=0)
    std = np.std(finite_delta, axis=0)
    scale = np.where(std > NUMERIC_EPS, std, 1.0)
    return {
        'method': VECTOR_FIT_METHOD,
        'score_direction': SCORE_DIRECTION,
        'delta_definition': DELTA_DEFINITION,
        'eps': NUMERIC_EPS,
        'n': int(finite_delta.shape[0]),
        'num_classes': int(finite_delta.shape[1]),
        'nonfinite_count': int(delta.shape[0] - finite_delta.shape[0]),
        'clean_delta_mean': mean.tolist(),
        'clean_delta_std': std.tolist(),
        'clean_delta_scale': scale.tolist(),
    }


def _validate_vector_fit(fit, num_classes):
    if fit is None:
        raise ValueError('Vector score requires clean-ID vector fit')
    if fit.get('method') != VECTOR_FIT_METHOD:
        raise ValueError(f'Unknown vector fit method: {fit.get("method")}')
    if int(fit.get('num_classes', -1)) != int(num_classes):
        raise ValueError(
            f'Vector fit num_classes {fit.get("num_classes")} != {num_classes}')


def vector_ood_score_from_cache(cache, score_rule, vector_fit=None):
    cache = select_response_step(cache)
    delta = np.asarray(cache['delta'], dtype=np.float64)
    probs = np.asarray(cache['target_probs'], dtype=np.float64)
    y_hat = np.asarray(cache['y_hat'], dtype=np.int64)
    row = _row_indices(delta)

    if score_rule == 'rest_positive_loss_increase_mean':
        return _rest_mean(np.clip(delta, 0.0, None), y_hat)
    if score_rule == 'pred_vs_rest_loss_increase_gap':
        pred_delta = delta[row, y_hat]
        rest_delta = _rest_mean(delta, y_hat)
        return rest_delta - pred_delta
    if score_rule == 'target_centered_loss_increase':
        centered_probs = probs - (1.0 / delta.shape[1])
        return np.sum(centered_probs * delta, axis=1)
    if score_rule == 'delta_l2_norm':
        return np.sqrt(np.mean(delta * delta, axis=1))
    if score_rule == 'clean_delta_z_l2':
        _validate_vector_fit(vector_fit, delta.shape[1])
        mean = np.asarray(vector_fit['clean_delta_mean'], dtype=np.float64)
        scale = np.asarray(vector_fit['clean_delta_scale'], dtype=np.float64)
        z = (delta - mean) / scale
        return np.sqrt(np.mean(z * z, axis=1))
    if score_rule == 'clean_delta_cosine_distance':
        _validate_vector_fit(vector_fit, delta.shape[1])
        mean = np.asarray(vector_fit['clean_delta_mean'], dtype=np.float64)
        dot = np.sum(delta * mean, axis=1)
        denom = (
            np.linalg.norm(delta, axis=1) * np.linalg.norm(mean)
            + NUMERIC_EPS
        )
        cosine = np.clip(dot / denom, -1.0, 1.0)
        return 1.0 - cosine
    raise ValueError(f'Unknown vector score rule: {score_rule}')


def _cache_array(cache, key):
    return np.asarray(cache[key], dtype=np.float64)


def perturbation_ood_score_from_cache(cache, score_rule):
    if score_rule not in PERTURBATION_SCORE_RULES:
        raise ValueError(f'Unknown perturbation score rule: {score_rule}')

    if score_rule == 'logit_l2':
        return _cache_array(cache, 'perturbation_logit_l2')
    if score_rule == 'prob_l1':
        return _cache_array(cache, 'perturbation_prob_l1')
    if score_rule == 'conf_drop':
        return -_cache_array(cache, 'perturbation_conf_delta')
    if score_rule == 'entropy_increase':
        return _cache_array(cache, 'perturbation_entropy_delta')
    raise ValueError(f'Unknown perturbation score rule: {score_rule}')
