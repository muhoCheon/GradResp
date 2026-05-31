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
SCORE_RULE_CHOICES = ACTIVE_SCORE_RULES + ['all']
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

CACHE_SCHEMA_VERSION = 5
SCORE_DIRECTION = 'higher_is_ood'
DELTA_DEFINITION = 'adapted_minus_base'
PERTURBATION_DEFINITION = 'perturbed_minus_clean'
PERTURBATION_SCORE_DIRECTION = SCORE_DIRECTION
NUMERIC_EPS = 1e-12
VECTOR_FIT_METHOD = 'clean_id_delta_vector'


def selected_score_rules(score_rule):
    if score_rule == 'all':
        return list(ACTIVE_SCORE_RULES)
    if score_rule not in ACTIVE_SCORE_RULES:
        raise ValueError(f'Unknown score rule: {score_rule}')
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
    raise ValueError(f'Unknown score rule: {score_rule}')


def ood_score_from_cache(cache, score_rule, vector_fit=None):
    if score_rule in VECTOR_SCORE_RULES:
        return vector_ood_score_from_cache(cache, score_rule, vector_fit)

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


def _row_indices(delta):
    return np.arange(delta.shape[0])


def _rest_mean(values, y_hat):
    mask = np.ones_like(values, dtype=bool)
    mask[_row_indices(values), y_hat] = False
    denom = max(values.shape[1] - 1, 1)
    return np.sum(np.where(mask, values, 0.0), axis=1) / denom


def fit_vector_score_reference(cache):
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
