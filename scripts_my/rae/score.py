"""RAE scoring kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch

from .config import (
    DEFAULT_REJECTION_POWER,
    DEFAULT_REJECTION_RULE,
    DEFAULT_VALIDATION_RULE,
    DEFAULT_VALIDATION_TEMPERATURE,
    NUMERIC_EPS,
)
from .gradients import (
    dense_candidate_directions_and_norms,
    forward_logits_features,
)
from .gradient_space import select_gradient_parameters


@dataclass
class TargetScoreBatch:
    pred: np.ndarray
    label: np.ndarray
    eid: np.ndarray
    best_class: np.ndarray
    q_best: np.ndarray
    v_best: np.ndarray
    q_max: np.ndarray
    v_pred: np.ndarray
    eid_pred: np.ndarray
    candidate_classes: np.ndarray
    q_c: np.ndarray
    v_c: np.ndarray
    e_c: np.ndarray
    rank_only_scores: np.ndarray
    same_positive_rates: np.ndarray
    accept_eid: np.ndarray | None = None
    reject_id_evidence: np.ndarray | None = None
    reject_ood_evidence: np.ndarray | None = None
    reject_k_mean: np.ndarray | None = None
    candidate_raw_grad_norm: np.ndarray | None = None
    candidate_effective_dim: np.ndarray | None = None
    candidate_direction_cos_to_pred: np.ndarray | None = None
    candidate_ref_proto_cos: np.ndarray | None = None
    v_c_label_shuffle: np.ndarray | None = None
    e_c_label_shuffle: np.ndarray | None = None
    eid_label_shuffle: np.ndarray | None = None

def candidate_classes_from_probs(probs: torch.Tensor,
                                 mode: str) -> torch.Tensor:
    if mode == 'all':
        classes = torch.arange(probs.shape[1], device=probs.device)
        return classes[None, :].expand(probs.shape[0], -1)
    if mode == 'pred':
        return probs.argmax(dim=1, keepdim=True)
    raise ValueError(f'Unknown candidate mode: {mode}')


def prepare_ref_grad_bank_for_scoring(bank: Dict,
                                      device: torch.device,
                                      *,
                                      label_shuffle_seed: int | None = None,
                                      reference_bank_chunk_size: int = 0,
                                      needs_ref_prototypes: bool = False) -> Dict:
    prepared = dict(bank)
    prepared['_labels_tensor'] = torch.as_tensor(
        bank['labels'], device=device, dtype=torch.long)
    if label_shuffle_seed is not None:
        shuffled = np.asarray(bank['labels'], dtype=np.int64).copy()
        rng = np.random.RandomState(int(label_shuffle_seed))
        rng.shuffle(shuffled)
        prepared['_shuffled_labels_tensor'] = torch.as_tensor(
            shuffled, device=device, dtype=torch.long)
    if bank['bank_type'] == 'classifier_compact':
        prepared['_features_tensor'] = torch.as_tensor(
            bank['features'], device=device)
        prepared['_probs_tensor'] = torch.as_tensor(bank['probs'], device=device)
    else:
        if needs_ref_prototypes:
            prepared['_ref_class_prototypes_tensor'] = (
                dense_reference_class_prototypes(bank, device)
            )
        if int(reference_bank_chunk_size) <= 0:
            prepared['_directions_tensor'] = torch.as_tensor(
                bank['directions'], device=device)
    return prepared


def dense_reference_class_prototypes(bank: Dict,
                                     device: torch.device,
                                     eps: float = NUMERIC_EPS) -> torch.Tensor:
    labels = np.asarray(bank['labels'], dtype=np.int64)
    directions = np.asarray(bank['directions'], dtype=np.float32)
    class_count = int(labels.max()) + 1
    sums = np.zeros((class_count, directions.shape[1]), dtype=np.float32)
    counts = np.bincount(labels, minlength=class_count).astype(np.float32)
    np.add.at(sums, labels, directions)
    nonempty = counts > 0
    sums[nonempty] /= counts[nonempty, None]
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    sums[nonempty] /= np.maximum(norms[nonempty], eps)
    sums[~nonempty] = 0.0
    return torch.as_tensor(sums, device=device)


def classifier_pairwise_k(target_features: torch.Tensor,
                          target_probs: torch.Tensor,
                          candidate_classes: torch.Tensor,
                          ref_features: torch.Tensor,
                          ref_probs: torch.Tensor,
                          ref_labels: torch.Tensor,
                          *,
                          include_bias: bool,
                          eps: float = NUMERIC_EPS) -> torch.Tensor:
    ref_res = ref_probs.clone()
    ref_res[torch.arange(ref_probs.shape[0], device=ref_probs.device),
            ref_labels] -= 1.0
    target_res = target_probs[:, None, :].expand(
        -1, candidate_classes.shape[1], -1).clone()
    target_res.scatter_add_(
        2,
        candidate_classes[:, :, None],
        -torch.ones(
            (*candidate_classes.shape, 1),
            dtype=target_res.dtype,
            device=target_res.device,
        ),
    )
    return classifier_pairwise_k_from_residuals(
        target_features,
        target_res,
        ref_features,
        ref_res,
        include_bias=include_bias,
        eps=eps,
    )


def classifier_pairwise_k_from_residuals(target_features: torch.Tensor,
                                         target_res: torch.Tensor,
                                         ref_features: torch.Tensor,
                                         ref_res: torch.Tensor,
                                         *,
                                         include_bias: bool,
                                         eps: float = NUMERIC_EPS) -> torch.Tensor:
    if target_res.ndim == 2:
        target_res = target_res[:, None, :]
    ref_res_norm = torch.linalg.norm(ref_res, dim=1).clamp_min(eps)
    ref_feat_norm_sq = (ref_features * ref_features).sum(dim=1)
    bias = 1.0 if include_bias else 0.0
    ref_total = ref_res_norm * torch.sqrt(ref_feat_norm_sq + bias).clamp_min(eps)

    residual_dot = torch.einsum('bkc,rc->bkr', target_res, ref_res)
    feature_dot = target_features @ ref_features.t()
    target_res_norm = torch.linalg.norm(target_res, dim=2).clamp_min(eps)
    target_feat_total = torch.sqrt(
        (target_features * target_features).sum(dim=1) + bias
    ).clamp_min(eps)
    denom = (
        target_res_norm[:, :, None] * target_feat_total[:, None, None] *
        ref_total[None, None, :]
    )
    return residual_dot * (feature_dot[:, None, :] + bias) / denom


def classifier_candidate_geometry(target_features: torch.Tensor,
                                  target_probs: torch.Tensor,
                                  candidate_classes: torch.Tensor,
                                  *,
                                  include_bias: bool,
                                  eps: float = NUMERIC_EPS):
    target_res = target_probs[:, None, :].expand(
        -1, candidate_classes.shape[1], -1).clone()
    target_res.scatter_add_(
        2,
        candidate_classes[:, :, None],
        -torch.ones(
            (*candidate_classes.shape, 1),
            dtype=target_res.dtype,
            device=target_res.device,
        ),
    )
    residual_norm = torch.linalg.norm(target_res, dim=2).clamp_min(eps)
    bias = 1.0 if include_bias else 0.0
    feature_norm_sq = (target_features * target_features).sum(dim=1)
    feature_factor = (feature_norm_sq + bias).clamp_min(eps)
    raw_norm = residual_norm * torch.sqrt(feature_factor)[:, None]

    residual_fourth = target_res.pow(4).sum(dim=2).clamp_min(eps)
    feature_fourth = (
        target_features.pow(4).sum(dim=1) + bias
    ).clamp_min(eps)
    effective_dim = (
        residual_norm.pow(4) * feature_factor[:, None].pow(2)
    ) / (residual_fourth * feature_fourth[:, None]).clamp_min(eps)

    pred_classes = target_probs.argmax(dim=1)
    pred_res = target_probs.clone()
    pred_res[torch.arange(target_probs.shape[0], device=target_probs.device),
             pred_classes] -= 1.0
    pred_res_norm = torch.linalg.norm(pred_res, dim=1).clamp_min(eps)
    residual_dot = torch.einsum('bkc,bc->bk', target_res, pred_res)
    cos_to_pred = residual_dot / (residual_norm * pred_res_norm[:, None])
    return raw_norm, effective_dim, cos_to_pred


def uniform_rejection_k(target_features: torch.Tensor,
                        target_probs: torch.Tensor,
                        ref_features: torch.Tensor,
                        ref_probs: torch.Tensor,
                        ref_labels: torch.Tensor,
                        *,
                        include_bias: bool,
                        eps: float = NUMERIC_EPS) -> torch.Tensor:
    uniform = torch.full_like(target_probs, 1.0 / target_probs.shape[1])
    target_res = target_probs - uniform
    ref_res = ref_probs.clone()
    ref_res[torch.arange(ref_probs.shape[0], device=ref_probs.device),
            ref_labels] -= 1.0
    return classifier_pairwise_k_from_residuals(
        target_features,
        target_res,
        ref_features,
        ref_res,
        include_bias=include_bias,
        eps=eps,
    ).squeeze(1)


def class_rejection_evidence_from_k(k_values: torch.Tensor,
                                    ref_labels: torch.Tensor,
                                    candidate_classes: torch.Tensor):
    ref_labels = ref_labels.to(device=k_values.device)
    candidate_classes = candidate_classes.to(device=k_values.device)
    same_mask = ref_labels[None, None, :].eq(candidate_classes[:, :, None])
    same_count = same_mask.sum(dim=2).to(torch.float64).clamp_min(1)
    k_by_class = k_values[:, None, :].expand(-1, candidate_classes.shape[1], -1)
    same_float = same_mask.to(torch.float64)
    k_mean = (k_by_class.to(torch.float64) * same_float).sum(dim=2)
    k_mean = k_mean / same_count
    id_evidence = (-k_mean).clamp_min(0.0).clamp_max(1.0)
    ood_evidence = (1.0 - id_evidence).clamp_min(0.0).clamp_max(1.0)
    return id_evidence, ood_evidence, k_mean


def batch_validation_scores_from_k(k_values: torch.Tensor,
                                   ref_labels: torch.Tensor,
                                   candidate_classes: torch.Tensor,
                                   *,
                                   validation_rule: str = DEFAULT_VALIDATION_RULE,
                                   temperature: float = DEFAULT_VALIDATION_TEMPERATURE):
    ref_labels = ref_labels.to(device=k_values.device)
    candidate_classes = candidate_classes.to(device=k_values.device)
    same_mask = ref_labels[None, None, :].eq(candidate_classes[:, :, None])
    other_mask = ~same_mask
    same_count = same_mask.sum(dim=2)
    other_count = other_mask.sum(dim=2)
    count_dtype = torch.float64
    denom = (same_count * other_count).to(count_dtype)
    valid = denom > 0

    sorted_other = k_values.masked_fill(~other_mask, float('inf')).sort(dim=2).values
    other_less_than_ref = torch.searchsorted(
        sorted_other.contiguous(), k_values.contiguous(), right=False)
    other_less_than_ref = other_less_than_ref.to(count_dtype)
    same_float = same_mask.to(count_dtype)
    rank_count = (other_less_than_ref * same_float).sum(dim=2)
    positive_rank_count = (
        other_less_than_ref * same_float * (k_values > 0).to(count_dtype)
    ).sum(dim=2)

    safe_denom = denom.clamp_min(1)
    zeros = torch.zeros_like(denom)
    pairwise_rank = torch.where(valid, positive_rank_count / safe_denom, zeros)
    rank_only = torch.where(valid, rank_count / safe_denom, zeros)

    same_positive_count = ((k_values > 0) & same_mask).sum(dim=2).to(count_dtype)
    safe_same_count = same_count.to(count_dtype).clamp_min(1)
    same_positive = torch.where(
        same_count > 0,
        same_positive_count / safe_same_count,
        torch.zeros_like(safe_same_count),
    )
    if validation_rule == 'pairwise_rank':
        validation = pairwise_rank
    else:
        k_float = k_values.to(count_dtype)
        if validation_rule == 'pairwise_margin':
            sorted_other_finite = torch.where(
                torch.isfinite(sorted_other),
                sorted_other,
                torch.zeros_like(sorted_other),
            ).to(count_dtype)
            prefix = torch.cat(
                [torch.zeros_like(sorted_other_finite[:, :, :1]),
                 sorted_other_finite.cumsum(dim=2)],
                dim=2,
            )
            sum_less = torch.gather(
                prefix, 2, other_less_than_ref.long())
            margin_sum = (
                (other_less_than_ref * k_float - sum_less).clamp_min(0.0) *
                same_float * (k_values > 0).to(count_dtype)
            ).sum(dim=2)
            validation = (margin_sum / (2.0 * safe_denom)).clamp_max(1.0)
        elif validation_rule in {'same_mean', 'mean_margin', 'soft_margin'}:
            same_mean = torch.where(
                same_count > 0,
                (k_float * same_float).sum(dim=2) /
                same_count.to(count_dtype).clamp_min(1),
                zeros,
            )
            other_float = other_mask.to(count_dtype)
            other_mean = torch.where(
                other_count > 0,
                (k_float * other_float).sum(dim=2) /
                other_count.to(count_dtype).clamp_min(1),
                zeros,
            )
            if validation_rule == 'same_mean':
                validation = same_mean.clamp_min(0.0).clamp_max(1.0)
            elif validation_rule == 'mean_margin':
                validation = (
                    (same_mean - other_mean).clamp_min(0.0) / 2.0
                ).clamp_max(1.0)
            else:
                if temperature <= 0:
                    raise ValueError('--validation-temperature must be positive')
                temp = float(temperature)
                validation = (
                    torch.sigmoid(same_mean / temp) *
                    torch.sigmoid((same_mean - other_mean) / temp)
                )
        else:
            raise ValueError(f'Unknown RAE validation rule: {validation_rule}')
        validation = torch.where(valid, validation, zeros)
    return validation, rank_only, same_positive


def dense_validation_scores_from_ref_chunks(
        directions: torch.Tensor,
        bank: Dict,
        ref_labels: torch.Tensor,
        candidate_classes: torch.Tensor,
        *,
        validation_rule: str = DEFAULT_VALIDATION_RULE,
        temperature: float = DEFAULT_VALIDATION_TEMPERATURE,
        reference_bank_chunk_size: int,
        shuffled_ref_labels: torch.Tensor | None = None):
    reference_bank_chunk_size = int(reference_bank_chunk_size)
    if reference_bank_chunk_size <= 0:
        raise ValueError('reference_bank_chunk_size must be positive')
    ref_dirs_source = np.asarray(bank['directions'])
    reference_count = int(ref_dirs_source.shape[0])
    batch_size, candidate_count, _ = directions.shape
    validations = []
    rank_scores = []
    same_positive_rates = []
    shuffled_validations = [] if shuffled_ref_labels is not None else None
    device = directions.device
    dtype = directions.dtype
    for candidate_idx in range(candidate_count):
        direction = directions[:, candidate_idx, :]
        k_chunks = []
        for start in range(0, reference_count, reference_bank_chunk_size):
            end = min(start + reference_bank_chunk_size, reference_count)
            ref_chunk = torch.as_tensor(
                ref_dirs_source[start:end], device=device, dtype=dtype)
            k_chunks.append(direction @ ref_chunk.t())
        k_values = torch.cat(k_chunks, dim=1).view(batch_size, 1, reference_count)
        classes = candidate_classes[:, candidate_idx:candidate_idx + 1]
        validation, rank_only, same_positive = batch_validation_scores_from_k(
            k_values,
            ref_labels,
            classes,
            validation_rule=validation_rule,
            temperature=temperature,
        )
        validations.append(validation.squeeze(1))
        rank_scores.append(rank_only.squeeze(1))
        same_positive_rates.append(same_positive.squeeze(1))
        if shuffled_validations is not None:
            shuffled_validation, _, _ = batch_validation_scores_from_k(
                k_values,
                shuffled_ref_labels,
                classes,
                validation_rule=validation_rule,
                temperature=temperature,
            )
            shuffled_validations.append(shuffled_validation.squeeze(1))
    stacked = (
        torch.stack(validations, dim=1),
        torch.stack(rank_scores, dim=1),
        torch.stack(same_positive_rates, dim=1),
    )
    if shuffled_validations is None:
        return (*stacked, None)
    return (*stacked, torch.stack(shuffled_validations, dim=1))


def validation_scores_from_k(k_values: np.ndarray,
                             ref_labels: np.ndarray,
                             candidate_classes: np.ndarray,
                             *,
                             validation_rule: str = DEFAULT_VALIDATION_RULE,
                             temperature: float = DEFAULT_VALIDATION_TEMPERATURE):
    if validation_rule != 'pairwise_rank':
        k_tensor = torch.as_tensor(k_values)
        classes_tensor = torch.as_tensor(candidate_classes, dtype=torch.long)
        squeeze_batch = False
        if k_tensor.ndim == 2:
            k_tensor = k_tensor.unsqueeze(0)
            classes_tensor = classes_tensor.unsqueeze(0)
            squeeze_batch = True
        validation, rank_only, same_positive = batch_validation_scores_from_k(
            k_tensor,
            torch.as_tensor(ref_labels, dtype=torch.long),
            classes_tensor,
            validation_rule=validation_rule,
            temperature=temperature,
        )
        if squeeze_batch:
            validation = validation.squeeze(0)
            rank_only = rank_only.squeeze(0)
            same_positive = same_positive.squeeze(0)
        return (
            validation.detach().cpu().numpy().astype(np.float64),
            rank_only.detach().cpu().numpy().astype(np.float64),
            same_positive.detach().cpu().numpy().astype(np.float64),
        )
    values = []
    rank_only = []
    same_positive = []
    for idx, class_id in enumerate(candidate_classes.astype(np.int64).tolist()):
        row = k_values[idx]
        same = row[ref_labels == class_id]
        other = row[ref_labels != class_id]
        if same.size == 0 or other.size == 0:
            values.append(0.0)
            rank_only.append(0.0)
            same_positive.append(0.0)
            continue
        comparisons = same[:, None] > other[None, :]
        positive = same[:, None] > 0
        values.append(float(np.mean(comparisons & positive)))
        rank_only.append(float(np.mean(comparisons)))
        same_positive.append(float(np.mean(same > 0)))
    return (
        np.asarray(values, dtype=np.float64),
        np.asarray(rank_only, dtype=np.float64),
        np.asarray(same_positive, dtype=np.float64),
    )


def ood_score_from_eid(eid, score_rule: str, eps: float = NUMERIC_EPS):
    eid = np.asarray(eid, dtype=np.float64)
    if score_rule == 'neglog_eid':
        return -np.log(eid + eps)
    if score_rule == 'neg_eid':
        return -eid
    raise ValueError(f'Unknown RAE score rule: {score_rule}')


def ood_score_from_split_scores(split_scores: Dict,
                                score_rule: str,
                                eps: float = NUMERIC_EPS):
    if score_rule in {'neglog_eid', 'neg_eid'}:
        return ood_score_from_eid(split_scores['eid'], score_rule, eps=eps)
    if score_rule == 'geom_effdim_mean':
        if 'candidate_effective_dim' not in split_scores:
            raise ValueError(
                'geom_effdim_mean requires candidate_effective_dim scores')
        return -np.asarray(split_scores['candidate_effective_dim']).mean(axis=1)
    if score_rule == 'geom_rawnorm_mean':
        if 'candidate_raw_grad_norm' not in split_scores:
            raise ValueError(
                'geom_rawnorm_mean requires candidate_raw_grad_norm scores')
        return np.asarray(split_scores['candidate_raw_grad_norm']).mean(axis=1)
    if score_rule == 'geom_cos_pred_mean':
        if 'candidate_direction_cos_to_pred' not in split_scores:
            raise ValueError(
                'geom_cos_pred_mean requires candidate_direction_cos_to_pred scores')
        return np.asarray(split_scores['candidate_direction_cos_to_pred']).mean(axis=1)
    if score_rule in {'geom_proto_cos_max', 'geom_proto_eid'}:
        if 'candidate_ref_proto_cos' not in split_scores:
            raise ValueError(
                f'{score_rule} requires candidate_ref_proto_cos scores')
        proto_cos = np.asarray(split_scores['candidate_ref_proto_cos'])
        if proto_cos.size == 0 or proto_cos.ndim != 2:
            raise ValueError(
                f'{score_rule} requires dense reference prototype scores')
        if score_rule == 'geom_proto_cos_max':
            return -proto_cos.max(axis=1)
        q_c = np.asarray(split_scores['q_c'])
        return -(q_c * np.maximum(proto_cos, 0.0)).max(axis=1)
    raise ValueError(f'Unknown RAE score rule: {score_rule}')


def score_target_batch(net: torch.nn.Module,
                       batch: Dict,
                       bank: Dict,
                       *,
                       gradient_space: str,
                       candidate_mode: str,
                       validation_rule: str,
                       validation_temperature: float,
                       rejection_rule: str = DEFAULT_REJECTION_RULE,
                       rejection_power: float = DEFAULT_REJECTION_POWER,
                       device: torch.device,
                       label_shuffle_seed: int | None = None,
                       selected_params: Sequence[torch.nn.Parameter] | None = None,
                       reference_bank_chunk_size: int = 0,
                       eps: float = NUMERIC_EPS) -> TargetScoreBatch:
    data = batch['data'].to(device)
    labels = batch['label'].detach().cpu().numpy().astype(np.int64)
    ref_labels = bank.get('_labels_tensor')
    if ref_labels is None:
        ref_labels = torch.as_tensor(bank['labels'], device=device).long()
    candidate_raw_grad_norm = None
    candidate_effective_dim = None
    candidate_direction_cos_to_pred = None
    candidate_ref_proto_cos = None
    if bank['bank_type'] == 'classifier_compact':
        with torch.no_grad():
            logits, features = forward_logits_features(net, data)
            probs = torch.softmax(logits, dim=1)
        candidates = candidate_classes_from_probs(probs, candidate_mode)
        ref_features = bank.get('_features_tensor')
        if ref_features is None:
            ref_features = torch.as_tensor(bank['features'], device=device)
        ref_probs = bank.get('_probs_tensor')
        if ref_probs is None:
            ref_probs = torch.as_tensor(bank['probs'], device=device)
        classifier_include_bias = bool(
            np.asarray(bank['classifier_has_bias']).item())
        (
            candidate_raw_grad_norm,
            candidate_effective_dim,
            candidate_direction_cos_to_pred,
        ) = classifier_candidate_geometry(
            features,
            probs,
            candidates,
            include_bias=classifier_include_bias,
            eps=eps,
        )
        k_batch = classifier_pairwise_k(
            features,
            probs,
            candidates,
            ref_features,
            ref_probs,
            ref_labels,
            include_bias=classifier_include_bias,
            eps=eps,
        )
    else:
        logits = net(data)
        probs = torch.softmax(logits, dim=1)
        candidates = candidate_classes_from_probs(probs, candidate_mode)
        if selected_params is None:
            selected_params = [
                param for _, param in select_gradient_parameters(net, gradient_space)
            ]
        directions, raw_norms = dense_candidate_directions_and_norms(
            net, data, candidates, selected_params, eps=eps)
        pred_for_geometry = probs.argmax(dim=1)
        pred_positions = candidates.eq(pred_for_geometry[:, None]).float()
        pred_dirs = (
            directions * pred_positions[:, :, None]
        ).sum(dim=1, keepdim=True)
        candidate_raw_grad_norm = raw_norms
        candidate_effective_dim = 1.0 / directions.pow(4).sum(
            dim=2).clamp_min(eps)
        candidate_direction_cos_to_pred = (directions * pred_dirs).sum(dim=2)
        ref_class_prototypes = bank.get('_ref_class_prototypes_tensor')
        if ref_class_prototypes is not None:
            if candidates.max() >= ref_class_prototypes.shape[0]:
                raise ValueError(
                    'candidate class exceeds reference prototype class count')
            candidate_ref_proto_cos = (
                directions * ref_class_prototypes[candidates]
            ).sum(dim=2)
        ref_dirs = bank.get('_directions_tensor')
        if ref_dirs is None and int(reference_bank_chunk_size) <= 0:
            ref_dirs = torch.as_tensor(bank['directions'], device=device)
        if ref_dirs is not None:
            k_batch = torch.einsum('bkd,rd->bkr', directions, ref_dirs)
        else:
            k_batch = None

    pred = probs.argmax(dim=1)
    q_max = probs.max(dim=1).values
    shuffled_ref_labels = None
    if label_shuffle_seed is not None:
        shuffled_ref_labels = bank.get('_shuffled_labels_tensor')
        if shuffled_ref_labels is None:
            shuffled_ref_labels_np = np.asarray(bank['labels'], dtype=np.int64).copy()
            rng = np.random.RandomState(int(label_shuffle_seed))
            rng.shuffle(shuffled_ref_labels_np)
            shuffled_ref_labels = torch.as_tensor(
                shuffled_ref_labels_np, device=device).long()
    if k_batch is None:
        (
            pairwise_validation,
            rank_only,
            same_positive,
            shuffled_pairwise_validation,
        ) = dense_validation_scores_from_ref_chunks(
            directions,
            bank,
            ref_labels,
            candidates,
            validation_rule=validation_rule,
            temperature=validation_temperature,
            reference_bank_chunk_size=reference_bank_chunk_size,
            shuffled_ref_labels=shuffled_ref_labels,
        )
    else:
        pairwise_validation, rank_only, same_positive = batch_validation_scores_from_k(
            k_batch,
            ref_labels,
            candidates,
            validation_rule=validation_rule,
            temperature=validation_temperature,
        )
        shuffled_pairwise_validation = None
    candidate_probs = probs.gather(1, candidates)
    validation = pairwise_validation
    accept_class_evidence = candidate_probs * validation
    accept_eid = accept_class_evidence.max(dim=1).values
    reject_id_evidence = None
    reject_ood_evidence = None
    reject_k_mean = None
    if rejection_rule == 'off':
        class_evidence = accept_class_evidence
        eid = accept_eid
    elif rejection_rule == 'uniform':
        if bank['bank_type'] != 'classifier_compact':
            raise ValueError(
                '--rejection-rule uniform requires --gradient-space classifier')
        if rejection_power < 0:
            raise ValueError('--rejection-power must be non-negative')
        reject_k = uniform_rejection_k(
            features,
            probs,
            ref_features,
            ref_probs,
            ref_labels,
            include_bias=classifier_include_bias,
            eps=eps,
        )
        reject_id_c, reject_ood_c, reject_k_mean_c = (
            class_rejection_evidence_from_k(reject_k, ref_labels, candidates))
        reject_scale = (reject_id_c + eps).pow(float(rejection_power))
        class_evidence = accept_class_evidence * reject_scale
        eid, best_idx = class_evidence.max(dim=1)
        reject_id_evidence = reject_id_c.gather(1, best_idx[:, None]).squeeze(1)
        reject_ood_evidence = reject_ood_c.gather(1, best_idx[:, None]).squeeze(1)
        reject_k_mean = reject_k_mean_c.gather(1, best_idx[:, None]).squeeze(1)
    else:
        raise ValueError(f'Unknown RAE rejection rule: {rejection_rule}')
    if rejection_rule == 'off':
        eid, best_idx = class_evidence.max(dim=1)
    best_class = candidates.gather(1, best_idx[:, None]).squeeze(1)
    q_best = candidate_probs.gather(1, best_idx[:, None]).squeeze(1)
    v_best = validation.gather(1, best_idx[:, None]).squeeze(1)

    pred_match = candidates.eq(pred[:, None])
    has_pred = pred_match.any(dim=1)
    pred_match_float = pred_match.to(validation.dtype)
    v_pred = (validation * pred_match_float).sum(dim=1)
    eid_pred = (class_evidence * pred_match_float).sum(dim=1)
    v_pred = torch.where(has_pred, v_pred, torch.zeros_like(v_pred))
    eid_pred = torch.where(has_pred, eid_pred, torch.zeros_like(eid_pred))

    shuffled_validation = None
    shuffled_evidence = None
    shuffled_eid = None
    if label_shuffle_seed is not None:
        if shuffled_pairwise_validation is None:
            shuffled_pairwise_validation, _, _ = batch_validation_scores_from_k(
                k_batch,
                shuffled_ref_labels,
                candidates,
                validation_rule=validation_rule,
                temperature=validation_temperature,
            )
        shuffled_validation = shuffled_pairwise_validation
        shuffled_evidence = candidate_probs * shuffled_validation
        if rejection_rule == 'uniform':
            shuffled_evidence = shuffled_evidence * reject_scale
        shuffled_eid = shuffled_evidence.max(dim=1).values
    return TargetScoreBatch(
        pred=pred.detach().cpu().numpy().astype(np.int64),
        label=labels,
        eid=eid.detach().cpu().numpy().astype(np.float64),
        best_class=best_class.detach().cpu().numpy().astype(np.int64),
        q_best=q_best.detach().cpu().numpy().astype(np.float64),
        v_best=v_best.detach().cpu().numpy().astype(np.float64),
        q_max=q_max.detach().cpu().numpy().astype(np.float64),
        v_pred=v_pred.detach().cpu().numpy().astype(np.float64),
        eid_pred=eid_pred.detach().cpu().numpy().astype(np.float64),
        candidate_classes=candidates.detach().cpu().numpy().astype(np.int64),
        q_c=candidate_probs.detach().cpu().numpy().astype(np.float64),
        v_c=validation.detach().cpu().numpy().astype(np.float64),
        e_c=class_evidence.detach().cpu().numpy().astype(np.float64),
        rank_only_scores=rank_only.detach().cpu().numpy().astype(np.float64),
        same_positive_rates=same_positive.detach().cpu().numpy().astype(np.float64),
        accept_eid=accept_eid.detach().cpu().numpy().astype(np.float64),
        reject_id_evidence=reject_id_evidence.detach().cpu().numpy().astype(
            np.float64) if reject_id_evidence is not None else None,
        reject_ood_evidence=reject_ood_evidence.detach().cpu().numpy().astype(
            np.float64) if reject_ood_evidence is not None else None,
        reject_k_mean=reject_k_mean.detach().cpu().numpy().astype(
            np.float64) if reject_k_mean is not None else None,
        candidate_raw_grad_norm=candidate_raw_grad_norm.detach().cpu().numpy().astype(
            np.float64) if candidate_raw_grad_norm is not None else None,
        candidate_effective_dim=candidate_effective_dim.detach().cpu().numpy().astype(
            np.float64) if candidate_effective_dim is not None else None,
        candidate_direction_cos_to_pred=(
            candidate_direction_cos_to_pred.detach().cpu().numpy().astype(
                np.float64)
            if candidate_direction_cos_to_pred is not None else None
        ),
        candidate_ref_proto_cos=(
            candidate_ref_proto_cos.detach().cpu().numpy().astype(np.float64)
            if candidate_ref_proto_cos is not None else None
        ),
        v_c_label_shuffle=shuffled_validation.detach().cpu().numpy().astype(
            np.float64) if shuffled_validation is not None else None,
        e_c_label_shuffle=shuffled_evidence.detach().cpu().numpy().astype(
            np.float64) if shuffled_evidence is not None else None,
        eid_label_shuffle=shuffled_eid.detach().cpu().numpy().astype(
            np.float64) if shuffled_eid is not None else None,
    )
