"""Diagnostic helpers and gate runners for standalone RAE.

The gates are intentionally split into two groups:

* online gates use a live model and small batches to check gradient mechanics;
* post-hoc gates use saved score artifacts and can be rerun later.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .artifacts import ensure_dir, write_csv, write_json
from .config import NUMERIC_EPS
from .gradient_space import select_gradient_parameters
from .gradients import (
    classifier_candidate_directions_dense,
    classifier_direction_dense,
    classifier_has_bias,
    forward_logits_features,
    normalized_grad_vector,
)
from .metrics import concat_score_tuples, metric_summary, score_tuple_from_ood
from .score import (
    candidate_classes_from_probs,
    classifier_pairwise_k,
    ood_score_from_eid,
    validation_scores_from_k,
)


SINGLE_RUN_DIAGNOSTIC_GATES = (
    'gate01',
    'gate02',
    'gate03',
    'gate04',
    'gate06',
    'gate07',
    'gate10',
)
EXPERIMENT_DIAGNOSTIC_GATES = ('gate08', 'gate09')
DEFERRED_GATES = ('gate05',)
KNOWN_DIAGNOSTIC_GATES = (
    SINGLE_RUN_DIAGNOSTIC_GATES + EXPERIMENT_DIAGNOSTIC_GATES +
    DEFERRED_GATES
)
DEFAULT_DIAGNOSTIC_STEP_SIZE = 1e-2


@dataclass
class DiagnosticResult:
    gate_id: str
    name: str
    status: str
    metrics: Mapping = field(default_factory=dict)
    thresholds: Mapping = field(default_factory=dict)
    artifacts: Mapping = field(default_factory=dict)
    message: str = ''

    def to_dict(self) -> Dict:
        return {
            'gate_id': self.gate_id,
            'name': self.name,
            'status': self.status,
            'metrics': _jsonable(self.metrics),
            'thresholds': _jsonable(self.thresholds),
            'artifacts': _jsonable(self.artifacts),
            'message': self.message,
        }


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def parse_diagnostic_gates(value: str | Iterable[str] | None,
                           *,
                           mode: str = 'all') -> List[str]:
    if mode == 'off':
        return []
    if value is None or str(value).strip().lower() in {'', 'all'}:
        gates = list(SINGLE_RUN_DIAGNOSTIC_GATES)
    else:
        items = value.split(',') if isinstance(value, str) else list(value)
        gates = []
        for item in items:
            key = str(item).strip().lower()
            if not key:
                continue
            if not key.isdigit():
                raise ValueError(
                    f'Unknown RAE diagnostic gate: {item}; use numbers such as '
                    '1,2,3,4,6,7,10 or all')
            gate = f'gate{int(key):02d}'
            if gate not in KNOWN_DIAGNOSTIC_GATES:
                raise ValueError(f'Unknown RAE diagnostic gate: {item}')
            if gate in DEFERRED_GATES:
                raise ValueError(
                    f'{gate} is deferred and cannot be selected until implemented')
            gates.append(gate)
    return sorted(dict.fromkeys(gates))


def _gate_file_name(result: DiagnosticResult) -> str:
    suffix = result.name.lower().replace(' ', '_').replace('/', '_')
    return f'{result.gate_id}_{suffix}.json'


def write_diagnostic_results(out_dir: Path,
                             results: Sequence[DiagnosticResult],
                             *,
                             context: Mapping | None = None,
                             required_gates: Sequence[str] = SINGLE_RUN_DIAGNOSTIC_GATES,
                             deferred_gates: Sequence[str] = DEFERRED_GATES) -> Dict:
    ensure_dir(out_dir)
    gates_dir = ensure_dir(out_dir / 'gates')
    rows = []
    result_dicts = []
    for result in results:
        result_path = gates_dir / _gate_file_name(result)
        payload = result.to_dict()
        write_json(result_path, payload)
        result_dicts.append(payload)
        rows.append({
            'gate_id': result.gate_id,
            'name': result.name,
            'status': result.status,
            'message': result.message,
            'artifact': str(result_path),
        })
    write_csv(
        out_dir / 'gates.csv',
        rows,
        fieldnames=['gate_id', 'name', 'status', 'message', 'artifact'],
    )
    status = _overall_status(result_dicts, required_gates)
    manifest = {
        'artifact': 'rae_diagnostics',
        'diagnostics_status': status,
        'required_gates': list(required_gates),
        'deferred_gates': list(deferred_gates),
        'results': result_dicts,
        'context': _jsonable(context or {}),
    }
    write_json(out_dir / 'diagnostics_manifest.json', manifest)
    return manifest


def _overall_status(results: Sequence[Mapping],
                    required_gates: Sequence[str]) -> str:
    by_gate = {}
    for result in results:
        by_gate.setdefault(str(result['gate_id']), []).append(str(result['status']))
    for gate in required_gates:
        if gate not in by_gate or any(status == 'skip' for status in by_gate[gate]):
            return 'incomplete'
    if any(status == 'fail' for statuses in by_gate.values() for status in statuses):
        return 'fail'
    if any(status == 'warn' for statuses in by_gate.values() for status in statuses):
        return 'warn'
    return 'pass'


def diagnostics_claim_status(diagnostics_manifest: Mapping | None,
                             *,
                             max_target_samples: int | None = None,
                             candidate_mode: str | None = None):
    reasons = []
    if max_target_samples:
        reasons.append('max_target_samples was set; subset runs are non-claim-bearing')
    if candidate_mode and candidate_mode not in {'all', 'pred'}:
        reasons.append(f'unknown candidate_mode={candidate_mode}')
    if diagnostics_manifest is None:
        reasons.append('diagnostics were not run')
    else:
        status = diagnostics_manifest.get('diagnostics_status')
        if status in {'fail', 'incomplete'}:
            reasons.append(f'diagnostics_status={status}')
    return len(reasons) == 0, reasons


def gate1_acceptance_delta(net: torch.nn.Module,
                           data: torch.Tensor,
                           class_id: int,
                           params,
                           *,
                           step_size: float = DEFAULT_DIAGNOSTIC_STEP_SIZE
                           ) -> Dict[str, float]:
    """Return CE before/after a small step along the acceptance direction."""
    net.zero_grad(set_to_none=True)
    label = torch.tensor([int(class_id)], dtype=torch.long, device=data.device)
    logits = net(data)
    before = F.cross_entropy(logits, label)
    direction = normalized_grad_vector(before, params)
    originals = [param.detach().clone() for param in params]
    offset = 0
    with torch.no_grad():
        for param in params:
            width = param.numel()
            param.add_(step_size * direction[offset:offset + width].view_as(param))
            offset += width
    try:
        after = F.cross_entropy(net(data), label)
    finally:
        with torch.no_grad():
            for param, original in zip(params, originals):
                param.copy_(original)
    return {
        'loss_before': float(before.detach().cpu().item()),
        'loss_after': float(after.detach().cpu().item()),
        'loss_delta': float((after - before).detach().cpu().item()),
        'passed': bool(after.detach().cpu().item() < before.detach().cpu().item()),
    }


def gate2_reference_sign_delta(net: torch.nn.Module,
                               data: torch.Tensor,
                               class_id: int,
                               ref_data: torch.Tensor,
                               ref_label: int,
                               params,
                               *,
                               step_size: float = DEFAULT_DIAGNOSTIC_STEP_SIZE
                               ) -> Dict[str, float]:
    """Check whether K sign matches small-step reference loss change sign."""
    target_label = torch.tensor(
        [int(class_id)], dtype=torch.long, device=data.device)
    ref_label_tensor = torch.tensor(
        [int(ref_label)], dtype=torch.long, device=ref_data.device)
    target_loss = F.cross_entropy(net(data), target_label)
    target_direction = normalized_grad_vector(target_loss, params)
    ref_loss = F.cross_entropy(net(ref_data), ref_label_tensor)
    ref_grads = torch.autograd.grad(
        ref_loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    ref_grad_flat = torch.cat([grad.reshape(-1) for grad in ref_grads])
    ref_direction = -F.normalize(ref_grad_flat, p=2, dim=0, eps=NUMERIC_EPS)
    k_value = float(torch.dot(target_direction, ref_direction).detach().cpu().item())
    first_order_delta = float(
        torch.dot(ref_grad_flat, target_direction).detach().cpu().item())
    originals = [param.detach().clone() for param in params]
    offset = 0
    with torch.no_grad():
        for param in params:
            width = param.numel()
            param.add_(step_size * target_direction[offset:offset + width].view_as(param))
            offset += width
    try:
        ref_after = F.cross_entropy(net(ref_data), ref_label_tensor)
    finally:
        with torch.no_grad():
            for param, original in zip(params, originals):
                param.copy_(original)
    delta = float((ref_after - ref_loss).detach().cpu().item())
    finite_matches = (
        (k_value > 0 and delta < 0) or
        (k_value < 0 and delta > 0) or
        abs(k_value) <= NUMERIC_EPS
    )
    first_order_matches = (
        (k_value > 0 and first_order_delta < 0) or
        (k_value < 0 and first_order_delta > 0) or
        abs(k_value) <= NUMERIC_EPS
    )
    return {
        'k': k_value,
        'reference_loss_delta': delta,
        'first_order_delta': first_order_delta,
        'finite_delta_matched': bool(finite_matches),
        'first_order_matched': bool(first_order_matches),
        'passed': bool(finite_matches or first_order_matches),
    }


def gate3_fc_factorization_error(k_fc: np.ndarray,
                                 feature_cosine: np.ndarray,
                                 residual_cosine: np.ndarray) -> Dict[str, float]:
    product = np.asarray(feature_cosine) * np.asarray(residual_cosine)
    diff = np.asarray(k_fc) - product
    return {
        'mae': float(np.mean(np.abs(diff))) if diff.size else 0.0,
        'max_abs': float(np.max(np.abs(diff))) if diff.size else 0.0,
        'corr': float(np.corrcoef(np.asarray(k_fc).ravel(), product.ravel())[0, 1])
        if diff.size > 1 else 1.0,
    }


def confidence_bins(q_max: np.ndarray,
                    values: np.ndarray,
                    labels: np.ndarray,
                    *,
                    bins: int = 10) -> Dict:
    q_max = np.asarray(q_max, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (q_max >= start) & (q_max <= end if end == 1.0 else q_max < end)
        if not np.any(mask):
            continue
        row = {
            'bin_start': float(start),
            'bin_end': float(end),
            'count': int(mask.sum()),
            'score_mean': float(values[mask].mean()),
            'score_median': float(np.median(values[mask])),
        }
        for label in sorted(set(labels[mask].tolist())):
            group = mask & (labels == label)
            row[f'label_{label}_count'] = int(group.sum())
            row[f'label_{label}_mean'] = float(values[group].mean())
        rows.append(row)
    return {'bins': rows}


def signed_support_summary(validation: np.ndarray,
                           rank_only: np.ndarray,
                           same_positive: np.ndarray) -> Dict[str, float]:
    validation = np.asarray(validation, dtype=np.float64)
    rank_only = np.asarray(rank_only, dtype=np.float64)
    same_positive = np.asarray(same_positive, dtype=np.float64)
    gap = rank_only - validation
    return {
        'validation_mean': float(validation.mean()) if validation.size else 0.0,
        'rank_only_mean': float(rank_only.mean()) if rank_only.size else 0.0,
        'same_positive_mean': float(same_positive.mean())
        if same_positive.size else 0.0,
        'signed_support_gap_mean': float(gap.mean()) if gap.size else 0.0,
    }


def reference_label_shuffle_summary(k_values: np.ndarray,
                                    ref_labels: np.ndarray,
                                    candidate_classes: np.ndarray,
                                    *,
                                    seed: int = 0) -> Dict[str, float]:
    """Gate 7: validation should drop when reference labels are shuffled."""
    rng = np.random.RandomState(seed)
    ref_labels = np.asarray(ref_labels, dtype=np.int64)
    shuffled = ref_labels.copy()
    rng.shuffle(shuffled)
    original_v, _, _ = validation_scores_from_k(
        np.asarray(k_values), ref_labels, np.asarray(candidate_classes))
    shuffled_v, _, _ = validation_scores_from_k(
        np.asarray(k_values), shuffled, np.asarray(candidate_classes))
    return {
        'original_validation_mean': float(original_v.mean())
        if original_v.size else 0.0,
        'shuffled_validation_mean': float(shuffled_v.mean())
        if shuffled_v.size else 0.0,
        'validation_drop': float(original_v.mean() - shuffled_v.mean())
        if original_v.size and shuffled_v.size else 0.0,
        'shuffle_seed': int(seed),
    }


def reference_size_stability_summary(rows) -> Dict[str, Dict[str, float]]:
    """Gate 8: summarize metric/evidence stability across sizes/seeds."""
    rows = list(rows)
    grouped = {}
    for row in rows:
        size = str(row['reference_per_class'])
        grouped.setdefault(size, []).append(row)
    summary = {}
    for size, items in grouped.items():
        numeric_keys = sorted({
            key
            for item in items
            for key, value in item.items()
            if key not in {'reference_per_class', 'seed'}
            and isinstance(value, (int, float, np.floating))
        })
        out = {'count': float(len(items))}
        for key in numeric_keys:
            values = np.asarray([float(item[key]) for item in items if key in item])
            if values.size:
                out[f'{key}_mean'] = float(values.mean())
                out[f'{key}_std'] = float(values.std(ddof=0))
        summary[size] = out
    return summary


def score_ablation_summary(score_npz: Path) -> Dict[str, float]:
    arrays = np.load(score_npz, allow_pickle=True)
    eid = arrays['eid'].astype(np.float64)
    ood_score = arrays['ood_score'].astype(np.float64)
    return {
        'n': int(eid.size),
        'eid_mean': float(eid.mean()) if eid.size else 0.0,
        'eid_median': float(np.median(eid)) if eid.size else 0.0,
        'ood_score_mean': float(ood_score.mean()) if ood_score.size else 0.0,
        'ood_score_median': float(np.median(ood_score)) if ood_score.size else 0.0,
    }


def summarize_score_dir(score_dir: Path, out_path: Path) -> Dict:
    score_files = sorted((score_dir / 'scores').glob('*.npz'))
    summary = {
        'score_dir': str(score_dir),
        'score_files': [str(path) for path in score_files],
        'ablation': {
            path.stem: score_ablation_summary(path)
            for path in score_files
        },
    }
    write_json(out_path, summary)
    return summary


def run_online_diagnostics(net: torch.nn.Module,
                           target_loader,
                           reference_loader,
                           *,
                           gradient_space: str,
                           device: torch.device,
                           gates: Sequence[str],
                           sample_count: int = 16,
                           step_size: float = DEFAULT_DIAGNOSTIC_STEP_SIZE
                           ) -> List[DiagnosticResult]:
    results: List[DiagnosticResult] = []
    needed = set(gates)
    if not needed.intersection({'gate01', 'gate02', 'gate03'}):
        return results

    try:
        target_batch = next(iter(target_loader))
        reference_batch = next(iter(reference_loader))
    except StopIteration:
        return [
            DiagnosticResult(
                gate_id=gate,
                name='online_gradient_mechanics',
                status='skip',
                message='target or reference loader was empty',
            )
            for gate in sorted(needed.intersection({'gate01', 'gate02', 'gate03'}))
        ]

    target_data = target_batch['data'].to(device)
    reference_data = reference_batch['data'].to(device)
    reference_labels = reference_batch['label'].to(device).long()
    with torch.no_grad():
        target_logits = net(target_data)
        target_pred = target_logits.argmax(dim=1)
    params = [
        param for _, param in select_gradient_parameters(net, gradient_space)
    ]
    n = min(int(sample_count), target_data.shape[0])
    ref_n = min(int(sample_count), reference_data.shape[0])

    if 'gate01' in needed:
        rows = [
            gate1_acceptance_delta(
                net,
                target_data[idx:idx + 1],
                int(target_pred[idx].item()),
                params,
                step_size=step_size,
            )
            for idx in range(n)
        ]
        passed = sum(bool(row['passed']) for row in rows)
        results.append(DiagnosticResult(
            gate_id='gate01',
            name='acceptance_delta',
            status='pass' if rows and passed == len(rows) else 'fail',
            metrics={
                'n': len(rows),
                'pass_rate': passed / len(rows) if rows else 0.0,
                'mean_loss_delta': float(np.mean([row['loss_delta'] for row in rows]))
                if rows else 0.0,
            },
            thresholds={'pass_rate': 1.0},
            message='acceptance direction should reduce target CE',
        ))

    if 'gate02' in needed:
        pair_n = min(n, ref_n)
        rows = [
            gate2_reference_sign_delta(
                net,
                target_data[idx:idx + 1],
                int(target_pred[idx].item()),
                reference_data[idx:idx + 1],
                int(reference_labels[idx].item()),
                params,
                step_size=step_size,
            )
            for idx in range(pair_n)
        ]
        passed = sum(bool(row['passed']) for row in rows)
        results.append(DiagnosticResult(
            gate_id='gate02',
            name='reference_sign_delta',
            status='pass' if rows and passed == len(rows) else 'fail',
            metrics={
                'n': len(rows),
                'pass_rate': passed / len(rows) if rows else 0.0,
                'mean_abs_k': float(np.mean([abs(row['k']) for row in rows]))
                if rows else 0.0,
                'mean_reference_loss_delta': float(np.mean([
                    row['reference_loss_delta'] for row in rows
                ])) if rows else 0.0,
                'finite_delta_match_rate': sum(
                    bool(row['finite_delta_matched']) for row in rows
                ) / len(rows) if rows else 0.0,
            },
            thresholds={'pass_rate': 1.0},
            message='K sign should predict first-order reference loss change',
        ))

    if 'gate03' in needed:
        if gradient_space != 'classifier':
            results.append(DiagnosticResult(
                gate_id='gate03',
                name='classifier_fc_factorization',
                status='skip',
                message='gate03 is defined for classifier gradient space',
            ))
        else:
            b = min(n, 2)
            r = min(ref_n, 8)
            with torch.no_grad():
                target_logits, target_features = forward_logits_features(
                    net, target_data[:b])
                target_probs = torch.softmax(target_logits, dim=1)
                ref_logits, ref_features = forward_logits_features(
                    net, reference_data[:r])
                ref_probs = torch.softmax(ref_logits, dim=1)
            candidate_classes = candidate_classes_from_probs(
                target_probs, 'all')[:, :min(3, target_probs.shape[1])]
            include_bias = classifier_has_bias(net)
            k_fc = classifier_pairwise_k(
                target_features,
                target_probs,
                candidate_classes,
                ref_features,
                ref_probs,
                reference_labels[:r],
                include_bias=include_bias,
            )
            target_dirs = classifier_candidate_directions_dense(
                target_features,
                target_probs,
                candidate_classes,
                include_bias=include_bias,
            )
            ref_dirs = classifier_direction_dense(
                ref_features,
                ref_probs,
                reference_labels[:r],
                include_bias=include_bias,
            )
            k_dense = torch.einsum('bkd,rd->bkr', target_dirs, ref_dirs)
            diff = (k_fc - k_dense).detach().cpu().numpy()
            results.append(DiagnosticResult(
                gate_id='gate03',
                name='classifier_fc_factorization',
                status='pass' if float(np.max(np.abs(diff))) < 1e-4 else 'fail',
                metrics={
                    'mae': float(np.mean(np.abs(diff))),
                    'max_abs': float(np.max(np.abs(diff))),
                    'n_targets': int(b),
                    'n_references': int(r),
                },
                thresholds={'max_abs': 1e-4},
                message='compact classifier K should match dense classifier K',
            ))

    return results


def run_posthoc_diagnostics(base_run_dir: Path,
                            out_dir: Path,
                            *,
                            scheme: str,
                            score_rules: Sequence[str],
                            gates: Sequence[str],
                            diagnostic_seed: int = 0,
                            reference_sizes: Sequence[int] | None = None,
                            mode: str = 'all') -> List[DiagnosticResult]:
    results: List[DiagnosticResult] = []
    needed = set(gates)
    for rule in score_rules:
        score_dir = base_run_dir / scheme / rule
        if 'gate04' in needed:
            results.append(_gate04_confidence_matched(score_dir, out_dir, rule))
        if 'gate06' in needed:
            results.append(_gate06_signed_support(score_dir, rule))
        if 'gate07' in needed:
            results.append(_gate07_label_shuffle(score_dir, rule, diagnostic_seed))
        if 'gate10' in needed:
            results.append(_gate10_score_ablation(score_dir, rule))
    if 'gate08' in needed:
        results.append(DiagnosticResult(
            gate_id='gate08',
            name='reference_size_stability',
            status='skip',
            metrics={'requested_reference_sizes': list(reference_sizes or [])},
            message='requires multiple completed runs with different reference sizes',
        ))
    if 'gate09' in needed:
        results.append(DiagnosticResult(
            gate_id='gate09',
            name='gradient_space_ablation',
            status='skip',
            message='requires completed runs for multiple gradient spaces',
        ))
    return results


def run_experiment_diagnostics(out_dir: Path,
                               metric_rows: Sequence[Mapping],
                               run_rows: Sequence[Mapping],
                               *,
                               reference_sizes: Sequence[int],
                               reference_seeds: Sequence[int],
                               gradient_spaces: Sequence[str],
                               candidate_modes: Sequence[str]
                               ) -> List[DiagnosticResult]:
    results = []
    results.append(_gate08_reference_size_grid(
        out_dir,
        metric_rows,
        run_rows,
        reference_sizes=reference_sizes,
        reference_seeds=reference_seeds,
        gradient_spaces=gradient_spaces,
        candidate_modes=candidate_modes,
    ))
    results.append(_gate09_gradient_space_ablation(
        out_dir,
        metric_rows,
        run_rows,
        gradient_spaces=gradient_spaces,
        candidate_modes=candidate_modes,
    ))
    return results


def experiment_required_gates(reference_sizes: Sequence[int],
                              reference_seeds: Sequence[int],
                              gradient_spaces: Sequence[str]) -> List[str]:
    required = []
    if len(set(int(v) for v in reference_sizes)) > 1 or len(set(
            int(v) for v in reference_seeds)) > 1:
        required.append('gate08')
    if len(set(str(v) for v in gradient_spaces)) > 1:
        required.append('gate09')
    return required


def _score_files(score_dir: Path) -> List[Path]:
    return sorted((score_dir / 'scores').glob('*.npz'))


def _load_npz(path: Path):
    return np.load(path, allow_pickle=True)


def _gate08_reference_size_grid(out_dir: Path,
                                metric_rows: Sequence[Mapping],
                                run_rows: Sequence[Mapping],
                                *,
                                reference_sizes: Sequence[int],
                                reference_seeds: Sequence[int],
                                gradient_spaces: Sequence[str],
                                candidate_modes: Sequence[str]
                                ) -> DiagnosticResult:
    expected_sizes = sorted({int(v) for v in reference_sizes})
    expected_seeds = sorted({int(v) for v in reference_seeds})
    if len(expected_sizes) <= 1 and len(expected_seeds) <= 1:
        return DiagnosticResult(
            gate_id='gate08',
            name='reference_size_stability',
            status='skip',
            metrics={
                'reference_per_class_grid': expected_sizes,
                'reference_seeds': expected_seeds,
                'gradient_spaces': list(dict.fromkeys(str(v) for v in gradient_spaces)),
                'candidate_modes': list(dict.fromkeys(str(v) for v in candidate_modes)),
            },
            message='only one reference size and seed were provided',
        )

    completed = [
        row for row in run_rows if str(row.get('run_status')) == 'complete'
    ]
    expected_run_count = (
        len(expected_sizes) *
        len(expected_seeds) *
        len(set(str(space) for space in gradient_spaces)) *
        len(set(str(mode) for mode in candidate_modes))
    )
    run_coverage = (
        len(completed) / expected_run_count if expected_run_count else 0.0
    )

    grouped = {}
    for row in metric_rows:
        key = (
            str(row.get('candidate_mode', '')),
            str(row['gradient_space']),
            str(row['score_rule']),
            str(row['ood_dataset']),
            int(row['reference_per_class']),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (
            candidate_mode,
            gradient_space,
            score_rule,
            ood_dataset,
            reference_per_class,
    ), rows in sorted(grouped.items()):
        for metric_name in ('AUROC', 'FPR@95', 'ACC'):
            values = [
                float(row[metric_name]) for row in rows
                if row.get(metric_name) not in {None, ''}
            ]
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            summary_rows.append({
                'candidate_mode': candidate_mode,
                'gradient_space': gradient_space,
                'score_rule': score_rule,
                'ood_dataset': ood_dataset,
                'reference_per_class': reference_per_class,
                'metric': metric_name,
                'count': int(arr.size),
                'mean': float(arr.mean()),
                'std': float(arr.std(ddof=0)),
                'min': float(arr.min()),
                'max': float(arr.max()),
            })

    trend_rows = []
    trend_groups = {}
    for row in summary_rows:
        key = (
            row['candidate_mode'],
            row['gradient_space'],
            row['score_rule'],
            row['ood_dataset'],
            row['metric'],
        )
        trend_groups.setdefault(key, []).append(row)
    for key, rows in sorted(trend_groups.items()):
        rows = sorted(rows, key=lambda item: int(item['reference_per_class']))
        if len(rows) < 2:
            continue
        first = rows[0]
        last = rows[-1]
        trend_rows.append({
            'candidate_mode': key[0],
            'gradient_space': key[1],
            'score_rule': key[2],
            'ood_dataset': key[3],
            'metric': key[4],
            'smallest_reference_per_class': int(first['reference_per_class']),
            'largest_reference_per_class': int(last['reference_per_class']),
            'largest_minus_smallest_mean': float(last['mean'] - first['mean']),
            'max_std': float(max(row['std'] for row in rows)),
        })

    summary_path = out_dir / 'gate08_reference_size_stability.csv'
    trend_path = out_dir / 'gate08_reference_size_trends.csv'
    write_csv(
        summary_path,
        summary_rows,
        fieldnames=[
            'candidate_mode', 'gradient_space', 'score_rule', 'ood_dataset',
            'reference_per_class', 'metric', 'count', 'mean', 'std', 'min', 'max',
        ],
    )
    write_csv(
        trend_path,
        trend_rows,
        fieldnames=[
            'candidate_mode', 'gradient_space', 'score_rule', 'ood_dataset', 'metric',
            'smallest_reference_per_class', 'largest_reference_per_class',
            'largest_minus_smallest_mean', 'max_std',
        ],
    )
    missing_runs = [
        row for row in run_rows if str(row.get('run_status')) != 'complete'
    ]
    status = 'pass' if run_coverage >= 1.0 and summary_rows else 'warn'
    return DiagnosticResult(
        gate_id='gate08',
        name='reference_size_stability',
        status=status,
        metrics={
            'reference_per_class_grid': expected_sizes,
            'reference_seeds': expected_seeds,
            'expected_run_count': int(expected_run_count),
            'completed_run_count': int(len(completed)),
            'run_coverage': float(run_coverage),
            'missing_run_count': int(len(missing_runs)),
            'summary_row_count': int(len(summary_rows)),
            'trend_row_count': int(len(trend_rows)),
        },
        artifacts={
            'summary_csv': str(summary_path),
            'trend_csv': str(trend_path),
        },
        message='reference size/seed stability grid was evaluated'
        if status == 'pass' else
        'reference size/seed grid is incomplete or produced no metric rows',
    )


def _gate09_gradient_space_ablation(out_dir: Path,
                                    metric_rows: Sequence[Mapping],
                                    run_rows: Sequence[Mapping],
                                    *,
                                    gradient_spaces: Sequence[str],
                                    candidate_modes: Sequence[str]
                                    ) -> DiagnosticResult:
    expected_spaces = [str(v) for v in gradient_spaces]
    unique_spaces = list(dict.fromkeys(expected_spaces))
    if len(unique_spaces) <= 1:
        return DiagnosticResult(
            gate_id='gate09',
            name='gradient_space_ablation',
            status='skip',
            metrics={
                'gradient_spaces': unique_spaces,
                'candidate_modes': list(dict.fromkeys(str(v) for v in candidate_modes)),
            },
            message='only one gradient space was provided',
        )

    baseline_space = unique_spaces[0]
    grouped = {}
    for row in metric_rows:
        key = (
            str(row.get('candidate_mode', '')),
            int(row['reference_per_class']),
            int(row['reference_seed']),
            str(row['score_rule']),
            str(row['ood_dataset']),
        )
        grouped.setdefault(key, {})[str(row['gradient_space'])] = row

    comparison_rows = []
    missing_comparisons = 0
    for key, by_space in sorted(grouped.items()):
        baseline = by_space.get(baseline_space)
        if baseline is None:
            missing_comparisons += max(0, len(unique_spaces) - 1)
            continue
        for space in unique_spaces[1:]:
            candidate = by_space.get(space)
            if candidate is None:
                missing_comparisons += 1
                continue
            comparison_rows.append({
                'candidate_mode': key[0],
                'reference_per_class': key[1],
                'reference_seed': key[2],
                'score_rule': key[3],
                'ood_dataset': key[4],
                'baseline_gradient_space': baseline_space,
                'gradient_space': space,
                'baseline_AUROC': float(baseline['AUROC']),
                'AUROC': float(candidate['AUROC']),
                'delta_AUROC': float(candidate['AUROC']) - float(baseline['AUROC']),
                'baseline_FPR@95': float(baseline['FPR@95']),
                'FPR@95': float(candidate['FPR@95']),
                'delta_FPR@95': float(candidate['FPR@95']) -
                float(baseline['FPR@95']),
                'baseline_ACC': float(baseline['ACC']),
                'ACC': float(candidate['ACC']),
                'delta_ACC': float(candidate['ACC']) - float(baseline['ACC']),
            })

    summary = {}
    for space in unique_spaces[1:]:
        rows = [row for row in comparison_rows if row['gradient_space'] == space]
        if not rows:
            continue
        summary[space] = {
            'count': len(rows),
            'mean_delta_AUROC': float(np.mean([row['delta_AUROC'] for row in rows])),
            'mean_delta_FPR@95': float(np.mean([
                row['delta_FPR@95'] for row in rows
            ])),
            'mean_delta_ACC': float(np.mean([row['delta_ACC'] for row in rows])),
        }

    comparison_path = out_dir / 'gate09_gradient_space_ablation.csv'
    write_csv(
        comparison_path,
        comparison_rows,
        fieldnames=[
            'candidate_mode', 'reference_per_class', 'reference_seed',
            'score_rule', 'ood_dataset',
            'baseline_gradient_space', 'gradient_space',
            'baseline_AUROC', 'AUROC', 'delta_AUROC',
            'baseline_FPR@95', 'FPR@95', 'delta_FPR@95',
            'baseline_ACC', 'ACC', 'delta_ACC',
        ],
    )
    completed_spaces = sorted({
        str(row.get('gradient_space'))
        for row in run_rows
        if str(row.get('run_status')) == 'complete'
    })
    status = (
        'pass'
        if comparison_rows and missing_comparisons == 0
        and set(unique_spaces).issubset(set(completed_spaces))
        else 'warn'
    )
    return DiagnosticResult(
        gate_id='gate09',
        name='gradient_space_ablation',
        status=status,
        metrics={
            'gradient_spaces': unique_spaces,
            'candidate_modes': list(dict.fromkeys(str(v) for v in candidate_modes)),
            'baseline_gradient_space': baseline_space,
            'completed_gradient_spaces': completed_spaces,
            'comparison_count': int(len(comparison_rows)),
            'missing_comparison_count': int(missing_comparisons),
            'summary': summary,
        },
        artifacts={'comparison_csv': str(comparison_path)},
        message='gradient-space ablation grid was evaluated'
        if status == 'pass' else
        'gradient-space grid is incomplete or produced missing comparisons',
    )


def _has_arrays(arrays, *keys: str) -> bool:
    return all(key in arrays.files and np.asarray(arrays[key]).size > 0
               for key in keys)


def _split_group(arrays) -> str:
    labels = np.asarray(arrays['label'])
    return 'ood' if labels.size and np.all(labels < 0) else 'id'


def _gate04_confidence_matched(score_dir: Path,
                               out_dir: Path,
                               rule: str,
                               *,
                               bins: int = 10) -> DiagnosticResult:
    score_files = _score_files(score_dir)
    if not score_files:
        return DiagnosticResult(
            gate_id='gate04',
            name=f'confidence_matched_separation/{rule}',
            status='skip',
            message='no score files found',
        )
    rows = []
    for path in score_files:
        arrays = _load_npz(path)
        if not _has_arrays(arrays, 'q_max', 'eid'):
            return DiagnosticResult(
                gate_id='gate04',
                name=f'confidence_matched_separation/{rule}',
                status='skip',
                message=f'{path} does not contain q_max/eid',
            )
        group = _split_group(arrays)
        q_max = np.asarray(arrays['q_max'], dtype=np.float64)
        eid = np.asarray(arrays['eid'], dtype=np.float64)
        edges = np.linspace(0.0, 1.0, int(bins) + 1)
        for start, end in zip(edges[:-1], edges[1:]):
            mask = (q_max >= start) & (q_max <= end if end == 1.0 else q_max < end)
            if not np.any(mask):
                continue
            rows.append({
                'split': path.stem,
                'group': group,
                'bin_start': float(start),
                'bin_end': float(end),
                'count': int(mask.sum()),
                'eid_mean': float(eid[mask].mean()),
                'eid_median': float(np.median(eid[mask])),
            })
    pair_gaps = []
    for start in sorted({row['bin_start'] for row in rows}):
        id_values = [
            row['eid_mean'] for row in rows
            if row['bin_start'] == start and row['group'] == 'id'
        ]
        ood_values = [
            row['eid_mean'] for row in rows
            if row['bin_start'] == start and row['group'] == 'ood'
        ]
        if id_values and ood_values:
            pair_gaps.append(float(np.mean(id_values) - np.mean(ood_values)))
    artifact = out_dir / 'gates' / f'gate04_{rule}_confidence_bins.csv'
    write_csv(
        artifact,
        rows,
        fieldnames=[
            'split', 'group', 'bin_start', 'bin_end', 'count',
            'eid_mean', 'eid_median',
        ],
    )
    if not pair_gaps:
        status = 'warn'
        message = 'no confidence bin contained both ID-side and OOD samples'
    else:
        mean_gap = float(np.mean(pair_gaps))
        status = 'pass' if mean_gap > 0.0 else 'fail'
        message = 'ID-side E_ID should exceed OOD E_ID within confidence bins'
    return DiagnosticResult(
        gate_id='gate04',
        name=f'confidence_matched_separation/{rule}',
        status=status,
        metrics={
            'matched_bin_count': len(pair_gaps),
            'mean_id_minus_ood_eid_gap': float(np.mean(pair_gaps))
            if pair_gaps else 0.0,
            'min_id_minus_ood_eid_gap': float(np.min(pair_gaps))
            if pair_gaps else 0.0,
        },
        artifacts={'confidence_bins_csv': str(artifact)},
        message=message,
    )


def _gate06_signed_support(score_dir: Path, rule: str) -> DiagnosticResult:
    score_files = _score_files(score_dir)
    summaries = []
    max_diff = 0.0
    for path in score_files:
        arrays = _load_npz(path)
        if not _has_arrays(arrays, 'q_c', 'v_c', 'rank_only_scores',
                           'same_positive_rates'):
            return DiagnosticResult(
                gate_id='gate06',
                name=f'signed_support/{rule}',
                status='skip',
                message=f'{path} does not contain class-wise diagnostic arrays',
            )
        q_c = np.asarray(arrays['q_c'], dtype=np.float64)
        v_c = np.asarray(arrays['v_c'], dtype=np.float64)
        rank = np.asarray(arrays['rank_only_scores'], dtype=np.float64)
        same_positive = np.asarray(arrays['same_positive_rates'], dtype=np.float64)
        signed_eid = np.max(q_c * v_c, axis=1)
        rank_eid = np.max(q_c * rank, axis=1)
        positive_eid = np.max(q_c * same_positive, axis=1)
        max_diff = max(max_diff, float(np.max(np.abs(rank_eid - signed_eid))))
        summaries.append({
            'split': path.stem,
            'group': _split_group(arrays),
            'n': int(q_c.shape[0]),
            'signed_eid_mean': float(signed_eid.mean()) if signed_eid.size else 0.0,
            'rank_only_eid_mean': float(rank_eid.mean()) if rank_eid.size else 0.0,
            'same_positive_eid_mean': float(positive_eid.mean())
            if positive_eid.size else 0.0,
            **signed_support_summary(v_c, rank, same_positive),
        })
    return DiagnosticResult(
        gate_id='gate06',
        name=f'signed_support/{rule}',
        status='pass' if max_diff > NUMERIC_EPS else 'warn',
        metrics={
            'max_rank_minus_signed_abs_diff': max_diff,
            'splits': summaries,
        },
        message='rank-only evidence should be distinguishable from signed validation',
    )


def _gate07_label_shuffle(score_dir: Path,
                          rule: str,
                          diagnostic_seed: int) -> DiagnosticResult:
    score_files = _score_files(score_dir)
    rows = []
    for path in score_files:
        arrays = _load_npz(path)
        if not _has_arrays(arrays, 'eid', 'eid_label_shuffle'):
            return DiagnosticResult(
                gate_id='gate07',
                name=f'label_shuffle/{rule}',
                status='skip',
                message=f'{path} does not contain eid_label_shuffle',
            )
        eid = np.asarray(arrays['eid'], dtype=np.float64)
        shuffled = np.asarray(arrays['eid_label_shuffle'], dtype=np.float64)
        if eid.shape != shuffled.shape:
            return DiagnosticResult(
                gate_id='gate07',
                name=f'label_shuffle/{rule}',
                status='skip',
                message=f'{path} has incompatible label-shuffle array shape',
            )
        rows.append({
            'split': path.stem,
            'group': _split_group(arrays),
            'n': int(eid.size),
            'eid_mean': float(eid.mean()) if eid.size else 0.0,
            'shuffled_eid_mean': float(shuffled.mean()) if shuffled.size else 0.0,
            'eid_minus_shuffled_mean': float(eid.mean() - shuffled.mean())
            if eid.size and shuffled.size else 0.0,
        })
    id_gaps = [
        row['eid_minus_shuffled_mean'] for row in rows if row['group'] == 'id'
    ]
    mean_id_gap = float(np.mean(id_gaps)) if id_gaps else 0.0
    return DiagnosticResult(
        gate_id='gate07',
        name=f'label_shuffle/{rule}',
        status='pass' if id_gaps and mean_id_gap >= 0.0 else 'warn',
        metrics={
            'shuffle_seed': int(diagnostic_seed),
            'mean_id_eid_minus_shuffled': mean_id_gap,
            'splits': rows,
        },
        message='reference-label shuffle should not improve ID-side evidence',
    )


def _gate10_score_ablation(score_dir: Path, rule: str) -> DiagnosticResult:
    score_files = _score_files(score_dir)
    if not score_files:
        return DiagnosticResult(
            gate_id='gate10',
            name=f'score_ablation/{rule}',
            status='skip',
            message='no score files found',
        )
    variants = {
        'eid': lambda arrays: np.asarray(arrays['eid'], dtype=np.float64),
        'q_max': lambda arrays: np.asarray(arrays['q_max'], dtype=np.float64),
        'v_best': lambda arrays: np.asarray(arrays['v_best'], dtype=np.float64),
        'eid_pred': lambda arrays: np.asarray(arrays['eid_pred'], dtype=np.float64),
    }
    rows = []
    for variant_name, extractor in variants.items():
        metrics = _variant_metric_rows(score_files, variant_name, extractor, rule)
        rows.extend(metrics)
    if _all_files_have(score_files, 'q_c', 'rank_only_scores'):
        rows.extend(_variant_metric_rows(
            score_files,
            'rank_only_eid',
            lambda arrays: np.max(
                np.asarray(arrays['q_c'], dtype=np.float64) *
                np.asarray(arrays['rank_only_scores'], dtype=np.float64),
                axis=1,
            ),
            rule,
        ))
    if _all_files_have(score_files, 'q_c', 'same_positive_rates'):
        rows.extend(_variant_metric_rows(
            score_files,
            'same_positive_eid',
            lambda arrays: np.max(
                np.asarray(arrays['q_c'], dtype=np.float64) *
                np.asarray(arrays['same_positive_rates'], dtype=np.float64),
                axis=1,
            ),
            rule,
        ))
    if _all_files_have(score_files, 'eid_label_shuffle'):
        rows.extend(_variant_metric_rows(
            score_files,
            'label_shuffle_eid',
            lambda arrays: np.asarray(arrays['eid_label_shuffle'], dtype=np.float64),
            rule,
        ))
    if not rows:
        return DiagnosticResult(
            gate_id='gate10',
            name=f'score_ablation/{rule}',
            status='skip',
            message='no ID/OOD metric pairs could be computed',
        )
    return DiagnosticResult(
        gate_id='gate10',
        name=f'score_ablation/{rule}',
        status='pass',
        metrics={'rows': rows},
        message='final score and ablation variants were evaluated',
    )


def _all_files_have(paths: Sequence[Path], *keys: str) -> bool:
    for path in paths:
        arrays = _load_npz(path)
        if not _has_arrays(arrays, *keys):
            return False
    return bool(paths)


def _variant_metric_rows(score_files: Sequence[Path],
                         variant_name: str,
                         evidence_extractor,
                         rule: str) -> List[Dict]:
    id_parts = []
    ood_parts = []
    for path in score_files:
        arrays = _load_npz(path)
        try:
            evidence = evidence_extractor(arrays)
        except KeyError:
            return []
        ood_score = ood_score_from_eid(evidence, rule)
        part = score_tuple_from_ood(arrays['pred'], ood_score, arrays['label'])
        if _split_group(arrays) == 'ood':
            ood_parts.append((path.stem, part))
        else:
            id_parts.append(part)
    if not id_parts or not ood_parts:
        return []
    metric_id = concat_score_tuples(id_parts)
    rows = []
    for split_name, part in ood_parts:
        fpr, auroc, aupr_in, aupr_out, acc = metric_summary(metric_id, part)
        rows.append({
            'variant': variant_name,
            'ood_split': split_name,
            'FPR@95': float(fpr),
            'AUROC': float(auroc),
            'AUPR_IN': float(aupr_in),
            'AUPR_OUT': float(aupr_out),
            'ACC': float(acc),
        })
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Summarize RAE diagnostics')
    parser.add_argument('score_dir', type=Path)
    parser.add_argument('--out', type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out = args.out or args.score_dir / 'diagnostics_summary.json'
    summarize_score_dir(args.score_dir, out)


if __name__ == '__main__':
    main()
