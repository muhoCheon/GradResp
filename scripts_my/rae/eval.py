#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .artifacts import (
    ensure_dir,
    file_sha256,
    diagnostics_dir,
    experiment_dir,
    ref_grad_bank_dir,
    read_json,
    reference_dir,
    run_dir,
    save_score_npz,
    score_rule_dir,
    selected_samples_hash,
    write_csv,
    write_json,
)
from .config import (
    CANDIDATE_MODES,
    CACHE_SCHEMA_VERSION,
    CLI_DEFAULT_SCORE_RULES,
    DEFAULT_CHECKPOINT,
    DEFAULT_REFERENCE_PER_CLASS_GRID_ARG,
    DEFAULT_REJECTION_POWER,
    DEFAULT_REJECTION_RULE,
    DEFAULT_VALIDATION_RULE,
    DEFAULT_VALIDATION_TEMPERATURE,
    GRADIENT_SPACES,
    MODEL_ARCH,
    NUM_CLASSES,
    REFERENCE_FILTERS,
    REJECTION_RULES,
    ROOT_DIR,
    SUPPORTED_DATASETS,
    SUPPORTED_SCHEMES,
    VALIDATION_RULES,
    ReferenceConfig,
    make_run_id,
    parse_csv_values,
    parse_score_rules,
)
from .data import (
    build_dataloaders,
    build_model,
    build_reference_loader,
    device_from_arg,
    set_seed,
    split_dataloaders,
    subset_loader,
)
from .diagnostics import (
    DEFAULT_DIAGNOSTIC_STEP_SIZE,
    DiagnosticResult,
    diagnostics_claim_status,
    experiment_required_gates,
    parse_diagnostic_gates,
    run_experiment_diagnostics,
    run_online_diagnostics,
    run_posthoc_diagnostics,
    summarize_score_dir,
    write_diagnostic_results,
)
from .gradients import compute_ref_grad_bank
from .gradient_space import select_gradient_parameters
from .metrics import (
    concat_score_tuples,
    format_metric_row,
    metric_summary,
    score_tuple_from_ood,
    write_metrics_csv,
)
from .reference import (
    build_reference_set,
    load_reference_set,
    reference_manifest_is_reusable,
    train_candidate_metadata_id,
    train_candidate_metadata_identity,
)
from .score import (
    ood_score_from_split_scores,
    prepare_ref_grad_bank_for_scoring,
    score_target_batch,
)


def reference_config_from_args(args) -> ReferenceConfig:
    return ReferenceConfig(
        dataset=args.dataset,
        per_class=int(args.reference_per_class),
        filter_name=args.reference_filter,
        min_confidence=float(args.reference_min_confidence),
        seed=int(args.reference_seed),
    )


def resolved_checkpoint(args) -> str:
    return args.checkpoint or DEFAULT_CHECKPOINT[args.dataset]


def current_model_identity(args) -> Tuple[str, str | None, Dict]:
    checkpoint = resolved_checkpoint(args)
    checkpoint_sha256 = file_sha256(checkpoint) if Path(checkpoint).exists() else None
    metadata_identity = train_candidate_metadata_identity(
        args.dataset,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        model_arch=MODEL_ARCH[args.dataset].__name__,
        num_classes=NUM_CLASSES[args.dataset],
    )
    return checkpoint, checkpoint_sha256, metadata_identity


def candidate_set_claim(candidate_mode: str) -> str:
    claims = {
        'all': 'all_classes',
        'pred': 'predicted_class_only',
    }
    if candidate_mode not in claims:
        raise ValueError(f'Unknown candidate mode: {candidate_mode}')
    return claims[candidate_mode]


def candidate_claim_scope(candidate_mode: str) -> str:
    scopes = {
        'all': 'exact_full_candidate_rae',
        'pred': 'predicted_class_rae',
    }
    if candidate_mode not in scopes:
        raise ValueError(f'Unknown candidate mode: {candidate_mode}')
    return scopes[candidate_mode]


def write_run_manifest(path: Path,
                       args,
                       *,
                       run_id: str,
                       checkpoint: str,
                       reference_manifest: Dict,
                       ref_grad_bank_manifest: Dict,
                       score_rules: List[str],
                       diagnostics_manifest: Dict | None = None,
                       elapsed: float | None = None) -> None:
    claim_bearing, claim_reasons = diagnostics_claim_status(
        diagnostics_manifest,
        max_target_samples=args.max_target_samples,
        candidate_mode=args.candidate_mode,
    )
    resource_adjusted = args.candidate_mode != 'all'
    manifest = {
        'artifact': 'rae_run',
        'dataset': args.dataset,
        'scheme': args.scheme,
        'baseline_protocol': args.baseline_protocol,
        'run_id': run_id,
        'score_rules': score_rules,
        'score_direction': 'higher_is_ood',
        'conf_boundary_transform': 'conf = -ood_score',
        'raw_eid_saved': True,
        'candidate_mode': args.candidate_mode,
        'candidate_set_claim': candidate_set_claim(args.candidate_mode),
        'claim_scope': candidate_claim_scope(args.candidate_mode),
        'exact_candidate_set': not resource_adjusted,
        'resource_adjusted_candidate_set': resource_adjusted,
        'validation_config': {
            'validation_rule': args.validation_rule,
            'validation_temperature': float(args.validation_temperature),
        },
        'rejection_config': {
            'rejection_rule': args.rejection_rule,
            'rejection_power': float(args.rejection_power),
        },
        'gradient_config': {
            'gradient_space': args.gradient_space,
        },
        'runtime_config': {
            'batch_size': int(args.batch_size),
            'reference_batch_size': int(args.reference_batch_size),
            'reference_bank_chunk_size': int(args.reference_bank_chunk_size),
            'num_workers': int(args.num_workers),
        },
        'reference_manifest': reference_manifest,
        'gradient_manifest': ref_grad_bank_manifest,
        'checkpoint': checkpoint,
        'checkpoint_resolved': str(Path(checkpoint).resolve()),
        'checkpoint_sha256': file_sha256(checkpoint)
        if Path(checkpoint).exists() else None,
        'model_arch': MODEL_ARCH[args.dataset].__name__,
        'num_classes': NUM_CLASSES[args.dataset],
        'max_target_samples': args.max_target_samples,
        'diagnostics_manifest': diagnostics_manifest,
        'claim_bearing': claim_bearing,
        'claim_bearing_reasons': claim_reasons,
        'elapsed_sec': elapsed,
    }
    write_json(path, manifest)


def build_reference(args, net, dataloaders, device: torch.device) -> Dict:
    ref_config = reference_config_from_args(args)
    out_dir = reference_dir(args, ref_config)
    checkpoint, checkpoint_sha256, metadata_identity = current_model_identity(args)
    if (out_dir / 'manifest.json').exists() and not args.rebuild_reference:
        manifest = read_json(out_dir / 'manifest.json')
        if reference_manifest_is_reusable(
                manifest,
                ref_config,
                expected_metadata_identity=metadata_identity,
                checkpoint_sha256=checkpoint_sha256,
                model_arch=MODEL_ARCH[args.dataset].__name__,
                num_classes=NUM_CLASSES[args.dataset]):
            return manifest
    metadata_dir = (
        Path(args.output_root) / 'train_candidate_metadata' / args.dataset /
        train_candidate_metadata_id(metadata_identity)
    )
    return build_reference_set(
        net,
        dataloaders['id']['train'],
        out_dir,
        ref_config,
        metadata_dir=metadata_dir,
        candidate_metadata_identity=metadata_identity,
        device=device,
        rebuild_metadata=args.rebuild_train_metadata,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        model_arch=MODEL_ARCH[args.dataset].__name__,
    )


def load_ref_grad_bank(path: Path) -> Dict:
    arrays = np.load(path / 'gradient_bank.npz', allow_pickle=True)
    ref_grad_bank = {key: arrays[key] for key in arrays.files}
    ref_grad_bank['bank_type'] = str(ref_grad_bank['bank_type'])
    if 'classifier_has_bias' in ref_grad_bank:
        ref_grad_bank['classifier_has_bias'] = np.asarray(
            ref_grad_bank['classifier_has_bias'])
    return ref_grad_bank


def load_reference_set_from_manifest(args, reference_manifest: Dict) -> Dict:
    ref_config = reference_config_from_args(args)
    _, checkpoint_sha256, metadata_identity = current_model_identity(args)
    expected_config_id = ref_config.id
    manifest_config_id = reference_manifest.get('reference_config_id')
    if manifest_config_id != expected_config_id:
        raise ValueError(
            'Reference manifest does not match current reference config: '
            f'manifest={manifest_config_id}, expected={expected_config_id}')
    if not reference_manifest_is_reusable(
            reference_manifest,
            ref_config,
            expected_metadata_identity=metadata_identity,
            checkpoint_sha256=checkpoint_sha256,
            model_arch=MODEL_ARCH[args.dataset].__name__,
            num_classes=NUM_CLASSES[args.dataset]):
        raise ValueError('Reference manifest is not reusable for the current config')

    ref = load_reference_set(reference_dir(args, ref_config))
    stored_manifest = ref['manifest']
    if not reference_manifest_is_reusable(
            stored_manifest,
            ref_config,
            expected_metadata_identity=metadata_identity,
            checkpoint_sha256=checkpoint_sha256,
            model_arch=MODEL_ARCH[args.dataset].__name__,
            num_classes=NUM_CLASSES[args.dataset]):
        raise ValueError('Stored reference artifact is not reusable')
    if stored_manifest.get('reference_config_id') != manifest_config_id:
        raise ValueError(
            'Stored reference config id does not match requested manifest: '
            f"stored={stored_manifest.get('reference_config_id')}, "
            f'manifest={manifest_config_id}')

    manifest_hash = reference_manifest.get('selected_sample_hash')
    stored_hash = stored_manifest.get('selected_sample_hash')
    if stored_hash != manifest_hash:
        raise ValueError(
            'Stored reference hash does not match requested manifest: '
            f'stored={stored_hash}, manifest={manifest_hash}')

    selected_hash = selected_samples_hash(ref['selected_indices'])
    if selected_hash != manifest_hash:
        raise ValueError(
            'Reference npz selected indices do not match manifest hash: '
            f'npz={selected_hash}, manifest={manifest_hash}')
    return ref


def expected_ref_grad_bank_type(gradient_space: str) -> str:
    return 'classifier_compact' if gradient_space == 'classifier' else 'dense'


def ref_grad_bank_manifest_is_reusable(manifest: Dict,
                                       args,
                                       reference_manifest: Dict,
                                       *,
                                       checkpoint: str,
                                       checkpoint_sha256: str | None) -> bool:
    if int(manifest.get('schema_version', 0)) != CACHE_SCHEMA_VERSION:
        return False
    if manifest.get('artifact') != 'rae_gradient_bank':
        return False
    if manifest.get('dataset') != args.dataset:
        return False
    gradient_config = manifest.get('gradient_config') or {}
    if gradient_config.get('gradient_space') != args.gradient_space:
        return False
    if manifest.get('reference_set_hash') != reference_manifest.get(
            'selected_sample_hash'):
        return False
    if manifest.get('reference_config_id') != reference_manifest.get(
            'reference_config_id'):
        return False
    if manifest.get('checkpoint_resolved') != str(Path(checkpoint).resolve()):
        return False
    if manifest.get('checkpoint_sha256') != checkpoint_sha256:
        return False
    if manifest.get('model_arch') != MODEL_ARCH[args.dataset].__name__:
        return False
    if int(manifest.get('num_classes', -1)) != NUM_CLASSES[args.dataset]:
        return False
    if manifest.get('bank_type') != expected_ref_grad_bank_type(args.gradient_space):
        return False
    expected_count = int(reference_manifest.get('selected_count', -1))
    if expected_count >= 0 and int(
            manifest.get('reference_count', -1)) != expected_count:
        return False
    return True


def ref_grad_bank_metadata(args, net, ref_grad_bank: Dict) -> Dict:
    metadata = {
        'bank_type': ref_grad_bank['bank_type'],
        'reference_count': int(np.asarray(ref_grad_bank['labels']).size),
    }
    if ref_grad_bank['bank_type'] == 'classifier_compact':
        metadata.update({
            'feature_dim': int(np.asarray(ref_grad_bank['features']).shape[1]),
            'classifier_has_bias': bool(
                np.asarray(ref_grad_bank['classifier_has_bias']).item()),
        })
    else:
        parameter_names = [name for name, _ in select_gradient_parameters(
            net, args.gradient_space)]
        metadata.update({
            'parameter_names': parameter_names,
            'parameter_count': int(np.asarray(ref_grad_bank['directions']).shape[1]),
        })
    return metadata


def build_ref_grad_bank(args, net, dataloaders, reference_manifest: Dict,
                        device: torch.device) -> Tuple[Dict, Dict]:
    out_dir = ref_grad_bank_dir(
        args, args.gradient_space, reference_manifest['selected_sample_hash'])
    checkpoint, checkpoint_sha256, _ = current_model_identity(args)
    if (out_dir / 'manifest.json').exists() and not args.rebuild_ref_grad_bank:
        manifest = read_json(out_dir / 'manifest.json')
        if (
                ref_grad_bank_manifest_is_reusable(
                    manifest,
                    args,
                    reference_manifest,
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_sha256) and
                (out_dir / 'gradient_bank.npz').exists()):
            return manifest, load_ref_grad_bank(out_dir)
    ref = load_reference_set_from_manifest(args, reference_manifest)
    ref_loader = build_reference_loader(
        dataloaders['id']['train'],
        ref['selected_indices'],
        batch_size=args.reference_batch_size,
        num_workers=args.num_workers,
    )
    ref_grad_bank = compute_ref_grad_bank(
        net,
        ref_loader,
        gradient_space=args.gradient_space,
        device=device,
    )
    manifest = {
        'artifact': 'rae_gradient_bank',
        'schema_version': CACHE_SCHEMA_VERSION,
        'dataset': args.dataset,
        'gradient_config': {
            'gradient_space': args.gradient_space,
            'gradient_config_id': args.gradient_space,
        },
        'reference_set_hash': reference_manifest['selected_sample_hash'],
        'reference_config_id': reference_manifest['reference_config_id'],
        'checkpoint': checkpoint,
        'checkpoint_resolved': str(Path(checkpoint).resolve()),
        'checkpoint_sha256': checkpoint_sha256,
        'model_arch': MODEL_ARCH[args.dataset].__name__,
        'num_classes': NUM_CLASSES[args.dataset],
    }
    manifest.update(ref_grad_bank_metadata(args, net, ref_grad_bank))
    ensure_dir(out_dir)
    np.savez(out_dir / 'gradient_bank.npz', **ref_grad_bank)
    write_json(out_dir / 'manifest.json', manifest)
    return manifest, ref_grad_bank


def score_loader(args,
                 net,
                 loader,
                 ref_grad_bank: Dict,
                 *,
                 split_name: str,
                 split_kind: str,
                 device: torch.device,
                 selected_params=None) -> Dict:
    loader = subset_loader(
        loader,
        args.max_target_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    records = {
        'pred': [],
        'label': [],
        'eid': [],
        'best_class': [],
        'q_best': [],
        'v_best': [],
        'q_max': [],
        'v_pred': [],
        'eid_pred': [],
        'candidate_classes': [],
        'q_c': [],
        'v_c': [],
        'e_c': [],
        'rank_only_scores': [],
        'same_positive_rates': [],
        'accept_eid': [],
        'reject_id_evidence': [],
        'reject_ood_evidence': [],
        'reject_k_mean': [],
        'candidate_raw_grad_norm': [],
        'candidate_effective_dim': [],
        'candidate_direction_cos_to_pred': [],
        'candidate_ref_proto_cos': [],
        'v_c_label_shuffle': [],
        'e_c_label_shuffle': [],
        'eid_label_shuffle': [],
    }
    label_shuffle_seed = (
        int(args.diagnostic_seed)
        if getattr(args, 'diagnostics', 'all') != 'off' else None
    )
    for batch in tqdm(loader, desc=f'RAE score {split_name}'):
        out = score_target_batch(
            net,
            batch,
            ref_grad_bank,
            gradient_space=args.gradient_space,
            candidate_mode=args.candidate_mode,
            validation_rule=args.validation_rule,
            validation_temperature=args.validation_temperature,
            rejection_rule=args.rejection_rule,
            rejection_power=args.rejection_power,
            device=device,
            label_shuffle_seed=label_shuffle_seed,
            selected_params=selected_params,
            reference_bank_chunk_size=args.reference_bank_chunk_size,
        )
        records['pred'].append(out.pred)
        records['label'].append(out.label)
        records['eid'].append(out.eid)
        records['best_class'].append(out.best_class)
        records['q_best'].append(out.q_best)
        records['v_best'].append(out.v_best)
        records['q_max'].append(out.q_max)
        records['v_pred'].append(out.v_pred)
        records['eid_pred'].append(out.eid_pred)
        records['candidate_classes'].append(out.candidate_classes)
        records['q_c'].append(out.q_c)
        records['v_c'].append(out.v_c)
        records['e_c'].append(out.e_c)
        records['rank_only_scores'].append(out.rank_only_scores)
        records['same_positive_rates'].append(out.same_positive_rates)
        if out.accept_eid is not None:
            records['accept_eid'].append(out.accept_eid)
        if out.reject_id_evidence is not None:
            records['reject_id_evidence'].append(out.reject_id_evidence)
        if out.reject_ood_evidence is not None:
            records['reject_ood_evidence'].append(out.reject_ood_evidence)
        if out.reject_k_mean is not None:
            records['reject_k_mean'].append(out.reject_k_mean)
        if out.candidate_raw_grad_norm is not None:
            records['candidate_raw_grad_norm'].append(out.candidate_raw_grad_norm)
        if out.candidate_effective_dim is not None:
            records['candidate_effective_dim'].append(out.candidate_effective_dim)
        if out.candidate_direction_cos_to_pred is not None:
            records['candidate_direction_cos_to_pred'].append(
                out.candidate_direction_cos_to_pred)
        if out.candidate_ref_proto_cos is not None:
            records['candidate_ref_proto_cos'].append(out.candidate_ref_proto_cos)
        if out.v_c_label_shuffle is not None:
            records['v_c_label_shuffle'].append(out.v_c_label_shuffle)
        if out.e_c_label_shuffle is not None:
            records['e_c_label_shuffle'].append(out.e_c_label_shuffle)
        if out.eid_label_shuffle is not None:
            records['eid_label_shuffle'].append(out.eid_label_shuffle)
    merged = {
        key: np.concatenate(values) if values else np.asarray([])
        for key, values in records.items()
    }
    if split_kind in {'nearood', 'farood'}:
        merged['score_label'] = -1 * np.ones_like(merged['label'], dtype=np.int64)
    else:
        merged['score_label'] = merged['label'].astype(np.int64)
    return merged


def save_split_scores(base_run_dir: Path,
                      scheme: str,
                      split_name: str,
                      split_scores: Dict,
                      score_rules: List[str]) -> Dict:
    parts = {}
    for rule in score_rules:
        ood_score = ood_score_from_split_scores(split_scores, rule)
        pred, conf, label = score_tuple_from_ood(
            split_scores['pred'], ood_score, split_scores['score_label'])
        out_dir = score_rule_dir(base_run_dir, scheme, rule) / 'scores'
        save_score_npz(
            out_dir / f'{split_name}.npz',
            pred,
            conf,
            label,
            ood_score=ood_score,
            eid=split_scores['eid'],
            best_class=split_scores['best_class'],
            v_best=split_scores['v_best'],
            q_best=split_scores['q_best'],
            q_max=split_scores['q_max'],
            v_pred=split_scores['v_pred'],
            eid_pred=split_scores['eid_pred'],
            candidate_classes=split_scores['candidate_classes'],
            q_c=split_scores['q_c'],
            v_c=split_scores['v_c'],
            e_c=split_scores['e_c'],
            rank_only_scores=split_scores['rank_only_scores'],
            same_positive_rates=split_scores['same_positive_rates'],
            accept_eid=split_scores.get('accept_eid'),
            reject_id_evidence=split_scores.get('reject_id_evidence'),
            reject_ood_evidence=split_scores.get('reject_ood_evidence'),
            reject_k_mean=split_scores.get('reject_k_mean'),
            candidate_raw_grad_norm=split_scores.get('candidate_raw_grad_norm'),
            candidate_effective_dim=split_scores.get('candidate_effective_dim'),
            candidate_direction_cos_to_pred=split_scores.get(
                'candidate_direction_cos_to_pred'),
            candidate_ref_proto_cos=split_scores.get('candidate_ref_proto_cos'),
            v_c_label_shuffle=split_scores.get('v_c_label_shuffle'),
            e_c_label_shuffle=split_scores.get('e_c_label_shuffle'),
            eid_label_shuffle=split_scores.get('eid_label_shuffle'),
        )
        parts[rule] = (pred, conf, label)
    return parts


def _candidate_column(arrays, key: str, positions: np.ndarray):
    if key not in arrays.files:
        return None
    values = np.asarray(arrays[key])
    if values.size == 0:
        return values
    if values.ndim != 2:
        return values
    rows = np.arange(values.shape[0])
    return values[rows, positions][:, None]


def _pred_positions(arrays) -> np.ndarray:
    pred = np.asarray(arrays['pred'], dtype=np.int64)
    candidates = np.asarray(arrays['candidate_classes'], dtype=np.int64)
    if candidates.ndim != 2:
        raise ValueError('candidate_classes must be a 2D array')
    matches = candidates == pred[:, None]
    if not np.all(matches.any(axis=1)):
        raise ValueError('all-candidate score artifact does not contain pred class')
    return np.argmax(matches, axis=1)


def derive_pred_split_scores_from_all(arrays) -> Dict:
    pred = np.asarray(arrays['pred'], dtype=np.int64)
    q_max = np.asarray(arrays['q_max'], dtype=np.float64)
    v_pred = np.asarray(arrays['v_pred'], dtype=np.float64)
    eid_pred = np.asarray(arrays['eid_pred'], dtype=np.float64)
    positions = _pred_positions(arrays)
    q_c = _candidate_column(arrays, 'q_c', positions)
    v_c = _candidate_column(arrays, 'v_c', positions)
    e_c = _candidate_column(arrays, 'e_c', positions)
    v_c_label_shuffle = _candidate_column(
        arrays, 'v_c_label_shuffle', positions)
    e_c_label_shuffle = _candidate_column(
        arrays, 'e_c_label_shuffle', positions)
    candidate_raw_grad_norm = _candidate_column(
        arrays, 'candidate_raw_grad_norm', positions)
    candidate_effective_dim = _candidate_column(
        arrays, 'candidate_effective_dim', positions)
    candidate_direction_cos_to_pred = _candidate_column(
        arrays, 'candidate_direction_cos_to_pred', positions)
    candidate_ref_proto_cos = _candidate_column(
        arrays, 'candidate_ref_proto_cos', positions)
    eid_label_shuffle = (
        e_c_label_shuffle[:, 0]
        if e_c_label_shuffle is not None and e_c_label_shuffle.ndim == 2
        else None
    )
    accept_eid = (
        (q_c[:, 0] * v_c[:, 0])
        if q_c is not None and v_c is not None and q_c.ndim == 2 and v_c.ndim == 2
        else eid_pred
    )
    return {
        'pred': pred,
        'label': np.asarray(arrays['label'], dtype=np.int64),
        'eid': eid_pred,
        'best_class': pred,
        'q_best': q_max,
        'v_best': v_pred,
        'q_max': q_max,
        'v_pred': v_pred,
        'eid_pred': eid_pred,
        'candidate_classes': pred[:, None],
        'q_c': q_c if q_c is not None else q_max[:, None],
        'v_c': v_c if v_c is not None else v_pred[:, None],
        'e_c': e_c if e_c is not None else eid_pred[:, None],
        'rank_only_scores': _candidate_column(
            arrays, 'rank_only_scores', positions),
        'same_positive_rates': _candidate_column(
            arrays, 'same_positive_rates', positions),
        'accept_eid': accept_eid,
        'candidate_raw_grad_norm': candidate_raw_grad_norm,
        'candidate_effective_dim': candidate_effective_dim,
        'candidate_direction_cos_to_pred': candidate_direction_cos_to_pred,
        'candidate_ref_proto_cos': candidate_ref_proto_cos,
        'v_c_label_shuffle': v_c_label_shuffle,
        'e_c_label_shuffle': e_c_label_shuffle,
        'eid_label_shuffle': eid_label_shuffle,
    }


def split_score_file_stem(split_kind: str, dataset_name: str) -> str:
    if split_kind == 'id':
        return 'id'
    safe_dataset = str(dataset_name).replace('/', '_')
    return f'{split_kind}_{safe_dataset}'


def split_name_from_score_file_stem(stem: str) -> str:
    if stem == 'id':
        return 'id'
    split_kind, dataset_name = stem.split('_', 1)
    return f'{split_kind}/{dataset_name}'


def clear_score_outputs(base_run_dir: Path,
                        scheme: str,
                        score_rules: List[str]) -> None:
    for rule in score_rules:
        rule_dir = score_rule_dir(base_run_dir, scheme, rule)
        scores_dir = rule_dir / 'scores'
        if scores_dir.exists():
            for path in scores_dir.glob('*.npz'):
                path.unlink()
        metrics_path = rule_dir / 'ood.csv'
        if metrics_path.exists():
            metrics_path.unlink()


def write_scheme_metrics(base_run_dir: Path,
                         scheme: str,
                         score_parts: Dict[str, Dict[str, tuple]],
                         score_rules: List[str]) -> None:
    for rule in score_rules:
        id_parts = [score_parts['id'][rule]]
        if scheme == 'fsood':
            for name in sorted(score_parts):
                if name.startswith('csid/'):
                    id_parts.append(score_parts[name][rule])
        metric_id = concat_score_tuples(id_parts)
        rows = []
        near_metrics = []
        far_metrics = []
        for name in sorted(score_parts):
            if name.startswith('nearood/'):
                metrics = metric_summary(metric_id, score_parts[name][rule])
                near_metrics.append(metrics)
                rows.append(format_metric_row(name.split('/', 1)[1], metrics))
        if near_metrics:
            rows.append(format_metric_row(
                'nearood', np.mean(np.asarray(near_metrics), axis=0)))
        for name in sorted(score_parts):
            if name.startswith('farood/'):
                metrics = metric_summary(metric_id, score_parts[name][rule])
                far_metrics.append(metrics)
                rows.append(format_metric_row(name.split('/', 1)[1], metrics))
        if far_metrics:
            rows.append(format_metric_row(
                'farood', np.mean(np.asarray(far_metrics), axis=0)))
        write_metrics_csv(score_rule_dir(base_run_dir, scheme, rule) / 'ood.csv', rows)


def save_derived_pred_score_files(source_run_dir: Path,
                                  pred_run_dir: Path,
                                  *,
                                  scheme: str,
                                  score_rules: List[str]) -> Dict[str, Dict[str, tuple]]:
    score_parts: Dict[str, Dict[str, tuple]] = {}
    for rule in score_rules:
        source_scores_dir = score_rule_dir(source_run_dir, scheme, rule) / 'scores'
        if not source_scores_dir.exists():
            raise FileNotFoundError(
                f'Cannot derive pred scores; missing {source_scores_dir}')
        for source_path in sorted(source_scores_dir.glob('*.npz')):
            with np.load(source_path, allow_pickle=True) as arrays:
                split_scores = derive_pred_split_scores_from_all(arrays)
            ood_score = ood_score_from_split_scores(split_scores, rule)
            pred, conf, label = score_tuple_from_ood(
                split_scores['pred'], ood_score, split_scores['label'])
            out_dir = score_rule_dir(pred_run_dir, scheme, rule) / 'scores'
            save_score_npz(
                out_dir / source_path.name,
                pred,
                conf,
                label,
                ood_score=ood_score,
                eid=split_scores['eid'],
                best_class=split_scores['best_class'],
                v_best=split_scores['v_best'],
                q_best=split_scores['q_best'],
                q_max=split_scores['q_max'],
                v_pred=split_scores['v_pred'],
                eid_pred=split_scores['eid_pred'],
                candidate_classes=split_scores['candidate_classes'],
                q_c=split_scores['q_c'],
                v_c=split_scores['v_c'],
                e_c=split_scores['e_c'],
                rank_only_scores=split_scores['rank_only_scores'],
                same_positive_rates=split_scores['same_positive_rates'],
                accept_eid=split_scores['accept_eid'],
                candidate_raw_grad_norm=split_scores.get('candidate_raw_grad_norm'),
                candidate_effective_dim=split_scores.get('candidate_effective_dim'),
                candidate_direction_cos_to_pred=split_scores.get(
                    'candidate_direction_cos_to_pred'),
                candidate_ref_proto_cos=split_scores.get('candidate_ref_proto_cos'),
                v_c_label_shuffle=split_scores.get('v_c_label_shuffle'),
                e_c_label_shuffle=split_scores.get('e_c_label_shuffle'),
                eid_label_shuffle=split_scores.get('eid_label_shuffle'),
            )
            split_name = split_name_from_score_file_stem(source_path.stem)
            score_parts.setdefault(split_name, {})[rule] = (pred, conf, label)
    return score_parts


def score_splits(args, net, dataloaders, ref_grad_bank: Dict, *,
                 base_run_dir: Path,
                 device: torch.device,
                 score_rules: List[str] | None = None) -> Dict:
    if score_rules is None:
        score_rules = parse_score_rules(args.score_rules)
    if getattr(args, 'overwrite', False):
        clear_score_outputs(base_run_dir, args.scheme, score_rules)
    label_shuffle_seed = (
        int(args.diagnostic_seed)
        if getattr(args, 'diagnostics', 'all') != 'off' else None
    )
    scoring_bank = prepare_ref_grad_bank_for_scoring(
        ref_grad_bank,
        device,
        label_shuffle_seed=label_shuffle_seed,
        reference_bank_chunk_size=(
            args.reference_bank_chunk_size
            if ref_grad_bank['bank_type'] != 'classifier_compact' else 0
        ),
        needs_ref_prototypes=bool(
            {'geom_proto_cos_max', 'geom_proto_eid'}.intersection(score_rules)
        ),
    )
    selected_params = None
    if scoring_bank['bank_type'] != 'classifier_compact':
        selected_params = [
            param for _, param in select_gradient_parameters(net, args.gradient_space)
        ]
    score_parts = {}
    for split_kind, dataset_name, _, _, loader in split_dataloaders(
            args.dataset, dataloaders, args.scheme):
        split_name = 'id' if split_kind == 'id' else f'{split_kind}/{dataset_name}'
        file_name = split_score_file_stem(split_kind, dataset_name)
        scores = score_loader(
            args,
            net,
            loader,
            scoring_bank,
            split_name=split_name,
            split_kind=split_kind,
            device=device,
            selected_params=selected_params,
        )
        score_parts[split_name] = save_split_scores(
            base_run_dir, args.scheme, file_name, scores, score_rules)
    write_scheme_metrics(base_run_dir, args.scheme, score_parts, score_rules)
    return {'processed_splits': sorted(score_parts)}


def reference_loader_from_manifest(args, dataloaders, reference_manifest: Dict):
    ref = load_reference_set_from_manifest(args, reference_manifest)
    return build_reference_loader(
        dataloaders['id']['train'],
        ref['selected_indices'],
        batch_size=args.reference_batch_size,
        num_workers=args.num_workers,
    )


def single_run_required_gates(gates: List[str],
                              *,
                              gradient_space: str | None = None) -> List[str]:
    excluded = {'gate05', 'gate08', 'gate09'}
    if gradient_space and gradient_space != 'classifier':
        excluded.add('gate03')
    return [
        gate for gate in gates
        if gate not in excluded
    ]


def parse_posthoc_diagnostic_gates(value: str | None, *,
                                   mode: str = 'all') -> List[str]:
    if mode == 'off':
        return []
    if value is None or str(value).strip().lower() in {'', 'all'}:
        return ['gate04', 'gate06', 'gate07', 'gate10']
    gates = parse_diagnostic_gates(value, mode=mode)
    online = sorted(set(gates).intersection({'gate01', 'gate02', 'gate03'}))
    if online:
        raise ValueError(
            'run-diagnostics is post-hoc only; online gates are not available: '
            f'{online}')
    return gates


def run_single(args) -> None:
    score_rules = parse_score_rules(args.score_rules)
    set_seed(args.seed)
    start = time.perf_counter()
    checkpoint = resolved_checkpoint(args)
    device = device_from_arg(args.device)
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
    run_id = make_run_id(args)
    base_run_dir = run_dir(args, run_id)
    if base_run_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f'Output directory already exists: {base_run_dir}. '
            'Use --overwrite or change the run configuration.')
    ensure_dir(base_run_dir)
    diagnostic_gates = parse_diagnostic_gates(
        args.diagnostic_gates,
        mode=args.diagnostics,
    )
    diagnostics_results = []
    if args.diagnostics != 'off':
        reference_loader = reference_loader_from_manifest(
            args, dataloaders, ref_manifest)
        diagnostics_results.extend(run_online_diagnostics(
            net,
            dataloaders['id']['train'],
            reference_loader,
            gradient_space=args.gradient_space,
            device=device,
            gates=diagnostic_gates,
            sample_count=args.diagnostic_samples,
            step_size=args.diagnostic_step_size,
        ))
    split_summary = score_splits(
        args,
        net,
        dataloaders,
        ref_grad_bank,
        base_run_dir=base_run_dir,
        device=device,
        score_rules=score_rules,
    )
    diag_dir = diagnostics_dir(args, run_id)
    ensure_dir(diag_dir)
    for rule in score_rules:
        summarize_score_dir(
            score_rule_dir(base_run_dir, args.scheme, rule),
            diag_dir / f'{args.scheme}_{rule}_summary.json',
        )
    diagnostics_manifest = None
    if args.diagnostics != 'off':
        diagnostics_results.extend(run_posthoc_diagnostics(
            base_run_dir,
            diag_dir,
            scheme=args.scheme,
            score_rules=score_rules,
            gates=diagnostic_gates,
            diagnostic_seed=args.diagnostic_seed,
            reference_sizes=parse_reference_sizes(args.diagnostic_reference_sizes),
            mode=args.diagnostics,
        ))
        diagnostics_manifest = write_diagnostic_results(
            diag_dir,
            diagnostics_results,
            required_gates=single_run_required_gates(
                diagnostic_gates, gradient_space=args.gradient_space),
            context={
                'run_dir': str(base_run_dir),
                'dataset': args.dataset,
                'scheme': args.scheme,
                'score_rules': score_rules,
                'validation_rule': args.validation_rule,
                'validation_temperature': float(args.validation_temperature),
                'rejection_rule': args.rejection_rule,
                'rejection_power': float(args.rejection_power),
                'diagnostics_mode': args.diagnostics,
                'diagnostic_seed': int(args.diagnostic_seed),
                'diagnostic_samples': int(args.diagnostic_samples),
                'diagnostic_step_size': float(args.diagnostic_step_size),
            },
        )
    elapsed = time.perf_counter() - start
    write_run_manifest(
        base_run_dir / 'run_manifest.json',
        args,
        run_id=run_id,
        checkpoint=checkpoint,
        reference_manifest=ref_manifest,
        ref_grad_bank_manifest=ref_grad_bank_manifest,
        score_rules=score_rules,
        diagnostics_manifest=diagnostics_manifest,
        elapsed=elapsed,
    )
    write_json(base_run_dir / 'score_summary.json', split_summary)


def run_build_reference(args) -> None:
    parse_score_rules(args.score_rules)
    set_seed(args.seed)
    device = device_from_arg(args.device)
    checkpoint = resolved_checkpoint(args)
    net = build_model(args.dataset, checkpoint).to(device)
    dataloaders = build_dataloaders(
        args.dataset,
        data_root=ROOT_DIR / 'data',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    build_reference(args, net, dataloaders, device)


def run_build_ref_grad_bank(args) -> None:
    parse_score_rules(args.score_rules)
    set_seed(args.seed)
    device = device_from_arg(args.device)
    checkpoint = resolved_checkpoint(args)
    net = build_model(args.dataset, checkpoint).to(device)
    dataloaders = build_dataloaders(
        args.dataset,
        data_root=ROOT_DIR / 'data',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    ref_config = reference_config_from_args(args)
    ref_manifest = read_json(reference_dir(args, ref_config) / 'manifest.json')
    build_ref_grad_bank(args, net, dataloaders, ref_manifest, device)


def run_score_splits(args) -> None:
    score_rules = parse_score_rules(args.score_rules)
    set_seed(args.seed)
    device = device_from_arg(args.device)
    checkpoint = resolved_checkpoint(args)
    net = build_model(args.dataset, checkpoint).to(device)
    dataloaders = build_dataloaders(
        args.dataset,
        data_root=ROOT_DIR / 'data',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    ref_config = reference_config_from_args(args)
    ref_manifest = read_json(reference_dir(args, ref_config) / 'manifest.json')
    _, ref_grad_bank = build_ref_grad_bank(
        args, net, dataloaders, ref_manifest, device)
    run_id = make_run_id(args)
    base_run_dir = run_dir(args, run_id)
    ensure_dir(base_run_dir)
    score_splits(
        args,
        net,
        dataloaders,
        ref_grad_bank,
        base_run_dir=base_run_dir,
        device=device,
        score_rules=score_rules,
    )


def parse_reference_sizes(value: str | None) -> List[int]:
    if not value:
        return []
    return [int(item) for item in parse_csv_values(value)]


def infer_diagnostics_output_dir(base_run_dir: Path,
                                 manifest: Dict,
                                 explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit)
    dataset = manifest.get('dataset', 'unknown')
    run_id = manifest.get('run_id', base_run_dir.name)
    try:
        if base_run_dir.parents[3].name == 'outputs':
            return base_run_dir.parents[4] / 'diagnostics' / dataset / run_id
    except IndexError:
        pass
    return base_run_dir / 'diagnostics'


def run_diagnostics(args) -> None:
    base_run_dir = Path(args.run_dir)
    manifest_path = base_run_dir / 'run_manifest.json'
    run_manifest = read_json(manifest_path) if manifest_path.exists() else {}
    scheme = args.scheme or run_manifest.get('scheme')
    if not scheme:
        raise ValueError('--scheme is required when run_manifest.json is absent')
    score_rules = (
        parse_score_rules(args.score_rules)
        if args.score_rules else list(
            run_manifest.get('score_rules', CLI_DEFAULT_SCORE_RULES))
    )
    diagnostic_gates = parse_posthoc_diagnostic_gates(
        args.diagnostic_gates,
        mode=args.diagnostics,
    )
    out_dir = infer_diagnostics_output_dir(base_run_dir, run_manifest, args.out_dir)
    results = run_posthoc_diagnostics(
        base_run_dir,
        out_dir,
        scheme=scheme,
        score_rules=score_rules,
        gates=diagnostic_gates,
        diagnostic_seed=args.diagnostic_seed,
        reference_sizes=parse_reference_sizes(args.diagnostic_reference_sizes),
        mode=args.diagnostics,
    )
    write_diagnostic_results(
        out_dir,
        results,
        required_gates=single_run_required_gates(
            diagnostic_gates,
            gradient_space=(
                run_manifest.get('gradient_config', {}) or {}
            ).get('gradient_space'),
        ),
        context={
            'run_dir': str(base_run_dir),
            'scheme': scheme,
            'score_rules': score_rules,
            'diagnostics_mode': args.diagnostics,
            'diagnostic_seed': int(args.diagnostic_seed),
            'rerun_posthoc_only': True,
        },
    )


def parse_int_grid(value: str | None, default: int) -> List[int]:
    if value is None or not str(value).strip():
        return [int(default)]
    return [int(item) for item in parse_csv_values(str(value))]


def parse_str_grid(value: str | None, default: str) -> List[str]:
    if value is None or not str(value).strip():
        return [str(default)]
    return parse_csv_values(str(value))


def parse_gradient_space_grid(value: str | None, default: str) -> List[str]:
    spaces = parse_str_grid(value, default)
    unknown = sorted(set(spaces) - set(GRADIENT_SPACES))
    if unknown:
        raise ValueError(f'Unknown gradient space(s): {unknown}')
    return list(dict.fromkeys(spaces))


def parse_candidate_mode_grid(value: str | None, default: str) -> List[str]:
    modes = parse_str_grid(value, default)
    unknown = sorted(set(modes) - set(CANDIDATE_MODES))
    if unknown:
        raise ValueError(f'Unknown candidate mode(s): {unknown}')
    return list(dict.fromkeys(modes))


def parse_validation_rule_grid(value: str | None, default: str) -> List[str]:
    rules = parse_str_grid(value, default)
    unknown = sorted(set(rules) - set(VALIDATION_RULES))
    if unknown:
        raise ValueError(f'Unknown validation rule(s): {unknown}')
    return list(dict.fromkeys(rules))


def parse_rejection_rule_grid(value: str | None, default: str) -> List[str]:
    rules = parse_str_grid(value, default)
    unknown = sorted(set(rules) - set(REJECTION_RULES))
    if unknown:
        raise ValueError(f'Unknown rejection rule(s): {unknown}')
    return list(dict.fromkeys(rules))


def safe_id_part(value: str) -> str:
    return ''.join(
        ch if ch.isalnum() or ch in {'-', '_', '.'} else '-'
        for ch in str(value)
    ).strip('-._') or 'x'


def default_experiment_id(args,
                          reference_sizes: List[int],
                          reference_seeds: List[int],
                          gradient_spaces: List[str],
                          candidate_modes: List[str],
                          validation_rules: List[str],
                          rejection_rules: List[str]) -> str:
    sizes = '-'.join(str(v) for v in reference_sizes)
    seeds = '-'.join(str(v) for v in reference_seeds)
    gradients = '-'.join(gradient_spaces)
    candidates = '-'.join(candidate_modes)
    validations = '-'.join(validation_rules)
    rejections = '-'.join(rejection_rules)
    rejection_power = str(args.rejection_power).replace('.', 'p')
    subset = ''
    if args.max_target_samples:
        subset = f'_subset{int(args.max_target_samples)}'
    return safe_id_part(
        f'exp_rpc{sizes}_refseed{seeds}_grad{gradients}'
        f'_cand{candidates}_val{validations}'
        f'_rej{rejections}_b{rejection_power}{subset}')


def child_experiment_args(args,
                          *,
                          experiment_id: str,
                          reference_per_class: int,
                          reference_seed: int,
                          gradient_space: str,
                          candidate_mode: str,
                          validation_rule: str,
                          rejection_rule: str):
    child = argparse.Namespace(**vars(args))
    child.command = 'run-single'
    child.reference_per_class = int(reference_per_class)
    child.reference_seed = int(reference_seed)
    child.gradient_space = gradient_space
    child.candidate_mode = candidate_mode
    child.validation_rule = validation_rule
    child.rejection_rule = rejection_rule
    child.experiment_id = experiment_id
    if child.diagnostics != 'off':
        child.diagnostics = 'all'
        child.diagnostic_gates = 'all'
    return child


def read_ood_metric_rows(run_path: Path,
                         *,
                         scheme: str,
                         score_rules: List[str],
                         run_info: Dict) -> List[Dict]:
    metric_names = ['FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC']
    rows = []
    for rule in score_rules:
        csv_path = score_rule_dir(run_path, scheme, rule) / 'ood.csv'
        if not csv_path.exists():
            continue
        with csv_path.open(newline='') as f:
            reader = csv.DictReader(f)
            for raw in reader:
                row = {
                    'run_id': run_info['run_id'],
                    'run_dir': str(run_path),
                    'reference_per_class': int(run_info['reference_per_class']),
                    'reference_seed': int(run_info['reference_seed']),
                    'gradient_space': run_info['gradient_space'],
                    'candidate_mode': run_info.get('candidate_mode', ''),
                    'validation_rule': run_info.get('validation_rule', ''),
                    'rejection_rule': run_info.get('rejection_rule', ''),
                    'rejection_power': float(
                        run_info.get('rejection_power', 1.0)),
                    'claim_scope': run_info.get('claim_scope', ''),
                    'score_rule': rule,
                    'ood_dataset': raw.get('dataset', ''),
                }
                for name in metric_names:
                    row[name] = float(raw[name]) if raw.get(name, '') != '' else ''
                rows.append(row)
    return rows


def collect_child_run_row(child, run_path: Path, action: str) -> Dict:
    manifest_path = run_path / 'run_manifest.json'
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    diagnostics = manifest.get('diagnostics_manifest') or {}
    run_id = manifest.get('run_id', make_run_id(child))
    return {
        'run_id': run_id,
        'run_dir': str(run_path),
        'run_action': action,
        'run_status': 'complete' if manifest_path.exists() else 'missing',
        'reference_per_class': int(child.reference_per_class),
        'reference_seed': int(child.reference_seed),
        'gradient_space': child.gradient_space,
        'candidate_mode': manifest.get('candidate_mode', child.candidate_mode),
        'validation_rule': (
            manifest.get('validation_config', {}) or {}
        ).get('validation_rule', child.validation_rule),
        'rejection_rule': (
            manifest.get('rejection_config', {}) or {}
        ).get('rejection_rule', child.rejection_rule),
        'rejection_power': float(
            (
                manifest.get('rejection_config', {}) or {}
            ).get('rejection_power', child.rejection_power)),
        'claim_scope': manifest.get(
            'claim_scope', candidate_claim_scope(child.candidate_mode)),
        'exact_candidate_set': bool(manifest.get('exact_candidate_set', False)),
        'resource_adjusted_candidate_set': bool(
            manifest.get('resource_adjusted_candidate_set', False)),
        'diagnostics_status': diagnostics.get('diagnostics_status', ''),
        'claim_bearing': bool(manifest.get('claim_bearing', False)),
    }


def run_or_reuse_experiment_child(child) -> tuple[Path, str]:
    child_run_dir = run_dir(child, make_run_id(child))
    manifest_path = child_run_dir / 'run_manifest.json'
    if manifest_path.exists() and not child.overwrite:
        return child_run_dir, 'reused'
    run_single(child)
    return child_run_dir, 'executed'


def copied_online_diagnostic_results(source_manifest: Dict,
                                     gates: List[str]) -> List[DiagnosticResult]:
    online_gates = set(gates).intersection({'gate01', 'gate02', 'gate03'})
    if not online_gates:
        return []
    source_diagnostics = source_manifest.get('diagnostics_manifest') or {}
    source_results = source_diagnostics.get('results') or []
    copied = []
    for raw in source_results:
        gate_id = str(raw.get('gate_id', ''))
        if gate_id not in online_gates:
            continue
        artifacts = dict(raw.get('artifacts') or {})
        artifacts['derived_from_candidate_mode'] = 'all'
        copied.append(DiagnosticResult(
            gate_id=gate_id,
            name=str(raw.get('name', '')),
            status=str(raw.get('status', '')),
            metrics=raw.get('metrics') or {},
            thresholds=raw.get('thresholds') or {},
            artifacts=artifacts,
            message=(
                str(raw.get('message', '')) +
                ' (copied from all-candidate run)'
            ).strip(),
        ))
    return copied


def can_copy_required_online_gates(source_manifest: Dict,
                                   diagnostic_gates: List[str],
                                   *,
                                   gradient_space: str) -> bool:
    required = set(single_run_required_gates(
        diagnostic_gates, gradient_space=gradient_space)).intersection(
            {'gate01', 'gate02', 'gate03'})
    if not required:
        return True
    copied = {
        result.gate_id
        for result in copied_online_diagnostic_results(source_manifest,
                                                       diagnostic_gates)
        if result.status != 'skip'
    }
    return required.issubset(copied)


def run_or_derive_pred_experiment_child(source_all_child,
                                        pred_child,
                                        source_all_run_dir: Path,
                                        score_rules: List[str]) -> tuple[Path, str]:
    pred_run_id = make_run_id(pred_child)
    pred_run_dir = run_dir(pred_child, pred_run_id)
    manifest_path = pred_run_dir / 'run_manifest.json'
    if manifest_path.exists() and not pred_child.overwrite:
        return pred_run_dir, 'reused'
    if pred_child.rejection_rule != 'off':
        run_single(pred_child)
        return pred_run_dir, 'executed'
    source_manifest_path = source_all_run_dir / 'run_manifest.json'
    if not source_manifest_path.exists():
        run_single(pred_child)
        return pred_run_dir, 'executed'
    source_manifest = read_json(source_manifest_path)
    diagnostic_gates = parse_diagnostic_gates(
        pred_child.diagnostic_gates,
        mode=pred_child.diagnostics,
    )
    if (
            pred_child.diagnostics != 'off' and
            not can_copy_required_online_gates(
                source_manifest,
                diagnostic_gates,
                gradient_space=pred_child.gradient_space)):
        run_single(pred_child)
        return pred_run_dir, 'executed'

    if pred_run_dir.exists() and not pred_child.overwrite:
        raise FileExistsError(
            f'Output directory already exists without run_manifest.json: '
            f'{pred_run_dir}. Use --overwrite or remove the partial artifact.')
    start = time.perf_counter()
    ensure_dir(pred_run_dir)
    if pred_child.overwrite:
        clear_score_outputs(pred_run_dir, pred_child.scheme, score_rules)
    score_parts = save_derived_pred_score_files(
        source_all_run_dir,
        pred_run_dir,
        scheme=pred_child.scheme,
        score_rules=score_rules,
    )
    write_scheme_metrics(pred_run_dir, pred_child.scheme, score_parts, score_rules)
    split_summary = {
        'processed_splits': sorted(score_parts),
        'derived_from_candidate_mode': 'all',
        'derived_from_run_id': source_manifest.get('run_id', make_run_id(source_all_child)),
        'derived_from_run_dir': str(source_all_run_dir),
    }

    diag_dir = diagnostics_dir(pred_child, pred_run_id)
    ensure_dir(diag_dir)
    for rule in score_rules:
        summarize_score_dir(
            score_rule_dir(pred_run_dir, pred_child.scheme, rule),
            diag_dir / f'{pred_child.scheme}_{rule}_summary.json',
        )
    diagnostics_manifest = None
    if pred_child.diagnostics != 'off':
        diagnostics_results = copied_online_diagnostic_results(
            source_manifest, diagnostic_gates)
        diagnostics_results.extend(run_posthoc_diagnostics(
            pred_run_dir,
            diag_dir,
            scheme=pred_child.scheme,
            score_rules=score_rules,
            gates=diagnostic_gates,
            diagnostic_seed=pred_child.diagnostic_seed,
            reference_sizes=parse_reference_sizes(
                pred_child.diagnostic_reference_sizes),
            mode=pred_child.diagnostics,
        ))
        diagnostics_manifest = write_diagnostic_results(
            diag_dir,
            diagnostics_results,
            required_gates=single_run_required_gates(
                diagnostic_gates, gradient_space=pred_child.gradient_space),
            context={
                'run_dir': str(pred_run_dir),
                'dataset': pred_child.dataset,
                'scheme': pred_child.scheme,
                'score_rules': score_rules,
                'validation_rule': pred_child.validation_rule,
                'validation_temperature': float(pred_child.validation_temperature),
                'rejection_rule': pred_child.rejection_rule,
                'rejection_power': float(pred_child.rejection_power),
                'diagnostics_mode': pred_child.diagnostics,
                'diagnostic_seed': int(pred_child.diagnostic_seed),
                'diagnostic_samples': int(pred_child.diagnostic_samples),
                'diagnostic_step_size': float(pred_child.diagnostic_step_size),
                'derived_from_candidate_mode': 'all',
                'derived_from_run_id': source_manifest.get('run_id'),
                'derived_from_run_dir': str(source_all_run_dir),
            },
        )
    write_run_manifest(
        pred_run_dir / 'run_manifest.json',
        pred_child,
        run_id=pred_run_id,
        checkpoint=source_manifest.get('checkpoint', resolved_checkpoint(pred_child)),
        reference_manifest=source_manifest['reference_manifest'],
        ref_grad_bank_manifest=source_manifest['gradient_manifest'],
        score_rules=score_rules,
        diagnostics_manifest=diagnostics_manifest,
        elapsed=time.perf_counter() - start,
    )
    write_json(pred_run_dir / 'score_summary.json', split_summary)
    return pred_run_dir, 'derived'


def write_experiment_manifest(path: Path,
                              args,
                              *,
                              experiment_id: str,
                              reference_sizes: List[int],
                              reference_seeds: List[int],
                              gradient_spaces: List[str],
                              candidate_modes: List[str],
                              validation_rules: List[str],
                              rejection_rules: List[str],
                              run_rows: List[Dict],
                              metric_rows: List[Dict],
                              diagnostics_manifest: Dict,
                              score_rules: List[str],
                              elapsed: float) -> None:
    required_gates = experiment_required_gates(
        reference_sizes, reference_seeds, gradient_spaces)
    reasons = []
    if args.max_target_samples:
        reasons.append('max_target_samples was set; subset experiments are non-claim-bearing')
    if not required_gates:
        reasons.append('no experiment grid axis had multiple values')
    if diagnostics_manifest.get('diagnostics_status') in {'fail', 'incomplete'}:
        reasons.append(
            f"diagnostics_status={diagnostics_manifest.get('diagnostics_status')}")
    missing = [row for row in run_rows if row['run_status'] != 'complete']
    if missing:
        reasons.append(f'{len(missing)} child run(s) were incomplete')
    non_claim_children = [
        row for row in run_rows if not bool(row.get('claim_bearing', False))
    ]
    if non_claim_children:
        reasons.append(f'{len(non_claim_children)} child run(s) were non-claim-bearing')
    manifest = {
        'artifact': 'rae_experiment',
        'dataset': args.dataset,
        'scheme': args.scheme,
        'baseline_protocol': args.baseline_protocol,
        'experiment_id': experiment_id,
        'score_rules': score_rules,
        'reference_per_class_grid': reference_sizes,
        'reference_seeds': reference_seeds,
        'gradient_spaces': gradient_spaces,
        'candidate_modes': candidate_modes,
        'validation_rules': validation_rules,
        'rejection_rules': rejection_rules,
        'rejection_power': float(args.rejection_power),
        'reference_filter': args.reference_filter,
        'reference_min_confidence': float(args.reference_min_confidence),
        'max_target_samples': args.max_target_samples,
        'child_run_count': len(run_rows),
        'metric_row_count': len(metric_rows),
        'required_gates': required_gates,
        'diagnostics_manifest': diagnostics_manifest,
        'claim_bearing': len(reasons) == 0,
        'claim_bearing_reasons': reasons,
        'elapsed_sec': elapsed,
    }
    write_json(path, manifest)


def run_all(args) -> None:
    score_rules = parse_score_rules(args.score_rules)
    start = time.perf_counter()
    reference_sizes = parse_int_grid(
        args.reference_per_class_grid, args.reference_per_class)
    reference_seeds = parse_int_grid(args.reference_seeds, args.reference_seed)
    gradient_spaces = parse_gradient_space_grid(
        args.gradient_spaces, args.gradient_space)
    candidate_modes = parse_candidate_mode_grid(
        args.candidate_modes, args.candidate_mode)
    validation_rules = parse_validation_rule_grid(
        args.validation_rules, args.validation_rule)
    rejection_rules = parse_rejection_rule_grid(
        args.rejection_rules, args.rejection_rule)
    experiment_id = args.experiment_id or default_experiment_id(
        args,
        reference_sizes,
        reference_seeds,
        gradient_spaces,
        candidate_modes,
        validation_rules,
        rejection_rules,
    )
    out_dir = experiment_dir(args, experiment_id)
    ensure_dir(out_dir)

    run_rows = []
    metric_rows = []
    for validation_rule in validation_rules:
        for rejection_rule in rejection_rules:
            for gradient_space in gradient_spaces:
                for reference_per_class in reference_sizes:
                    for reference_seed in reference_seeds:
                        all_child = None
                        all_run_dir = None
                        if 'all' in candidate_modes:
                            all_child = child_experiment_args(
                                args,
                                experiment_id=experiment_id,
                                reference_per_class=reference_per_class,
                                reference_seed=reference_seed,
                                gradient_space=gradient_space,
                                candidate_mode='all',
                                validation_rule=validation_rule,
                                rejection_rule=rejection_rule,
                            )
                            all_run_dir, action = run_or_reuse_experiment_child(
                                all_child)
                            run_row = collect_child_run_row(
                                all_child, all_run_dir, action)
                            run_rows.append(run_row)
                            metric_rows.extend(read_ood_metric_rows(
                                all_run_dir,
                                scheme=args.scheme,
                                score_rules=score_rules,
                                run_info=run_row,
                            ))

                        for candidate_mode in candidate_modes:
                            if candidate_mode == 'all':
                                continue
                            child = child_experiment_args(
                                args,
                                experiment_id=experiment_id,
                                reference_per_class=reference_per_class,
                                reference_seed=reference_seed,
                                gradient_space=gradient_space,
                                candidate_mode=candidate_mode,
                                validation_rule=validation_rule,
                                rejection_rule=rejection_rule,
                            )
                            if candidate_mode == 'pred' and all_child is not None:
                                child_run_dir, action = (
                                    run_or_derive_pred_experiment_child(
                                        all_child,
                                        child,
                                        all_run_dir,
                                        score_rules,
                                    ))
                            else:
                                child_run_dir, action = run_or_reuse_experiment_child(
                                    child)
                            run_row = collect_child_run_row(
                                child, child_run_dir, action)
                            run_rows.append(run_row)
                            metric_rows.extend(read_ood_metric_rows(
                                child_run_dir,
                                scheme=args.scheme,
                                score_rules=score_rules,
                                run_info=run_row,
                            ))

    write_csv(
        out_dir / 'runs.csv',
        run_rows,
        fieldnames=[
            'run_id', 'run_dir', 'run_action', 'run_status',
            'reference_per_class', 'reference_seed', 'gradient_space',
            'candidate_mode', 'validation_rule',
            'rejection_rule', 'rejection_power',
            'claim_scope',
            'exact_candidate_set', 'resource_adjusted_candidate_set',
            'diagnostics_status', 'claim_bearing',
        ],
    )
    write_csv(
        out_dir / 'metrics.csv',
        metric_rows,
        fieldnames=[
            'run_id', 'run_dir', 'reference_per_class', 'reference_seed',
            'gradient_space', 'candidate_mode', 'validation_rule', 'claim_scope',
            'rejection_rule', 'rejection_power',
            'score_rule', 'ood_dataset',
            'FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC',
        ],
    )
    experiment_results = run_experiment_diagnostics(
        out_dir,
        metric_rows,
        run_rows,
        reference_sizes=reference_sizes,
        reference_seeds=reference_seeds,
        gradient_spaces=gradient_spaces,
        candidate_modes=candidate_modes,
    )
    required_gates = experiment_required_gates(
        reference_sizes, reference_seeds, gradient_spaces)
    diagnostics_manifest = write_diagnostic_results(
        out_dir,
        experiment_results,
        required_gates=required_gates,
        context={
            'experiment_id': experiment_id,
            'dataset': args.dataset,
            'scheme': args.scheme,
            'score_rules': score_rules,
            'reference_per_class_grid': reference_sizes,
            'reference_seeds': reference_seeds,
            'gradient_spaces': gradient_spaces,
            'candidate_modes': candidate_modes,
            'validation_rules': validation_rules,
            'rejection_rules': rejection_rules,
            'rejection_power': float(args.rejection_power),
        },
    )
    write_experiment_manifest(
        out_dir / 'experiment_manifest.json',
        args,
        experiment_id=experiment_id,
        reference_sizes=reference_sizes,
        reference_seeds=reference_seeds,
        gradient_spaces=gradient_spaces,
        candidate_modes=candidate_modes,
        validation_rules=validation_rules,
        rejection_rules=rejection_rules,
        run_rows=run_rows,
        metric_rows=metric_rows,
        diagnostics_manifest=diagnostics_manifest,
        score_rules=score_rules,
        elapsed=time.perf_counter() - start,
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--dataset', choices=SUPPORTED_DATASETS, default='cifar10')
    parser.add_argument('--scheme', choices=SUPPORTED_SCHEMES, default='fsood')
    parser.add_argument(
        '--baseline-protocol',
        default='eval_api',
    )
    parser.add_argument('--checkpoint')
    parser.add_argument('--output-root', default='results_test/rae')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--reference-batch-size', type=int, default=256)
    parser.add_argument(
        '--reference-bank-chunk-size',
        type=int,
        default=1024,
        help=(
            'Dense last_block/all scoring reference-bank chunk size. '
            'Use 0 to load the full dense reference bank on GPU.'
        ),
    )
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument(
        '--gradient-space',
        choices=GRADIENT_SPACES,
        default='classifier',
    )
    parser.add_argument('--reference-per-class', type=int, default=16)
    parser.add_argument(
        '--reference-filter',
        choices=REFERENCE_FILTERS,
        default='correct',
    )
    parser.add_argument('--reference-min-confidence', type=float, default=0.9)
    parser.add_argument('--reference-seed', type=int, default=0)
    parser.add_argument('--rebuild-train-metadata', action='store_true')
    parser.add_argument('--rebuild-reference', action='store_true')
    parser.add_argument(
        '--rebuild-gradient-bank',
        dest='rebuild_ref_grad_bank',
        action='store_true',
    )
    parser.add_argument(
        '--candidate-mode',
        choices=CANDIDATE_MODES,
        default='all',
    )
    parser.add_argument(
        '--validation-rule',
        choices=VALIDATION_RULES,
        default=DEFAULT_VALIDATION_RULE,
        help='Reference validation rule used for V_c.',
    )
    parser.add_argument(
        '--validation-temperature',
        type=float,
        default=DEFAULT_VALIDATION_TEMPERATURE,
        help='Temperature for --validation-rule soft_margin.',
    )
    parser.add_argument(
        '--rejection-rule',
        choices=REJECTION_RULES,
        default=DEFAULT_REJECTION_RULE,
        help='Optional semantic rejection evidence rule.',
    )
    parser.add_argument(
        '--rejection-power',
        type=float,
        default=DEFAULT_REJECTION_POWER,
        help='Power for multiplying rejection conflict into final ID evidence.',
    )
    parser.add_argument(
        '--score-rules',
        default=','.join(CLI_DEFAULT_SCORE_RULES),
        help='Comma-separated subset of neglog_eid,neg_eid.',
    )
    parser.add_argument(
        '--max-target-samples',
        type=int,
        help='Smoke/subset only. Runs with this set are not claim-bearing.',
    )
    parser.add_argument(
        '--diagnostics',
        choices=('off', 'all'),
        default='all',
        help='Run RAE diagnostic gates and write diagnostics_manifest.json.',
    )
    parser.add_argument(
        '--diagnostic-gates',
        default='all',
        help=(
            'Comma-separated numeric child gates such as 1,2,3,4,6,7,10; '
            'or all. Gate 8/9 are parent run-all comparisons.'),
    )
    parser.add_argument('--diagnostic-samples', type=int, default=16)
    parser.add_argument(
        '--diagnostic-step-size',
        type=float,
        default=DEFAULT_DIAGNOSTIC_STEP_SIZE,
        help='Finite step size for online Gate 1/2 diagnostics.',
    )
    parser.add_argument('--diagnostic-seed', type=int, default=0)
    parser.add_argument(
        '--diagnostic-reference-sizes',
        default=DEFAULT_REFERENCE_PER_CLASS_GRID_ARG,
        help='Expected reference sizes for Gate 8 manifests/comparisons.',
    )


def parse_args(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description='RAE evaluator')
    subparsers = parser.add_subparsers(dest='command', required=True)

    single = subparsers.add_parser('run-single')
    add_common_args(single)

    full = subparsers.add_parser('run-all')
    add_common_args(full)
    full.add_argument(
        '--experiment-id',
        help='Optional parent experiment artifact id. Defaults to a grid-derived id.',
    )
    full.add_argument(
        '--reference-per-class-grid',
        default=DEFAULT_REFERENCE_PER_CLASS_GRID_ARG,
        help=(
            'Comma-separated reference sizes. Defaults to '
            f'{DEFAULT_REFERENCE_PER_CLASS_GRID_ARG}.'),
    )
    full.add_argument(
        '--reference-seeds',
        help='Comma-separated reference seeds, e.g. 0,1,2. Defaults to --reference-seed.',
    )
    full.add_argument(
        '--gradient-spaces',
        default='classifier,last_block',
        help='Comma-separated gradient spaces. Defaults to classifier,last_block.',
    )
    full.add_argument(
        '--candidate-modes',
        default='all',
        help='Comma-separated candidate modes. Defaults to all.',
    )
    full.add_argument(
        '--validation-rules',
        default=DEFAULT_VALIDATION_RULE,
        help='Comma-separated validation rules. Defaults to pairwise_rank.',
    )
    full.add_argument(
        '--rejection-rules',
        default=DEFAULT_REJECTION_RULE,
        help='Comma-separated rejection rules. Defaults to off.',
    )

    for name in ['build-reference', 'build-gradient-bank', 'score-splits']:
        sub = subparsers.add_parser(name)
        add_common_args(sub)

    diag = subparsers.add_parser('run-diagnostics')
    diag.add_argument('--run-dir', required=True, type=Path)
    diag.add_argument('--out-dir', type=Path)
    diag.add_argument('--scheme', choices=SUPPORTED_SCHEMES)
    diag.add_argument(
        '--score-rules',
        help='Comma-separated subset of neglog_eid,neg_eid. Defaults to run manifest.',
    )
    diag.add_argument(
        '--diagnostic-gates',
        default='all',
        help='Comma-separated numeric single-run gates such as 4,6,7,10; or all.',
    )
    diag.add_argument('--diagnostic-seed', type=int, default=0)
    diag.add_argument(
        '--diagnostic-reference-sizes',
        default=DEFAULT_REFERENCE_PER_CLASS_GRID_ARG,
    )
    diag.set_defaults(diagnostics='all')
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == 'run-single':
        run_single(args)
    elif args.command == 'run-all':
        run_all(args)
    elif args.command == 'build-reference':
        run_build_reference(args)
    elif args.command == 'build-gradient-bank':
        run_build_ref_grad_bank(args)
    elif args.command == 'score-splits':
        run_score_splits(args)
    elif args.command == 'run-diagnostics':
        run_diagnostics(args)
    else:
        raise ValueError(f'Unknown command: {args.command}')


if __name__ == '__main__':
    main(sys.argv[1:])
