"""Reference-set construction for standalone RAE."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from .artifacts import (
    ensure_dir,
    read_json,
    selected_samples_hash,
    write_csv,
    write_json,
)
from .config import CACHE_SCHEMA_VERSION, NUM_CLASSES, ReferenceConfig


TRAIN_CANDIDATE_METADATA_SCHEMA_VERSION = 2
TRAIN_CANDIDATE_METADATA_FILE = 'candidates.npz'
TRAIN_CANDIDATE_METADATA_MANIFEST_FILE = 'manifest.json'
SELECTED_SAMPLES_CSV_FILE = 'selected_samples.csv'


def train_candidate_metadata_identity(dataset: str,
                                      *,
                                      checkpoint: str,
                                      checkpoint_sha256: str | None,
                                      model_arch: str,
                                      num_classes: int) -> Dict:
    return {
        'artifact': 'rae_train_candidate_metadata',
        'schema_version': TRAIN_CANDIDATE_METADATA_SCHEMA_VERSION,
        'dataset': dataset,
        'checkpoint_resolved': str(Path(checkpoint).resolve()),
        'checkpoint_sha256': checkpoint_sha256,
        'model_arch': model_arch,
        'num_classes': int(num_classes),
        'source_split': 'id_train',
    }


def train_candidate_metadata_id(identity: Dict) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _normalize_train_candidate_metadata(metadata: Dict) -> Dict:
    metadata = dict(metadata)
    required = [
        'scan_index',
        'index',
        'labels',
        'pred',
        'conf',
        'correct',
        'image_name',
    ]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError('missing RAE train candidate metadata field(s): ' +
                         ', '.join(missing))
    normalized = {key: np.asarray(metadata[key]) for key in required}
    normalized['scan_index'] = normalized['scan_index'].astype(np.int64)
    normalized['index'] = normalized['index'].astype(np.int64)
    normalized['labels'] = normalized['labels'].astype(np.int64)
    normalized['pred'] = normalized['pred'].astype(np.int64)
    normalized['conf'] = normalized['conf'].astype(np.float64)
    normalized['correct'] = normalized['correct'].astype(np.bool_)
    normalized['image_name'] = normalized['image_name'].astype(str)
    expected = len(normalized['labels'])
    lengths = {
        key: len(normalized[key])
        for key in required
    }
    if any(length != expected for length in lengths.values()):
        raise ValueError(f'inconsistent RAE candidate field lengths: {lengths}')
    return normalized


@torch.no_grad()
def collect_train_candidates(net: torch.nn.Module,
                             train_loader,
                             *,
                             device: torch.device) -> Dict:
    net.eval()
    labels_all = []
    pred_all = []
    conf_all = []
    index_all = []
    scan_index_all = []
    image_names = []
    scan_offset = 0
    for batch in tqdm(train_loader, desc='RAE train candidates'):
        data = batch['data'].to(device)
        batch_labels = batch['label'].to(device).long()
        batch_size = int(batch_labels.numel())
        logits = net(data)
        probs = torch.softmax(logits, dim=1)
        batch_conf, batch_pred = probs.max(dim=1)
        labels_all.append(batch_labels.cpu().numpy())
        pred_all.append(batch_pred.cpu().numpy())
        conf_all.append(batch_conf.cpu().numpy())
        index_all.append(
            batch['index'].detach().cpu().numpy().astype(np.int64)
            if isinstance(batch['index'], torch.Tensor) else
            np.asarray(batch['index'], dtype=np.int64))
        scan_index_all.append(
            np.arange(scan_offset, scan_offset + batch_size, dtype=np.int64))
        image_names.append(np.asarray([str(value) for value in batch['image_name']]))
        scan_offset += batch_size
    labels = np.concatenate(labels_all).astype(np.int64)
    pred = np.concatenate(pred_all).astype(np.int64)
    conf = np.concatenate(conf_all).astype(np.float64)
    return _normalize_train_candidate_metadata({
        'scan_index': np.concatenate(scan_index_all).astype(np.int64),
        'labels': labels,
        'pred': pred,
        'conf': conf,
        'index': np.concatenate(index_all).astype(np.int64),
        'image_name': np.concatenate(image_names).astype(str),
        'correct': pred == labels,
    })


def save_train_candidate_metadata(output_dir: Path,
                                  identity: Dict,
                                  metadata: Dict) -> Dict:
    output_dir = ensure_dir(output_dir)
    metadata_id = train_candidate_metadata_id(identity)
    np.savez_compressed(
        output_dir / TRAIN_CANDIDATE_METADATA_FILE,
        scan_index=metadata['scan_index'],
        index=metadata['index'],
        labels=metadata['labels'],
        pred=metadata['pred'],
        conf=metadata['conf'],
        correct=metadata['correct'],
        image_name=metadata['image_name'],
    )
    manifest = {
        'artifact': 'rae_train_candidate_metadata',
        'schema_version': TRAIN_CANDIDATE_METADATA_SCHEMA_VERSION,
        'metadata_id': metadata_id,
        'dataset': identity['dataset'],
        'identity': identity,
        'num_candidates': int(len(metadata['labels'])),
        'fields': [
            'scan_index',
            'index',
            'labels',
            'pred',
            'conf',
            'correct',
            'image_name',
        ],
        'metadata_file': TRAIN_CANDIDATE_METADATA_FILE,
    }
    write_json(output_dir / TRAIN_CANDIDATE_METADATA_MANIFEST_FILE, manifest)
    return {
        'metadata_dir': str(output_dir),
        'metadata_path': str(output_dir / TRAIN_CANDIDATE_METADATA_FILE),
        'manifest_path': str(output_dir / TRAIN_CANDIDATE_METADATA_MANIFEST_FILE),
        'manifest': manifest,
        'identity': identity,
        'metadata_id': metadata_id,
        'metadata': metadata,
        'reused': False,
    }


def load_train_candidate_metadata(output_dir: Path, identity: Dict) -> Dict | None:
    manifest_path = output_dir / TRAIN_CANDIDATE_METADATA_MANIFEST_FILE
    metadata_path = output_dir / TRAIN_CANDIDATE_METADATA_FILE
    if not manifest_path.exists() or not metadata_path.exists():
        return None
    manifest = read_json(manifest_path)
    if manifest.get('identity') != identity:
        return None
    arrays = np.load(metadata_path, allow_pickle=False)
    metadata = _normalize_train_candidate_metadata({
        key: arrays[key] for key in arrays.files
    })
    return {
        'metadata_dir': str(output_dir),
        'metadata_path': str(metadata_path),
        'manifest_path': str(manifest_path),
        'manifest': manifest,
        'identity': identity,
        'metadata_id': manifest.get('metadata_id',
                                    train_candidate_metadata_id(identity)),
        'metadata': metadata,
        'reused': True,
    }


def load_or_build_train_candidate_metadata(output_dir: Path,
                                           identity: Dict,
                                           net: torch.nn.Module,
                                           train_loader,
                                           *,
                                           device: torch.device,
                                           rebuild: bool = False) -> Dict:
    if not rebuild:
        cached = load_train_candidate_metadata(output_dir, identity)
        if cached is not None:
            return cached
    metadata = collect_train_candidates(net, train_loader, device=device)
    return save_train_candidate_metadata(output_dir, identity, metadata)


def reference_filter_mask(candidates: Dict,
                          filter_name: str,
                          min_confidence: float) -> np.ndarray:
    mask = np.ones_like(candidates['labels'], dtype=bool)
    if filter_name in {'correct', 'correct_high_confidence'}:
        mask &= candidates['correct']
    if filter_name in {'high_confidence', 'correct_high_confidence'}:
        mask &= candidates['conf'] >= float(min_confidence)
    if filter_name not in {
            'all',
            'correct',
            'high_confidence',
            'correct_high_confidence',
    }:
        raise ValueError(f'Unknown RAE reference filter: {filter_name}')
    return mask


def sample_reference_indices(candidates: Dict,
                             config: ReferenceConfig,
                             *,
                             num_classes: int) -> Dict:
    candidates = _normalize_train_candidate_metadata(candidates)
    if config.per_class <= 0:
        raise ValueError(f'reference per_class must be positive: {config.per_class}')
    rng = np.random.RandomState(config.seed)
    mask = reference_filter_mask(
        candidates, config.filter_name, config.min_confidence)
    selected_positions = []
    counts = {}
    available_counts = {}
    shortfalls = {}
    for class_id in range(num_classes):
        positions = np.where(mask & (candidates['labels'] == class_id))[0]
        available_counts[str(class_id)] = int(positions.size)
        if positions.size < config.per_class:
            shortfalls[str(class_id)] = {
                'available': int(positions.size),
                'requested': int(config.per_class),
            }
            continue
        if positions.size > config.per_class:
            positions = rng.choice(
                positions, size=config.per_class, replace=False)
        positions = np.asarray(sorted(positions.tolist()), dtype=np.int64)
        counts[str(class_id)] = int(positions.size)
        selected_positions.extend(positions.tolist())
    if shortfalls:
        missing = ', '.join(
            f'{class_id}({info["available"]}/{info["requested"]})'
            for class_id, info in shortfalls.items())
        raise RuntimeError(
            'Not enough RAE reference samples after filtering for classes: ' +
            missing)
    selected_positions = np.asarray(selected_positions, dtype=np.int64)
    return {
        'selected_positions': selected_positions,
        'selected_indices': candidates['index'][selected_positions],
        'counts': counts,
        'available_counts': available_counts,
        'quota_satisfied': True,
        'quota_shortfalls': {},
    }


def _selected_sample_rows(candidates: Dict, selection: Dict):
    candidates = _normalize_train_candidate_metadata(candidates)
    rows = []
    for order, position in enumerate(selection['selected_positions']):
        position = int(position)
        rows.append({
            'selection_order': int(order),
            'candidate_position': position,
            'scan_index': int(candidates['scan_index'][position]),
            'dataset_index': int(candidates['index'][position]),
            'label': int(candidates['labels'][position]),
            'pred': int(candidates['pred'][position]),
            'conf': float(candidates['conf'][position]),
            'correct': bool(candidates['correct'][position]),
            'image_name': str(candidates['image_name'][position]),
        })
    return rows


def save_reference_set(output_dir: Path,
                       config: ReferenceConfig,
                       candidates: Dict,
                       selection: Dict,
                       *,
                       metadata_record: Dict,
                       dataset: str,
                       checkpoint: str,
                       checkpoint_sha256: str | None,
                       model_arch: str) -> Dict:
    ensure_dir(output_dir)
    candidates = _normalize_train_candidate_metadata(candidates)
    positions = selection['selected_positions']
    selected_indices = selection['selected_indices'].astype(np.int64)
    np.savez(
        output_dir / 'reference_set.npz',
        selected_positions=positions.astype(np.int64),
        selected_indices=selected_indices,
        labels=candidates['labels'][positions].astype(np.int64),
        pred=candidates['pred'][positions].astype(np.int64),
        conf=candidates['conf'][positions].astype(np.float64),
        image_name=candidates['image_name'][positions],
    )
    sample_rows = _selected_sample_rows(candidates, selection)
    selected_samples_path = output_dir / SELECTED_SAMPLES_CSV_FILE
    write_csv(
        selected_samples_path,
        sample_rows,
        fieldnames=[
            'selection_order',
            'candidate_position',
            'scan_index',
            'dataset_index',
            'label',
            'pred',
            'conf',
            'correct',
            'image_name',
        ],
    )
    metadata_manifest = metadata_record.get('manifest', {})
    manifest = {
        'artifact': 'rae_reference_set',
        'schema_version': CACHE_SCHEMA_VERSION,
        'dataset': dataset,
        'num_classes': NUM_CLASSES[dataset],
        'reference_config': asdict(config),
        'reference_config_id': config.id,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': checkpoint_sha256,
        'model_arch': model_arch,
        'selected_count': int(selected_indices.size),
        'selected_sample_hash': selected_samples_hash(selected_indices),
        'per_class_counts': selection['counts'],
        'per_class_available_counts': selection['available_counts'],
        'quota_per_class': int(config.per_class),
        'quota_satisfied': bool(selection.get('quota_satisfied', False)),
        'quota_shortfalls': selection.get('quota_shortfalls', {}),
        'train_candidate_metadata': {
            'metadata_id': metadata_record.get('metadata_id'),
            'metadata_dir': metadata_record.get('metadata_dir'),
            'metadata_path': metadata_record.get('metadata_path'),
            'manifest_path': metadata_record.get('manifest_path'),
            'identity': metadata_record.get('identity'),
            'reused': bool(metadata_record.get('reused', False)),
            'num_candidates': metadata_manifest.get('num_candidates'),
        },
        'selected_samples_csv': {
            'path': str(selected_samples_path),
            'num_selected': int(len(sample_rows)),
        },
        'filter_pass_count': int(
            reference_filter_mask(
                candidates, config.filter_name,
                config.min_confidence).sum()),
    }
    write_json(output_dir / 'manifest.json', manifest)
    return manifest


def build_reference_set(net: torch.nn.Module,
                        train_loader,
                        output_dir: Path,
                        config: ReferenceConfig,
                        *,
                        metadata_dir: Path,
                        candidate_metadata_identity: Dict,
                        device: torch.device,
                        rebuild_metadata: bool = False,
                        checkpoint: str,
                        checkpoint_sha256: str | None,
                        model_arch: str) -> Dict:
    metadata_record = load_or_build_train_candidate_metadata(
        metadata_dir,
        candidate_metadata_identity,
        net,
        train_loader,
        device=device,
        rebuild=rebuild_metadata,
    )
    candidates = metadata_record['metadata']
    selection = sample_reference_indices(
        candidates, config, num_classes=NUM_CLASSES[config.dataset])
    return save_reference_set(
        output_dir,
        config,
        candidates,
        selection,
        metadata_record=metadata_record,
        dataset=config.dataset,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        model_arch=model_arch,
    )


def load_reference_set(reference_dir: Path) -> Dict:
    manifest = read_json(reference_dir / 'manifest.json')
    arrays = np.load(reference_dir / 'reference_set.npz', allow_pickle=True)
    return {
        'manifest': manifest,
        'selected_indices': arrays['selected_indices'].astype(np.int64),
        'selected_positions': arrays['selected_positions'].astype(np.int64),
        'labels': arrays['labels'].astype(np.int64),
        'pred': arrays['pred'].astype(np.int64),
        'conf': arrays['conf'].astype(np.float64),
        'image_name': arrays['image_name'].astype(str),
    }


def reference_quota_shortfalls(manifest: Dict) -> Dict:
    config = manifest.get('reference_config') or {}
    requested = int(manifest.get('quota_per_class') or config.get('per_class') or 0)
    counts = manifest.get('per_class_counts') or {}
    if requested <= 0 or not counts:
        return {}
    shortfalls = {}
    for class_id, count in counts.items():
        count = int(count)
        if count < requested:
            shortfalls[str(class_id)] = {
                'available': count,
                'requested': requested,
            }
    return shortfalls


def reference_manifest_is_reusable(manifest: Dict,
                                   config: ReferenceConfig,
                                   *,
                                   require_metadata: bool = True,
                                   expected_metadata_identity: Dict | None = None,
                                   checkpoint_sha256: str | None = None,
                                   model_arch: str | None = None,
                                   num_classes: int | None = None) -> bool:
    if int(manifest.get('schema_version', 0)) != CACHE_SCHEMA_VERSION:
        return False
    if manifest.get('reference_config') != asdict(config):
        return False
    if checkpoint_sha256 is not None and manifest.get(
            'checkpoint_sha256') != checkpoint_sha256:
        return False
    if model_arch is not None and manifest.get('model_arch') != model_arch:
        return False
    if num_classes is not None and int(
            manifest.get('num_classes', -1)) != int(num_classes):
        return False
    if require_metadata and not manifest.get('train_candidate_metadata'):
        return False
    if expected_metadata_identity is not None:
        metadata = manifest.get('train_candidate_metadata') or {}
        if metadata.get('identity') != expected_metadata_identity:
            return False
    if manifest.get('quota_satisfied') is not True:
        return False
    return not reference_quota_shortfalls(manifest)
