"""Reference configuration and train metadata helpers for TARR."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
except ImportError:
    torch = None
    F = None
    DataLoader = None
    Subset = None

REFERENCE_FILTERS = [
    'all',
    'correct',
    'high_confidence',
    'correct_high_confidence',
    'correct_confidence_stratified',
]

TRAIN_CANDIDATE_METADATA_SCHEMA_VERSION = 2
TRAIN_CANDIDATE_METADATA_FILE = 'candidates.npz'
TRAIN_CANDIDATE_METADATA_MANIFEST_FILE = 'manifest.json'
SELECTED_SAMPLES_CSV_FILE = 'selected_samples.csv'
ANCHOR_SET_FILE = 'anchor_set.npz'


def _require_torch():
    if torch is None or F is None:
        raise RuntimeError('torch is required for reference tensor selection')


def _is_torch_tensor(value):
    return torch is not None and isinstance(value, torch.Tensor)


@dataclass(frozen=True)
class ReferenceConfig:
    id: str
    source: str = 'train'
    per_class: int = 4
    filter: str = 'all'
    min_confidence: float = 0.9
    seed: int = 0
    selection_policy: str = 'class_balanced_random'

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ReferenceSelection:
    data: torch.Tensor
    label: torch.Tensor
    selected_candidate_indices: np.ndarray
    selected_metadata: dict

    def __iter__(self):
        yield self.data
        yield self.label

    def to_dict(self):
        return {
            'data': self.data,
            'label': self.label,
            'selected_candidate_indices': self.selected_candidate_indices,
            'selected_metadata': self.selected_metadata,
        }


def reference_config_id(source, per_class, filter_name, min_confidence, seed):
    conf = f'{min_confidence:g}'.replace('.', 'p').replace('-', 'm')
    return f'{source}_{filter_name}_rpc{per_class}_mc{conf}_rs{seed}'


def default_reference_config(args):
    return ReferenceConfig(
        id=reference_config_id('train', args.reference_per_class,
                               args.reference_filter,
                               args.reference_min_confidence, args.seed),
        per_class=args.reference_per_class,
        filter=args.reference_filter,
        min_confidence=args.reference_min_confidence,
        seed=args.seed,
    )


def parse_reference_config(value):
    config_id = ''
    raw_fields = value
    if ':' in value:
        config_id, raw_fields = value.split(':', 1)
    fields = {}
    for item in raw_fields.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f'Invalid --reference-config field: {item}')
        key, raw_value = item.split('=', 1)
        fields[key.strip()] = raw_value.strip()
    source = fields.get('source', 'train')
    per_class = int(fields['per_class'])
    filter_name = fields.get('filter', 'all')
    if filter_name not in REFERENCE_FILTERS:
        raise ValueError(f'Unknown reference filter: {filter_name}')
    min_confidence = float(fields.get('min_confidence', 0.9))
    seed = int(fields.get('seed', 0))
    if not config_id:
        config_id = reference_config_id(source, per_class, filter_name,
                                        min_confidence, seed)
    return ReferenceConfig(config_id, source, per_class, filter_name,
                           min_confidence, seed)


def parse_reference_configs(args):
    values = getattr(args, 'reference_config', None) or []
    configs = [parse_reference_config(value)
               for value in values] if values else [default_reference_config(args)]
    ids = [config.id for config in configs]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate reference config ids: {ids}')
    return configs


def selected_reference_hash(labels, data):
    digest = hashlib.sha256()
    digest.update(labels.detach().cpu().numpy().astype('int64').tobytes())
    digest.update(data.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def candidate_indices_hash(candidate_indices):
    values = np.asarray(candidate_indices, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(values.tobytes())
    return digest.hexdigest()


def file_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_path(path):
    return str(Path(path).expanduser().resolve())


def train_candidate_metadata_identity(dataset, train_imglist_path,
                                      checkpoint_path, model_arch, num_classes,
                                      preprocessor_identity,
                                      train_imglist_sha256=None,
                                      checkpoint_sha256=None):
    """Return the stable identity for train candidate prediction metadata."""
    train_imglist_path = _resolved_path(train_imglist_path)
    checkpoint_path = _resolved_path(checkpoint_path)
    if train_imglist_sha256 is None:
        train_imglist_sha256 = file_sha256(train_imglist_path)
    if checkpoint_sha256 is None:
        checkpoint_sha256 = file_sha256(checkpoint_path)
    return {
        'schema_version': TRAIN_CANDIDATE_METADATA_SCHEMA_VERSION,
        'artifact_type': 'train_candidate_metadata',
        'dataset': str(dataset),
        'source': 'train',
        'train_imglist_path': train_imglist_path,
        'train_imglist_sha256': train_imglist_sha256,
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': checkpoint_sha256,
        'model_arch': str(model_arch),
        'num_classes': int(num_classes),
        'preprocessor_identity': str(preprocessor_identity),
    }


def train_candidate_metadata_id(identity):
    payload = json.dumps(identity, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _dataset_path_part(dataset):
    dataset = str(dataset).strip()
    if not dataset:
        raise ValueError('train candidate metadata identity is missing dataset')
    parts = Path(dataset).parts
    if Path(dataset).is_absolute() or any(part in ('', '.', '..')
                                          for part in parts):
        raise ValueError(f'invalid dataset path component: {dataset!r}')
    return dataset


def train_candidate_metadata_dir(metadata_root, identity):
    dataset = _dataset_path_part(identity.get('dataset'))
    return Path(metadata_root) / dataset / train_candidate_metadata_id(identity)


def _train_candidate_metadata_paths(metadata_root, identity):
    metadata_dir = train_candidate_metadata_dir(metadata_root, identity)
    return (
        metadata_dir,
        metadata_dir / TRAIN_CANDIDATE_METADATA_MANIFEST_FILE,
        metadata_dir / TRAIN_CANDIDATE_METADATA_FILE,
    )


def _normalize_train_candidate_metadata(metadata):
    required = [
        'scan_index',
        'dataset_index',
        'label',
        'pred',
        'confidence',
        'entropy',
        'margin',
        'energy',
        'correct',
    ]
    metadata = dict(metadata)
    if 'dataset_index' not in metadata and 'index' in metadata:
        metadata['dataset_index'] = metadata['index']
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError('missing train candidate metadata field(s): ' +
                         ', '.join(missing))
    normalized = {key: np.asarray(value) for key, value in metadata.items()}
    normalized['scan_index'] = normalized['scan_index'].astype(np.int64)
    normalized['dataset_index'] = normalized['dataset_index'].astype(np.int64)
    normalized['label'] = normalized['label'].astype(np.int64)
    normalized['pred'] = normalized['pred'].astype(np.int64)
    normalized['confidence'] = normalized['confidence'].astype(np.float64)
    normalized['entropy'] = normalized['entropy'].astype(np.float64)
    normalized['margin'] = normalized['margin'].astype(np.float64)
    normalized['energy'] = normalized['energy'].astype(np.float64)
    if 'ce_loss' in normalized:
        normalized['ce_loss'] = normalized['ce_loss'].astype(np.float64)
    else:
        normalized['ce_loss'] = np.full(
            len(normalized['label']), np.nan, dtype=np.float64)
    normalized['correct'] = normalized['correct'].astype(np.bool_)
    lengths = {key: len(normalized[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f'inconsistent candidate field lengths: {lengths}')
    if len(normalized['ce_loss']) != lengths['label']:
        raise ValueError(
            'inconsistent candidate field lengths: ce_loss has '
            f'{len(normalized["ce_loss"])} values, expected {lengths["label"]}')
    return normalized


def _train_candidate_metadata_record(metadata_dir, manifest_path, metadata_path,
                                     identity, manifest, metadata):
    metadata_id = (
        manifest.get('metadata_id') or manifest.get('candidate_id')
        or train_candidate_metadata_id(identity))
    return {
        'metadata_dir': str(metadata_dir),
        'metadata_path': str(metadata_path),
        'metadata_id': metadata_id,
        'candidate_id': metadata_id,
        'manifest_path': str(manifest_path),
        'manifest': manifest,
        'identity': manifest.get('identity', identity),
        'metadata': metadata,
        'candidates': metadata,
    }


def _load_train_candidate_metadata_exact(metadata_root, identity, paths):
    metadata_dir, manifest_path, metadata_path = paths(metadata_root, identity)
    if not manifest_path.exists() or not metadata_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('identity') != identity:
        return None
    metadata = load_train_candidate_metadata_file(metadata_path)
    return _train_candidate_metadata_record(metadata_dir, manifest_path,
                                            metadata_path, identity, manifest,
                                            metadata)


def load_train_candidate_metadata(metadata_root, identity):
    return _load_train_candidate_metadata_exact(
        metadata_root, identity, _train_candidate_metadata_paths)


def save_train_candidate_metadata(metadata_root, identity, metadata):
    metadata = _normalize_train_candidate_metadata(metadata)
    metadata_dir, manifest_path, metadata_path = _train_candidate_metadata_paths(
        metadata_root, identity)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(metadata_path, **metadata)
    metadata_id = train_candidate_metadata_id(identity)
    manifest = {
        'schema_version': TRAIN_CANDIDATE_METADATA_SCHEMA_VERSION,
        'artifact_type': 'train_candidate_metadata',
        'metadata_id': metadata_id,
        'candidate_id': metadata_id,
        'dataset': identity.get('dataset'),
        'identity': identity,
        'num_candidates': int(len(metadata['label'])),
        'fields': sorted(metadata.keys()),
        'metadata_file': TRAIN_CANDIDATE_METADATA_FILE,
        'created_at_unix': time.time(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) +
                             '\n')
    return _train_candidate_metadata_record(metadata_dir, manifest_path,
                                            metadata_path, identity, manifest,
                                            metadata)


def _batch_indices(batch, batch_size, scan_offset):
    if 'index' in batch:
        return batch['index'].detach().cpu().numpy().astype(np.int64)
    return np.arange(scan_offset, scan_offset + batch_size, dtype=np.int64)


def _batch_image_names(batch, batch_size):
    image_names = batch.get('image_name')
    if image_names is None:
        return np.asarray([''] * batch_size)
    return np.asarray(list(image_names))


def _candidate_diagnostics(logits):
    _require_torch()
    probs = torch.softmax(logits, dim=1)
    confidence, pred = torch.max(probs, dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    if probs.shape[1] > 1:
        top2 = torch.topk(probs, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
    else:
        margin = confidence
    energy = -torch.logsumexp(logits, dim=1)
    return confidence, pred, entropy, margin, energy


def build_train_candidate_metadata(net, data_loader, identity):
    """Scan the train loader once and build prediction metadata."""
    _require_torch()
    device = next(net.parameters()).device
    scan_indices = []
    dataset_indices = []
    labels = []
    preds = []
    confidences = []
    entropies = []
    margins = []
    energies = []
    ce_losses = []
    image_names = []
    scan_offset = 0

    was_training = net.training
    net.eval()
    with torch.no_grad():
        for batch in data_loader:
            data = batch['data'].to(device)
            label = batch['label'].to(device)
            batch_size = int(label.numel())
            logits = net(data)
            confidence, pred, entropy, margin, energy = _candidate_diagnostics(
                logits)
            ce_loss = F.cross_entropy(logits, label, reduction='none')

            scan_indices.append(
                np.arange(scan_offset,
                          scan_offset + batch_size,
                          dtype=np.int64))
            dataset_indices.append(_batch_indices(batch, batch_size, scan_offset))
            labels.append(label.detach().cpu().numpy().astype(np.int64))
            preds.append(pred.detach().cpu().numpy().astype(np.int64))
            confidences.append(
                confidence.detach().cpu().numpy().astype(np.float64))
            entropies.append(entropy.detach().cpu().numpy().astype(np.float64))
            margins.append(margin.detach().cpu().numpy().astype(np.float64))
            energies.append(energy.detach().cpu().numpy().astype(np.float64))
            ce_losses.append(
                ce_loss.detach().cpu().numpy().astype(np.float64))
            image_names.append(_batch_image_names(batch, batch_size))
            scan_offset += batch_size
    if was_training:
        net.train()

    label = np.concatenate(labels) if labels else np.asarray([], dtype=np.int64)
    pred = np.concatenate(preds) if preds else np.asarray([], dtype=np.int64)
    metadata = {
        'scan_index':
        np.concatenate(scan_indices)
        if scan_indices else np.asarray([], dtype=np.int64),
        'dataset_index':
        np.concatenate(dataset_indices)
        if dataset_indices else np.asarray([], dtype=np.int64),
        'label':
        label,
        'pred':
        pred,
        'confidence':
        np.concatenate(confidences)
        if confidences else np.asarray([], dtype=np.float64),
        'entropy':
        np.concatenate(entropies)
        if entropies else np.asarray([], dtype=np.float64),
        'margin':
        np.concatenate(margins)
        if margins else np.asarray([], dtype=np.float64),
        'energy':
        np.concatenate(energies)
        if energies else np.asarray([], dtype=np.float64),
        'ce_loss':
        np.concatenate(ce_losses)
        if ce_losses else np.asarray([], dtype=np.float64),
        'correct':
        pred == label,
        'image_name':
        np.concatenate(image_names) if image_names else np.asarray([]),
    }
    return _normalize_train_candidate_metadata(metadata)


def load_or_build_train_candidate_metadata(metadata_root, identity, net,
                                           data_loader, rebuild=False):
    if not rebuild:
        cached = load_train_candidate_metadata(metadata_root, identity)
        if cached is not None:
            return cached
    metadata = build_train_candidate_metadata(net, data_loader, identity)
    return save_train_candidate_metadata(metadata_root, identity, metadata)


def train_candidate_filter_mask(metadata, config):
    metadata = _normalize_train_candidate_metadata(metadata)
    if config.filter == 'all':
        return np.ones(len(metadata['label']), dtype=np.bool_)
    if config.filter == 'correct':
        return metadata['correct'].astype(np.bool_)
    if config.filter == 'high_confidence':
        return metadata['confidence'] >= config.min_confidence
    if config.filter == 'correct_high_confidence':
        return (metadata['correct'].astype(np.bool_)
                & (metadata['confidence'] >= config.min_confidence))
    if config.filter == 'correct_confidence_stratified':
        return metadata['correct'].astype(np.bool_)
    raise ValueError(f'Unknown reference filter: {config.filter}')


def _reservoir_select(indices, k, seed):
    rng = random.Random(seed)
    selected = []
    seen = 0
    for candidate in indices:
        seen += 1
        if len(selected) < k:
            selected.append(int(candidate))
        else:
            replace_idx = rng.randrange(seen)
            if replace_idx < k:
                selected[replace_idx] = int(candidate)
    return selected


def _stratified_confidence_select(metadata, indices, k, seed, class_id):
    confidence = metadata['confidence'][indices]
    order = np.lexsort((metadata['scan_index'][indices], confidence))
    ordered = indices[order]
    if k == 1:
        strata = [ordered[len(ordered) // 3: max(len(ordered) // 3,
                                                2 * len(ordered) // 3)]]
        if len(strata[0]) == 0:
            strata = [ordered]
        allocation = [1]
    elif k == 2:
        strata = np.array_split(ordered, 3)
        allocation = [0, 1, 1]
    else:
        strata = np.array_split(ordered, 3)
        base = k // 3
        allocation = [base, base, base]
        for stratum_id in [1, 2, 0]:
            if sum(allocation) >= k:
                break
            allocation[stratum_id] += 1

    rng = random.Random(f'{seed}:{class_id}:correct_confidence_stratified')
    selected = []
    for stratum, quota in zip(strata, allocation):
        if quota <= 0:
            continue
        pool = [int(value) for value in stratum]
        if len(pool) < quota:
            selected.extend(pool)
        else:
            selected.extend(rng.sample(pool, quota))

    if len(selected) < k:
        selected_set = set(selected)
        remaining = [int(value) for value in ordered
                     if int(value) not in selected_set]
        selected.extend(rng.sample(remaining, k - len(selected)))
    return selected


def _group_candidate_indices_by_class(metadata, mask, num_classes):
    filtered_indices = np.flatnonzero(mask)
    if len(filtered_indices) == 0:
        empty = np.asarray([], dtype=np.int64)
        offsets = np.zeros(num_classes, dtype=np.int64)
        return empty, offsets, offsets

    labels = metadata['label'][filtered_indices]
    order = np.lexsort((metadata['scan_index'][filtered_indices], labels))
    grouped_indices = filtered_indices[order]
    grouped_labels = labels[order]
    class_ids = np.arange(num_classes, dtype=grouped_labels.dtype)
    starts = np.searchsorted(grouped_labels, class_ids, side='left')
    ends = np.searchsorted(grouped_labels, class_ids, side='right')
    return grouped_indices, starts, ends


def _exclude_candidate_indices_from_mask(mask, excluded_candidate_indices):
    if excluded_candidate_indices is None:
        return mask, 0
    excluded = np.asarray(excluded_candidate_indices, dtype=np.int64)
    if excluded.size == 0:
        return mask, 0
    if np.any(excluded < 0) or np.any(excluded >= len(mask)):
        raise ValueError(
            'excluded candidate index outside train metadata range '
            f'0..{len(mask) - 1}')
    excluded = np.unique(excluded)
    mask = mask.copy()
    mask[excluded] = False
    return mask, int(len(excluded))


def select_train_candidate_indices(metadata,
                                   config,
                                   num_classes,
                                   excluded_candidate_indices=None):
    """Select candidate row indices for a reference config.

    Existing filters keep the previous class-wise reservoir sampling behavior.
    `correct_confidence_stratified` uses correct candidates spread across each
    class confidence distribution and does not apply `min_confidence`.
    """
    metadata = _normalize_train_candidate_metadata(metadata)
    if config.per_class <= 0:
        raise ValueError(f'per_class must be positive: {config.per_class}')
    mask = train_candidate_filter_mask(metadata, config)
    mask, excluded_count = _exclude_candidate_indices_from_mask(
        mask, excluded_candidate_indices)
    grouped_indices, starts, ends = _group_candidate_indices_by_class(
        metadata, mask, num_classes)
    selected = []
    selected_labels = []
    missing = []
    for class_id in range(num_classes):
        class_indices = grouped_indices[starts[class_id]:ends[class_id]]
        if len(class_indices) < config.per_class:
            missing.append(str(class_id))
            continue
        if config.filter == 'correct_confidence_stratified':
            class_selected = _stratified_confidence_select(
                metadata, class_indices, config.per_class, config.seed,
                class_id)
        else:
            class_selected = _reservoir_select(class_indices, config.per_class,
                                               config.seed)
        selected.extend(class_selected)
        selected_labels.extend([class_id] * len(class_selected))
    if missing:
        suffix = ' after exclusions' if excluded_count else ''
        raise RuntimeError(
            'Not enough reference samples' + suffix + ' for classes: ' +
            ', '.join(missing))
    return (
        np.asarray(selected, dtype=np.int64),
        np.asarray(selected_labels, dtype=np.int64),
    )


def selected_train_candidate_metadata(metadata, candidate_indices):
    selected = {}
    metadata = _normalize_train_candidate_metadata(metadata)
    for key, value in metadata.items():
        array = np.asarray(value)
        if len(array) == len(metadata['label']):
            selected[key] = array[candidate_indices]
    return selected


def _sample_data(sample):
    if isinstance(sample, dict):
        return sample['data']
    if isinstance(sample, (tuple, list)) and sample:
        return sample[0]
    raise TypeError(f'Unsupported dataset sample type: {type(sample)!r}')


def _load_reference_samples_by_dataset_index(data_loader, dataset_indices):
    _require_torch()
    dataset = getattr(data_loader, 'dataset', None)
    if dataset is None or not hasattr(dataset, '__getitem__'):
        raise TypeError('data_loader does not expose an indexable dataset')
    dataset_indices = [int(index) for index in dataset_indices]
    batch_size = int(getattr(data_loader, 'batch_size', 0) or 0)
    num_workers = int(getattr(data_loader, 'num_workers', 0) or 0)
    if batch_size > 1:
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'collate_fn': getattr(data_loader, 'collate_fn', None),
            'pin_memory': False,
        }
        if num_workers > 0:
            prefetch_factor = getattr(data_loader, 'prefetch_factor', None)
            loader_kwargs['prefetch_factor'] = prefetch_factor or 2
            loader_kwargs['persistent_workers'] = bool(
                getattr(data_loader, 'persistent_workers', False))
        selected_batches = []
        subset_loader = DataLoader(Subset(dataset, dataset_indices),
                                   **loader_kwargs)
        for batch in subset_loader:
            data = _sample_data(batch)
            if not isinstance(data, torch.Tensor):
                data = torch.as_tensor(data)
            selected_batches.append(data.cpu().clone())
        if not selected_batches:
            raise RuntimeError('No selected reference samples were loaded')
        return torch.cat(selected_batches, dim=0)

    selected_data = []
    for dataset_index in dataset_indices:
        sample = dataset[dataset_index]
        data = _sample_data(sample)
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        selected_data.append(data.cpu().clone())
    return torch.stack(selected_data)


def _load_reference_samples_by_scan(data_loader, selected_scan_indices):
    _require_torch()
    scan_to_output = {
        int(scan_index): output_idx
        for output_idx, scan_index in enumerate(selected_scan_indices)
    }
    selected_data = [None] * len(selected_scan_indices)
    scan_offset = 0
    remaining = len(scan_to_output)
    for batch in data_loader:
        data = batch['data'].cpu()
        batch_size = int(data.shape[0])
        for batch_idx in range(batch_size):
            scan_index = scan_offset + batch_idx
            output_idx = scan_to_output.get(scan_index)
            if output_idx is None:
                continue
            selected_data[output_idx] = data[batch_idx].clone()
            remaining -= 1
        if remaining == 0:
            break
        scan_offset += batch_size
    if remaining:
        missing = [
            str(int(scan_index))
            for scan_index, output_idx in scan_to_output.items()
            if selected_data[output_idx] is None
        ]
        raise RuntimeError(
            'Selected reference scan indices missing from loader: ' +
            ', '.join(missing))
    return torch.stack(selected_data)


def select_reference_tensors_from_train_metadata(data_loader, metadata, config,
                                                 num_classes,
                                                 excluded_candidate_indices=None):
    """Return selected reference data and labels using candidate metadata."""
    _require_torch()
    metadata = _normalize_train_candidate_metadata(metadata)
    candidate_indices, labels = select_train_candidate_indices(
        metadata, config, num_classes, excluded_candidate_indices)
    selected_scan_indices = metadata['scan_index'][candidate_indices]
    selected_dataset_indices = metadata['dataset_index'][candidate_indices]
    try:
        data = _load_reference_samples_by_dataset_index(
            data_loader, selected_dataset_indices)
    except (KeyError, IndexError, TypeError, RuntimeError, ValueError):
        data = _load_reference_samples_by_scan(data_loader, selected_scan_indices)
    return ReferenceSelection(
        data=data,
        label=torch.tensor(labels, dtype=torch.long),
        selected_candidate_indices=candidate_indices,
        selected_metadata=selected_train_candidate_metadata(
            metadata, candidate_indices),
    )


def select_reference_from_train_metadata(metadata_record, data_loader, config,
                                         num_classes):
    if not isinstance(metadata_record, dict):
        metadata_record = getattr(metadata_record, '__dict__', {})
    metadata = (metadata_record.get('metadata')
                or metadata_record.get('train_candidate_metadata')
                or metadata_record.get('candidates'))
    if metadata is None:
        metadata_path = metadata_record.get('metadata_path')
        if metadata_path is None:
            raise ValueError('train metadata record is missing metadata_path')
        metadata = load_train_candidate_metadata_file(metadata_path)
    return select_reference_tensors_from_train_metadata(data_loader, metadata,
                                                       config, num_classes)


def load_train_candidate_metadata_file(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = {key: data[key] for key in data.files}
    return _normalize_train_candidate_metadata(metadata)


def _csv_scalar(value):
    if _is_torch_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        else:
            return json.dumps(value.tolist(), sort_keys=True)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, (bool, int, float, str)):
        return value
    if value is None:
        return ''
    return str(value)


def _selection_count(selected_metadata, selected_candidate_indices,
                     selected_labels):
    counts = []
    if selected_candidate_indices is not None:
        counts.append(len(selected_candidate_indices))
    if selected_labels is not None:
        counts.append(len(selected_labels))
    for value in selected_metadata.values():
        array = np.asarray(value)
        if array.ndim > 0:
            counts.append(len(array))
    return max(counts) if counts else 0


def selected_samples_csv_rows(selected_metadata,
                              selected_candidate_indices=None,
                              selected_labels=None):
    """Return rows suitable for selected_samples.csv."""
    selected_metadata = {
        key: np.asarray(value)
        for key, value in (selected_metadata or {}).items()
    }
    row_count = _selection_count(selected_metadata, selected_candidate_indices,
                                 selected_labels)
    rows = []
    for row_index in range(row_count):
        row = {'selection_order': row_index}
        if selected_candidate_indices is not None:
            row['selected_candidate_index'] = _csv_scalar(
                np.asarray(selected_candidate_indices)[row_index])
        if selected_labels is not None:
            row['selected_label'] = _csv_scalar(
                np.asarray(selected_labels)[row_index])
        for key, values in selected_metadata.items():
            if values.ndim == 0 or len(values) <= row_index:
                continue
            row[key] = _csv_scalar(values[row_index])
        rows.append(row)
    return rows


def _selected_samples_fieldnames(rows):
    preferred = [
        'selection_order',
        'selected_candidate_index',
        'selected_label',
        'preview_path',
        'scan_index',
        'dataset_index',
        'image_name',
        'label',
        'pred',
        'correct',
        'confidence',
        'entropy',
        'margin',
        'energy',
        'ce_loss',
    ]
    present = {key for row in rows for key in row}
    ordered = [key for key in preferred if key in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered or preferred


def write_selected_samples_csv(path,
                               selected_metadata,
                               selected_candidate_indices=None,
                               selected_labels=None,
                               row_updates=None):
    """Write selected reference rows to selected_samples.csv."""
    path = Path(path)
    rows = selected_samples_csv_rows(
        selected_metadata,
        selected_candidate_indices=selected_candidate_indices,
        selected_labels=selected_labels,
    )
    if row_updates is not None:
        if len(row_updates) != len(rows):
            raise ValueError(
                'row_updates length does not match selected_samples rows: '
                f'{len(row_updates)} != {len(rows)}')
        for row, updates in zip(rows, row_updates):
            row.update(updates)
    fieldnames = _selected_samples_fieldnames(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        'path': str(path),
        'num_selected': len(rows),
        'fields': fieldnames,
    }


def write_reference_set_selected_samples_csv(path, metadata, candidate_indices,
                                             selected_labels=None):
    selected_metadata = selected_train_candidate_metadata(metadata,
                                                          candidate_indices)
    return write_selected_samples_csv(
        path,
        selected_metadata,
        selected_candidate_indices=candidate_indices,
        selected_labels=selected_labels,
    )


def _safe_preview_basename(value):
    basename = Path(str(value)).name
    safe = ''.join(ch if ch.isalnum() or ch in '.-_' else '_' for ch in basename)
    return safe or 'sample'


def _preview_source_path(data_root, train_data_dir, image_name):
    image_name = str(image_name)
    if Path(image_name).is_absolute():
        raise ValueError(f'preview image_name must be relative: {image_name}')
    return Path(data_root) / train_data_dir / image_name


def _print_json_or_line(summary, print_json, line):
    if print_json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        print(line)


def _format_train_metadata_line(record):
    manifest = record.get('manifest') or {}
    identity = record.get('identity') or manifest.get('identity') or {}
    action = 'reused' if record.get('reused') else 'built'
    return (
        f"[train-candidate-metadata] {action} "
        f"id={record.get('candidate_id')} "
        f"dataset={identity.get('dataset')} "
        f"n={manifest.get('num_candidates')} "
        f"path={record.get('metadata_dir')}")


def _format_reference_set_line(record, config):
    manifest = record.get('manifest') or {}
    preview = record.get('preview') or manifest.get('preview') or {}
    action = 'reused' if record.get('reused') else 'built'
    return (
        f"[reference-set] {action} "
        f"id={record.get('reference_set_id')} "
        f"config={config.id} "
        f"seed={config.seed} "
        f"n={manifest.get('num_reference')} "
        f"preview={preview.get('num_copied', 0) if preview.get('enabled') else 0} "
        f"path={record.get('reference_set_dir')}")


def _format_anchor_set_line(record, config):
    manifest = record.get('manifest') or {}
    preview = record.get('preview') or manifest.get('preview') or {}
    action = 'reused' if record.get('reused') else 'built'
    return (
        f"[anchor-set] {action} "
        f"id={record.get('anchor_set_id')} "
        f"config={config.id} "
        f"seed={config.seed} "
        f"n={manifest.get('num_anchor')} "
        f"probe={manifest.get('probe_reference_set_id')} "
        f"preview={preview.get('num_copied', 0) if preview.get('enabled') else 0} "
        f"path={record.get('anchor_set_dir')}")


def write_reference_preview(preview_dir,
                            data_root,
                            train_data_dir,
                            selected_metadata,
                            selected_labels,
                            preview_per_class):
    """Copy selected reference images into preview/class_* folders."""
    preview_dir = Path(preview_dir)
    selected_metadata = {
        key: np.asarray(value)
        for key, value in (selected_metadata or {}).items()
    }
    selected_labels = np.asarray(selected_labels)
    if 'image_name' not in selected_metadata:
        raise ValueError('selected metadata is missing image_name for preview')
    preview_per_class = int(preview_per_class)
    if preview_per_class < 0:
        raise ValueError('--preview-per-class must be non-negative')
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    copied_by_class = {}
    row_updates = [{'preview_path': ''} for _ in range(len(selected_labels))]
    num_copied = 0
    for row_index, label in enumerate(selected_labels):
        class_id = int(label)
        class_count = copied_by_class.get(class_id, 0)
        if preview_per_class and class_count >= preview_per_class:
            continue
        image_name = selected_metadata['image_name'][row_index]
        source = _preview_source_path(data_root, train_data_dir, image_name)
        if not source.exists():
            raise FileNotFoundError(f'preview source image not found: {source}')
        pred = int(selected_metadata.get('pred', selected_labels)[row_index])
        confidence = float(
            selected_metadata.get('confidence',
                                  np.full(len(selected_labels), np.nan))[row_index])
        correct = bool(
            selected_metadata.get('correct',
                                  np.zeros(len(selected_labels), dtype=bool))[row_index])
        class_dir = preview_dir / f'class_{class_id:03d}'
        class_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f'{class_count:03d}_label{class_id}_pred{pred}'
            f'_conf{confidence:.3f}_correct{int(correct)}_'
            f'{_safe_preview_basename(image_name)}')
        target = class_dir / filename
        shutil.copy2(source, target)
        row_updates[row_index]['preview_path'] = str(target.relative_to(
            preview_dir.parent))
        copied_by_class[class_id] = class_count + 1
        num_copied += 1
    return {
        'enabled': True,
        'per_class': preview_per_class,
        'path': str(preview_dir.relative_to(preview_dir.parent)),
        'num_copied': num_copied,
        'row_updates': row_updates,
    }


def select_reference_tensors_from_metadata(data_loader, metadata, config,
                                           num_classes):
    return select_reference_tensors_from_train_metadata(data_loader, metadata,
                                                       config, num_classes)


def select_reference_from_train_candidate_metadata(train_metadata, data_loader,
                                                   config, num_classes):
    return select_reference_from_train_metadata(train_metadata, data_loader,
                                               config, num_classes)


def _identity_digest(identity):
    payload = json.dumps(identity, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def anchor_set_id(identity):
    return _identity_digest(identity)


def anchor_set_root(args):
    root = getattr(args, 'anchor_set_root', None)
    if root:
        return Path(root)
    return Path(args.output_root) / 'anchor_sets'


def _load_train_metadata_record_from_path(metadata_path):
    metadata_path = Path(metadata_path).expanduser().resolve()
    manifest_path = metadata_path.parent / TRAIN_CANDIDATE_METADATA_MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(
            f'train_candidate_metadata manifest not found: {manifest_path}')
    with manifest_path.open() as f:
        metadata_manifest = json.load(f)
    metadata = load_train_candidate_metadata_file(metadata_path)
    record = {
        'metadata_dir': metadata_path.parent,
        'metadata_path': metadata_path,
        'manifest_path': manifest_path,
        'manifest': metadata_manifest,
        'identity': metadata_manifest.get('identity'),
        'candidate_id': metadata_manifest.get('candidate_id'),
        'reused': True,
    }
    return metadata, record


def _parse_selected_candidate_index(raw_value, csv_path, line_number):
    value = str(raw_value).strip()
    if not value:
        raise ValueError(
            f'missing selected_candidate_index in {csv_path}:{line_number}')
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f'invalid selected_candidate_index in {csv_path}:{line_number}: '
            f'{value!r}') from exc
    if parsed < 0:
        raise ValueError(
            f'negative selected_candidate_index in {csv_path}:{line_number}: '
            f'{parsed}')
    return parsed


def load_selected_candidate_indices_csv(path):
    path = Path(path)
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if 'selected_candidate_index' not in fieldnames:
            raise ValueError(
                f'{path} is missing selected_candidate_index column')
        values = [
            _parse_selected_candidate_index(
                row.get('selected_candidate_index'), path, line_number)
            for line_number, row in enumerate(reader, start=2)
        ]
    return np.asarray(values, dtype=np.int64)


def _probe_reference_set_id(manifest):
    probe_id = manifest.get('reference_set_id')
    if probe_id:
        return probe_id
    identity = manifest.get('identity')
    if identity:
        return _identity_digest(identity)
    raise ValueError('probe reference_set manifest is missing reference_set_id')


def _probe_train_candidate_id(manifest):
    identity = manifest.get('identity') or {}
    candidate_info = identity.get('train_candidate_metadata') or {}
    return candidate_info.get('candidate_id')


def load_probe_reference_set_info(probe_reference_set_dir):
    probe_dir = Path(probe_reference_set_dir).expanduser().resolve()
    selected_samples_path = probe_dir / SELECTED_SAMPLES_CSV_FILE
    manifest_path = probe_dir / 'manifest.json'
    if not selected_samples_path.exists():
        raise FileNotFoundError(
            f'probe selected_samples.csv not found: {selected_samples_path}')
    if not manifest_path.exists():
        raise FileNotFoundError(
            f'probe reference_set manifest not found: {manifest_path}')
    with manifest_path.open() as f:
        manifest = json.load(f)
    selected_candidate_indices = load_selected_candidate_indices_csv(
        selected_samples_path)
    excluded_indices = np.unique(selected_candidate_indices)
    return {
        'probe_reference_set_dir': probe_dir,
        'selected_samples_path': selected_samples_path,
        'manifest_path': manifest_path,
        'manifest': manifest,
        'probe_reference_set_id': _probe_reference_set_id(manifest),
        'selected_candidate_indices': selected_candidate_indices,
        'excluded_candidate_indices': excluded_indices,
        'excluded_probe_selected_candidate_hash':
        candidate_indices_hash(selected_candidate_indices),
        'excluded_probe_selected_candidate_count':
        int(len(selected_candidate_indices)),
        'excluded_probe_unique_candidate_count': int(len(excluded_indices)),
        'probe_train_candidate_id': _probe_train_candidate_id(manifest),
    }


def _validate_probe_train_metadata(probe_info, train_metadata_record):
    probe_candidate_id = probe_info.get('probe_train_candidate_id')
    current_candidate_id = train_metadata_record.get('candidate_id')
    if (probe_candidate_id and current_candidate_id
            and probe_candidate_id != current_candidate_id):
        raise ValueError(
            '--probe-reference-set-dir was built from train_candidate_metadata '
            f'{probe_candidate_id}, but --metadata uses {current_candidate_id}')


def _feature_backed_reference_bank(eval_mod,
                                   net,
                                   reference_data,
                                   reference_label,
                                   batch_size,
                                   progress,
                                   progress_desc):
    features = []
    losses = []
    confidences = []
    predictions = []
    entropies = []
    margins = []
    energies = []
    with torch.no_grad():
        iterator = range(0, reference_label.numel(), batch_size)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=progress_desc)
        for start in iterator:
            end = start + batch_size
            logits, feature = net(reference_data[start:end], return_feature=True)
            diag = eval_mod.logit_diagnostics(logits)
            loss = F.cross_entropy(
                logits,
                reference_label[start:end],
                reduction='none',
            )
            features.append(feature.detach())
            losses.append(loss.detach())
            confidences.append(diag['conf'].detach())
            predictions.append(diag['pred'].detach())
            entropies.append(diag['entropy'].detach())
            margins.append(diag['margin'].detach())
            energies.append(diag['energy'].detach())
    prediction = torch.cat(predictions)
    return {
        'features': torch.cat(features),
        'label': reference_label,
        'base_reference_sample_diag': {
            'loss': torch.cat(losses),
            'confidence': torch.cat(confidences),
            'entropy': torch.cat(entropies),
            'margin': torch.cat(margins),
            'energy': torch.cat(energies),
            'prediction': prediction,
            'correct': prediction == reference_label,
        },
    }


def _feature_backed_artifact_payload(bank):
    payload = {
        'features':
        bank['features'].detach().cpu().numpy(),
        'labels':
        bank['label'].detach().cpu().numpy().astype(np.int64),
        'selected_reference_hash':
        np.asarray(bank['selected_reference_hash']),
        'base_reference_loss':
        bank['base_reference_sample_diag']['loss'].detach().cpu().numpy(),
        'base_reference_confidence':
        bank['base_reference_sample_diag']
        ['confidence'].detach().cpu().numpy(),
        'base_reference_entropy':
        bank['base_reference_sample_diag']['entropy'].detach().cpu().numpy(),
        'base_reference_margin':
        bank['base_reference_sample_diag']['margin'].detach().cpu().numpy(),
        'base_reference_energy':
        bank['base_reference_sample_diag']['energy'].detach().cpu().numpy(),
        'base_reference_prediction':
        bank['base_reference_sample_diag']
        ['prediction'].detach().cpu().numpy(),
        'base_reference_correct':
        bank['base_reference_sample_diag']['correct'].detach().cpu().numpy(),
    }
    metadata = bank.get('selected_metadata') or {}
    for source, target in (
        ('scan_index', 'selected_scan_index'),
        ('dataset_index', 'selected_dataset_index'),
        ('index', 'selected_dataset_index'),
    ):
        if source in metadata and target not in payload:
            payload[target] = np.asarray(metadata[source])
    return payload


def _anchor_set_matches(output_dir, identity):
    manifest_path = Path(output_dir) / 'manifest.json'
    if not manifest_path.exists():
        return False
    with manifest_path.open() as f:
        manifest = json.load(f)
    if manifest.get('identity') != identity:
        return False
    return (Path(output_dir) / ANCHOR_SET_FILE).exists()


def load_anchor_set(output_dir, identity):
    output_dir = Path(output_dir)
    if not _anchor_set_matches(output_dir, identity):
        return None
    with np.load(output_dir / ANCHOR_SET_FILE, allow_pickle=False) as data:
        bank = {key: data[key] for key in data.files}
    with (output_dir / 'manifest.json').open() as f:
        manifest = json.load(f)
    return {
        'anchor_set_dir': output_dir,
        'anchor_set_path': output_dir / ANCHOR_SET_FILE,
        'manifest_path': output_dir / 'manifest.json',
        'manifest': manifest,
        'identity': identity,
        'bank': bank,
        'reused': True,
    }


def save_anchor_set(output_dir, identity, bank):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _feature_backed_artifact_payload(bank)
    np.savez_compressed(output_dir / ANCHOR_SET_FILE, **payload)
    set_id = anchor_set_id(identity)
    manifest = {
        'schema_version': 1,
        'artifact_type': 'anchor_set',
        'identity': identity,
        'anchor_set_id': set_id,
        'probe_reference_set_id': identity['probe_reference_set_id'],
        'excluded_probe_selected_candidate_hash':
        identity['excluded_probe_selected_candidate_hash'],
        'anchor_probe_disjoint': bool(identity['anchor_probe_disjoint']),
        'num_anchor': int(bank['label'].numel()),
        'created_at_unix': time.time(),
        'fields': sorted(payload.keys()),
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return {
        'anchor_set_dir': output_dir,
        'anchor_set_path': output_dir / ANCHOR_SET_FILE,
        'manifest_path': output_dir / 'manifest.json',
        'manifest': manifest,
        'identity': identity,
        'reused': False,
    }


def _cli_build_train_metadata(args):
    from argparse import Namespace
    from scripts_my.tarr import eval as eval_mod

    if args.train_candidate_batch_size <= 0:
        raise ValueError('--train-candidate-batch-size must be positive.')
    eval_mod.set_seed(args.seed)
    checkpoint = args.checkpoint or eval_mod.DEFAULT_CHECKPOINT[args.dataset]
    net = eval_mod.build_model(args.dataset)
    net.load_state_dict(eval_mod.load_checkpoint(checkpoint))
    net.cuda()
    net.eval()
    stage_args = Namespace(
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        train_candidate_metadata_root=args.train_candidate_metadata_root,
        train_candidate_batch_size=args.train_candidate_batch_size,
        batch_size=args.train_candidate_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        rebuild_train_candidate_metadata=args.rebuild_train_candidate_metadata,
        no_progress=args.no_progress,
    )
    loader = eval_mod.make_runtime_train_loader(
        args.dataset,
        eval_mod.train_candidate_batch_size(stage_args),
        args.num_workers,
    )
    record = eval_mod.load_or_build_train_candidate_metadata(
        net, loader, stage_args)
    summary = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in record.items()
        if key != 'metadata'
    }
    _print_json_or_line(
        summary,
        args.print_json,
        _format_train_metadata_line(summary),
    )
    return 0


def _cli_build_reference_set(args):
    from argparse import Namespace
    from scripts_my.tarr import eval as eval_mod

    metadata = load_train_candidate_metadata_file(args.metadata)
    config = parse_reference_config(args.reference_config)
    if args.runtime_mode != 'classifier_feature_cache':
        raise ValueError(
            'build-reference-set currently writes feature-backed reference_set '
            'artifacts and requires --runtime-mode classifier_feature_cache.')
    checkpoint = args.checkpoint or eval_mod.DEFAULT_CHECKPOINT[args.dataset]
    metadata_path = Path(args.metadata).expanduser().resolve()
    manifest_path = metadata_path.parent / TRAIN_CANDIDATE_METADATA_MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(
            f'train_candidate_metadata manifest not found: {manifest_path}')
    with manifest_path.open() as f:
        metadata_manifest = json.load(f)
    train_metadata_record = {
        'metadata_dir': metadata_path.parent,
        'metadata_path': metadata_path,
        'manifest_path': manifest_path,
        'manifest': metadata_manifest,
        'identity': metadata_manifest.get('identity'),
        'candidate_id': metadata_manifest.get('candidate_id'),
        'reused': True,
    }

    eval_mod.set_seed(args.seed)
    net = eval_mod.build_model(args.dataset)
    net.load_state_dict(eval_mod.load_checkpoint(checkpoint))
    net.cuda()
    net.eval()
    classifier = eval_mod.classifier_layer(net)
    classifier_name = eval_mod.classifier_layer_name(net)
    loader = eval_mod.make_runtime_train_loader(
        args.dataset,
        args.reference_set_batch_size,
        args.num_workers,
    )
    selection = select_reference_tensors_from_train_metadata(
        loader, metadata, config, eval_mod.NUM_CLASSES[args.dataset])
    reference_data = selection.data.cuda(non_blocking=True)
    reference_label = selection.label.cuda(non_blocking=True)
    selected_hash = selected_reference_hash(selection.label, selection.data)

    stage_args = Namespace(
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        reference_set_root=args.reference_set_root,
        seed=args.seed,
    )
    identity = eval_mod.reference_set_identity(
        stage_args,
        train_metadata_record,
        config,
        selected_hash,
        classifier_name,
        args.runtime_mode,
    )
    set_id = eval_mod.reference_set_id(identity)
    output_dir = (eval_mod.reference_set_root(stage_args) / args.dataset /
                  config.id / f'seed{config.seed}' / set_id)
    train_spec = eval_mod.dataset_spec(args.dataset, 'id', 'train')
    data_root = eval_mod.ROOT_DIR / 'data'

    def finalize_reference_set_artifacts(record, reused):
        preview_summary = {'enabled': False}
        row_updates = [{
            'preview_path': ''
        } for _ in range(len(selection.label))]
        if args.write_preview:
            preview_summary = write_reference_preview(
                output_dir / 'preview',
                data_root,
                train_spec['data_dir'],
                selection.selected_metadata,
                selection.label.detach().cpu().numpy(),
                args.preview_per_class,
            )
            row_updates = preview_summary.pop('row_updates')
        csv_summary = write_selected_samples_csv(
            output_dir / SELECTED_SAMPLES_CSV_FILE,
            selection.selected_metadata,
            selected_candidate_indices=selection.selected_candidate_indices,
            selected_labels=selection.label.detach().cpu().numpy(),
            row_updates=row_updates,
        )
        manifest_path = output_dir / 'manifest.json'
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        manifest['selected_samples_csv'] = csv_summary
        manifest['preview'] = preview_summary
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) +
                                 '\n')
        record = dict(record)
        record['manifest'] = manifest
        record['selected_samples_csv'] = csv_summary
        record['preview'] = preview_summary
        record['reused'] = bool(reused)
        return record

    if (not args.rebuild_reference_set
            and eval_mod.load_reference_set(output_dir, identity) is not None):
        summary = {
            'reference_set_dir': str(output_dir),
            'reference_set_path': str(output_dir / eval_mod.REFERENCE_SET_FILE),
            'manifest_path': str(output_dir / 'manifest.json'),
            'reference_set_id': set_id,
            'identity': identity,
            'reused': True,
        }
        summary = finalize_reference_set_artifacts(summary, reused=True)
        _print_json_or_line(
            summary,
            args.print_json,
            _format_reference_set_line(summary, config),
        )
        return 0

    bank = _feature_backed_reference_bank(
        eval_mod,
        net,
        reference_data,
        reference_label,
        args.reference_set_batch_size,
        args.progress,
        f'Build TARR reference_set {config.id}',
    )
    bank['selected_reference_hash'] = selected_hash
    bank['selected_metadata'] = selection.selected_metadata
    record = eval_mod.save_reference_set(output_dir, identity, bank)
    record['reference_set_id'] = set_id
    record = finalize_reference_set_artifacts(record, reused=False)
    _print_json_or_line(
        record,
        args.print_json,
        _format_reference_set_line(record, config),
    )
    return 0


def _cli_build_anchor_set(args):
    from argparse import Namespace
    from scripts_my.tarr import eval as eval_mod

    metadata, train_metadata_record = _load_train_metadata_record_from_path(
        args.metadata)
    config = parse_reference_config(args.reference_config)
    if args.runtime_mode != 'classifier_feature_cache':
        raise ValueError(
            'build-anchor-set currently writes feature-backed anchor_set '
            'artifacts and requires --runtime-mode classifier_feature_cache.')

    probe_info = load_probe_reference_set_info(args.probe_reference_set_dir)
    _validate_probe_train_metadata(probe_info, train_metadata_record)

    checkpoint = args.checkpoint or eval_mod.DEFAULT_CHECKPOINT[args.dataset]
    eval_mod.set_seed(args.seed)
    net = eval_mod.build_model(args.dataset)
    net.load_state_dict(eval_mod.load_checkpoint(checkpoint))
    net.cuda()
    net.eval()
    classifier_name = eval_mod.classifier_layer_name(net)
    loader = eval_mod.make_runtime_train_loader(
        args.dataset,
        args.anchor_set_batch_size,
        args.num_workers,
    )
    selection = select_reference_tensors_from_train_metadata(
        loader,
        metadata,
        config,
        eval_mod.NUM_CLASSES[args.dataset],
        excluded_candidate_indices=probe_info['excluded_candidate_indices'],
    )
    overlap = np.intersect1d(selection.selected_candidate_indices,
                             probe_info['excluded_candidate_indices'])
    if len(overlap):
        raise RuntimeError(
            'anchor selection overlaps probe selected_candidate_index rows: ' +
            ', '.join(str(int(value)) for value in overlap[:20]))

    anchor_probe_disjoint = True
    reference_data = selection.data.cuda(non_blocking=True)
    reference_label = selection.label.cuda(non_blocking=True)
    selected_hash = selected_reference_hash(selection.label, selection.data)

    stage_args = Namespace(
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        seed=args.seed,
    )
    identity = eval_mod.reference_set_identity(
        stage_args,
        train_metadata_record,
        config,
        selected_hash,
        classifier_name,
        args.runtime_mode,
    )
    identity = dict(identity)
    identity['artifact_type'] = 'anchor_set'
    identity['probe_reference_set_id'] = probe_info['probe_reference_set_id']
    identity['excluded_probe_selected_candidate_hash'] = (
        probe_info['excluded_probe_selected_candidate_hash'])
    identity['anchor_probe_disjoint'] = anchor_probe_disjoint
    identity['excluded_probe_selected_candidate_count'] = (
        probe_info['excluded_probe_selected_candidate_count'])

    set_id = anchor_set_id(identity)
    output_dir = (anchor_set_root(args) / args.dataset / config.id /
                  f'seed{config.seed}' / set_id)
    train_spec = eval_mod.dataset_spec(args.dataset, 'id', 'train')
    data_root = eval_mod.ROOT_DIR / 'data'

    def finalize_anchor_set_artifacts(record, reused):
        preview_summary = {'enabled': False}
        row_updates = [{
            'preview_path': ''
        } for _ in range(len(selection.label))]
        if args.write_preview:
            preview_summary = write_reference_preview(
                output_dir / 'preview',
                data_root,
                train_spec['data_dir'],
                selection.selected_metadata,
                selection.label.detach().cpu().numpy(),
                args.preview_per_class,
            )
            row_updates = preview_summary.pop('row_updates')
        csv_summary = write_selected_samples_csv(
            output_dir / SELECTED_SAMPLES_CSV_FILE,
            selection.selected_metadata,
            selected_candidate_indices=selection.selected_candidate_indices,
            selected_labels=selection.label.detach().cpu().numpy(),
            row_updates=row_updates,
        )
        manifest_path = output_dir / 'manifest.json'
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        manifest['artifact_type'] = 'anchor_set'
        manifest['anchor_set_id'] = set_id
        manifest['probe_reference_set_id'] = probe_info['probe_reference_set_id']
        manifest['excluded_probe_selected_candidate_hash'] = (
            probe_info['excluded_probe_selected_candidate_hash'])
        manifest['anchor_probe_disjoint'] = anchor_probe_disjoint
        manifest['selected_samples_csv'] = csv_summary
        manifest['preview'] = preview_summary
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) +
                                 '\n')
        record = dict(record)
        record['manifest'] = manifest
        record['selected_samples_csv'] = csv_summary
        record['preview'] = preview_summary
        record['anchor_set_id'] = set_id
        record['reused'] = bool(reused)
        return record

    if (not args.rebuild_anchor_set
            and load_anchor_set(output_dir, identity) is not None):
        summary = {
            'anchor_set_dir': str(output_dir),
            'anchor_set_path': str(output_dir / ANCHOR_SET_FILE),
            'manifest_path': str(output_dir / 'manifest.json'),
            'anchor_set_id': set_id,
            'identity': identity,
            'reused': True,
        }
        summary = finalize_anchor_set_artifacts(summary, reused=True)
        _print_json_or_line(
            summary,
            args.print_json,
            _format_anchor_set_line(summary, config),
        )
        return 0

    bank = _feature_backed_reference_bank(
        eval_mod,
        net,
        reference_data,
        reference_label,
        args.anchor_set_batch_size,
        args.progress,
        f'Build TARR anchor_set {config.id}',
    )
    bank['selected_reference_hash'] = selected_hash
    bank['selected_metadata'] = selection.selected_metadata
    record = save_anchor_set(output_dir, identity, bank)
    record['anchor_set_id'] = set_id
    record = finalize_anchor_set_artifacts(record, reused=False)
    _print_json_or_line(
        record,
        args.print_json,
        _format_anchor_set_line(record, config),
    )
    return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    train = subparsers.add_parser(
        'build-train-metadata',
        help='build train_candidate_metadata for a dataset/checkpoint',
    )
    train.add_argument(
        '--dataset',
        required=True,
        choices=['cifar10', 'cifar100', 'imagenet', 'imagenet200'],
    )
    train.add_argument('--checkpoint')
    train.add_argument('--output-root', default='results_test/tarr')
    train.add_argument(
        '--train-candidate-metadata-root',
        help=('Root for canonical metadata artifacts. Defaults to '
              '<output-root>/train_candidate_metadata.'),
    )
    train.add_argument('--train-candidate-batch-size', type=int, default=256)
    train.add_argument('--num-workers', type=int, default=8)
    train.add_argument('--seed', type=int, default=0)
    train.add_argument('--rebuild-train-candidate-metadata',
                       action='store_true')
    train.add_argument('--no-progress', action='store_true')
    train.add_argument('--print-json', action='store_true')
    train.set_defaults(func=_cli_build_train_metadata)

    reference_set = subparsers.add_parser(
        'build-reference-set',
        help='select reference rows from existing train metadata',
    )
    reference_set.add_argument(
        '--dataset',
        required=True,
        choices=['cifar10', 'cifar100', 'imagenet', 'imagenet200'],
    )
    reference_set.add_argument('--checkpoint')
    reference_set.add_argument('--output-root', default='results_test/tarr')
    reference_set.add_argument('--reference-set-root')
    reference_set.add_argument('--reference-set-batch-size',
                               type=int,
                               default=512)
    reference_set.add_argument('--runtime-mode',
                               default='classifier_feature_cache',
                               choices=['classifier_feature_cache'])
    reference_set.add_argument('--num-workers', type=int, default=8)
    reference_set.add_argument('--seed', type=int, default=0)
    reference_set.add_argument('--rebuild-reference-set', action='store_true')
    reference_set.add_argument('--progress', action='store_true')
    reference_set.add_argument('--print-json', action='store_true')
    reference_set.add_argument('--write-preview', action='store_true')
    reference_set.add_argument('--preview-per-class', type=int, default=8)
    reference_set.add_argument(
        '--metadata',
        required=True,
        help='path to candidates.npz',
    )
    reference_set.add_argument(
        '--reference-config',
        required=True,
        help='reference config, e.g. per_class=4,filter=correct,seed=0',
    )
    reference_set.set_defaults(func=_cli_build_reference_set)

    anchor_set = subparsers.add_parser(
        'build-anchor-set',
        help='select anchor rows disjoint from a probe reference_set',
    )
    anchor_set.add_argument(
        '--dataset',
        required=True,
        choices=['cifar10', 'cifar100', 'imagenet', 'imagenet200'],
    )
    anchor_set.add_argument('--checkpoint')
    anchor_set.add_argument('--output-root', default='results_test/tarr')
    anchor_set.add_argument('--anchor-set-root')
    anchor_set.add_argument('--anchor-set-batch-size',
                            '--reference-set-batch-size',
                            dest='anchor_set_batch_size',
                            type=int,
                            default=512)
    anchor_set.add_argument('--runtime-mode',
                            default='classifier_feature_cache',
                            choices=['classifier_feature_cache'])
    anchor_set.add_argument('--num-workers', type=int, default=8)
    anchor_set.add_argument('--seed', type=int, default=0)
    anchor_set.add_argument('--rebuild-anchor-set', action='store_true')
    anchor_set.add_argument('--progress', action='store_true')
    anchor_set.add_argument('--print-json', action='store_true')
    anchor_set.add_argument('--write-preview', action='store_true')
    anchor_set.add_argument('--preview-per-class', type=int, default=8)
    anchor_set.add_argument(
        '--metadata',
        required=True,
        help='path to candidates.npz',
    )
    anchor_set.add_argument(
        '--reference-config',
        required=True,
        help='anchor config, e.g. per_class=4,filter=correct,seed=0',
    )
    anchor_set.add_argument(
        '--probe-reference-set-dir',
        required=True,
        help='path to the probe reference_set directory containing selected_samples.csv',
    )
    anchor_set.set_defaults(func=_cli_build_anchor_set)
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
