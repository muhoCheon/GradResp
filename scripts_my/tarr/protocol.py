"""Helpers for TARR artifact paths and run/scheme manifests."""

from __future__ import annotations

import json
from pathlib import Path

SUPPORTED_DATASETS = ('cifar10', 'cifar100', 'imagenet', 'imagenet200')
TTA_RESPONSE_DIR = 'tta_response'
SCORE_RESULTS_DIR = 'score_results'

DISPLAY_LABELS = {
    'cifar10': 'CIFAR-10',
    'cifar100': 'CIFAR-100',
    'imagenet': 'ImageNet',
    'imagenet200': 'ImageNet-200',
    'cinic10': 'CINIC-10',
    'cifar10c': 'CIFAR-10-C',
    'cifar100c': 'CIFAR-100-C',
    'imagenet_v2': 'ImageNet-V2',
    'imagenet_c': 'ImageNet-C',
    'imagenet_r': 'ImageNet-R',
    'tin': 'Tiny ImageNet',
    'mnist': 'MNIST',
    'svhn': 'SVHN',
    'texture': 'DTD',
    'textures': 'DTD',
    'places365': 'Places365',
    'ssb_hard': 'SSB-hard',
    'ninco': 'NINCO',
    'inaturalist': 'iNaturalist',
    'openimage_o': 'OpenImage-O',
}

NEAR_DATASETS = {
    'cifar10': ('cifar100', 'tin'),
    'cifar100': ('cifar10', 'tin'),
    'imagenet': ('ssb_hard', 'ninco'),
    'imagenet200': ('ssb_hard', 'ninco'),
}

FAR_DATASETS = {
    'cifar10': ('mnist', 'svhn', 'texture', 'places365'),
    'cifar100': ('mnist', 'svhn', 'texture', 'places365'),
    'imagenet': ('inaturalist', 'textures', 'openimage_o'),
    'imagenet200': ('inaturalist', 'textures', 'openimage_o'),
}

EXPECTED_CSID_BY_PROTOCOL = {
    ('cifar10', 'main_py'): ('cinic10',),
    ('cifar10', 'eval_api'): ('cifar10c',),
    ('cifar100', 'main_py'): ('cifar100c',),
    ('cifar100', 'eval_api'): ('cifar100c',),
    ('imagenet', 'main_py'): ('imagenet_v2', 'imagenet_c', 'imagenet_r'),
    ('imagenet', 'eval_api'): ('imagenet_v2', 'imagenet_c', 'imagenet_r'),
    ('imagenet200', 'main_py'): ('imagenet_v2', 'imagenet_c', 'imagenet_r'),
    ('imagenet200', 'eval_api'): ('imagenet_v2', 'imagenet_c', 'imagenet_r'),
}


def supported_dataset_names():
    return list(SUPPORTED_DATASETS)


def display_label(name):
    return DISPLAY_LABELS.get(name, str(name))


def expected_csid_datasets(dataset, baseline_protocol):
    key = (dataset, baseline_protocol)
    if key not in EXPECTED_CSID_BY_PROTOCOL:
        raise ValueError(
            f'Unsupported protocol pairing: dataset={dataset}, '
            f'baseline_protocol={baseline_protocol}')
    return list(EXPECTED_CSID_BY_PROTOCOL[key])


def near_dataset_names(dataset):
    if dataset not in NEAR_DATASETS:
        raise ValueError(f'Unsupported dataset: {dataset}')
    return list(NEAR_DATASETS[dataset])


def far_dataset_names(dataset):
    if dataset not in FAR_DATASETS:
        raise ValueError(f'Unsupported dataset: {dataset}')
    return list(FAR_DATASETS[dataset])


def uses_evaluator_csid_loaders(dataset):
    return dataset in {'imagenet', 'imagenet200'}


def parse_dataset_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    if isinstance(value, (list, tuple)):
        names = []
        for item in value:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                names.extend(
                    parse_dataset_list(
                        item.get('name') or item.get('dataset')
                        or item.get('dataset_name')))
        return names
    if isinstance(value, dict):
        for key in ['datasets', 'names', 'dataset_names']:
            names = parse_dataset_list(value.get(key))
            if names:
                return names
        return parse_dataset_list(
            value.get('name') or value.get('dataset')
            or value.get('dataset_name'))
    return []


def _dataset_keys(value):
    if not isinstance(value, dict):
        return []
    metadata_keys = {
        'datasets',
        'names',
        'dataset_names',
        'name',
        'dataset',
        'dataset_name',
    }
    return [key for key in value.keys() if key not in metadata_keys]


def reference_cache_dir(run_dir, scheme, reference_config_id):
    return (Path(run_dir) / scheme / 'references' / str(reference_config_id) /
            TTA_RESPONSE_DIR)


def scheme_cache_dir(run_dir, scheme):
    return Path(run_dir) / scheme / TTA_RESPONSE_DIR


def reference_score_results_dir(run_dir, scheme, reference_config_id):
    return (Path(run_dir) / scheme / 'references' / str(reference_config_id) /
            SCORE_RESULTS_DIR)


def scheme_score_results_dir(run_dir, scheme):
    return Path(run_dir) / scheme / SCORE_RESULTS_DIR


def score_result_path(owner_dir, score_rule):
    return Path(owner_dir) / SCORE_RESULTS_DIR / score_rule / 'ood.csv'


def _reject_unsupported_tta_response_dir(cache_dir):
    cache_dir = Path(cache_dir)
    if cache_dir.name == 'response_cache':
        raise ValueError(
            f'Unsupported TTA artifact directory: {cache_dir}. '
            f'Use the canonical {TTA_RESPONSE_DIR}/ directory.')


def require_canonical_cache_dir(cache_dir):
    cache_dir = Path(cache_dir)
    _reject_unsupported_tta_response_dir(cache_dir)
    return cache_dir


def resolve_cache_dir(cache_dir=None,
                      run_dir=None,
                      scheme=None,
                      reference_config_id=None):
    if cache_dir is not None:
        return require_canonical_cache_dir(cache_dir)
    if run_dir is None or scheme is None:
        raise ValueError(
            'Pass either --tta-response-dir or both --run-dir and --scheme.')
    if reference_config_id:
        return require_canonical_cache_dir(
            reference_cache_dir(run_dir, scheme, reference_config_id))
    return require_canonical_cache_dir(scheme_cache_dir(run_dir, scheme))


def cache_context(cache_dir):
    cache_dir = Path(cache_dir)
    _reject_unsupported_tta_response_dir(cache_dir)
    if cache_dir.name == TTA_RESPONSE_DIR:
        owner_dir = cache_dir.parent
    else:
        owner_dir = cache_dir

    reference_config_id = None
    if owner_dir.parent.name == 'references':
        reference_config_id = owner_dir.name
        scheme_dir = owner_dir.parent.parent
    else:
        scheme_dir = owner_dir

    return {
        'cache_dir': cache_dir,
        'owner_dir': owner_dir,
        'scheme_dir': scheme_dir,
        'run_dir': scheme_dir.parent,
        'scheme': scheme_dir.name,
        'reference_config_id': reference_config_id,
    }


def _manifest_paths(run_dir, scheme):
    run_dir = Path(run_dir)
    scheme_dir = run_dir / scheme
    return [
        scheme_dir / 'scheme_manifest.json',
        run_dir / 'scheme_manifest.json',
        run_dir / 'run_manifest.json',
    ]


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
    unique_paths = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def _read_json(path):
    try:
        with path.open() as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _extract_csid_datasets(node):
    if isinstance(node, dict):
        for key in [
                'csid_datasets',
                'csid_dataset',
                'fsood_csid_datasets',
                'fsood_csid_dataset',
        ]:
            names = parse_dataset_list(node.get(key))
            if names:
                return names

        if 'csid' in node:
            csid_node = node['csid']
            names = parse_dataset_list(csid_node)
            if not names:
                names = _dataset_keys(csid_node)
            if names:
                return names

        for key in ['splits', 'datasets', TTA_RESPONSE_DIR, 'tta_response_files']:
            names = _extract_csid_datasets(node.get(key))
            if names:
                return names

        for value in node.values():
            names = _extract_csid_datasets(value)
            if names:
                return names
    elif isinstance(node, list):
        for item in node:
            names = _extract_csid_datasets(item)
            if names:
                return names
    return []


def csid_datasets_from_run_manifest(run_dir, scheme):
    for path in _manifest_paths(run_dir, scheme):
        manifest = _read_json(path)
        if manifest is None:
            continue
        names = _extract_csid_datasets(manifest)
        if names:
            return names
    return []


def csid_datasets_from_cache_manifest(cache_dir):
    for path in _cache_manifest_paths(cache_dir):
        manifest = _read_json(path)
        if manifest is None:
            continue
        names = _extract_csid_datasets(manifest)
        if names:
            return names
    return []


def _extract_ood_datasets(node, split):
    if isinstance(node, dict):
        for key in [f'{split}_datasets', f'{split}_dataset_names']:
            names = parse_dataset_list(node.get(key))
            if names:
                return names

        names = parse_dataset_list(
            node.get('resolved_dataset_names', {}).get(split))
        if names:
            return names

        dataset_manifest = node.get('dataset_manifest', {})
        if isinstance(dataset_manifest, dict):
            names = _dataset_keys(
                dataset_manifest.get('ood', {}).get(split, {}))
            if names:
                return names

        ood_node = node.get('ood', {})
        if isinstance(ood_node, dict):
            names = parse_dataset_list(ood_node.get(split))
            if not names:
                names = _dataset_keys(ood_node.get(split, {}))
            if names:
                return names

        for key in [
                'protocol_config',
                'score',
                'score_config',
                SCORE_RESULTS_DIR,
                'splits',
                'datasets',
                TTA_RESPONSE_DIR,
                'tta_response_files',
        ]:
            names = _extract_ood_datasets(node.get(key), split)
            if names:
                return names

        for value in node.values():
            names = _extract_ood_datasets(value, split)
            if names:
                return names
    elif isinstance(node, list):
        for item in node:
            names = _extract_ood_datasets(item, split)
            if names:
                return names
    return []


def ood_datasets_from_run_manifest(run_dir, scheme, split=None):
    splits = [split] if split else ['near', 'far']
    result = {}
    for current_split in splits:
        for path in _manifest_paths(run_dir, scheme):
            manifest = _read_json(path)
            if manifest is None:
                continue
            names = _extract_ood_datasets(manifest, current_split)
            if names:
                result[current_split] = names
                break
    if split:
        return result.get(split, [])
    return result


def ood_datasets_from_cache_manifest(cache_dir, split=None):
    splits = [split] if split else ['near', 'far']
    result = {}
    for current_split in splits:
        for path in _cache_manifest_paths(cache_dir):
            manifest = _read_json(path)
            if manifest is None:
                continue
            names = _extract_ood_datasets(manifest, current_split)
            if names:
                result[current_split] = names
                break
    if split:
        return result.get(split, [])
    return result
