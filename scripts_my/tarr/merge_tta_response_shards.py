#!/usr/bin/env python
"""Merge target-sharded TARR tta_response artifacts.

This utility combines multiple run directories produced with the same TARR
configuration and disjoint target shards. It only merges Stage 3
``tta_response`` artifacts; Stage 4 should be run on the merged run directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from scripts_my.tarr.protocol import TTA_RESPONSE_DIR


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def dataset_entries(response_dir):
    response_dir = Path(response_dir)
    entries = {}
    for path in sorted(response_dir.iterdir()):
        if path.name == 'manifest.json':
            continue
        if path.is_file() and path.suffix == '.npz':
            entries[path.stem] = path
        elif path.is_dir() and (path / 'manifest.json').exists():
            entries[path.name] = path
    return entries


def load_single_npz(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def is_sample_axis_value(value, sample_count):
    return getattr(value, 'ndim', 0) >= 1 and int(value.shape[0]) == sample_count


def merge_single_npz(inputs, output_path):
    parts = [load_single_npz(path) for path in inputs]
    key_sets = [set(part) for part in parts]
    if len(set(frozenset(keys) for keys in key_sets)) != 1:
        raise ValueError(f'Input single-npz keys differ for {output_path}')
    keys = sorted(key_sets[0])
    sample_counts = [int(part['pred'].shape[0]) for part in parts]
    payload = {}
    for key in keys:
        values = [part[key] for part in parts]
        if all(is_sample_axis_value(value, count)
               for value, count in zip(values, sample_counts)):
            payload[key] = np.concatenate(values, axis=0)
        else:
            first = values[0]
            if key == 'target_shard_index':
                payload[key] = np.asarray(-1, dtype=np.int64)
                continue
            if key == 'target_shard_count':
                payload[key] = np.asarray(len(inputs), dtype=np.int64)
                continue
            for value in values[1:]:
                if np.asarray(first).shape != np.asarray(value).shape:
                    raise ValueError(
                        f'Non-sample key {key!r} has incompatible shapes '
                        f'while merging {output_path}')
                if not np.array_equal(first, value):
                    raise ValueError(
                        f'Non-sample key {key!r} differs while merging '
                        f'{output_path}')
            payload[key] = first
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return {
        'storage': 'single_npz',
        'path': str(output_path),
        'num_samples': int(sum(sample_counts)),
    }


def normalize_merged_part_metadata(source_path, target_path, *, shard_count):
    payload = load_single_npz(source_path)
    if 'target_shard_index' in payload:
        payload['target_shard_index'] = np.asarray(-1, dtype=np.int64)
    if 'target_shard_count' in payload:
        payload['target_shard_count'] = np.asarray(shard_count, dtype=np.int64)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target_path, **payload)


def merge_sharded_dirs(inputs, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_shards = []
    sample_offset = 0
    shard_index = 0
    first_manifest = None
    for dataset_dir in inputs:
        manifest = read_json(Path(dataset_dir) / 'manifest.json')
        if not manifest:
            raise FileNotFoundError(Path(dataset_dir) / 'manifest.json')
        if not manifest.get('complete', False):
            raise ValueError(f'Incomplete shard manifest: {dataset_dir}')
        if first_manifest is None:
            first_manifest = manifest
        for shard in manifest.get('shards', []):
            source = Path(dataset_dir) / shard['path']
            count = int(shard.get('num_samples', shard['end'] - shard['start']))
            target_name = f'part_{shard_index:06d}.npz'
            target = output_dir / target_name
            normalize_merged_part_metadata(
                source,
                target,
                shard_count=len(inputs))
            merged_shards.append({
                'path': target_name,
                'start': int(sample_offset),
                'end': int(sample_offset + count),
                'num_samples': count,
            })
            sample_offset += count
            shard_index += 1
    output_manifest = dict(first_manifest or {})
    output_manifest.update({
        'complete': True,
        'num_samples': int(sample_offset),
        'num_shards': len(merged_shards),
        'shards': merged_shards,
        'merged_from': [str(path) for path in inputs],
    })
    write_json(output_dir / 'manifest.json', output_manifest)
    return {
        'storage': 'sharded_npz',
        'manifest': str(output_dir / 'manifest.json'),
        'num_samples': int(sample_offset),
        'num_shards': len(merged_shards),
    }


def merge_dataset(inputs, output_response_dir, dataset_name):
    paths = [dataset_entries(path)[dataset_name] for path in inputs]
    if all(path.is_file() for path in paths):
        return merge_single_npz(paths, output_response_dir / f'{dataset_name}.npz')
    if all(path.is_dir() for path in paths):
        return merge_sharded_dirs(paths, output_response_dir / dataset_name)
    raise ValueError(
        f'Mixed single/sharded tta_response storage for dataset {dataset_name}')


def classify_dataset_names(dataset, names):
    csid = []
    near = []
    far = []
    if dataset == 'cifar100':
        csid = ['cifar100c']
        near = ['cifar10', 'tin']
        far = ['mnist', 'svhn', 'texture', 'places365']
    elif dataset == 'imagenet200':
        csid = ['imagenet_v2', 'imagenet_c', 'imagenet_r']
        near = ['ssb_hard', 'ninco']
        far = ['inaturalist', 'textures', 'openimage_o']
    elif dataset == 'cifar10':
        csid = ['cifar10c']
        near = ['cifar100', 'tin']
        far = ['mnist', 'svhn', 'texture', 'places365']
    elif dataset == 'imagenet':
        csid = ['imagenet_v2', 'imagenet_c', 'imagenet_r']
        near = ['ssb_hard', 'ninco']
        far = ['inaturalist', 'textures', 'openimage_o']
    name_set = set(names)
    return {
        'id': [dataset] if dataset in name_set else [],
        'csid': [name for name in csid if name in name_set],
        'ood': {
            'near': [name for name in near if name in name_set],
            'far': [name for name in far if name in name_set],
        },
    }


def merged_target_shard_metadata(input_run_dirs):
    return {
        'count': len(input_run_dirs),
        'index': -1,
        'merged': True,
        'input_run_dirs': [str(path) for path in input_run_dirs],
        'rule': 'merged target shards',
    }


def record_to_manifest_value(record):
    if record.get('storage') == 'single_npz':
        return record['path']
    return record


def response_manifest_files(response_files):
    result = {'id': {}, 'csid': {}, 'ood': {'near': {}, 'far': {}}}
    for split in ['id', 'csid']:
        for name, record in response_files.get(split, {}).items():
            result[split][name] = record_to_manifest_value(record)
    for split in ['near', 'far']:
        for name, record in response_files.get('ood', {}).get(split, {}).items():
            result['ood'][split][name] = record_to_manifest_value(record)
    return result


def response_counts(response_files):
    result = {'id': {}, 'csid': {}, 'ood': {'near': {}, 'far': {}}}
    for split in ['id', 'csid']:
        for name, record in response_files.get(split, {}).items():
            result[split][name] = int(record.get('num_samples', 0))
    for split in ['near', 'far']:
        for name, record in response_files.get('ood', {}).get(split, {}).items():
            result['ood'][split][name] = int(record.get('num_samples', 0))
    return result


def update_manifest_for_merge(
        manifest,
        *,
        output_run_dir,
        scheme,
        reference_config_id,
        response_files,
        input_run_dirs):
    merged = copy.deepcopy(manifest)
    shard_meta = merged_target_shard_metadata(input_run_dirs)
    processed_counts = response_counts(response_files)
    tta_response_files = response_manifest_files(response_files)
    merged.update({
        'run_id': output_run_dir.name,
        'cache_run_id': output_run_dir.name,
        'output_dir': str(output_run_dir),
        'target_shard': shard_meta,
        'target_shard_count': len(input_run_dirs),
        'target_shard_index': -1,
        'processed_counts': {reference_config_id: processed_counts},
        'tta_response_files': {reference_config_id: tta_response_files},
        'merged_tta_response': True,
        'merged_from': [str(path) for path in input_run_dirs],
    })
    if scheme:
        merged['scheme'] = scheme
    artifact_identity = merged.get('artifact_identity')
    if isinstance(artifact_identity, dict):
        artifact_identity['target_shard'] = shard_meta
        artifact_identity['is_merged_tta_response'] = True
    timing = merged.get('timing')
    if isinstance(timing, dict):
        timing['target_shard'] = shard_meta
        timing['target_shard_count'] = len(input_run_dirs)
        timing['target_shard_index'] = -1
    return merged


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True,
                        choices=['cifar10', 'cifar100', 'imagenet', 'imagenet200'])
    parser.add_argument('--scheme', default='fsood', choices=['ood', 'fsood'])
    parser.add_argument('--reference-config-id', required=True)
    parser.add_argument('--output-run-dir', required=True)
    parser.add_argument('--input-run-dir', action='append', required=True)
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_run_dir = Path(args.output_run_dir)
    output_reference_dir = (
        output_run_dir / args.scheme / 'references' / args.reference_config_id)
    output_response_dir = output_reference_dir / TTA_RESPONSE_DIR
    if output_run_dir.exists() and args.overwrite:
        shutil.rmtree(output_run_dir)
    elif output_run_dir.exists():
        raise FileExistsError(
            f'Output run dir already exists: {output_run_dir}. '
            'Pass --overwrite to replace it.')

    input_response_dirs = [
        Path(path) / args.scheme / 'references' / args.reference_config_id /
        TTA_RESPONSE_DIR
        for path in args.input_run_dir
    ]
    input_run_dirs = [Path(path) for path in args.input_run_dir]
    for path in input_response_dirs:
        if not path.exists():
            raise FileNotFoundError(path)
    entry_sets = [dataset_entries(path) for path in input_response_dirs]
    dataset_names = sorted(set().union(*(set(entries) for entries in entry_sets)))
    for name in dataset_names:
        missing = [
            str(path) for path, entries in zip(input_response_dirs, entry_sets)
            if name not in entries
        ]
        if missing:
            raise ValueError(f'Dataset {name!r} missing from inputs: {missing}')

    response_files = {'id': {}, 'csid': {}, 'ood': {'near': {}, 'far': {}}}
    for name in dataset_names:
        record = merge_dataset(input_response_dirs, output_response_dir, name)
        split_info = classify_dataset_names(args.dataset, [name])
        if split_info['id']:
            response_files['id'][name] = record
        elif split_info['csid']:
            response_files['csid'][name] = record
        elif split_info['ood']['near']:
            response_files['ood']['near'][name] = record
        elif split_info['ood']['far']:
            response_files['ood']['far'][name] = record
        else:
            response_files.setdefault('other', {})[name] = record

    split_names = classify_dataset_names(args.dataset, dataset_names)
    reference_manifest = {
        'schema_version': 1,
        'merged_tta_response': True,
        'dataset': args.dataset,
        'scheme': args.scheme,
        'reference_config_id': args.reference_config_id,
        'input_run_dirs': [str(path) for path in args.input_run_dir],
        'dataset_names': split_names,
        'tta_response_files': response_files,
    }
    write_json(output_reference_dir / 'manifest.json', reference_manifest)

    first_run_manifest = read_json(input_run_dirs[0] / 'run_manifest.json')
    first_scheme_manifest = read_json(
        input_run_dirs[0] / args.scheme / 'scheme_manifest.json')
    if not first_run_manifest:
        raise FileNotFoundError(input_run_dirs[0] / 'run_manifest.json')
    if not first_scheme_manifest:
        raise FileNotFoundError(
            input_run_dirs[0] / args.scheme / 'scheme_manifest.json')
    write_json(
        output_run_dir / 'run_manifest.json',
        update_manifest_for_merge(
            first_run_manifest,
            output_run_dir=output_run_dir,
            scheme=None,
            reference_config_id=args.reference_config_id,
            response_files=response_files,
            input_run_dirs=input_run_dirs))
    write_json(
        output_run_dir / args.scheme / 'scheme_manifest.json',
        update_manifest_for_merge(
            first_scheme_manifest,
            output_run_dir=output_run_dir,
            scheme=args.scheme,
            reference_config_id=args.reference_config_id,
            response_files=response_files,
            input_run_dirs=input_run_dirs))
    print(f'output_run_dir: {output_run_dir}')
    print(f'datasets: {", ".join(dataset_names)}')


if __name__ == '__main__':
    main()
