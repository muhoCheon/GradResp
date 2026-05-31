#!/usr/bin/env python
"""Prepare and optionally run TARR matrix jobs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from scripts_my.tarr.protocol import (  # noqa: E402
    SCORE_RESULTS_DIR,
    TTA_RESPONSE_DIR,
)


def load_reference_filters():
    """Read REFERENCE_FILTERS without importing torch-heavy reference.py."""
    reference_path = ROOT_DIR / 'scripts_my/tarr/reference.py'
    tree = ast.parse(reference_path.read_text())
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name)
                        and target.id == 'REFERENCE_FILTERS'
                        for target in node.targets)):
            return list(ast.literal_eval(node.value))
    raise RuntimeError(f'REFERENCE_FILTERS not found in {reference_path}')


REFERENCE_FILTERS = load_reference_filters()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True,
                        choices=['cifar10', 'cifar100', 'imagenet', 'imagenet200'])
    parser.add_argument('--suite', default='main_and_eval_api',
                        choices=[
                            'smoke_protocol_check',
                            'main_py_s0',
                            'eval_api_s012',
                            'main_and_eval_api',
                        ])
    parser.add_argument('--output-root', default='results_test/tarr')
    parser.add_argument('--jobs-root', default='results_test/tarr/protocol_jobs')
    parser.add_argument(
        '--reference-config',
        action='append',
        default=[],
        help=('Reference config spec passed through to eval.py. May be repeated; '
              'example: id:per_class=16,filter=correct,seed=0.'),
    )
    parser.add_argument('--reference-per-class', type=int, default=16)
    parser.add_argument('--reference-filter', default='all',
                        choices=REFERENCE_FILTERS)
    parser.add_argument('--reference-min-confidence', type=float, default=0.9)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--score-rule', default='all')
    parser.add_argument('--scheme', default='both', choices=['ood', 'fsood', 'both'])
    parser.add_argument('--objective', default='predicted_label_ce')
    parser.add_argument('--update-scope', default='classifier')
    parser.add_argument('--runtime-mode', default='auto')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--reference-batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--gpus', default='')
    parser.add_argument('--max-parallel', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--rerun-failed', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--execute', action='store_true',
                        help=('Actually run prepared jobs. By default only '
                              'command/status files are written.'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--max-id-samples', type=int, default=0)
    parser.add_argument('--max-ood-samples', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser


def protocols_and_seeds(suite):
    if suite == 'smoke_protocol_check':
        return [('main_py', 0), ('eval_api', 0)]
    if suite == 'main_py_s0':
        return [('main_py', 0)]
    if suite == 'eval_api_s012':
        return [('eval_api', seed) for seed in [0, 1, 2]]
    if suite == 'main_and_eval_api':
        return [('main_py', 0)] + [('eval_api', seed) for seed in [0, 1, 2]]
    raise ValueError(f'Unknown suite: {suite}')


def short_token(value, fallback='default'):
    if not value:
        return fallback
    digest = hashlib.sha1('\0'.join(value).encode()).hexdigest()[:8]
    return f'g{len(value)}_{digest}'


def lr_tag(value):
    return f'{value:g}'.replace('.', 'p').replace('-', 'm')


def run_id(args, protocol, seed):
    tta_id = (
        f'{args.objective}_s{args.steps}_lr{lr_tag(args.lr)}'
        f'_{args.update_scope}_{args.runtime_mode}'
    )
    ref_id = short_token(args.reference_config)
    score_id = args.score_rule
    return (
        f'{args.dataset}_{protocol}_{args.scheme}'
        f'__tta-{tta_id}__refs-{ref_id}__score-{score_id}__seed{seed}'
    )


def output_dir(args, protocol, seed, rid):
    return (Path(args.output_root) / 'outputs' / args.dataset / protocol /
            f'seed{seed}' / rid)


def command_for_job(args, protocol, seed, rid):
    cmd = [
        sys.executable,
        str(ROOT_DIR / 'scripts_my/tarr/eval.py'),
        '--dataset',
        args.dataset,
        '--baseline-protocol',
        protocol,
        '--output-root',
        args.output_root,
        '--run-id',
        rid,
        '--reference-per-class',
        str(args.reference_per_class),
        '--reference-filter',
        args.reference_filter,
        '--reference-min-confidence',
        str(args.reference_min_confidence),
        '--objective',
        args.objective,
        '--steps',
        str(args.steps),
        '--lr',
        str(args.lr),
        '--update-scope',
        args.update_scope,
        '--runtime-mode',
        args.runtime_mode,
        '--score-rule',
        args.score_rule,
        '--scheme',
        args.scheme,
        '--batch-size',
        str(args.batch_size),
        '--reference-batch-size',
        str(args.reference_batch_size),
        '--num-workers',
        str(args.num_workers),
        '--seed',
        str(seed),
        '--save-tta-response',
        '--no-progress',
    ]
    for reference_config in args.reference_config:
        cmd += ['--reference-config', reference_config]
    if args.max_samples:
        cmd += ['--max-samples', str(args.max_samples)]
    if args.max_id_samples:
        cmd += ['--max-id-samples', str(args.max_id_samples)]
    if args.max_ood_samples:
        cmd += ['--max-ood-samples', str(args.max_ood_samples)]
    if args.overwrite:
        cmd += ['--overwrite']
    return cmd


def command_hash(cmd):
    return hashlib.sha256('\0'.join(cmd).encode()).hexdigest()


def validate_reference_config_filters(values, allowed_filters):
    allowed = set(allowed_filters)
    for value in values:
        raw_fields = value.split(':', 1)[1] if ':' in value else value
        fields = {}
        for item in raw_fields.split(','):
            item = item.strip()
            if not item:
                continue
            if '=' not in item:
                raise ValueError(f'Invalid --reference-config field: {item}')
            key, raw_value = item.split('=', 1)
            fields[key.strip()] = raw_value.strip()
        filter_name = fields.get('filter', 'all')
        if filter_name not in allowed:
            raise ValueError(
                f'Unknown reference filter {filter_name!r}. '
                f'Expected one of: {", ".join(allowed_filters)}')


def read_status(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_status(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def tta_response_dir_exists(cache_dir):
    cache_dir = Path(cache_dir)
    if cache_dir.name == 'response_cache':
        raise FileNotFoundError(
            f'Unsupported TTA response directory {cache_dir}; use '
            f'{TTA_RESPONSE_DIR}/.')
    return cache_dir.exists()


def metric_exists_for_reference(scheme_dir, score_rule, reference_config_id):
    if reference_config_id:
        owner_dir = scheme_dir / 'references' / reference_config_id
    else:
        owner_dir = scheme_dir
    metric_path = owner_dir / SCORE_RESULTS_DIR / score_rule / 'ood.csv'
    if metric_path.exists():
        return True
    return False


def artifacts_pass(out_dir, scheme):
    run_manifest = out_dir / 'run_manifest.json'
    run_info = out_dir / 'run_info.md'
    if not run_manifest.exists() or not run_info.exists():
        return False
    manifest_data = json.loads(run_manifest.read_text())
    schemes = ['ood', 'fsood'] if scheme == 'both' else [scheme]
    for item in schemes:
        scheme_dir = out_dir / item
        manifest = scheme_dir / 'scheme_manifest.json'
        if not manifest.exists():
            return False
        scheme_data = json.loads(manifest.read_text())
        score_rules = scheme_data.get('expanded_score_rules', [])
        ref_configs = scheme_data.get('reference_configs',
                                      manifest_data.get('reference_configs', []))
        ref_ids = []
        if isinstance(ref_configs, list):
            ref_ids = [ref.get('id') for ref in ref_configs
                       if isinstance(ref, dict) and ref.get('id')]
        elif isinstance(ref_configs, dict):
            ref_ids = list(ref_configs.keys())
        if (scheme_dir / 'references').exists() and not ref_ids:
            ref_ids = [path.name for path in (scheme_dir / 'references').iterdir()
                       if path.is_dir()]
        if ref_ids:
            for ref_id in ref_ids:
                cache_dir = (
                    scheme_dir / 'references' / ref_id / TTA_RESPONSE_DIR)
                if not tta_response_dir_exists(cache_dir):
                    return False
                for score_rule in score_rules:
                    if not metric_exists_for_reference(scheme_dir, score_rule,
                                                       ref_id):
                        return False
        else:
            cache_dir = scheme_dir / TTA_RESPONSE_DIR
            if not tta_response_dir_exists(cache_dir):
                return False
            for score_rule in score_rules:
                if not metric_exists_for_reference(scheme_dir, score_rule, None):
                    return False
    return True


def should_skip(status_path, out_dir, cmd, args):
    status = read_status(status_path)
    if not status:
        return False
    if args.rerun_failed and status.get('status') == 'fail':
        return False
    return (args.resume and status.get('status') == 'pass'
            and status.get('command_hash') == command_hash(cmd)
            and artifacts_pass(out_dir, args.scheme))


def write_command_file(path, cmd):
    path.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd ' +
                    shlex.quote(str(ROOT_DIR)) + '\n' +
                    ' '.join(shlex.quote(part) for part in cmd) + '\n')
    path.chmod(path.stat().st_mode | 0o111)


def write_prepared_status(status_path, args, rid, protocol, seed, cmd, out_dir):
    status = read_status(status_path) or {}
    if status.get('status') == 'pass' and status.get('command_hash') == command_hash(cmd):
        return
    write_status(status_path, {
        'status': 'prepared',
        'run_id': rid,
        'dataset': args.dataset,
        'baseline_protocol': protocol,
        'seed': seed,
        'command_hash': command_hash(cmd),
        'command': cmd,
        'output_dir': str(out_dir),
        'scheme': args.scheme,
        'reference_config': args.reference_config,
        'max_samples': args.max_samples,
        'max_id_samples': args.max_id_samples,
        'max_ood_samples': args.max_ood_samples,
    })


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        validate_reference_config_filters(args.reference_config,
                                          REFERENCE_FILTERS)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.max_parallel != 1:
        raise NotImplementedError(
            'This lightweight matrix runner currently executes sequentially. '
            'Use multiple invocations or a scheduler for parallel jobs.')
    gpus = [item.strip() for item in args.gpus.split(',') if item.strip()]
    jobs_root = Path(args.jobs_root) / args.dataset / args.suite

    for job_index, (protocol, seed) in enumerate(protocols_and_seeds(args.suite)):
        rid = run_id(args, protocol, seed)
        job_dir = jobs_root / rid
        status_path = job_dir / 'run_status.json'
        out_dir = output_dir(args, protocol, seed, rid)
        cmd = command_for_job(args, protocol, seed, rid)
        env = None
        if gpus:
            env = dict(**os.environ)
            env['CUDA_VISIBLE_DEVICES'] = gpus[job_index % len(gpus)]

        if args.dry_run:
            print(' '.join(shlex.quote(part) for part in cmd))
            continue

        job_dir.mkdir(parents=True, exist_ok=True)
        write_command_file(job_dir / 'command.sh', cmd)
        (job_dir / 'run.log').touch(exist_ok=True)
        if should_skip(status_path, out_dir, cmd, args):
            print(f'skip: {rid}')
            continue
        if not args.execute:
            write_prepared_status(status_path, args, rid, protocol, seed, cmd,
                                  out_dir)
            print(f'prepared: {rid}')
            continue

        start = time.time()
        write_status(status_path, {
            'status': 'running',
            'run_id': rid,
            'dataset': args.dataset,
            'baseline_protocol': protocol,
            'seed': seed,
            'command_hash': command_hash(cmd),
            'output_dir': str(out_dir),
            'started_at': start,
        })
        with (job_dir / 'run.log').open('w') as log:
            proc = subprocess.run(cmd, cwd=str(ROOT_DIR), stdout=log,
                                  stderr=subprocess.STDOUT, env=env,
                                  check=False)
        elapsed = time.time() - start
        passed = proc.returncode == 0 and artifacts_pass(out_dir, args.scheme)
        write_status(status_path, {
            'status': 'pass' if passed else 'fail',
            'run_id': rid,
            'dataset': args.dataset,
            'baseline_protocol': protocol,
            'seed': seed,
            'returncode': proc.returncode,
            'runtime_sec': elapsed,
            'command_hash': command_hash(cmd),
            'output_dir': str(out_dir),
        })
        print(f'{"pass" if passed else "fail"}: {rid}')
        if args.fail_fast and not passed:
            raise SystemExit(proc.returncode or 1)


if __name__ == '__main__':
    main()
