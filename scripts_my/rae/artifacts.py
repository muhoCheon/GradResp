"""Artifact path and serialization helpers for standalone RAE."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .config import ReferenceConfig


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Mapping) -> None:
    ensure_dir(path.parent)
    with path.open('w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def selected_samples_hash(indices: Sequence[int]) -> str:
    arr = np.asarray(indices, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(arr.tobytes())
    return digest.hexdigest()


def output_root(args) -> Path:
    return Path(args.output_root)


def reference_dir(args, config: ReferenceConfig) -> Path:
    return (
        output_root(args) / 'reference_sets' / config.dataset / config.id /
        f'seed{config.seed}'
    )


def ref_grad_bank_dir(args, gradient_space: str,
                      reference_id: str) -> Path:
    return (
        output_root(args) / 'gradient_banks' / args.dataset / gradient_space /
        reference_id
    )


def run_dir(args, run_id: str) -> Path:
    return (
        output_root(args) / 'outputs' / args.dataset /
        args.baseline_protocol / f'seed{args.seed}' / run_id
    )


def diagnostics_dir(args, run_id: str) -> Path:
    return output_root(args) / 'diagnostics' / args.dataset / run_id


def experiment_dir(args, experiment_id: str) -> Path:
    return output_root(args) / 'experiments' / args.dataset / experiment_id


def score_rule_dir(base_run_dir: Path, scheme: str, score_rule: str) -> Path:
    return base_run_dir / scheme / score_rule


def save_score_npz(path: Path,
                   pred,
                   conf,
                   label,
                   *,
                   ood_score=None,
                   eid=None,
                   best_class=None,
                   v_best=None,
                   q_best=None,
                   **extra_arrays) -> None:
    ensure_dir(path.parent)
    payload = {
        'pred': np.asarray(pred, dtype=np.int64),
        'conf': np.asarray(conf, dtype=np.float64),
        'label': np.asarray(label, dtype=np.int64),
    }
    optional = {
        'ood_score': ood_score,
        'eid': eid,
        'best_class': best_class,
        'v_best': v_best,
        'q_best': q_best,
    }
    optional.update(extra_arrays)
    for key, value in optional.items():
        if value is not None:
            payload[key] = np.asarray(value)
    np.savez(path, **payload)


def write_csv(path: Path, rows: Iterable[Mapping], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
