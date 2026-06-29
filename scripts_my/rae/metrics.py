"""Metric and score-file helpers for standalone RAE."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openood.evaluators.metrics import compute_all_metrics

from .artifacts import write_csv


def score_tuple_from_ood(pred, ood_score, label):
    pred = np.asarray(pred, dtype=np.int64)
    ood_score = np.asarray(ood_score, dtype=np.float64)
    label = np.asarray(label, dtype=np.int64)
    conf = -ood_score
    return pred, conf, label


def openood_conf_from_ood_score(ood_score):
    return -np.asarray(ood_score, dtype=np.float64)


def concat_score_tuples(parts):
    return (
        np.concatenate([part[0] for part in parts]),
        np.concatenate([part[1] for part in parts]),
        np.concatenate([part[2] for part in parts]),
    )


def metric_summary(id_scores, ood_scores):
    pred = np.concatenate([id_scores[0], ood_scores[0]])
    conf = np.concatenate([id_scores[1], ood_scores[1]])
    label = np.concatenate([
        id_scores[2],
        -1 * np.ones_like(ood_scores[2], dtype=np.int64),
    ])
    return compute_all_metrics(conf, label, pred)


def format_metric_row(dataset_name, metrics):
    fpr, auroc, aupr_in, aupr_out, acc = metrics
    return {
        'dataset': dataset_name,
        'FPR@95': f'{100 * fpr:.2f}',
        'AUROC': f'{100 * auroc:.2f}',
        'AUPR_IN': f'{100 * aupr_in:.2f}',
        'AUPR_OUT': f'{100 * aupr_out:.2f}',
        'ACC': f'{100 * acc:.2f}',
    }


def write_metrics_csv(path: Path, rows):
    fieldnames = ['dataset', 'FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC']
    write_csv(path, rows, fieldnames)
