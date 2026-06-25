"""Post-hoc ordering grid search for TARR A/R probe scores.

This script uses saved ``score_results`` artifacts and evaluates whether any
score direction or accept/reject sign combination induces the desired ordering:

    clean ID < csID < near-OOD < far-OOD

All metrics use higher score = more OOD-like.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


FRESH_RUNS = {
    'imagenet200_correct_rpc32': {
        'score_root': (
            'results_test/tarr/outputs/imagenet200/eval_api/seed0/'
            'imagenet200_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_'
            'correct_rpc32_refseed0_merged8/fsood/references/correct_rpc32/'
            'score_results'
        ),
        'clean': ['imagenet200'],
        'csid': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
        'near': ['ssb_hard', 'ninco'],
        'far': ['inaturalist', 'textures', 'openimage_o'],
        'group1_both_best': 65.545,
    },
    'imagenet200_correcthigh09_rpc16_minbank': {
        'score_root': (
            'results_test/tarr/outputs/imagenet200/eval_api/seed0/'
            'imagenet200_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_'
            'correcthigh09_rpc16_refseed0_minbank_pce_uniform_merged8/'
            'fsood/references/correcthigh09_rpc16/score_results'
        ),
        'clean': ['imagenet200'],
        'csid': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
        'near': ['ssb_hard', 'ninco'],
        'far': ['inaturalist', 'textures', 'openimage_o'],
        'group1_both_best': 65.545,
    },
    'cifar100_all_rpc16': {
        'score_root': (
            'results_test/tarr/outputs/cifar100/eval_api/seed0/'
            'cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_'
            'all_rpc16_refseed0_merged8/fsood/references/all_rpc16/'
            'score_results'
        ),
        'clean': ['cifar100'],
        'csid': ['cifar100c'],
        'near': ['cifar10', 'tin'],
        'far': ['mnist', 'svhn', 'texture', 'places365'],
        'group1_both_best': 66.20,
    },
    'cifar100_correct_rpc32': {
        'score_root': (
            'results_test/tarr/outputs/cifar100/eval_api/seed0/'
            'cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_'
            'correct_rpc32_refseed0_merged8/fsood/references/correct_rpc32/'
            'score_results'
        ),
        'clean': ['cifar100'],
        'csid': ['cifar100c'],
        'near': ['cifar10', 'tin'],
        'far': ['mnist', 'svhn', 'texture', 'places365'],
        'group1_both_best': 66.20,
    },
}


PAIR_RULE_GROUPS = {
    'eff_pos': ('accept_efficiency', 'reject_efficiency'),
    'eff_abs': ('accept_abs_ref_efficiency', 'reject_abs_ref_efficiency'),
    'eff_pred': ('accept_pred_ref_efficiency', 'reject_pred_ref_efficiency'),
    'eff_target_weighted': (
        'accept_target_weighted_ref_efficiency',
        'reject_target_weighted_ref_efficiency',
    ),
    'ref_pos': (
        'accept_pos_ref_loss_delta_mean',
        'reject_pos_ref_loss_delta_mean',
    ),
    'ref_signed': (
        'accept_signed_ref_loss_delta_mean',
        'reject_signed_ref_loss_delta_mean',
    ),
    'ref_abs': (
        'accept_abs_ref_loss_delta_mean',
        'reject_abs_ref_loss_delta_mean',
    ),
    'ref_pred': (
        'accept_pred_ref_loss_delta',
        'reject_pred_ref_loss_delta',
    ),
    'ref_target_weighted': (
        'accept_target_weighted_ref_loss_delta',
        'reject_target_weighted_ref_loss_delta',
    ),
    'target_objective': (
        'accept_target_objective_delta',
        'reject_target_objective_delta',
    ),
}


PAIR_FORMULAS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    '+A+R': lambda a, r: a + r,
    '+A-R': lambda a, r: a - r,
    '-A+R': lambda a, r: -a + r,
    '-A-R': lambda a, r: -a - r,
    'max(+A,+R)': lambda a, r: np.maximum(a, r),
    'min(+A,+R)': lambda a, r: np.minimum(a, r),
    'max(+A,-R)': lambda a, r: np.maximum(a, -r),
    'min(+A,-R)': lambda a, r: np.minimum(a, -r),
    'max(-A,+R)': lambda a, r: np.maximum(-a, r),
    'min(-A,+R)': lambda a, r: np.minimum(-a, r),
    'max(-A,-R)': lambda a, r: np.maximum(-a, -r),
    'min(-A,-R)': lambda a, r: np.minimum(-a, -r),
}


@dataclass(frozen=True)
class ScoreLeaf:
    rule: str
    step: str
    variant: str
    path: Path


def auc_higher_positive(negative: np.ndarray, positive: np.ndarray) -> float:
    """AUROC where larger scores should belong to ``positive``."""
    scores = np.concatenate([negative, positive]).astype(np.float64)
    labels = np.concatenate([
        np.zeros(negative.shape[0], dtype=np.int8),
        np.ones(positive.shape[0], dtype=np.int8),
    ])
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    i = 0
    while i < scores.shape[0]:
        j = i + 1
        while j < scores.shape[0] and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    n_pos = labels.sum()
    n_neg = labels.shape[0] - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float(
        (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


def parse_leaf(path: Path, score_root: Path) -> ScoreLeaf | None:
    if path.name != 'scores':
        return None
    rel = path.relative_to(score_root)
    parts = rel.parts
    if any(part.startswith('id_side_') for part in parts):
        return None
    if len(parts) == 2:
        return ScoreLeaf(rule=parts[0], step='final', variant='legacy',
                         path=path.parent)
    if len(parts) == 4 and parts[1].startswith('step_'):
        return ScoreLeaf(rule=parts[0], step=parts[1][len('step_'):],
                         variant=parts[2], path=path.parent)
    return None


def score_leaves(score_root: Path) -> list[ScoreLeaf]:
    leaves = []
    for scores_dir in score_root.glob('**/scores'):
        leaf = parse_leaf(scores_dir, score_root)
        if leaf is not None:
            leaves.append(leaf)
    return leaves


class ScoreStore:
    def __init__(self, split_names: dict[str, list[str]]):
        self.split_names = split_names

    def dataset_score(self, leaf: ScoreLeaf, dataset: str) -> np.ndarray:
        npz_path = leaf.path / 'scores' / f'{dataset}.npz'
        with np.load(npz_path) as data:
            # score artifacts store confidence = -ood_score.
            return -data['conf'].astype(np.float64, copy=False)

    def split_score(self, leaf: ScoreLeaf, split: str) -> np.ndarray:
        return np.concatenate([
            self.dataset_score(leaf, dataset)
            for dataset in self.split_names[split]
        ])

    def combined_id_score(self, leaf: ScoreLeaf) -> np.ndarray:
        return np.concatenate([
            self.split_score(leaf, 'clean'),
            self.split_score(leaf, 'csid'),
        ])

    def split_arrays(self, leaf: ScoreLeaf) -> dict[str, np.ndarray]:
        return {
            split: self.split_score(leaf, split)
            for split in ['clean', 'csid', 'near', 'far']
        }


def metrics_from_scores(clean: np.ndarray, csid: np.ndarray,
                        near: np.ndarray, far: np.ndarray,
                        group1_both_best: float | None) -> dict[str, float]:
    id_both = np.concatenate([clean, csid])
    clean_csid = 100.0 * auc_higher_positive(clean, csid)
    csid_near = 100.0 * auc_higher_positive(csid, near)
    near_far = 100.0 * auc_higher_positive(near, far)
    near_both = 100.0 * auc_higher_positive(id_both, near)
    far_both = 100.0 * auc_higher_positive(id_both, far)
    both_avg = (near_both + far_both) / 2.0
    ordering_min = min(clean_csid, csid_near, near_far)
    ordering_avg = (clean_csid + csid_near + near_far) / 3.0
    clean_mean = float(np.mean(clean))
    csid_mean = float(np.mean(csid))
    near_mean = float(np.mean(near))
    far_mean = float(np.mean(far))
    mean_order_edges = [
        clean_mean < csid_mean,
        csid_mean < near_mean,
        near_mean < far_mean,
    ]
    out = {
        'clean_mean': clean_mean,
        'csid_mean': csid_mean,
        'near_mean': near_mean,
        'far_mean': far_mean,
        'mean_order_pass': float(all(mean_order_edges)),
        'mean_order_violations': float(3 - sum(mean_order_edges)),
        'clean_csid_auc': clean_csid,
        'csid_near_auc': csid_near,
        'near_far_auc': near_far,
        'ordering_min_auc': ordering_min,
        'ordering_avg_auc': ordering_avg,
        'fsood_near_auc': near_both,
        'fsood_far_auc': far_both,
        'fsood_both_avg': both_avg,
    }
    if group1_both_best is not None:
        out['group1_gap'] = both_avg - group1_both_best
    return out


def candidate_row(run_name: str, candidate_type: str, score_name: str,
                  base_rule: str, sign_formula: str, step: str,
                  accept_variant: str, reject_variant: str,
                  clean: np.ndarray, csid: np.ndarray,
                  near: np.ndarray, far: np.ndarray,
                  group1_both_best: float | None) -> dict[str, object]:
    row: dict[str, object] = {
        'run': run_name,
        'candidate_type': candidate_type,
        'score_name': score_name,
        'base_rule': base_rule,
        'sign_formula': sign_formula,
        'step': step,
        'accept_variant': accept_variant,
        'reject_variant': reject_variant,
    }
    row.update(metrics_from_scores(clean, csid, near, far, group1_both_best))
    return row


def evaluate_single_scores(run_name: str, leaves: list[ScoreLeaf],
                           store: ScoreStore,
                           group1_both_best: float | None) -> list[dict[str, object]]:
    rows = []
    for leaf in leaves:
        try:
            clean = store.split_score(leaf, 'clean')
            csid = store.split_score(leaf, 'csid')
            near = store.split_score(leaf, 'near')
            far = store.split_score(leaf, 'far')
        except FileNotFoundError:
            continue
        for sign, label in [(1.0, '+score'), (-1.0, '-score')]:
            rows.append(candidate_row(
                run_name=run_name,
                candidate_type='single',
                score_name=f'{label}:{leaf.rule}',
                base_rule=leaf.rule,
                sign_formula=label,
                step=leaf.step,
                accept_variant=accept_variant(leaf.variant),
                reject_variant=reject_variant(leaf.variant),
                clean=sign * clean,
                csid=sign * csid,
                near=sign * near,
                far=sign * far,
                group1_both_best=group1_both_best,
            ))
    return rows


def accept_variant(variant: str) -> str:
    if variant.startswith('accept_'):
        return variant.split('__', 1)[0]
    return ''


def reject_variant(variant: str) -> str:
    if variant.startswith('reject_'):
        return variant
    if '__reject_' in variant:
        return 'reject_' + variant.split('__reject_', 1)[1]
    return ''


def leaves_by_rule_step(leaves: list[ScoreLeaf]) -> dict[tuple[str, str], list[ScoreLeaf]]:
    out: dict[tuple[str, str], list[ScoreLeaf]] = {}
    for leaf in leaves:
        out.setdefault((leaf.rule, leaf.step), []).append(leaf)
    return out


def evaluate_pair_scores(run_name: str, leaves: list[ScoreLeaf],
                         store: ScoreStore,
                         group1_both_best: float | None) -> list[dict[str, object]]:
    rows = []
    index = leaves_by_rule_step(leaves)
    for group_name, (accept_rule, reject_rule) in PAIR_RULE_GROUPS.items():
        print(f'  pair group {group_name}', flush=True)
        steps = sorted({
            step for rule, step in index
            if rule in {accept_rule, reject_rule}
        })
        for step in steps:
            accept_leaves = index.get((accept_rule, step), [])
            reject_leaves = index.get((reject_rule, step), [])
            accept_arrays = []
            reject_arrays = []
            for leaf in accept_leaves:
                try:
                    accept_arrays.append((leaf, store.split_arrays(leaf)))
                except FileNotFoundError:
                    continue
            for leaf in reject_leaves:
                try:
                    reject_arrays.append((leaf, store.split_arrays(leaf)))
                except FileNotFoundError:
                    continue
            for accept_leaf, accept_splits in accept_arrays:
                for reject_leaf, reject_splits in reject_arrays:
                    for formula_name, formula in PAIR_FORMULAS.items():
                        for split in ['clean', 'csid', 'near', 'far']:
                            if (accept_splits[split].shape
                                    != reject_splits[split].shape):
                                raise ValueError(
                                    f'shape mismatch for {split}: '
                                    f'{accept_leaf.path} '
                                    f'{accept_splits[split].shape} vs '
                                    f'{reject_leaf.path} '
                                    f'{reject_splits[split].shape}')
                        rows.append(candidate_row(
                            run_name=run_name,
                            candidate_type='pair',
                            score_name=f'{group_name}:{formula_name}',
                            base_rule=f'{accept_rule}__{reject_rule}',
                            sign_formula=formula_name,
                            step=step,
                            accept_variant=accept_leaf.variant,
                            reject_variant=reject_leaf.variant,
                            clean=formula(accept_splits['clean'],
                                          reject_splits['clean']),
                            csid=formula(accept_splits['csid'],
                                         reject_splits['csid']),
                            near=formula(accept_splits['near'],
                                         reject_splits['near']),
                            far=formula(accept_splits['far'],
                                        reject_splits['far']),
                            group1_both_best=group1_both_best,
                        ))
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'run',
        'candidate_type',
        'score_name',
        'base_rule',
        'sign_formula',
        'step',
        'accept_variant',
        'reject_variant',
        'clean_mean',
        'csid_mean',
        'near_mean',
        'far_mean',
        'mean_order_pass',
        'mean_order_violations',
        'clean_csid_auc',
        'csid_near_auc',
        'near_far_auc',
        'ordering_min_auc',
        'ordering_avg_auc',
        'fsood_near_auc',
        'fsood_far_auc',
        'fsood_both_avg',
        'group1_gap',
    ]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    f'{value:.6f}' if isinstance(value, float) else value
                )
                for key, value in row.items()
            })


def append_rows(path: Path, rows: list[dict[str, object]], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'run',
        'candidate_type',
        'score_name',
        'base_rule',
        'sign_formula',
        'step',
        'accept_variant',
        'reject_variant',
        'clean_mean',
        'csid_mean',
        'near_mean',
        'far_mean',
        'mean_order_pass',
        'mean_order_violations',
        'clean_csid_auc',
        'csid_near_auc',
        'near_far_auc',
        'ordering_min_auc',
        'ordering_avg_auc',
        'fsood_near_auc',
        'fsood_far_auc',
        'fsood_both_avg',
        'group1_gap',
    ]
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    f'{value:.6f}' if isinstance(value, float) else value
                )
                for key, value in row.items()
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path('results_test/tarr/summary/'
                                     'tarr_probe_ordering_grid.csv'))
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--run', action='append', choices=sorted(FRESH_RUNS),
                        help='Run key to process. Can be passed multiple times.')
    args = parser.parse_args()

    if args.output.exists():
        args.output.unlink()
    all_rows: list[dict[str, object]] = []
    wrote_header = False
    selected_runs = args.run or list(FRESH_RUNS)
    for run_name in selected_runs:
        config = FRESH_RUNS[run_name]
        score_root = Path(config['score_root'])
        if not score_root.exists():
            raise FileNotFoundError(score_root)
        split_names = {
            'clean': list(config['clean']),
            'csid': list(config['csid']),
            'near': list(config['near']),
            'far': list(config['far']),
        }
        leaves = score_leaves(score_root)
        print(f'Processing {run_name}: {len(leaves)} score leaves',
              flush=True)
        store = ScoreStore(split_names)
        group1_both_best = float(config['group1_both_best'])
        run_rows = []
        print('  single score signs', flush=True)
        run_rows.extend(evaluate_single_scores(
            run_name, leaves, store, group1_both_best))
        run_rows.extend(evaluate_pair_scores(
            run_name, leaves, store, group1_both_best))
        append_rows(args.output, run_rows, write_header=not wrote_header)
        wrote_header = True
        all_rows.extend(run_rows)
        print(f'Processed {run_name}: {len(run_rows)} rows', flush=True)

    rows = all_rows
    print(f'Wrote {len(rows)} rows to {args.output}')

    for run_name in selected_runs:
        run_rows = [row for row in rows if row['run'] == run_name]
        print(f'\n## {run_name}')
        for title, key in [
                ('Top ordering_min_auc', 'ordering_min_auc'),
                ('Top csid_near_auc', 'csid_near_auc'),
                ('Top fsood_both_avg', 'fsood_both_avg')]:
            print(title)
            for row in sorted(
                    run_rows,
                    key=lambda item: float(item[key]),
                    reverse=True)[:args.top_k]:
                print(
                    f"{key}={row[key]:.2f} "
                    f"ord_min={row['ordering_min_auc']:.2f} "
                    f"csid_near={row['csid_near_auc']:.2f} "
                    f"both={row['fsood_both_avg']:.2f} "
                    f"gap={row.get('group1_gap', float('nan')):.2f} "
                    f"{row['candidate_type']} {row['score_name']} "
                    f"step={row['step']} A={row['accept_variant']} "
                    f"R={row['reject_variant']}"
                )


if __name__ == '__main__':
    main()
