# TARR Soft View-Consistency Objective Plan

> Archive note: This plan is historical. Use docs_my/TARR/ and scripts_my/tarr/ as the current source of truth unless this file is explicitly referenced for context.

## Summary

TARR의 다음 active plan은 target augmentation views의 soft prediction distribution을 이용하는 TTA objective를 검증하는 것이다.

공통 gate는 다음과 같다.

```text
clean ID ~= csID
semantic OOD != clean ID + csID
```

기존 broad steps/lr/reference sweep은 반복하지 않는다. CIFAR-10 `eval_api` full objective screen에서 시작하고, 통과한 objective만 promoted refinement, CIFAR-100 transfer, robustness 순서로 확장한다.

## Objective Relations

| Method/objective | View usage | Objective meaning | TARR scope |
| --- | --- | --- | --- |
| Tent / `entropy` | Single original view | Target prediction entropy minimization. | Single-view baseline objective. |
| MEMO / `memo_marginal_entropy` | Multiple augmentation views | Entropy minimization on the average prediction across views. | Primary soft view objective candidate. |
| CoTTA-inspired consistency | Multiple augmentation views | Augmentation-averaged prediction and soft cross-view consistency as a stabilizing idea. | Only the averaged-prediction/consistency idea is used. No teacher, EMA, or stochastic restoration implementation. |

## Code State

- `adaptation.py` supports `predicted_label_ce`, `entropy`, `memo_marginal_entropy`, `view_consistency_js`, `view_consistency_kl`, and `entropy_consistency`.
- `eval.py` builds differentiable gaussian pixel/feature views for soft view objectives.
- Hard label-match objectives and diagnostics are not part of the current design.
- `scoring.py`, `cache.py`, and `reports.py` use response cache schema v5 and keep perturbation diagnostic scores separate from active `--score-rule all`.
- Current perturbation diagnostic scores are `logit_l2`, `prob_l1`, `conf_drop`, and `entropy_increase`.

## Active Experiment Plan

### Phase 1: CIFAR-10 `eval_api` Full Objective Screen

Run full OOD and FSOOD for each objective:

```text
dataset: cifar10
baseline_protocol: eval_api
scheme: ood, fsood
fsood_csid_dataset: cifar10c
objectives:
  - memo
  - view_consistency_js
  - view_consistency_kl
  - entropy_consistency
score_rule: all
save_response_cache: true
response_cache_schema: v5
view_response: pixel
view_kind: gaussian
view_count: 8
```

Required outputs:

- `ood.csv` for every active score rule.
- `alignment_summary.csv`.
- `perturbation_summary.csv`.
- `perturbation_alignment_summary.csv`.
- clean-only, csID-only, and both-ID-side FSOOD rescore.
- strict `cache.py validate` result.

### Phase 2: Promoted Refinement

Only Phase 1 candidates that pass the common gate can be refined. Refinement is limited to learning rate, step count, view count, and one reference filter or reference-size adjustment. The objective formula, score direction, protocol, and csID identity must stay fixed while refining.

### Phase 3: CIFAR-100 Transfer

Transfer the promoted CIFAR-10 recipe to CIFAR-100 `eval_api` OOD/FSOOD before CIFAR-100-specific tuning. Strict transfer, resource-adjusted transfer, and CIFAR-100 tuned results must be reported separately.

### Phase 4: Robustness

For promoted settings, run reference seeds and view seeds/counts as predeclared robustness checks. Report mean/variance and do not select the best seed as the claim row.

## Test Plan

- Static:
  - `conda run -n openood python -m py_compile scripts_my/tarr/*.py`
  - `conda run -n openood python scripts_my/tarr/eval.py --help`
  - `conda run -n openood python scripts_my/tarr/cache.py --help`
  - `conda run -n openood python scripts_my/tarr/reports.py --help`
- Unit/synthetic:
  - KL/JS/entropy-consistency toy tensors produce finite losses.
  - `eps=0`, `repeats=1`, or non-gaussian soft view objective usage fails clearly.
  - removed hard label-match objectives and stale score rules are rejected.
- Full:
  - CIFAR-10 OOD/FSOOD full objective screen first.
  - Promoted refinement second.
  - CIFAR-100 OOD/FSOOD transfer third.
  - Robustness last.

## Assumptions

- Existing older-schema caches are historical and are not modified.
- Response cache schema v5 is the current claim-valid cache schema for new soft view-consistency runs.
- View diagnostic score families are diagnostic-only unless promoted by a predeclared protocol.
- csID results cannot be used to tune thresholds or policy for a claim-bearing row.
 
