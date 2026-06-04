# TARR Soft View-Consistency Objective 전환 계획

> Archive note: This plan is historical. Use docs_my/TARR/ and scripts_my/tarr/ as the current source of truth unless this file is explicitly referenced for context.

## Summary
hard argmax-matching 기반 TTA objective와 diagnostic을 모두 제거한다. 새 claim 후보는 argmax 일치율이 아니라 perturbation view의 soft distribution 정보를 쓰는 objective로만 구성한다.

- `memo_marginal_entropy`: augmentation/perturbation view 평균 출력 `p_bar`의 entropy 최소화
- `view_consistency_kl`, `view_consistency_js`: view별 prediction distribution을 평균 prediction에 맞춤
- `entropy_consistency`: view별 entropy 변동성 최소화

문서에는 관련성을 명확히 적는다: Tent는 single-view entropy minimization, MEMO는 augmentation view 평균 출력 entropy minimization, CoTTA는 augmentation-averaged prediction/consistency 아이디어와 연결된다. CoTTA의 teacher/EMA/stochastic restoration 전체를 구현하는 것은 아니다.

## Code Changes
- `scripts_my/tarr/adaptation.py`
  - public `OBJECTIVES`를 다음으로 정리한다:
    - `predicted_label_ce`
    - `entropy`
    - `memo_marginal_entropy`
    - `view_consistency_kl`
    - `view_consistency_js`
    - `entropy_consistency`
  - hard argmax-matching objective variants 제거.
  - `RUNTIME_IMPL_VERSION` bump.
  - `tta_config_id()`에 새 objective token 추가: `memo`, `vkl`, `vjs`, `entcons`.
  - hard-match threshold 관련 config 제거.

- `scripts_my/tarr/eval.py`
  - hard-match threshold CLI 제거.
  - hard argmax-match update/gate/skip 로직 제거.
  - perturbation argmax-match diagnostic 제거.
  - differentiable view-logit helper 추가:
    - pixel gaussian: `net(data + noise)`
    - feature gaussian: `classifier(feature + noise)`
    - objective path에서는 `torch.no_grad()`나 detached logits를 쓰지 않음.
  - 새 losses:
    - `memo_marginal_entropy`: `H(mean_i p_i)`
    - `view_consistency_kl`: `mean_i KL(p_i || stopgrad(p_bar))`
    - `view_consistency_js`: `H(p_bar) - mean_i H(p_i)`
    - `entropy_consistency`: `Var_i H(p_i)`
  - view objectives는 `--perturbation-response pixel|feature`, `--perturbation-kind gaussian`, `--perturbation-eps > 0`, `--perturbation-repeats >= 2`를 요구한다.

- `scripts_my/tarr/scoring.py`, `cache.py`, `reports.py`
  - response cache schema를 v5로 bump.
  - schema v5에서 제거:
    - `tta_update_weight`
    - `tta_update_applied`
    - `tta_skipped`
    - perturbation argmax-match fields
  - perturbation diagnostic score에서 제거:
    - perturbation argmax-match rule
    - hard argmax-match diagnostic rule
  - 남는 perturbation diagnostic score:
    - `logit_l2`
    - `prob_l1`
    - `conf_drop`
    - `entropy_increase`
  - validation/report는 schema v5를 current claim-valid schema로 처리하고, hard argmax-match field가 있는 older cache는 historical/diagnostic로만 취급한다.

## Documentation
- `implementation.md`
  - TTA objective table 추가:
    - `entropy`: Tent-style single-view entropy minimization
    - `memo_marginal_entropy`: MEMO-style average-view prediction entropy minimization
    - `view_consistency_kl/js`: CoTTA의 augmentation-averaged prediction/consistency 아이디어와 연결된 soft consistency objective
    - `entropy_consistency`: view별 uncertainty stability objective
  - hard argmax-matching objective와 diagnostic은 현재 TARR design에서 사용하지 않는다고 정리한다.
  - response cache schema v5 field policy와 perturbation objective constraints를 정리한다.

- `experiments.md`
  - active 실험 phase를 soft view objective 기준으로 재작성한다.
  - stale broad-sweep phase는 active plan에서 제거한다.
  - result table 컬럼:
    - Dataset, Protocol, Scheme, Run ID, Objective, Perturbation, Reference, Score
    - Clean response, csID response, OOD response
    - Clean-csID alignment, Near/Far AUROC, Avg FPR95, Gate, Decision

- `ablations.md`
  - objective ablation을 `CE/entropy/MEMO/KL/JS/entropy-consistency` 중심으로 정리한다.
  - promotion gate는 항상 `clean ~= csID`와 `semantic OOD separation`을 함께 요구한다.
  - hard argmax-match 계열은 ablation 후보에서도 제외한다.

- `notes.md`
  - open issue를 `csID-aligned soft view objective`와 `perturbation-response gate` 중심으로 축소한다.
  - hard argmax-match 관련 TODO나 실험 후보는 제거한다.

## Experiment Phases
- Phase 0: schema v5 full-run readiness
  - static checks, help checks, cache validate/rescore/report path 확인.
  - preflight가 필요하면 `/tmp`에서만 수행하고 `results_test/tarr/outputs`에는 full run만 남긴다.

- Phase 1: CIFAR-10 eval_api full objective screen
  - GPU0: `memo_marginal_entropy`, `view_consistency_js`
  - GPU1: `view_consistency_kl`, `entropy_consistency`
  - fixed defaults: pixel gaussian, repeats 8, steps 5, lr 1e-2, classifier update, freeze BN, `score_rule=all`.
  - reference configs: `all_rpc32`, `highconf_rpc32`, `correcthigh_rpc32`, `correct_confidence_stratified_rpc32`.

- Phase 2: CIFAR-10 promoted setting refinement
  - top 1-2 objectives only.
  - compare pixel vs feature gaussian perturbation.
  - optional steps 10 only if Phase 1 passes clean/csID alignment.

- Phase 3: CIFAR-100 transfer
  - no retuning except dataset-specific reference size if necessary.
  - same objective, perturbation recipe, score rule selection policy.

- Phase 4: robustness
  - CIFAR-10 seeds 0/1/2 for selected setting.
  - CIFAR-100 seeds 0/1/2 only if transfer is promising.
  - final claim uses FSOOD `both`; `clean` and `csid` remain diagnostic.

## Subagent Allocation
- Agent 1: `eval.py` and `adaptation.py`
  - objective list cleanup, hard argmax-match removal, differentiable view helper, MEMO/KL/JS/entropy-consistency losses.

- Agent 2: `scoring.py`, `cache.py`, `reports.py`
  - schema v5, hard argmax-match diagnostic removal, perturbation diagnostic score cleanup, validation/report compatibility.

- Agent 3: docs
  - implementation/experiments/ablations/notes update, MEMO/Tent/CoTTA relation table.

- Agent 4: experiment runner
  - after implementation, run only full jobs in `results_test/tarr/outputs`, monitor GPUs, collect diagnostics.

- Parent coordinator
  - patch integration, static checks, phase gate decisions, final result table.

## Tests
- Static:
  - `conda run -n openood python -m py_compile scripts_my/tarr/*.py`
  - `conda run -n openood python scripts_my/tarr/eval.py --help`
  - `conda run -n openood python scripts_my/tarr/cache.py --help`
  - `conda run -n openood python scripts_my/tarr/reports.py --help`

- Unit/synthetic:
  - KL/JS/entropy-consistency toy tensors produce finite losses.
  - `eps=0`, `repeats=1`, non-gaussian objective usage fails clearly.
  - removed hard argmax-match objectives are rejected by CLI.
  - removed perturbation argmax-match score rules are rejected.

- Cleanup checks:
  - `rg "hard argmax-match objective identifiers" scripts_my/tarr docs_my/TARR`
  - result should be empty except explicit historical notes if intentionally kept outside canonical docs.

- Full-run acceptance:
  - `run_manifest.json` says full run.
  - no `max_*samples` limit.
  - schema v5 cache validates.
  - `alignment_summary.csv` and `perturbation_alignment_summary.csv` exist.
  - FSOOD main metric is `both`.

## References
- MEMO: https://arxiv.org/abs/2110.09506
- Tent: https://arxiv.org/abs/2006.10726
- CoTTA: https://arxiv.org/abs/2203.13591
