# TARR CIFAR-100 18-Config TTA Broad Search Plan

## Summary

CIFAR-100에서 `refseed=0` 기준 18개 TTA 설정을 full FSOOD로 실행한다. CIFAR-10과 동일한 objective/step-lr/perturbation 설정을 사용하고, CIFAR-100 smoke 결과에 따라 **GPU당 Stage 3 process 최대 5개**를 사용한다.

Primary criterion은 FSOOD `both` score_result 개선이며, diagnostics로 `clean < csID << semantic OOD` ordering, clean-only AUROC, csID-only AUROC를 함께 본다. `refseed=1/2` 확장은 CIFAR-100 refseed0 결과 및 다른 dataset refseed0 완료 후 진행한다.

## Key Settings

- Dataset/protocol:
  ```text
  dataset=cifar100
  baseline_protocol=eval_api
  scheme=fsood
  refseed=0
  ```

- Runtime defaults:
  ```text
  batch_size=512
  reference_set_batch_size=2048
  tta_response_shard_size=1024
  debug_output_mode=none
  num_workers=0
  update_scope=classifier
  runtime_mode=auto
  freeze_bn_stats=true
  score_rule=all
  ```

- Reference configs:
  - Every run uses all 15 prebuilt CIFAR-100 `reference_set` configs for `refseed=0`.
  - Use CIFAR/CIFAR-100 threshold naming:
    ```text
    all_rpc8/16/32
    correct_rpc8/16/32
    highconf09_rpc8/16/32
    correcthigh09_rpc8/16/32
    strat_rpc8/16/32
    ```

- Perturbation:
  - `predicted_label_ce`, `entropy`: `perturbation_response=none`
  - `memo_marginal_entropy`, `view_consistency_js`, `view_consistency_kl`, `entropy_consistency`:
    ```text
    perturbation_response=pixel
    perturbation_kind=gaussian
    perturbation_eps=0.01
    perturbation_repeats=4
    perturbation_seed=0
    ```

## Run Grid and Scheduling

Run IDs:

```text
cifar100_eval_api_fsood_plce_s5_lr1e2_refseed0
cifar100_eval_api_fsood_plce_s30_lr1e2_refseed0
cifar100_eval_api_fsood_plce_s10_lr3e2_refseed0

cifar100_eval_api_fsood_ent_s5_lr1e2_refseed0
cifar100_eval_api_fsood_ent_s30_lr1e2_refseed0
cifar100_eval_api_fsood_ent_s10_lr3e2_refseed0

cifar100_eval_api_fsood_memo_s5_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar100_eval_api_fsood_memo_s30_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar100_eval_api_fsood_memo_s10_lr3e2_pixgauss_eps1e2_r4_refseed0

cifar100_eval_api_fsood_vcjs_s5_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar100_eval_api_fsood_vcjs_s30_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar100_eval_api_fsood_vcjs_s10_lr3e2_pixgauss_eps1e2_r4_refseed0

cifar100_eval_api_fsood_vckl_s5_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar100_eval_api_fsood_vckl_s30_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar100_eval_api_fsood_vckl_s10_lr3e2_pixgauss_eps1e2_r4_refseed0

cifar100_eval_api_fsood_hcons_s5_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar100_eval_api_fsood_hcons_s30_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar100_eval_api_fsood_hcons_s10_lr3e2_pixgauss_eps1e2_r4_refseed0
```

Scheduling uses up to 5 Stage 3 processes per GPU:

- Wave 1:
  - GPU0: `plce_s5`, `plce_s30`, `plce_s10`, `ent_s5`, `ent_s30`
  - GPU1: `ent_s10`, `memo_s5`, `memo_s30`, `memo_s10`, `vcjs_s5`

- Wave 2:
  - GPU0: `vcjs_s30`, `vcjs_s10`, `vckl_s5`, `vckl_s30`, `vckl_s10`
  - GPU1: `hcons_s5`, `hcons_s30`, `hcons_s10`

This preserves the objective-major order while using the measured CIFAR-100 concurrency. Stage 4 should run after each wave with low CPU concurrency; do not overlap many Stage 4 jobs if host RAM or IO pressure appears.

## Implementation Changes

- Add a CIFAR-100 queue script modeled after the CIFAR-10 18-grid script:
  - `results_test/tarr/job_scripts/tarr_cifar100_18grid_wave.sh`
  - Role names:
    ```text
    gpu0_wave1
    gpu1_wave1
    gpu0_wave2
    gpu1_wave2
    ```
  - Each role starts its assigned Stage 3 runs in parallel, waits for completion, then runs Stage 4 sequentially for each completed run.

- Stage 4 per run:
  ```text
  cache.py validate
  cache.py score --fsood-id-side both --score-rule all
  cache.py score --fsood-id-side clean --score-rule all
  cache.py score --fsood-id-side csid --score-rule all
  cache.py score --vector-score-rule all
  reports.py diagnostics
  reports.py collect-score
  ```

- For soft-view runs also run:
  ```text
  cache.py score --perturbation-score-rule all
  ```

- Update experiment docs:
  - Add CIFAR-100 command log entries.
  - Add result table with:
    ```text
    dataset, run_id, objective, steps, lr, perturbation,
    reference_config_id, score_rule,
    both_avg_auroc, both_avg_fpr95,
    clean_avg_auroc, csid_avg_auroc,
    clean_mean_score, csid_mean_score, ood_mean_score,
    ordering_status, baseline_delta, decision
    ```
  - Record `results_test/tarr/summary/cifar100_parallel_smoke_workers0.csv` as the concurrency justification.

## Analysis and Decisions

- Internal baseline:
  ```text
  predicted_label_ce, steps=5, lr=1e-2, refseed=0
  ```
  Use the best active score/reference row from this run as CIFAR-100 internal TARR baseline.

- Decision labels:
  ```text
  promising
  failed
  needs_refseed
  needs_perturbation_refinement
  ```

- Promote a setting if:
  - FSOOD `both` avg AUROC improves over CIFAR-100 internal baseline, or
  - `both` is comparable but csID-only AUROC and score ordering improve clearly.

- Perturbation refinement rule:
  - Only refine if best soft-view row is within `0.5pp` of best perturbation-free row, or improves csID/order clearly.
  - If triggered, refine only the best 1-2 soft-view objectives:
    ```text
    pixel eps=0.005 repeats=4
    pixel eps=0.02 repeats=4
    pixel eps=0.01 repeats=8
    feature eps=0.01 repeats=4
    ```

## Test and Acceptance Criteria

- Preflight:
  - Confirm CIFAR-100 has 45 prebuilt `reference_set` artifacts.
  - Confirm no old CIFAR-100 broad queue is running.
  - Confirm GPUs are idle before starting Wave 1.

- Full-run acceptance:
  - 18 run directories exist under:
    ```text
    results_test/tarr/outputs/cifar100/eval_api/seed0/
    ```
  - Each run has `tta_response` for clean ID, csID, near OOD, and far OOD splits.
  - `cache.py validate` passes for all 15 reference configs per run.
  - `collect-score` writes:
    ```text
    results_test/tarr/summary/cifar100_eval_api_score_results.csv
    ```
  - Best-by-run summary is generated for CIFAR-100, matching the CIFAR-10 analysis style.

## Execution Commands

Run Wave 1 on both GPUs:

```bash
GPU_ID=0 bash results_test/tarr/job_scripts/tarr_cifar100_18grid_wave.sh gpu0_wave1
GPU_ID=1 bash results_test/tarr/job_scripts/tarr_cifar100_18grid_wave.sh gpu1_wave1
```

After Wave 1 Stage 3/4 completes, run Wave 2:

```bash
GPU_ID=0 bash results_test/tarr/job_scripts/tarr_cifar100_18grid_wave.sh gpu0_wave2
GPU_ID=1 bash results_test/tarr/job_scripts/tarr_cifar100_18grid_wave.sh gpu1_wave2
```

Monitor logs:

```bash
tail -f results_test/tarr/job_logs/cifar100_eval_api_fsood_*_stage3.log
tail -f results_test/tarr/job_logs/cifar100_eval_api_fsood_*_stage4.log
```

Primary summary output:

```text
results_test/tarr/summary/cifar100_eval_api_score_results.csv
```

## Assumptions

- Stage 1 and Stage 2 CIFAR-100 artifacts for `refseed=0` already exist.
- CIFAR-100 broad search uses sharded `tta_response`; no single response cache.
- GPU당 5개는 Stage 3-only measured default; reduce to 4 if soft-view or Stage 4 overlap causes memory/IO pressure.
- `refseed=1/2` is intentionally deferred until all target datasets have refseed0 results or a dataset-wise best needs robustness.

## Completed Run Log

All 18 `refseed=0` full FSOOD runs completed with sharded `tta_response`.

| wave | command | start | end | status |
| --- | --- | --- | --- | --- |
| GPU0 wave1 | `GPU_ID=0 bash results_test/tarr/job_scripts/tarr_cifar100_18grid_wave.sh gpu0_wave1` | 2026-05-27 14:20 KST | 2026-05-27 18:59 KST | completed |
| GPU1 wave1 | `GPU_ID=1 bash results_test/tarr/job_scripts/tarr_cifar100_18grid_wave.sh gpu1_wave1` | 2026-05-27 14:20 KST | 2026-05-27 22:13 KST | completed |
| GPU0 wave2 | `GPU_ID=0 bash results_test/tarr/job_scripts/tarr_cifar100_18grid_wave.sh gpu0_wave2` | 2026-05-27 20:00 KST | 2026-05-28 05:09 KST | completed |
| GPU1 wave2 | `GPU_ID=1 bash results_test/tarr/job_scripts/tarr_cifar100_18grid_wave.sh gpu1_wave2` | 2026-05-27 22:15 KST | 2026-05-28 04:34 KST | completed |

Final collection:

```bash
conda run --no-capture-output -n openood python scripts_my/tarr/reports.py collect-score \
  --dataset cifar100 \
  --baseline-protocol eval_api \
  --runs-root results_test/tarr/outputs \
  --output-csv results_test/tarr/summary/cifar100_eval_api_score_results.csv
```

Outputs:

```text
results_test/tarr/summary/cifar100_eval_api_score_results.csv
results_test/tarr/summary/cifar100_18grid_best_by_run.csv
results_test/tarr/summary/cifar100_compare_group1_all_rpc8_positive_loss_increase_mean.csv
results_test/tarr/summary/cifar100_compare_group1_best_hcons_s30_all_rpc8_positive_loss_increase_mean_avg.csv
```

## Results

Internal baseline:

```text
run_id: cifar100_eval_api_fsood_plce_s5_lr1e2_refseed0
reference_config_id: all_rpc8
score_rule: positive_loss_increase_mean
both avg AUROC: 63.325
both avg FPR95: 75.130
clean-only avg AUROC: 77.515
csID-only avg AUROC: 50.560
```

Top active-score rows:

| rank | run_id | reference_config_id | score_rule | both avg AUROC | both avg FPR95 | clean avg AUROC | csID avg AUROC | baseline delta | ordering |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `cifar100_eval_api_fsood_hcons_s30_lr1e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 64.880 | 75.470 | 74.825 | 55.925 | +1.555 | `clean<csid<ood` |
| 2 | `cifar100_eval_api_fsood_hcons_s5_lr1e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 64.840 | 75.525 | 74.905 | 55.780 | +1.515 | `clean<csid<ood` |
| 3 | `cifar100_eval_api_fsood_hcons_s10_lr3e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 64.805 | 75.485 | 74.795 | 55.815 | +1.480 | `clean<csid<ood` |
| 4 | `cifar100_eval_api_fsood_vckl_s5_lr1e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 64.255 | 75.345 | 71.430 | 57.795 | +0.930 | `clean<csid<ood` |
| 5 | `cifar100_eval_api_fsood_vcjs_s5_lr1e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 64.255 | 75.345 | 71.430 | 57.795 | +0.930 | `clean<csid<ood` |

Decision:

- `entropy_consistency` is the best CIFAR-100 `refseed=0` branch by FSOOD `both` AUROC.
- `hcons_s30`, `hcons_s5`, and `hcons_s10` are effectively tied; the larger TTA budget gives the best point estimate.
- `view_consistency_kl/js` improves csID-only AUROC more strongly than the internal baseline, but its `both` AUROC is lower than `entropy_consistency`.
- `predicted_label_ce` longer budgets are only marginally above the internal baseline.
- `entropy` and `memo_marginal_entropy` branches are below the internal baseline in this grid.

Group 1 comparison for the best row (`hcons_s30`, `all_rpc8`, `positive_loss_increase_mean`):

- TARR avg AUROC: `64.880`.
- Above KNN by `+1.315pp`, MSP by `+1.395pp`, EBO by `+1.275pp`, MLS by `+1.130pp`.
- Below RMDS by `-1.320pp`.

Promotion:

```text
needs_refseed: hcons_s30, hcons_s5, hcons_s10
promising diagnostic branch: vckl_s5/vcjs_s5 for csID AUROC
failed/no expansion: entropy, memo_marginal_entropy, PLCE-only budget sweep
```
