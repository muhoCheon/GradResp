# TARR ImageNet-200 18-Config TTA Broad Search Plan

## Summary

ImageNet-200에서 `refseed=0` 기준 18개 TTA 설정을 full FSOOD로 실행한다. CIFAR-10/CIFAR-100과 동일한 objective/step-lr/perturbation 설정을 사용하고, ImageNet-200 smoke 결과에 따라 **GPU당 Stage 3 process 최대 5개**를 사용한다.

Primary criterion은 FSOOD `both` score_result 개선이며, diagnostics로 `clean < csID << semantic OOD` ordering, clean-only AUROC, csID-only AUROC를 함께 본다. `refseed=1/2` 확장은 ImageNet-200 refseed0 결과 및 다른 dataset refseed0 완료 후 진행한다.

## Key Settings

- Dataset/protocol:
  ```text
  dataset=imagenet200
  baseline_protocol=eval_api
  scheme=fsood
  refseed=0
  ```

- Runtime defaults:
  ```text
  batch_size=64
  reference_set_batch_size=1024
  tta_response_shard_size=1024
  debug_output_mode=none
  num_workers=2
  update_scope=classifier
  runtime_mode=auto
  freeze_bn_stats=true
  score_rule=all
  ```

- Reference configs:
  - Every run uses all 15 prebuilt ImageNet-200 `reference_set` configs for `refseed=0`.
  - Use ImageNet-200 threshold naming:
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
imagenet200_eval_api_fsood_plce_s5_lr1e2_refseed0
imagenet200_eval_api_fsood_plce_s30_lr1e2_refseed0
imagenet200_eval_api_fsood_plce_s10_lr3e2_refseed0

imagenet200_eval_api_fsood_ent_s5_lr1e2_refseed0
imagenet200_eval_api_fsood_ent_s30_lr1e2_refseed0
imagenet200_eval_api_fsood_ent_s10_lr3e2_refseed0

imagenet200_eval_api_fsood_memo_s5_lr1e2_pixgauss_eps1e2_r4_refseed0
imagenet200_eval_api_fsood_memo_s30_lr1e2_pixgauss_eps1e2_r4_refseed0
imagenet200_eval_api_fsood_memo_s10_lr3e2_pixgauss_eps1e2_r4_refseed0

imagenet200_eval_api_fsood_vcjs_s5_lr1e2_pixgauss_eps1e2_r4_refseed0
imagenet200_eval_api_fsood_vcjs_s30_lr1e2_pixgauss_eps1e2_r4_refseed0
imagenet200_eval_api_fsood_vcjs_s10_lr3e2_pixgauss_eps1e2_r4_refseed0

imagenet200_eval_api_fsood_vckl_s5_lr1e2_pixgauss_eps1e2_r4_refseed0
imagenet200_eval_api_fsood_vckl_s30_lr1e2_pixgauss_eps1e2_r4_refseed0
imagenet200_eval_api_fsood_vckl_s10_lr3e2_pixgauss_eps1e2_r4_refseed0

imagenet200_eval_api_fsood_hcons_s5_lr1e2_pixgauss_eps1e2_r4_refseed0
imagenet200_eval_api_fsood_hcons_s30_lr1e2_pixgauss_eps1e2_r4_refseed0
imagenet200_eval_api_fsood_hcons_s10_lr3e2_pixgauss_eps1e2_r4_refseed0
```

Scheduling uses up to 5 Stage 3 processes per GPU:

- Wave 1:
  - GPU0: `plce_s5`, `plce_s30`, `plce_s10`, `ent_s5`, `ent_s30`
  - GPU1: `ent_s10`, `memo_s5`, `memo_s30`, `memo_s10`, `vcjs_s5`

- Wave 2:
  - GPU0: `vcjs_s30`, `vcjs_s10`, `vckl_s5`, `vckl_s30`, `vckl_s10`
  - GPU1: `hcons_s5`, `hcons_s30`, `hcons_s10`

This preserves the objective-major order while using the measured ImageNet-200 concurrency. Stage 4 should run after each wave with low CPU concurrency; do not overlap many Stage 4 jobs if host RAM or IO pressure appears.

## Implementation Changes

- Add ImageNet-200 queue script:
  - `results_test/tarr/job_scripts/tarr_imagenet200_18grid_wave.sh`
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
  - Add ImageNet-200 command log entries.
  - Add ImageNet-200 result table with:
    ```text
    dataset, run_id, objective, steps, lr, perturbation,
    reference_config_id, score_rule,
    both_avg_auroc, both_avg_fpr95,
    clean_avg_auroc, csid_avg_auroc,
    clean_mean_score, csid_mean_score, ood_mean_score,
    ordering_status, baseline_delta, decision
    ```
  - Record `results_test/tarr/summary/imagenet200_parallel_smoke_workers2.csv` as the concurrency justification.

## Analysis and Decisions

- Internal baseline:
  ```text
  predicted_label_ce, steps=5, lr=1e-2, refseed=0
  ```
  Use the best active score/reference row from this run as ImageNet-200 internal TARR baseline.

- Decision labels:
  ```text
  promising
  failed
  needs_refseed
  needs_perturbation_refinement
  ```

- Promote a setting if:
  - FSOOD `both` avg AUROC improves over ImageNet-200 internal baseline, or
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
  - Confirm ImageNet-200 has 45 prebuilt `reference_set` artifacts.
  - Confirm no old ImageNet-200 broad queue is running.
  - Confirm GPUs are idle before starting Wave 1.
  - Confirm disk capacity before starting the full 18-grid queue.

- Full-run acceptance:
  - 18 run directories exist under:
    ```text
    results_test/tarr/outputs/imagenet200/eval_api/seed0/
    ```
  - Each run has `tta_response` for clean ID, csID, near OOD, and far OOD splits.
  - `cache.py validate` passes for all 15 reference configs per run.
  - `collect-score` writes:
    ```text
    results_test/tarr/summary/imagenet200_eval_api_score_results.csv
    ```
  - Best-by-run summary is generated for ImageNet-200, matching the CIFAR-10/CIFAR-100 analysis style.

## Execution Commands

Run Wave 1 on both GPUs:

```bash
GPU_ID=0 bash results_test/tarr/job_scripts/tarr_imagenet200_18grid_wave.sh gpu0_wave1
GPU_ID=1 bash results_test/tarr/job_scripts/tarr_imagenet200_18grid_wave.sh gpu1_wave1
```

After Wave 1 Stage 3/4 completes, run Wave 2:

```bash
GPU_ID=0 bash results_test/tarr/job_scripts/tarr_imagenet200_18grid_wave.sh gpu0_wave2
GPU_ID=1 bash results_test/tarr/job_scripts/tarr_imagenet200_18grid_wave.sh gpu1_wave2
```

Monitor logs:

```bash
tail -f results_test/tarr/job_logs/imagenet200_eval_api_fsood_*_stage3.log
tail -f results_test/tarr/job_logs/imagenet200_eval_api_fsood_*_stage4.log
```

Primary summary output:

```text
results_test/tarr/summary/imagenet200_eval_api_score_results.csv
```

## Assumptions

- Stage 1 and Stage 2 ImageNet-200 artifacts for `refseed=0` already exist.
- ImageNet-200 broad search uses sharded `tta_response`; no single response cache.
- GPU당 5개는 Stage 3-only measured default; reduce to 4 if soft-view or Stage 4 overlap causes memory/IO pressure.
- `refseed=1/2` is intentionally deferred until all target datasets have refseed0 results or a dataset-wise best needs robustness.
