# TARR CIFAR-10 18-Config TTA Broad Search Plan

## Summary

CIFAR-10부터 `refseed=0` 기준으로 TTA objective/step-lr grid를 full FSOOD로 실행한다. 기준은 FSOOD `both` score_result 개선이며, diagnostic으로 `clean < csID << semantic OOD` score ordering을 본다.

Grid는 다음 18개로 고정한다.

```text
objectives:
  predicted_label_ce
  entropy
  memo_marginal_entropy
  view_consistency_js
  view_consistency_kl
  entropy_consistency

step/lr per objective:
  s5_lr1e2
  s30_lr1e2
  s10_lr3e2
```

CIFAR-10은 resource smoke 결과에 따라 **GPU당 Stage 3 process 최대 4개**를 허용한다. 단, objective별 해석을 유지하기 위해 실행은 objective block 단위로 관리한다.

## Execution Strategy

- Dataset/order:
  - Start with `cifar10`, `eval_api`, `fsood`, `refseed=0`.
  - CIFAR-10 결과 분석 후 CIFAR-100으로 넘어간다.
  - ImageNet-200/ImageNet-1K는 CIFAR-10/100에서 promising한 후보만 transfer한다.

- CIFAR-10 runtime defaults:
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
  - Every run uses all 15 prebuilt CIFAR-10 `reference_set` configs for the selected `refseed`.
  - Initial run uses only `refseed=0`.
  - `refseed=1/2` are reserved for best or near-best candidates.

- Perturbation settings:
  - For `predicted_label_ce` and `entropy`:
    ```text
    perturbation_response=none
    ```
  - For `memo_marginal_entropy`, `view_consistency_js`, `view_consistency_kl`, `entropy_consistency`:
    ```text
    perturbation_response=pixel
    perturbation_kind=gaussian
    perturbation_eps=0.01
    perturbation_repeats=4
    perturbation_seed=0
    ```

## Run List

Run IDs:

```text
cifar10_eval_api_fsood_plce_s5_lr1e2_refseed0
cifar10_eval_api_fsood_plce_s30_lr1e2_refseed0
cifar10_eval_api_fsood_plce_s10_lr3e2_refseed0

cifar10_eval_api_fsood_ent_s5_lr1e2_refseed0
cifar10_eval_api_fsood_ent_s30_lr1e2_refseed0
cifar10_eval_api_fsood_ent_s10_lr3e2_refseed0

cifar10_eval_api_fsood_memo_s5_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar10_eval_api_fsood_memo_s30_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar10_eval_api_fsood_memo_s10_lr3e2_pixgauss_eps1e2_r4_refseed0

cifar10_eval_api_fsood_vcjs_s5_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar10_eval_api_fsood_vcjs_s30_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar10_eval_api_fsood_vcjs_s10_lr3e2_pixgauss_eps1e2_r4_refseed0

cifar10_eval_api_fsood_vckl_s5_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar10_eval_api_fsood_vckl_s30_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar10_eval_api_fsood_vckl_s10_lr3e2_pixgauss_eps1e2_r4_refseed0

cifar10_eval_api_fsood_hcons_s5_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar10_eval_api_fsood_hcons_s30_lr1e2_pixgauss_eps1e2_r4_refseed0
cifar10_eval_api_fsood_hcons_s10_lr3e2_pixgauss_eps1e2_r4_refseed0
```

Scheduling:

- Wave 1:
  - GPU0: `predicted_label_ce` 3 runs
  - GPU1: `entropy` 3 runs
- Wave 2:
  - GPU0: `memo_marginal_entropy` 3 runs
  - GPU1: `view_consistency_js` 3 runs
- Wave 3:
  - GPU0: `view_consistency_kl` 3 runs
  - GPU1: `entropy_consistency` 3 runs

This keeps objective blocks intact while staying below the measured safe limit of 4 processes/GPU.

## Stage 4 and Analysis

After every Stage 3 run:

```text
cache.py validate
cache.py score --fsood-id-side both --score-rule all
cache.py score --fsood-id-side clean --score-rule all
cache.py score --fsood-id-side csid --score-rule all
cache.py score --vector-score-rule all
reports.py diagnostics
reports.py collect-score
```

For perturbation objectives also run:

```text
cache.py score --perturbation-score-rule all
```

Result table columns:

```text
dataset, run_id, objective, steps, lr, perturbation,
reference_config_id, score_rule,
both_avg_auroc, both_avg_fpr95,
clean_avg_auroc, csid_avg_auroc,
clean_mean_score, csid_mean_score, ood_mean_score,
ordering_status, runtime, decision
```

Decision labels:

```text
promising
failed
needs_refseed
needs_perturbation_refinement
```

## Perturbation Refinement Rule

Only refine perturbation options if at least one soft-view objective shows either:

```text
both_avg_auroc >= best perturbation-free candidate - 0.5pp
```

or

```text
csid_avg_auroc / clean<csID<<OOD ordering clearly improves
```

If soft-view is weak, do not expand perturbation immediately.

If refinement is triggered, test only the best one or two soft-view objectives with:

```text
perturbation_eps: 0.005, 0.01, 0.02
perturbation_repeats: 4, 8
perturbation_response: pixel, feature
```

Do not run the full Cartesian product. Start with:

```text
best_soft_objective + best_step_lr:
  pixel eps=0.005 repeats=4
  pixel eps=0.02  repeats=4
  pixel eps=0.01  repeats=8
  feature eps=0.01 repeats=4
```

## Subagent Allocation

- Agent 1: GPU0 runner
  - Runs Wave 1 GPU0, Wave 2 GPU0, Wave 3 GPU0.
  - Owns Stage 3 logs and failure handling for GPU0 jobs.

- Agent 2: GPU1 runner
  - Runs Wave 1 GPU1, Wave 2 GPU1, Wave 3 GPU1.
  - Owns Stage 3 logs and failure handling for GPU1 jobs.

- Agent 3: scorer/analyzer
  - Runs Stage 4 after each completed run.
  - Maintains collected score CSV and extracts best row per run.
  - Produces the clean/csID/OOD ordering summary.

- Agent 4: documentation tracker
  - Updates `experiments.md` with command log, result table, decisions, and artifact paths.
  - Records failed branches and next perturbation-refinement decisions.

- Parent coordinator
  - Enforces no more than 4 Stage 3 processes per GPU.
  - Starts the next wave only after the current wave has finished Stage 3 and critical Stage 4 scoring.
  - Decides whether perturbation refinement is justified.

## Assumptions

- Stage 1 and Stage 2 CIFAR-10 artifacts for `refseed=0` already exist.
- Initial search is CIFAR-10 only.
- `refseed=1/2` is not run until a candidate is selected as promising.
- Soft-view objectives use pixel gaussian perturbation by default.
- Stage 4 is allowed to run immediately after Stage 3, but with low CPU concurrency to avoid RAM pressure.
