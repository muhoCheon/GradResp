# TARR Experiments

This document tracks the current dataset-wise TTA broad search for the
four-stage TARR artifact pipeline. Old experiment tables are historical and
live in `legacy_experiments.md`.

## Claim-Bearing Result Requirements

A result is claim-bearing only when it is produced by the full four-stage TARR pipeline:

```text
train_candidate_metadata -> reference_set -> tta_response -> score_result
```

For the Probe/Anchor plan, the existing `reference_set` is the
Probe/Measurement set `P`. The optional new `anchor_set` is Anchor `A`, stored
under `anchor_sets/`, and regularizes the target update only. Scores remain
measured on Probe `P`.

Stage requirements:

- Stage 1, `train_candidate_metadata`: built from the ID train split only, using the declared dataset, train imglist, checkpoint, model, and preprocessor.
- Stage 2P, `reference_set` / Probe `P`: selected only from the Stage 1 metadata. The reference filter, per-class count, confidence threshold, and reference seed must be recorded.
- Stage 2A, `anchor_set` / Anchor `A`, when enabled: selected only from Stage 1 metadata, stored under `anchor_sets/`, and disjoint from the paired Probe `P`.
- Stage 3, `tta_response`: run on the full target splits required by the scheme. For FSOOD, this means clean ID, csID, near semantic OOD, and far semantic OOD. `--steps` is the maximum update step count; `--save-steps` records the saved `response_steps` and defaults to the final step only. No `--max-id-samples` or `--max-ood-samples` limit may be used for a claim-bearing row.
- Stage 4, `score_result`: computed from the saved `tta_response` and written under `score_results/<score_rule>/` for the reported score rule, selected `response_step`, and FSOOD ID-side setting.

Identity and validation requirements:

- The manifests must match the reported `dataset`, `scheme`, `baseline_protocol`, csID identity, checkpoint identity, and imglist SHA256.
- `ood_score` must be stored in the OOD-like direction. OpenOOD `conf` is `-ood_score`.
- Claim-bearing rows must record the selected `response_step`; if multiple `response_steps` were saved, the reported row is tied to exactly one Stage 4 `--response-step`.
- Smoke or subset runs are code/runtime checks only. They can justify implementation or resource settings, but they are not used for performance claims.

Inline scoring and offline scoring are two ways to run Stage 4. They produce the same `score_result` artifact when the `tta_response` input and scoring config match.

## Current Goal

Find the best TARR TTA setting separately for each dataset. The primary internal criterion is whether FSOOD `both` score_result improves over the default TARR setting for the same dataset: `predicted_label_ce`, `steps=5`, final `response_step=5`, `lr=1e-2`, `refseed=0`.

Diagnostics track whether the score distribution follows:

```text
clean < csID << semantic OOD
```

Primary datasets, all with `baseline_protocol=eval_api` and `scheme=fsood`:

| Dataset CLI name | Display name | clean ID | csID | semantic OOD |
| --- | --- | --- | --- | --- |
| `cifar10` | CIFAR-10 | `cifar10` | `cifar10c` | near: `cifar100`, `tin`; far: `mnist`, `svhn`, `texture`, `places365` |
| `cifar100` | CIFAR-100 | `cifar100` | `cifar100c` | near: `cifar10`, `tin`; far: `mnist`, `svhn`, `texture`, `places365` |
| `imagenet200` | ImageNet-200 | `imagenet200` | `imagenet_v2`, `imagenet_c`, `imagenet_r` | near: `ssb_hard`, `ninco`; far: `inaturalist`, `textures`, `openimage_o` |
| `imagenet` | ImageNet-1K | `imagenet` | `imagenet_v2`, `imagenet_c`, `imagenet_r` | near: `ssb_hard`, `ninco`; far: `inaturalist`, `textures`, `openimage_o` |

Terms:

- `clean ID`: the uncorrupted in-distribution test split.
- `csID`: covariate-shifted ID. It should remain semantically in-distribution but differs in corruption, style, or collection source.
- `semantic OOD`: classes or visual concepts outside the ID label space. FSOOD reports near and far semantic OOD groups.

Dataset-wise best settings are selected independently. Do not force a single global TTA setting unless it is competitive on every dataset.

## Execution Order

Run one dataset family at a time. Use both GPUs to split TTA candidates for the
same dataset, finish Stage 3/4 and analysis for that dataset, then move to the
next dataset.

Canonical broad-search order:

```text
cifar10 -> cifar100 -> imagenet200 -> imagenet
```

Rationale:

- Dataset-specific runtime and memory settings are easier to tune and audit.
- A failed or memory-heavy dataset cannot obscure progress from another dataset.
- The internal baseline and best TTA setting are decided per dataset before
  spending compute on larger datasets.

Within a dataset, parallelize by TTA candidate, not by dataset:

```text
GPU0: subset of TTA candidates for the active dataset
GPU1: remaining TTA candidates for the same active dataset
```

For CIFAR-10, the current resource benchmark supports running four Stage 3
processes per GPU with `num_workers=0`, sharded `tta_response`, and
`debug_output_mode=none`. Use this as the default CIFAR-10 broad-search
concurrency. Five processes per GPU can improve raw throughput slightly, but
four is the safer default before Stage 4 scoring/reporting is added.

For CIFAR-100, the measured Stage 3 smoke with `predicted_label_ce`, `steps=5`,
`batch_size=512`, `reference_set_batch_size=2048`, sharded `tta_response`, and
`num_workers=0` completed up to five processes per GPU. Five processes gave the
best raw Stage 3 throughput in the smoke, with max VRAM about `3.0GiB` on GPU0.
Use five processes per GPU for CIFAR-100 Stage 3-only waves; reduce to four if
Stage 4 scoring is overlapped heavily or if a soft-view wave shows memory
pressure.

For ImageNet-200, the measured Stage 3 smoke with `predicted_label_ce`,
`steps=5`, `batch_size=64`, `reference_set_batch_size=1024`, sharded
`tta_response`, and `num_workers=2` supports five processes per GPU as the best
Stage 3-only setting. The smoke reached about `80%` average GPU utilization with
about `3.7GiB` max VRAM. Use five processes per GPU for ImageNet-200 Stage 3
waves when Stage 4 is not overlapped; reduce to four if host IO pressure,
worker contention, or Stage 4 overlap appears.

Use low-concurrency CPU scoring/report generation after Stage 3 artifacts are
written. Do not run many CPU-heavy Stage 4 jobs while GPU Stage 3 jobs are
memory-bound.

## Execution Unit

One full broad-search execution unit is:

```text
dataset + eval_api + fsood + TTA config + response_step + reference seed
```

Each execution unit bundles 15 prebuilt `reference_set` configs and then runs
Stage 4 immediately after Stage 3 for the selected `response_step`.

Required post-run artifacts for a completed execution unit:

- `tta_response` for clean ID, csID, near OOD, and far OOD target splits.
- Active `score_result` for `both`, `clean`, and `csid` FSOOD ID-side scoring at the selected `response_step`.
- Vector diagnostic `score_result`.
- Perturbation diagnostic `score_result` for soft-view runs.
- Diagnostics for each reference config.
- `collect-score` dataset summary.
- `compare-group1` output for selected exact score rules.

## TTA Candidate Grid

The current broad-search grid has 18 TTA settings. CIFAR-10 and CIFAR-100 were
both run with the full grid at `refseed=0`.

Step/lr grid used for every objective:

```text
s5_lr1e2:  steps=5,  lr=1e-2
s30_lr1e2: steps=30, lr=1e-2
s10_lr3e2: steps=10, lr=3e-2
```

Objective grid:

| Objective | TTA IDs | Perturbation |
| --- | --- | --- |
| `predicted_label_ce` | `plce_s5_lr1e2`, `plce_s30_lr1e2`, `plce_s10_lr3e2` | none |
| `entropy` | `ent_s5_lr1e2`, `ent_s30_lr1e2`, `ent_s10_lr3e2` | none |
| `memo_marginal_entropy` | `memo_s5_lr1e2`, `memo_s30_lr1e2`, `memo_s10_lr3e2` | pixel gaussian, eps `0.01`, repeats `4`, seed `0` |
| `view_consistency_js` | `vcjs_s5_lr1e2`, `vcjs_s30_lr1e2`, `vcjs_s10_lr3e2` | pixel gaussian, eps `0.01`, repeats `4`, seed `0` |
| `view_consistency_kl` | `vckl_s5_lr1e2`, `vckl_s30_lr1e2`, `vckl_s10_lr3e2` | pixel gaussian, eps `0.01`, repeats `4`, seed `0` |
| `entropy_consistency` | `hcons_s5_lr1e2`, `hcons_s30_lr1e2`, `hcons_s10_lr3e2` | pixel gaussian, eps `0.01`, repeats `4`, seed `0` |

Current findings:

- CIFAR-10 favors `entropy` and `memo_marginal_entropy`. The best row is `ent_s30_lr1e2`; `ent_s10_lr3e2` and `memo_s30_lr1e2` are near-best.
- CIFAR-100 favors `entropy_consistency`. The best row is `hcons_s30_lr1e2`; `hcons_s5_lr1e2` and `hcons_s10_lr3e2` are near-best.
- CIFAR-100 `view_consistency_js/kl` is not the best by FSOOD `both` AUROC, but it improves csID-only AUROC and remains a diagnostic promising branch.
- One global TTA objective is not supported by current evidence. Select dataset-wise candidates independently.

Transfer policy:

- CIFAR-10 and CIFAR-100: full 18-config grid is already completed for `refseed=0`.
- ImageNet-200: run the full 18-config grid at `refseed=0` using the measured five-process/GPU setting with `num_workers=2`.
- ImageNet-1K: run only selected candidates first. Full 18-config broad search is disk-expensive and should not be the default.
- Refseed robustness is run only after the dataset-wise `refseed=0` screen identifies promising candidates.

## Probe/Anchor 2x2 Plan

Acceptance/Rejection Probe and Anchor `A` are separate improvement axes.
Anchor `A` should first use ID train samples disjoint from Probe `P`; do not
optimize Anchor selection before the basic 2x2 result is understood.

| Setting | Acceptance/Rejection | Anchor `A` | Purpose |
| --- | --- | --- | --- |
| Baseline TARR | off | off | Existing Probe `P` measurement baseline. |
| Anchor only | off | on | Test whether ID-surface regularization stabilizes the existing objective. |
| Acceptance/Rejection only | on | off | Test whether acceptance-vs-rejection compatibility contrast adds signal. |
| Acceptance/Rejection + Anchor | on | on | Test whether the two axes are complementary. |

Smoke commands are code/runtime checks only and must use sample limits. Full
ablation rows remove sample limits, run Stage 4 for `both`, `clean`, and
`csid`, and pass the same validation gates as the current four-stage pipeline.

Status in this patch: implementation and smoke-test commands are added, but
full Probe/Anchor ablation experiments are not run here. The existing
dataset-wise best table is unchanged until full rows pass validation.

## Reference Seed Plan

Each Stage 3 run uses one reference seed and 15 reference configs. Run ids include the reference seed:

```text
<dataset>_eval_api_fsood_<tta_id>_refseed0
<dataset>_eval_api_fsood_<tta_id>_refseed1
<dataset>_eval_api_fsood_<tta_id>_refseed2
```

Start with `refseed0` for all datasets/TTA candidates. Extend to `refseed1/2` only for promising candidates or when robustness is needed for a dataset-wise best.

Reference coverage preflight:

| Dataset | Expected prebuilt `reference_set` artifacts | Current status |
| --- | ---: | --- |
| `cifar10` | 45 | verified |
| `cifar100` | 45 | verified |
| `imagenet200` | 45 | verified |
| `imagenet` | 45 | verified |

## Command Log

Record the exact command for each broad-search execution unit.

| timestamp | agent | dataset | run_id | command | status | runtime | output_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-27 02:33 KST | Codex | `cifar10` | Wave 1 GPU0, `predicted_label_ce` block | `GPU_ID=0 bash results_test/tarr/job_scripts/tarr_cifar10_18grid_wave.sh gpu0_wave1` | completed | Stage 3/4 done by 03:25 KST | `results_test/tarr/outputs/cifar10/eval_api/seed0/cifar10_eval_api_fsood_plce_*_refseed0` | 3 runs in parallel on GPU0. |
| 2026-05-27 02:33 KST | Codex | `cifar10` | Wave 1 GPU1, `entropy` block | `GPU_ID=1 bash results_test/tarr/job_scripts/tarr_cifar10_18grid_wave.sh gpu1_wave1` | completed | Stage 3/4 done by 03:42 KST | `results_test/tarr/outputs/cifar10/eval_api/seed0/cifar10_eval_api_fsood_ent_*_refseed0` | 3 runs in parallel on GPU1. |
| 2026-05-27 03:44 KST | Codex | `cifar10` | Wave 2 GPU0, `memo_marginal_entropy` block | `GPU_ID=0 bash results_test/tarr/job_scripts/tarr_cifar10_18grid_wave.sh gpu0_wave2` | completed | Stage 3/4 done by 08:20 KST | `results_test/tarr/outputs/cifar10/eval_api/seed0/cifar10_eval_api_fsood_memo_*_refseed0` | Pixel gaussian eps `0.01`, repeats `4`. |
| 2026-05-27 03:44 KST | Codex | `cifar10` | Wave 2 GPU1, `view_consistency_js` block | `GPU_ID=1 bash results_test/tarr/job_scripts/tarr_cifar10_18grid_wave.sh gpu1_wave2` | completed | Stage 3/4 done by 08:43 KST | `results_test/tarr/outputs/cifar10/eval_api/seed0/cifar10_eval_api_fsood_vcjs_*_refseed0` | Pixel gaussian eps `0.01`, repeats `4`. |
| 2026-05-27 08:21 KST | Codex | `cifar10` | Wave 3 GPU0, `view_consistency_kl` block | `GPU_ID=0 bash results_test/tarr/job_scripts/tarr_cifar10_18grid_wave.sh gpu0_wave3` | completed | Stage 3/4 done by 12:37 KST | `results_test/tarr/outputs/cifar10/eval_api/seed0/cifar10_eval_api_fsood_vckl_*_refseed0` | Pixel gaussian eps `0.01`, repeats `4`. |
| 2026-05-27 08:43 KST | Codex | `cifar10` | Wave 3 GPU1, `entropy_consistency` block | `GPU_ID=1 bash results_test/tarr/job_scripts/tarr_cifar10_18grid_wave.sh gpu1_wave3` | completed | Stage 3/4 done by 12:59 KST | `results_test/tarr/outputs/cifar10/eval_api/seed0/cifar10_eval_api_fsood_hcons_*_refseed0` | Pixel gaussian eps `0.01`, repeats `4`. |
| 2026-05-27 13:27 KST | Codex | `cifar100` | Stage 3 parallel smoke | `GPU_ID=0 NUM_WORKERS=0 PARALLEL_COUNTS='1 2 3 4' bash results_test/tarr/job_scripts/tarr_cifar100_parallel_smoke.sh`; then `PARALLEL_COUNTS='5'` | completed | `n=1..5`: 235s, 275s, 319s, 369s, 426s | `results_test/tarr/summary/cifar100_parallel_smoke_workers0.csv` | Non-claim-bearing. Smoke outputs deleted after each wave. |
| 2026-05-27 14:20 KST | Codex | `cifar100` | Wave 1 GPU0, `predicted_label_ce` plus partial `entropy` block | `GPU_ID=0 bash results_test/tarr/job_scripts/tarr_cifar100_18grid_wave.sh gpu0_wave1` | completed | Stage 3/4 done by 18:59 KST | `results_test/tarr/outputs/cifar100/eval_api/seed0/cifar100_eval_api_fsood_plce_*_refseed0`; `.../ent_s5_*`; `.../ent_s30_*` | Five Stage 3 processes on GPU0. |
| 2026-05-27 14:20 KST | Codex | `cifar100` | Wave 1 GPU1, remaining `entropy`, `memo`, and `vcjs_s5` | `GPU_ID=1 bash results_test/tarr/job_scripts/tarr_cifar100_18grid_wave.sh gpu1_wave1` | completed | Stage 3/4 done by 22:13 KST | `results_test/tarr/outputs/cifar100/eval_api/seed0/cifar100_eval_api_fsood_ent_s10_*`; `.../memo_*`; `.../vcjs_s5_*` | Five Stage 3 processes on GPU1. |
| 2026-05-27 20:00 KST | Codex | `cifar100` | Wave 2 GPU0, remaining `vcjs` and `vckl` block | `GPU_ID=0 bash results_test/tarr/job_scripts/tarr_cifar100_18grid_wave.sh gpu0_wave2` | completed | Stage 3/4 done by 05:09 KST on 2026-05-28 | `results_test/tarr/outputs/cifar100/eval_api/seed0/cifar100_eval_api_fsood_vcjs_s30_*`; `.../vcjs_s10_*`; `.../vckl_*` | Stage 4 dominated total runtime. |
| 2026-05-27 22:15 KST | Codex | `cifar100` | Wave 2 GPU1, `entropy_consistency` block | `GPU_ID=1 bash results_test/tarr/job_scripts/tarr_cifar100_18grid_wave.sh gpu1_wave2` | completed | Stage 3/4 done by 04:34 KST on 2026-05-28 | `results_test/tarr/outputs/cifar100/eval_api/seed0/cifar100_eval_api_fsood_hcons_*_refseed0` | Best CIFAR-100 branch came from this wave. |
| 2026-05-28 11:37 KST | Codex | `imagenet200` | Stage 3 parallel smoke, workers 0 | `GPU_ID=0 NUM_WORKERS=0 PARALLEL_COUNTS='1 2 3' bash results_test/tarr/job_scripts/tarr_imagenet200_parallel_smoke.sh`; then `PARALLEL_COUNTS='4 5'` | completed | `n=1..5`: 116s, 130s, 159s, 183s, 299s | `results_test/tarr/summary/imagenet200_parallel_smoke_workers0.csv` | Non-claim-bearing. Smoke outputs deleted after each wave. |
| 2026-05-28 11:53 KST | Codex | `imagenet200` | Stage 3 parallel smoke, workers 2 | `GPU_ID=0 NUM_WORKERS=2 PARALLEL_COUNTS='4' bash results_test/tarr/job_scripts/tarr_imagenet200_parallel_smoke.sh`; then `PARALLEL_COUNTS='5'` | completed | `n=4`: 175s; `n=5`: 201s | `results_test/tarr/summary/imagenet200_parallel_smoke_workers2.csv` | Non-claim-bearing. Best measured ImageNet-200 Stage 3 setting is five processes per GPU with `num_workers=2`. |
| 2026-05-28 14:25 KST | Codex | `imagenet200` | Wave 1 GPU0, `predicted_label_ce` plus partial `entropy` block | `GPU_ID=0 bash results_test/tarr/job_scripts/tarr_imagenet200_18grid_wave.sh gpu0_wave1` | completed | Stage 3/4 done by 19:58 KST | `results_test/tarr/outputs/imagenet200/eval_api/seed0/imagenet200_eval_api_fsood_plce_*_refseed0`; `.../ent_s5_*`; `.../ent_s30_*` | Five Stage 3 processes on GPU0, `num_workers=2`. |
| 2026-05-28 14:26 KST | Codex | `imagenet200` | Wave 1 GPU1, remaining `entropy`, `memo`, and `vcjs_s5` | `GPU_ID=1 bash results_test/tarr/job_scripts/tarr_imagenet200_18grid_wave.sh gpu1_wave1` | completed | Stage 3/4 done by 2026-05-29 04:54 KST | `results_test/tarr/outputs/imagenet200/eval_api/seed0/imagenet200_eval_api_fsood_ent_s10_*`; `.../memo_*`; `.../vcjs_s5_*` | Five Stage 3 processes on GPU1, `num_workers=2`. |
| 2026-05-28 20:32 KST | Codex | `imagenet200` | Wave 2 GPU0, remaining `vcjs` and `vckl` block | `GPU_ID=0 bash results_test/tarr/job_scripts/tarr_imagenet200_18grid_wave.sh gpu0_wave2` | completed | Stage 3/4/collect-score done by 2026-05-29 07:26 KST | `results_test/tarr/outputs/imagenet200/eval_api/seed0/imagenet200_eval_api_fsood_vcjs_s30_*`; `.../vcjs_s10_*`; `.../vckl_*` | Five Stage 3 processes on GPU0, `num_workers=2`. |
| 2026-05-28 20:32 KST | Codex | `imagenet200` | Wave 2 GPU1, `entropy_consistency` block | `GPU_ID=1 bash results_test/tarr/job_scripts/tarr_imagenet200_18grid_wave.sh gpu1_wave2` | completed | Stage 3/4/collect-score done by 2026-05-29 04:54 KST | `results_test/tarr/outputs/imagenet200/eval_api/seed0/imagenet200_eval_api_fsood_hcons_*_refseed0` | Three Stage 3 processes on GPU1, `num_workers=2`. |

Command-log requirements:

- Include working directory and full command.
- Include GPU id, batch size, reference-set batch size, shard size, and debug mode.
- Mark smoke/subset commands explicitly as non-claim-bearing.
- Link failures to the branch decision log when they prune or modify the grid.

## Dataset-Wise Best

Primary metric is FSOOD `both` avg AUROC. First tie-breaker is FSOOD `both` avg FPR95. Diagnostics are clean-only AUROC, csID-only AUROC, and score ordering.

| dataset | selected_tta_id | selected_reference_config_id | selected_score_rule | refseed0_result | refseed_mean_std | baseline_comparison | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cifar10` | `ent_s30_lr1e2` | `strat_rpc8` | `predicted_class_loss_decrease` | both avg AUROC `78.725`, avg FPR95 `56.09` | TODO | `+0.525` over internal TARR baseline `plce_s5_lr1e2` | needs refseed robustness | Best refseed0 active-score row. `ent_s10_lr3e2` is nearly tied. `memo_s30_lr1e2` is the best soft-view row and has better csID-only AUROC than entropy. |
| `cifar100` | `hcons_s30_lr1e2_pixgauss_eps1e2_r4` | `all_rpc8` | `positive_loss_increase_mean` | both avg AUROC `64.880`, avg FPR95 `75.47` | TODO | `+1.555` over internal TARR baseline `plce_s5_lr1e2` | needs refseed robustness | `hcons_s5` and `hcons_s10_lr3e2` are nearly tied. Best row improves csID-only AUROC from `50.56` to `55.925` and preserves `clean<csid<ood` ordering. |
| `imagenet200` | `plce_s5_lr1e2` | `highconf09_rpc8` | `predicted_class_loss_decrease` | both avg AUROC `58.700`, avg FPR95 `82.705` | TODO | `+0.000`; no tested setting improved over the default TARR setting | failed to improve | Best row is the default PLCE baseline itself. All other 17 configs are below baseline on both AUROC. |
| `imagenet` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

Promotion rules:

- A row must be a full canonical run with validation passing.
- A setting is promising if FSOOD `both` improves over internal TARR baseline, or `both` is comparable while csID-only and score ordering improve clearly.
- A setting fails if clean-only improves but csID-only remains collapsed, semantic OOD does not separate from csID, or runtime cost is high without metric/alignment gain.
- Reference-seed robustness is evidence, not a license to select the luckiest seed silently.

## Baseline Comparison Plan

Internal TARR baseline per dataset:

```text
predicted_label_ce, steps=5, response_step=5, lr=1e-2, refseed=0, active score_rule=all
```

External baseline comparison uses Group 1 via `reports.py compare-group1`. Because `compare-group1` expects one exact score rule, run it for selected active score rules after `collect-score` identifies the relevant TARR rows.

Primary external baselines:

```text
MSP, MLS, EBO, GradNorm, RMDS, KNN
```

## CIFAR-10 18-Config Broad Search

Run summary:

- Dataset: `cifar10`, `baseline_protocol=eval_api`, `scheme=fsood`, `refseed=0`.
- Reference configs per run: all 15 prebuilt CIFAR-10 `reference_set` configs.
- Stage 3 storage: sharded `tta_response`, shard size `1024`, `debug_output_mode=none`.
- Runtime defaults: `batch_size=512`, `reference_set_batch_size=2048`, `num_workers=0`.
- Summary artifacts:
  - `results_test/tarr/summary/cifar10_eval_api_score_results.csv`
  - `results_test/tarr/summary/cifar10_18grid_best_by_run_ood_direction.csv`
  - `results_test/tarr/summary/cifar10_compare_group1_strat_rpc8_predicted_class_loss_decrease.csv`
  - `results_test/tarr/summary/cifar10_compare_group1_correct_rpc32_positive_loss_increase_mean.csv`
  - `results_test/tarr/summary/cifar10_compare_group1_best_ent_s30_strat_rpc8_predicted_class_loss_decrease_avg.csv`

Best active-score row per run:

Each row below is the best `score_result` selected for that run across all 15
reference configs and all collected score rules. It is not an exhaustive listing
of every reference config tested.

| run_id | ref | score_rule | both AUROC | FPR95 | clean AUROC | csID AUROC | ordering | delta | decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `cifar10_eval_api_fsood_plce_s5_lr1e2_refseed0` | `strat_rpc8` | `predicted_class_loss_decrease` | 78.200 | 56.310 | 89.985 | 67.595 | `clean<csID<OOD` | +0.000 | baseline |
| `cifar10_eval_api_fsood_plce_s30_lr1e2_refseed0` | `strat_rpc8` | `predicted_class_loss_decrease` | 78.500 | 56.120 | 90.155 | 68.005 | `clean<csID<OOD` | +0.300 | promising |
| `cifar10_eval_api_fsood_plce_s10_lr3e2_refseed0` | `strat_rpc8` | `predicted_class_loss_decrease` | 78.485 | 56.120 | 90.145 | 67.995 | `clean<csID<OOD` | +0.285 | promising |
| `cifar10_eval_api_fsood_ent_s5_lr1e2_refseed0` | `strat_rpc8` | `positive_loss_decrease_mean` | 78.460 | 56.265 | 90.220 | 67.870 | `clean<csID<OOD` | +0.260 | promising |
| `cifar10_eval_api_fsood_ent_s30_lr1e2_refseed0` | `strat_rpc8` | `predicted_class_loss_decrease` | 78.725 | 56.090 | 90.200 | 68.400 | `clean<csID<OOD` | +0.525 | needs_refseed |
| `cifar10_eval_api_fsood_ent_s10_lr3e2_refseed0` | `strat_rpc8` | `predicted_class_loss_decrease` | 78.710 | 56.105 | 90.195 | 68.375 | `clean<csID<OOD` | +0.510 | needs_refseed |
| `cifar10_eval_api_fsood_memo_s5_lr1e2_pixgauss_eps1e2_r4_refseed0` | `strat_rpc8` | `positive_loss_decrease_mean` | 78.435 | 56.250 | 90.220 | 67.835 | `clean<csID<OOD` | +0.235 | promising |
| `cifar10_eval_api_fsood_memo_s30_lr1e2_pixgauss_eps1e2_r4_refseed0` | `correct_rpc32` | `positive_loss_increase_mean` | 78.705 | 57.340 | 89.770 | 68.740 | `clean<csID<OOD` | +0.505 | needs_refseed |
| `cifar10_eval_api_fsood_memo_s10_lr3e2_pixgauss_eps1e2_r4_refseed0` | `correct_rpc32` | `positive_loss_increase_mean` | 78.690 | 57.405 | 89.765 | 68.725 | `clean<csID<OOD` | +0.490 | promising |
| `cifar10_eval_api_fsood_vcjs_s5_lr1e2_pixgauss_eps1e2_r4_refseed0` | `correct_rpc32` | `positive_loss_increase_mean` | 75.960 | 65.175 | 88.205 | 64.935 | `clean<csID<OOD` | -2.240 | failed |
| `cifar10_eval_api_fsood_vcjs_s30_lr1e2_pixgauss_eps1e2_r4_refseed0` | `correct_rpc32` | `positive_loss_increase_mean` | 76.210 | 53.730 | 88.690 | 64.980 | `clean<csID<OOD` | -1.990 | failed |
| `cifar10_eval_api_fsood_vcjs_s10_lr3e2_pixgauss_eps1e2_r4_refseed0` | `correct_rpc32` | `positive_loss_increase_mean` | 76.255 | 54.015 | 88.760 | 65.010 | `clean<csID<OOD` | -1.945 | failed |
| `cifar10_eval_api_fsood_vckl_s5_lr1e2_pixgauss_eps1e2_r4_refseed0` | `correct_rpc32` | `positive_loss_increase_mean` | 75.960 | 70.305 | 88.210 | 64.935 | `clean<csID<OOD` | -2.240 | failed |
| `cifar10_eval_api_fsood_vckl_s30_lr1e2_pixgauss_eps1e2_r4_refseed0` | `correct_rpc32` | `positive_loss_increase_mean` | 76.210 | 53.720 | 88.690 | 64.980 | `clean<csID<OOD` | -1.990 | failed |
| `cifar10_eval_api_fsood_vckl_s10_lr3e2_pixgauss_eps1e2_r4_refseed0` | `correct_rpc32` | `positive_loss_increase_mean` | 76.260 | 53.850 | 88.760 | 65.005 | `clean<csID<OOD` | -1.940 | failed |
| `cifar10_eval_api_fsood_hcons_s5_lr1e2_pixgauss_eps1e2_r4_refseed0` | `correct_rpc32` | `positive_loss_decrease_mean` | 76.050 | 86.055 | 88.260 | 65.060 | `clean<csID<OOD` | -2.150 | failed |
| `cifar10_eval_api_fsood_hcons_s30_lr1e2_pixgauss_eps1e2_r4_refseed0` | `correct_rpc32` | `positive_loss_increase_mean` | 76.365 | 85.855 | 88.590 | 65.365 | `clean<csID<OOD` | -1.835 | failed |
| `cifar10_eval_api_fsood_hcons_s10_lr3e2_pixgauss_eps1e2_r4_refseed0` | `correct_rpc32` | `positive_loss_increase_mean` | 76.430 | 80.245 | 88.715 | 65.375 | `clean<csID<OOD` | -1.770 | failed |

Interpretation:

- Best perturbation-free row: `entropy`, `steps=30`, `lr=1e-2`, `strat_rpc8`, `predicted_class_loss_decrease`.
- Best soft-view row: `memo_marginal_entropy`, `steps=30`, `lr=1e-2`, `correct_rpc32`, `positive_loss_increase_mean`.
- `memo_s30` is within `0.02pp` of the best perturbation-free row and has the best csID-only AUROC among the top rows (`68.740`), so it qualifies for refseed robustness and limited perturbation refinement.
- `view_consistency_js`, `view_consistency_kl`, and `entropy_consistency` are not competitive in this screen. Do not expand their perturbation grid unless later datasets contradict this CIFAR-10 result.
- External Group 1 comparison was generated for the two selected score/reference families. The best TARR row is essentially tied with KNN and above RMDS/MSP/EBO/MLS on average AUROC. Because the KNN margin is only `+0.005pp` and TARR FPR95 is worse than KNN, treat this as competitive parity rather than a clear external SOTA claim.

External Group 1 comparison for the best CIFAR-10 row:

| method / setting | avg AUROC | TARR AUROC - baseline AUROC |
| --- | ---: | ---: |
| TARR best: `entropy`, `steps=30`, `lr=1e-2`, `strat_rpc8`, `predicted_class_loss_decrease` | 78.725 | 0.000 |
| KNN | 78.720 | +0.005 |
| RMDS | 78.290 | +0.435 |
| EBO | 77.940 | +0.785 |
| MLS | 77.815 | +0.910 |
| IODIN | 77.660 | +1.065 |
| MSP | 77.475 | +1.250 |
| ReAct | 77.080 | +1.645 |
| ODIN | 75.630 | +3.095 |
| Gram | 75.115 | +3.610 |
| MDS | 73.980 | +4.745 |
| SCALE | 73.825 | +4.900 |
| DICE | 73.340 | +5.385 |
| SHE | 73.145 | +5.580 |
| KLM | 72.020 | +6.705 |
| ASH | 68.630 | +10.095 |
| GradNorm | 55.005 | +23.720 |
| Residual | 50.000 | +28.725 |

Conclusion: CIFAR-10 TARR best is competitive with the strongest Group 1 baseline and beats most post-hoc baselines in AUROC, but the KNN gap is too small to claim a robust external improvement without refseed/checkpoint robustness.

## CIFAR-100 18-Config Broad Search

Run summary:

- Dataset: `cifar100`, `baseline_protocol=eval_api`, `scheme=fsood`, `refseed=0`.
- Reference configs per run: all 15 prebuilt CIFAR-100 `reference_set` configs.
- Stage 3 storage: sharded `tta_response`, shard size `1024`, `debug_output_mode=none`.
- Runtime defaults used for this broad run: `batch_size=512`, `reference_set_batch_size=2048`, `num_workers=0`.
- Summary artifacts:
  - `results_test/tarr/summary/cifar100_eval_api_score_results.csv`
  - `results_test/tarr/summary/cifar100_18grid_best_by_run.csv`
  - `results_test/tarr/summary/cifar100_compare_group1_all_rpc8_positive_loss_increase_mean.csv`
  - `results_test/tarr/summary/cifar100_compare_group1_best_hcons_s30_all_rpc8_positive_loss_increase_mean_avg.csv`

Best active-score row per run:

Each row below is the best `score_result` selected for that run across all 15
reference configs and all collected score rules. The repeated `all_rpc8` /
`positive_loss_increase_mean` entries mean that this pair was the best row for
each listed CIFAR-100 run, not that the other reference configs were skipped.

| run_id | ref | score_rule | both AUROC | FPR95 | clean AUROC | csID AUROC | ordering | delta | decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `cifar100_eval_api_fsood_plce_s5_lr1e2_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 63.325 | 75.130 | 77.515 | 50.560 | `clean<csID<OOD` | +0.000 | baseline |
| `cifar100_eval_api_fsood_plce_s30_lr1e2_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 63.405 | 75.160 | 77.205 | 50.990 | `clean<csID<OOD` | +0.080 | minor_gain |
| `cifar100_eval_api_fsood_plce_s10_lr3e2_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 63.350 | 75.150 | 77.135 | 50.940 | `clean<csID<OOD` | +0.025 | minor_gain |
| `cifar100_eval_api_fsood_ent_s5_lr1e2_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 62.920 | 75.300 | 76.560 | 50.650 | `clean<csID<OOD` | -0.405 | failed |
| `cifar100_eval_api_fsood_ent_s30_lr1e2_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 63.190 | 75.650 | 76.735 | 51.000 | `clean<csID<OOD` | -0.135 | failed |
| `cifar100_eval_api_fsood_ent_s10_lr3e2_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 63.160 | 75.630 | 76.775 | 50.910 | `clean<csID<OOD` | -0.165 | failed |
| `cifar100_eval_api_fsood_memo_s5_lr1e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 62.870 | 75.465 | 76.425 | 50.665 | `clean<csID<OOD` | -0.455 | failed |
| `cifar100_eval_api_fsood_memo_s30_lr1e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 63.165 | 75.855 | 76.635 | 51.040 | `clean<csID<OOD` | -0.160 | failed |
| `cifar100_eval_api_fsood_memo_s10_lr3e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 63.125 | 75.850 | 76.690 | 50.920 | `clean<csID<OOD` | -0.200 | failed |
| `cifar100_eval_api_fsood_vcjs_s5_lr1e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 64.255 | 75.345 | 71.430 | 57.795 | `clean<csID<OOD` | +0.930 | promising_diagnostic |
| `cifar100_eval_api_fsood_vcjs_s30_lr1e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 64.210 | 74.975 | 70.885 | 58.210 | `clean<csID<OOD` | +0.885 | promising_diagnostic |
| `cifar100_eval_api_fsood_vcjs_s10_lr3e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 64.225 | 75.105 | 71.105 | 58.035 | `clean<csID<OOD` | +0.900 | promising_diagnostic |
| `cifar100_eval_api_fsood_vckl_s5_lr1e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 64.255 | 75.345 | 71.430 | 57.795 | `clean<csID<OOD` | +0.930 | promising_diagnostic |
| `cifar100_eval_api_fsood_vckl_s30_lr1e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 64.210 | 74.970 | 70.885 | 58.210 | `clean<csID<OOD` | +0.885 | promising_diagnostic |
| `cifar100_eval_api_fsood_vckl_s10_lr3e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 64.225 | 75.110 | 71.105 | 58.035 | `clean<csID<OOD` | +0.900 | promising_diagnostic |
| `cifar100_eval_api_fsood_hcons_s5_lr1e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 64.840 | 75.525 | 74.905 | 55.780 | `clean<csID<OOD` | +1.515 | needs_refseed |
| `cifar100_eval_api_fsood_hcons_s30_lr1e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 64.880 | 75.470 | 74.825 | 55.925 | `clean<csID<OOD` | +1.555 | needs_refseed |
| `cifar100_eval_api_fsood_hcons_s10_lr3e2_pixgauss_eps1e2_r4_refseed0` | `all_rpc8` | `positive_loss_increase_mean` | 64.805 | 75.485 | 74.795 | 55.815 | `clean<csID<OOD` | +1.480 | needs_refseed |

Interpretation:

- Best row: `entropy_consistency`, `steps=30`, `lr=1e-2`, `all_rpc8`, `positive_loss_increase_mean`.
- The best row improves internal TARR baseline by `+1.555pp` both AUROC and improves csID-only AUROC from `50.560` to `55.925`.
- `hcons_s5`, `hcons_s10_lr3e2`, and `hcons_s30` are tightly clustered. Treat them as robustness candidates rather than over-interpreting one step/lr point.
- `view_consistency_js` and `view_consistency_kl` are weaker than entropy-consistency on both AUROC, but they improve csID-only AUROC more strongly (`57.8-58.2`). Keep them as diagnostic/promising branches, not primary candidates.
- Longer `predicted_label_ce` budgets give only minor gains (`+0.025` to `+0.080pp`) over the internal baseline.
- `entropy` and `memo_marginal_entropy` are below the internal baseline on CIFAR-100, unlike CIFAR-10.
- The CIFAR-10/CIFAR-100 contrast supports selecting TTA settings dataset-wise instead of forcing one global TTA objective.

External Group 1 comparison for the best CIFAR-100 row:

| method / setting | avg AUROC | TARR AUROC - baseline AUROC |
| --- | ---: | ---: |
| TARR best: `entropy_consistency`, `steps=30`, `lr=1e-2`, pixel gaussian eps `0.01`, repeats `4`, `all_rpc8`, `positive_loss_increase_mean` | 64.880 | 0.000 |
| RMDS | 66.200 | -1.320 |
| DICE | 64.570 | +0.310 |
| SCALE | 64.430 | +0.450 |
| ASH | 64.025 | +0.855 |
| IODIN | 63.760 | +1.120 |
| MLS | 63.750 | +1.130 |
| ReAct | 63.715 | +1.165 |
| EBO | 63.605 | +1.275 |
| KNN | 63.565 | +1.315 |
| MSP | 63.485 | +1.395 |

Conclusion: CIFAR-100 TARR best beats KNN/MSP/EBO/MLS and most Group 1 post-hoc baselines in AUROC, but it does not beat RMDS. This is a strong dataset-wise TARR improvement and a competitive baseline comparison, not an external SOTA claim.

## ImageNet-200 18-Config Broad Search

Run summary:

- Dataset: `imagenet200`, `baseline_protocol=eval_api`, `scheme=fsood`, `refseed=0`.
- Reference configs per run: all 15 prebuilt ImageNet-200 `reference_set` configs.
- Runtime defaults: `batch_size=64`, `reference_set_batch_size=1024`, `num_workers=2`.
- Stage 3 storage: sharded `tta_response`, shard size `1024`, `debug_output_mode=none`.
- Schedule: two waves, up to five Stage 3 processes per GPU.
- Summary artifacts:
  - `results_test/tarr/summary/imagenet200_eval_api_score_results.csv`
  - `results_test/tarr/summary/imagenet200_18grid_best_by_run.csv`
  - `results_test/tarr/summary/imagenet200_compare_group1_best_plce_s5_highconf09_rpc8_predicted_class_loss_decrease_avg.csv`

Best active-score row per run:

Each row below is the best active `score_result` selected for that run across
all 15 reference configs and all collected active score rules.

| run_id | ref | score_rule | both AUROC | FPR95 | clean AUROC | csID AUROC | ordering | delta | decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `imagenet200_eval_api_fsood_plce_s5_lr1e2_refseed0` | `highconf09_rpc8` | `predicted_class_loss_decrease` | 58.700 | 82.705 | 76.130 | 54.090 | `clean>csid>ood` | +0.000 | baseline |
| `imagenet200_eval_api_fsood_plce_s30_lr1e2_refseed0` | `highconf09_rpc8` | `predicted_class_loss_decrease` | 57.355 | 84.035 | 71.255 | 53.675 | `clean>csid>ood` | -1.345 | failed |
| `imagenet200_eval_api_fsood_plce_s10_lr3e2_refseed0` | `highconf09_rpc8` | `predicted_class_loss_decrease` | 57.350 | 84.000 | 71.170 | 53.695 | `clean>csid>ood` | -1.350 | failed |
| `imagenet200_eval_api_fsood_ent_s5_lr1e2_refseed0` | `highconf09_rpc8` | `positive_loss_decrease_mean` | 56.930 | 82.285 | 74.670 | 52.230 | `clean>csid>ood` | -1.770 | failed |
| `imagenet200_eval_api_fsood_ent_s30_lr1e2_refseed0` | `correcthigh09_rpc16` | `target_weighted_loss_increase` | 56.300 | 91.340 | 71.585 | 52.255 | `clean>csid>ood` | -2.400 | failed |
| `imagenet200_eval_api_fsood_ent_s10_lr3e2_refseed0` | `correcthigh09_rpc16` | `target_weighted_loss_increase` | 56.165 | 91.195 | 71.565 | 52.090 | `clean>csid>ood` | -2.535 | failed |
| `imagenet200_eval_api_fsood_memo_s5_lr1e2_pixgauss_eps1e2_r4_refseed0` | `highconf09_rpc8` | `positive_loss_decrease_mean` | 56.955 | 82.295 | 74.645 | 52.270 | `clean>csid>ood` | -1.745 | failed |
| `imagenet200_eval_api_fsood_memo_s30_lr1e2_pixgauss_eps1e2_r4_refseed0` | `correcthigh09_rpc16` | `target_weighted_loss_increase` | 56.295 | 91.325 | 71.595 | 52.250 | `clean>csid>ood` | -2.405 | failed |
| `imagenet200_eval_api_fsood_memo_s10_lr3e2_pixgauss_eps1e2_r4_refseed0` | `correcthigh09_rpc16` | `target_weighted_loss_increase` | 56.175 | 91.200 | 71.575 | 52.095 | `clean>csid>ood` | -2.525 | failed |
| `imagenet200_eval_api_fsood_vcjs_s5_lr1e2_pixgauss_eps1e2_r4_refseed0` | `highconf09_rpc32` | `positive_loss_increase_mean` | 54.795 | 83.885 | 67.465 | 51.440 | `not_ordered` | -3.905 | failed |
| `imagenet200_eval_api_fsood_vcjs_s30_lr1e2_pixgauss_eps1e2_r4_refseed0` | `highconf09_rpc32` | `positive_loss_increase_mean` | 54.650 | 83.840 | 66.950 | 51.395 | `not_ordered` | -4.050 | failed |
| `imagenet200_eval_api_fsood_vcjs_s10_lr3e2_pixgauss_eps1e2_r4_refseed0` | `highconf09_rpc32` | `positive_loss_increase_mean` | 54.700 | 83.855 | 67.255 | 51.380 | `not_ordered` | -4.000 | failed |
| `imagenet200_eval_api_fsood_vckl_s5_lr1e2_pixgauss_eps1e2_r4_refseed0` | `highconf09_rpc32` | `positive_loss_increase_mean` | 54.795 | 83.875 | 67.460 | 51.440 | `not_ordered` | -3.905 | failed |
| `imagenet200_eval_api_fsood_vckl_s30_lr1e2_pixgauss_eps1e2_r4_refseed0` | `highconf09_rpc32` | `positive_loss_increase_mean` | 54.650 | 83.835 | 66.950 | 51.395 | `not_ordered` | -4.050 | failed |
| `imagenet200_eval_api_fsood_vckl_s10_lr3e2_pixgauss_eps1e2_r4_refseed0` | `highconf09_rpc32` | `positive_loss_increase_mean` | 54.700 | 83.825 | 67.250 | 51.380 | `not_ordered` | -4.000 | failed |
| `imagenet200_eval_api_fsood_hcons_s5_lr1e2_pixgauss_eps1e2_r4_refseed0` | `highconf09_rpc8` | `predicted_class_loss_increase` | 55.170 | 94.010 | 68.580 | 51.615 | `not_ordered` | -3.530 | failed |
| `imagenet200_eval_api_fsood_hcons_s30_lr1e2_pixgauss_eps1e2_r4_refseed0` | `highconf09_rpc8` | `predicted_class_loss_increase` | 55.395 | 93.950 | 69.495 | 51.660 | `not_ordered` | -3.305 | failed |
| `imagenet200_eval_api_fsood_hcons_s10_lr3e2_pixgauss_eps1e2_r4_refseed0` | `highconf09_rpc8` | `predicted_class_loss_increase` | 55.265 | 93.950 | 69.160 | 51.590 | `not_ordered` | -3.435 | failed |

Interpretation:

- Best row: the default `predicted_label_ce`, `steps=5`, `lr=1e-2`, `highconf09_rpc8`, `predicted_class_loss_decrease`.
- No tested ImageNet-200 TTA setting improved over the internal default TARR baseline. The nearest alternatives are still `1.3pp` or more below baseline.
- Soft-view objectives are clearly worse on ImageNet-200 in this grid. Do not expand pixel-gaussian perturbation for ImageNet-200 before trying a different ImageNet-specific strategy.
- The selected score directions often show `clean>csid>ood`, which is the reverse of the preferred positive OOD-like ordering. Treat these rows as AUROC-oriented diagnostics, not evidence that the current ImageNet-200 score semantics are well aligned.

External Group 1 comparison for the best ImageNet-200 row:

| method / setting | avg AUROC | TARR AUROC - baseline AUROC |
| --- | ---: | ---: |
| TARR best: `predicted_label_ce`, `steps=5`, `lr=1e-2`, `highconf09_rpc8`, `predicted_class_loss_decrease` | 58.700 | 0.000 |
| SCALE | 65.545 | -6.845 |
| ASH | 65.110 | -6.410 |
| KLM | 62.900 | -4.200 |
| GradNorm | 61.710 | -3.010 |
| MSP | 60.230 | -1.530 |
| ReAct | 59.140 | -0.440 |
| IODIN | 58.340 | +0.360 |
| MLS | 58.270 | +0.430 |
| EBO | 57.550 | +1.150 |
| ODIN | 57.290 | +1.410 |
| Residual | 50.000 | +8.700 |

Conclusion: ImageNet-200 TARR does not produce a promising external comparison. The best TARR row beats IODIN/MLS/EBO/ODIN but is below SCALE, ASH, KLM, GradNorm, MSP, and ReAct. Combined with the internal broad-search result where no TTA candidate improves over the default TARR setting, ImageNet-scale experiments need a revised strategy before ImageNet-1K transfer.

## Branch Decision Log

Use this log for decisions that affect the search tree.

| date | dataset | decision | evidence | effect_on_grid_or_results | owner |
| --- | --- | --- | --- | --- | --- |
| 2026-05-27 | `cifar10` | Promote `entropy s30/lr1e-2`, `entropy s10/lr3e-2`, and `memo s30/lr1e-2` to refseed robustness candidates. | Refseed0 best active rows: `entropy s30` both AUROC `78.725`, `entropy s10` `78.710`, `memo s30` `78.705`; baseline `plce s5` is `78.200`. | Run refseed1/2 for these candidates before CIFAR-100 transfer. | Codex |
| 2026-05-27 | `cifar10` | Do not expand JS/KL/entropy-consistency perturbation grid. | Best JS/KL/HCons rows are `1.77-2.24pp` below baseline. | Keep only MEMO for limited perturbation refinement if needed. | Codex |
| 2026-05-27 | `cifar100` | Run CIFAR-100 refseed0 before CIFAR-10 refseed1/2 robustness. Use five Stage 3 processes per GPU as the measured Stage 3-only default. | CIFAR-100 smoke completed `n=5` with wall `426s`, average GPU util `87.59%`, max VRAM `3015MiB`, no failures. | Prepare CIFAR-100 refseed0 broad queue; keep Stage 4 low concurrency. | Codex |
| 2026-05-28 | `cifar100` | Promote `entropy_consistency` `s30/lr1e-2`, `s5/lr1e-2`, and `s10/lr3e-2` to refseed robustness candidates. | Refseed0 best active rows: `hcons s30` both AUROC `64.880`, `hcons s5` `64.840`, `hcons s10` `64.805`; internal baseline `plce s5` is `63.325`. | Defer `refseed1/2` until other dataset refseed0 screens complete, then run robustness for these candidates. | Codex |
| 2026-05-28 | `cifar100` | Keep `view_consistency_js/kl` as diagnostic promising branches, not primary candidates. | JS/KL both AUROC improves baseline by about `+0.89-0.93pp`, and csID-only AUROC reaches `57.8-58.2`, but both AUROC is below entropy-consistency. | Consider JS/KL only if later datasets favor view consistency or if csID separation becomes the main target. | Codex |
| 2026-05-28 | `cifar100` | Do not expand CIFAR-100 `entropy`, `memo`, or PLCE-only grids immediately. | `entropy` and `memo` are below baseline; PLCE longer budgets improve by only `+0.025-0.080pp`. | Spend next compute on `imagenet200` refseed0 transfer or deferred robustness, not these branches. | Codex |
| 2026-05-28 | `imagenet200` | Run the full 18-config refseed0 broad grid. | ImageNet-200 smoke supports five Stage 3 processes/GPU with `num_workers=2`; 45 reference sets already exist for `refseed=0`. | Use `tarr_imagenet200_18grid_wave.sh` in two waves, then analyze against the ImageNet-200 PLCE s5 baseline. | Codex |
| 2026-05-29 | `imagenet200` | Do not promote any ImageNet-200 refseed0 TTA branch. | Best row is the default `plce s5/lr1e-2` baseline itself: both AUROC `58.700`. All other 17 configs are at least `1.345pp` below baseline. | Do not spend refseed1/2 on this ImageNet-200 grid. Revisit ImageNet-scale strategy before ImageNet-1K transfer. | Codex |

## Artifact Matrix

| Stage | Artifact | Experiment variables | Reuse rule |
| --- | --- | --- | --- |
| 1 | `train_candidate_metadata` | Dataset, train imglist, checkpoint, model arch/classes, preprocessing identity, metadata schema | Reuse only when the identity exactly matches. |
| 2P | `reference_set` / Probe `P` | Source, filter, per-class size, confidence threshold, seed | Reuse only with the same Stage 1 input and selected-sample identity. |
| 2A | `anchor_set` / Anchor `A` | Source, filter, per-class size, confidence threshold, seed, paired Probe exclusion | Reuse only with the same Stage 1 input, selected-sample identity, and Probe exclusion identity. |
| 3 | `tta_response` | Scheme, target split, TTA objective, max `steps`, saved `response_steps`, lr, update scope, runtime mode, perturbation config | New Stage 3 run when target protocol or TTA config changes. |
| 4 | `score_result` | Score rule, selected `response_step`, ID-side aggregation for FSOOD, score direction | Recompute from matching `tta_response` without rerunning TTA. |

Batch sizes and worker counts are runtime settings. They can support cost claims only when reported with the same protocol, hardware, target count, and artifact identities.

## Canonical Output Paths

```text
results_test/tarr/train_candidate_metadata/<dataset>/<candidate_id>/manifest.json
results_test/tarr/train_candidate_metadata/<dataset>/<candidate_id>/candidates.npz

results_test/tarr/reference_sets/<dataset>/<reference_config_id>/seed<seed>/<reference_set_id>/manifest.json
results_test/tarr/reference_sets/<dataset>/<reference_config_id>/seed<seed>/<reference_set_id>/reference_set.npz
results_test/tarr/reference_sets/<dataset>/<reference_config_id>/seed<seed>/<reference_set_id>/selected_samples.csv

results_test/tarr/anchor_sets/<dataset>/<reference_config_id>/seed<seed>/<anchor_set_id>/manifest.json
results_test/tarr/anchor_sets/<dataset>/<reference_config_id>/seed<seed>/<anchor_set_id>/anchor_set.npz
results_test/tarr/anchor_sets/<dataset>/<reference_config_id>/seed<seed>/<anchor_set_id>/selected_samples.csv

results_test/tarr/outputs/<dataset>/<baseline_protocol>/seed<seed>/<run_id>/run_manifest.json
results_test/tarr/outputs/<dataset>/<baseline_protocol>/seed<seed>/<run_id>/run_info.md
results_test/tarr/outputs/<dataset>/<baseline_protocol>/seed<seed>/<run_id>/<scheme>/scheme_manifest.json
results_test/tarr/outputs/<dataset>/<baseline_protocol>/seed<seed>/<run_id>/<scheme>/references/<reference_config_id>/tta_response/<target_dataset_name>/manifest.json
results_test/tarr/outputs/<dataset>/<baseline_protocol>/seed<seed>/<run_id>/<scheme>/references/<reference_config_id>/tta_response/<target_dataset_name>/part_*.npz
results_test/tarr/outputs/<dataset>/<baseline_protocol>/seed<seed>/<run_id>/<scheme>/references/<reference_config_id>/score_results/<score_rule>/scores/<target_dataset_name>.npz
results_test/tarr/outputs/<dataset>/<baseline_protocol>/seed<seed>/<run_id>/<scheme>/references/<reference_config_id>/score_results/<score_rule>/ood.csv
```

Diagnostics are written under:

```text
results_test/tarr/outputs/<dataset>/<baseline_protocol>/seed<seed>/<run_id>/<scheme>/references/<reference_config_id>/diagnostics/
```

Expected diagnostics:

- `score_summary.csv`
- `target_summary.csv`
- `delta_summary.csv`
- `runtime_summary.csv`
- `reference_summary.csv`
- `alignment_summary.csv`
- `perturbation_summary.csv` when perturbation fields exist
- `perturbation_alignment_summary.csv` when perturbation score fields exist
- `score_direction.csv`
- `failure_cases.csv`

## Claim-Bearing Metadata Checklist

Every reported row must include:

- Dataset, scheme, model/checkpoint, and baseline protocol.
- Actual csID dataset names for FSOOD.
- Full/subset status.
- `train_candidate_metadata_id`.
- `reference_set_id` and `reference_config_id`.
- `anchor_set_id`, paired `reference_config_id`, anchor loss weight, and Probe/Anchor disjointness identity when Anchor `A` is enabled.
- TTA config: objective, max `steps`, saved `response_steps`, lr, update scope, runtime mode, view/perturbation config.
- `tta_response` storage, shard count, and target split list.
- Score rule, selected `response_step`, and `score_result_id`.
- Score direction and OpenOOD confidence transform.
- AUROC/FPR95 metrics and output paths.
- Runtime by stage.

## Baseline Comparison Policy

Baseline comparison is performed after a dataset-wise best row is selected.
The main comparison should use Group 1 baselines from the same dataset, scheme,
baseline protocol, csID identity, target split list, and AUROC/FPR95 fields as
the TARR run. Compare only full canonical TARR runs against full baseline runs;
subset/smoke rows are excluded. If a baseline uses a different checkpoint,
imglist, csID identity, or metric field definition, keep it as context only
and do not use it for the primary claim.

## Stage 3 Runtime Defaults

Use these settings as the starting point for new full Stage 3 runs. They are
operational defaults, not claim criteria. Dataset-specific benchmark sections
below explain why each value was chosen.

| Dataset | `batch_size` | `reference_set_batch_size` | `tta_response_shard_size` | `debug_output_mode` | `num_workers` start |
| --- | ---: | ---: | ---: | --- | ---: |
| `cifar10` | 512 | 2048 | 1024 | `none` for broad queues | 0 |
| `cifar100` | 512 | 2048 | 1024 | `none` | 0 |
| `imagenet200` | 64 | 1024 | 256 for smoke, 1024 for full | `none` | 2 |
| `imagenet` | 64 | 1024 | 1024 | `none` | 4 |

If host RAM remains high, reduce `num_workers` first. If Stage 3 response
buffers remain large, reduce `reference_set_batch_size` before reducing
`batch_size`.

## CIFAR-10 Parallelism Benchmark

Smoke benchmark condition:

```text
dataset=cifar10, scheme=fsood, objective=predicted_label_ce,
steps=5, lr=1e-2, max-id-samples=1024, max-ood-samples=1024,
batch_size=512, reference_set_batch_size=2048,
tta_response_shard_size=1024, debug_output_mode=none, num_workers=0
```

| Stage 3 processes per GPU | wall sec | completed | avg GPU util | max VRAM MiB | throughput speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 191 | 1 | 16.82 | 557 | 1.00x |
| 2 | 226 | 2 | 67.81 | 1097 | 1.69x |
| 3 | 270 | 3 | 83.73 | 1636 | 2.12x |
| 4 | 313 | 4 | 91.20 | 2175 | 2.44x |
| 5 | 361 | 5 | 94.61 | 2715 | 2.65x |
| 6 | 409 | 6 | 96.34 | 3254 | 2.80x |

Decision: use four CIFAR-10 Stage 3 processes per GPU as the default broad
queue. Use five only for Stage 3-only throughput tests after confirming host RAM
and Stage 4 scheduling are isolated.

## CIFAR-100 Parallelism Benchmark

Smoke benchmark condition:

```text
dataset=cifar100, scheme=fsood, objective=predicted_label_ce,
steps=5, lr=1e-2, max-id-samples=1024, max-ood-samples=1024,
batch_size=512, reference_set_batch_size=2048,
tta_response_shard_size=1024, debug_output_mode=none, num_workers=0
```

The smoke script deletes temporary run outputs after each wave. The benchmark
therefore measures Stage 3 throughput without keeping claim-bearing artifacts.

| Stage 3 processes per GPU | wall sec | completed | avg GPU util | max VRAM MiB | throughput speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 235 | 1 | 16.77 | 617 | 1.00x |
| 2 | 275 | 2 | 61.81 | 1217 | 1.71x |
| 3 | 319 | 3 | 77.42 | 1816 | 2.21x |
| 4 | 369 | 4 | 83.13 | 2415 | 2.55x |
| 5 | 426 | 5 | 87.59 | 3015 | 2.76x |

Decision: use five CIFAR-100 Stage 3 processes per GPU with `num_workers=0` as
the Stage 3-only broad-search default. Stage 4 scoring/reporting is CPU/IO
heavy, so run it after each wave with low concurrency. Reduce to four processes
per GPU if soft-view runs or overlapping Stage 4 jobs create host memory or IO
pressure.

## ImageNet-200 Parallelism Benchmark

Smoke benchmark condition:

```text
dataset=imagenet200, scheme=fsood, objective=predicted_label_ce,
steps=5, lr=1e-2, max-id-samples=256, max-ood-samples=256,
batch_size=64, reference_set_batch_size=1024,
tta_response_shard_size=256, debug_output_mode=none
```

The smoke script deletes temporary run outputs after each wave. The benchmark
therefore measures Stage 3 throughput without keeping claim-bearing artifacts.

| Stage 3 processes per GPU | num_workers | wall sec | completed | avg GPU util | max VRAM MiB | throughput speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 116 | 1 | 18.17 | 737 | 1.00x |
| 2 | 0 | 130 | 2 | 53.64 | 1457 | 1.78x |
| 3 | 0 | 159 | 3 | 65.06 | 2178 | 2.19x |
| 4 | 0 | 183 | 4 | 76.52 | 2897 | 2.54x |
| 5 | 0 | 299 | 5 | 55.38 | 3617 | 1.94x |
| 4 | 2 | 175 | 4 | 77.65 | 2933 | 2.65x |
| 5 | 2 | 201 | 5 | 80.39 | 3661 | 2.89x |

Decision: use five ImageNet-200 Stage 3 processes per GPU with `num_workers=2`
as the Stage 3-only broad-search default. If Stage 4 is overlapped or host IO
pressure appears, reduce to four processes per GPU. Use sharded `tta_response`
and keep `debug_output_mode=none`.

## Acceptance/Rejection Probe Current Claim Path

Date: 2026-06-02, revised 2026-06-03.

The class-hypothesis acceptance-search branch was removed from the current code
and documentation. It produced some empirical signal, especially on CIFAR-100,
but it is hard to defend as a paper claim and expensive to ablate cleanly. Those
old results are therefore treated as non-claim-bearing exploratory evidence and
are not used as promoted configurations.

Current documented acceptance probes for claim-bearing A/R response-bank runs:

```text
predicted_label_ce
entropy_min
view_consistency
```

Do not include `topk_ce` or `allclass_ce` in default/documented accept banks.

Current documented semantic rejection probes:

```text
entropy_max
uniform
```

`logit_suppression` can remain only as a separate evidence/energy suppression
branch. It is not a semantic rejection probe and should not be mixed into the
default semantic bank.

With `--save-steps`, A/R stores accept and reject responses for each saved
update count. Class-wise accept/reject deltas use `[N,S,C]`; scalar fields such
as `accept_target_objective_delta` and `reject_target_objective_delta` use `[N,S]`.
Response-bank runs additionally store accept and reject bank axes, for example
`[N,S,A,C]`, `[N,S,R,C]`, `[N,S,A]`, and `[N,S,R]`. The bank field names are
`accept_ref_loss_delta_bank`, `reject_ref_loss_delta_bank`,
`accept_target_objective_delta_bank`, and `reject_target_objective_delta_bank`.
`reject_target_entropy_delta` is diagnostic only and is not the canonical
efficiency numerator. Stage 4 must score one selected `response_step` and
record it with the claim-bearing row.

Use the delta-based efficiency definition for A/R interpretation:
`ref_loss_delta` is after-before reference CE loss, `target_objective_delta` is
after-before branch-specific target objective,
`ref_delta_penalty(mode) = scalarize(ref_loss_delta, mode)`, and
`efficiency = -target_objective_delta / (eps + ref_delta_penalty(mode))`.
The reference side is always Probe CE; the target side follows the selected
accept/reject branch objective.

Stage 3 runner flags for banked A/R:

```text
--tta-mode ar_bank
--accept-probe-types predicted_label_ce,entropy_min,view_consistency
--reject-probe-types entropy_max,uniform
```

Stage 4 branch-bank scoring flags:

```text
reject_efficiency:       --reject-branch all
accept_efficiency:       --accept-branch all
ar_efficiency_contrast:  --accept-branch all --reject-branch all --branch-combine cross
```

Interpretation policy:

- `entropy_max` and `uniform` are the main semantic rejection probes.
- `logit_suppression` remains only an evidence/energy suppression probe.
- `reject_efficiency` is the primary A/R score.
- Anchor-only and full-batch A/R+Anchor CE did not show a clear gain in the
  completed focused tests.

Claim-compatible results retained from the previous focused runs:

| dataset | branch | reference | score_rule | both AUROC | clean AUROC | csID AUROC | decision |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| cifar10 | default TARR | `correct_rpc32` | `positive_loss_increase_mean` | 78.02 | 89.63 | 67.58 | A/R not promoted from CIFAR-10 |
| cifar100 | claim-compatible A/R branch | `correct_rpc32` | `reject_efficiency` | 61.86 | 71.50 | 53.18 | improves csID ordering but does not beat default TARR |
| imagenet200 | `predicted_label_ce + entropy_max` | `correct_rpc32` | `reject_efficiency` | 63.24 | 74.12 | 60.38 | promoted for near/both analysis |

Current decision:

```text
Primary A/R claim path:
  acceptance: predicted_label_ce, entropy_min, view_consistency
  rejection: entropy_max or uniform
  score: reject_efficiency initially; ar_efficiency_contrast for paired
         branch-bank claim analysis

Suppressive regularizer branch:
  logit_suppression can remain as a separate branch, but only as
  a raw-logit suppressive regularizer.

Do not promote:
  class-hypothesis acceptance search
  anchor-only CE/distill/param_reg
  full-batch A/R + Anchor CE
  direct positive-reference-delta contrast as primary score
  discarded Stage 4 score-construction variants
```

Full-spectrum audit after this revision:

- CIFAR-100 needs to be rerun with claim-compatible A/R branches before it can
  be used as positive evidence.
- ImageNet-200 still supports the semantic A/R idea for near-OOD and csID
  ordering, but far-OOD remains weak versus ASH.
- ImageNet-1K should remain blocked until ImageNet-200 far-OOD improves.

Recommended next actions:

1. Rerun CIFAR-100 using only the retained acceptance probes and semantic
   rejection probes.
2. Add a far-shift-sensitive Stage 3 response field, such as energy response,
   feature drift, activation sparsity, or reference activation stability.
3. Keep `reject_efficiency` as the primary score unless a new Stage 3 response
   justifies a new score.

## Stage 4-Only Score Construction Audit

Date: 2026-06-02.

Saved A/R `tta_response` was reused to test whether score construction alone
could improve csID/far-OOD separation without rerunning Stage 3. The retained
result is:

```text
Primary A/R score: reject_efficiency
```

For step-wise `tta_response`, this audit applies after Stage 4 selects the
declared `response_step`; it is not a license to pick an unreported best step.

The attempted Stage 4 score-construction variants did not improve over
`reject_efficiency` and are not used as score rules or calibration methods in
the current code. They were removed from the current search space.

Decision:

```text
Keep:
  reject_efficiency

Drop from current search:
  direct positive-reference-delta contrast as primary score
  discarded Stage 4 score-construction variants
  response-shape score variants
```

Reason:

```text
Score-only construction was not enough. To improve far-OOD while preserving
near-OOD, the Stage 3 TTA/probe response itself needs an additional
far-shift-sensitive measurement. Do not continue Stage 4 formula search
with the current saved response fields.
```

Next TTA implication:

```text
semantic branch:
  current A/R reject_efficiency

far-shift branch:
  new Stage 3 response field such as feature drift, activation sparsity,
  energy response, or reference activation stability

final score:
  simple combination of semantic branch and far-shift branch after the new
  response field is validated
```

## Fresh A/R Response-Bank Runtime Check

Date: 2026-06-04.

Goal:

Run the new `target_objective_delta` response-bank schema from Stage 3 on
ImageNet-200 and CIFAR-100, then compare broad Stage 4 `probe_all` scores
against Group 1 post-hoc baselines.

Attempted focused full commands:

```text
ImageNet-200:
  reference = correct_rpc32
  accept bank = predicted_label_ce, entropy_min, view_consistency
  reject bank = entropy_max, uniform
  steps/save_steps = 30 / 5,10,30
  lr = 1e-2

CIFAR-100:
  reference = correct_rpc32
  accept bank = predicted_label_ce, entropy_min, view_consistency
  reject bank = entropy_max, uniform
  steps/save_steps = 30 / 5,10,30
  lr = 1e-2
```

Both commands used `--perturbation-response pixel --perturbation-kind gaussian
--perturbation-eps 0.01 --perturbation-repeats 4` because
`view_consistency` requires perturbation views.

Result:

The runs were intentionally interrupted before completion. They were not used
for AUROC comparison.

Observed runtime issue:

```text
single-process full FSOOD run:
  low GPU memory use
  low GPU utilization
  very long target-loop wall time
```

The CIFAR-100 run completed several FSOOD loaders but then entered a large
137-batch split, making the focused run several hours long. The ImageNet-200
run entered a 469-batch split and was also projected to take hours.

Decision:

Do not use single-process full FSOOD `eval.py run-response` as the default
execution path for the semantic response bank. Before claim-bearing full
coverage runs, use a small max-sample precheck to validate schema/scoring, then
run full coverage through a parallel runner or target-sharded execution path.

This is a runtime finding only. It does not change the scientific scoring
decision.

Completed precheck:

```text
ImageNet-200:
  run_id = imagenet200_eval_api_fsood_arbank_semantic_precheck_s5_lr1e2_correct_rpc32_refseed0
  reference = correct_rpc32
  max_id_samples = 8
  max_ood_samples = 8
  response_steps = [5]
  accept_ref_loss_delta_bank shape = [8,1,3,200]
  reject_ref_loss_delta_bank shape = [8,1,2,200]
  score_results/ood.csv count after probe_all both/clean/csid = 395

CIFAR-100:
  run_id = cifar100_eval_api_fsood_arbank_semantic_precheck_s5_lr1e2_correct_rpc32_refseed0
  reference = correct_rpc32
  max_id_samples = 8
  max_ood_samples = 8
  response_steps = [5]
  accept_ref_loss_delta_bank shape = [8,1,3,100]
  reject_ref_loss_delta_bank shape = [8,1,2,100]
  score_results/ood.csv count after probe_all both/clean/csid = 395
```

Precheck decision:

The new response-bank schema and Stage 4 `probe_all` expansion work on both
ImageNet-200 and CIFAR-100. The remaining blocker is not correctness but
runtime: full target coverage needs a parallel execution plan before it is
reasonable to run the full SOTA comparison.

Target-shard execution path:

Implemented target-shard support in `scripts_my/tarr/eval.py` and merge support
in `scripts_my/tarr/merge_tta_response_shards.py`.

New Stage 3 options:

```text
--target-shard-count <N>
--target-shard-index <i>
```

Shard rule:

```text
global_sample_index % target_shard_count == target_shard_index
```

After all shards finish, merge them:

```bash
python scripts_my/tarr/merge_tta_response_shards.py \
  --dataset cifar100 \
  --scheme fsood \
  --reference-config-id correct_rpc32 \
  --output-run-dir <merged_run_dir> \
  --input-run-dir <shard0_run_dir> \
  --input-run-dir <shard1_run_dir> \
  --overwrite
```

Then run normal validation and scoring on the merged run:

```bash
python scripts_my/tarr/cache.py validate \
  --run-dir <merged_run_dir> \
  --scheme fsood \
  --reference-config-id correct_rpc32 \
  --expect-dataset cifar100

python scripts_my/tarr/cache.py score \
  --run-dir <merged_run_dir> \
  --scheme fsood \
  --reference-config-id correct_rpc32 \
  --dataset cifar100 \
  --fsood-id-side both \
  --response-step all \
  --score-rule probe_all \
  --overwrite
```

Completed 2-shard merge smoke:

```text
Dataset = CIFAR-100
Reference = correct_rpc32
Shard count = 2
Per-shard sample limit = max_id_samples=4, max_ood_samples=4
Merged run = cifar100_eval_api_fsood_arbank_semantic_shardtest_s5_lr1e2_correct_rpc32_refseed0_merged2

cache.py validate:
  OK 8 TTA response dataset(s)

Merged ID response shape:
  accept_ref_loss_delta_bank = [8,1,3,100]
  reject_ref_loss_delta_bank = [8,1,2,100]
  accept_target_objective_delta_bank = [8,1,3]
  reject_target_objective_delta_bank = [8,1,2]
  reject_target_entropy_delta_bank = [8,1,2]
  target_shard_index = -1
  target_shard_count = 2

Stage 4:
  probe_all both/clean/csid succeeded
  score_manifest count = 360
  ood.csv count = 360
```

Operational runner:

```bash
scripts_my/runners/tarr_ar_bank_sharded_focused.sh \
  imagenet200 \
  correct_rpc32 \
  "per_class=32,filter=correct,seed=0" \
  8
```

## Fresh A/R Response-Bank Full Focused Results

Date: 2026-06-05.

`both avg` is the simple mean of FSOOD `nearood` AUROC and `farood` AUROC.
Group 1 gap is measured against the best post-hoc Group 1 baseline for the
same dataset and `both avg` definition. A focused result above Group 1 is a
candidate for reference-grid and refseed follow-up, not a final SOTA claim.

Execution log:

| Date | Dataset | Reference | Run | Stage 3 | Stage 4 | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-06-05 | CIFAR-100 | `correct_rpc32` | `cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_correct_rpc32_refseed0_merged8` | 8 target shards, fresh response bank | `probe_all`, `both/clean/csid` | complete |
| 2026-06-05 | CIFAR-100 | `all_rpc8` | `cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_all_rpc8_refseed0_merged8` | 8 target shards, fresh response bank | `probe_all`, `both/clean/csid` | complete reference follow-up |
| 2026-06-05 | CIFAR-100 | `all_rpc16` | `cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_all_rpc16_refseed0_merged8` | 8 target shards, fresh response bank | `probe_all`, `both/clean/csid` | complete |
| 2026-06-05 | CIFAR-100 | `correct_rpc8` | `cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_correct_rpc8_refseed0_minbank_pce_uniform_merged8` | 8 target shards, reduced fresh response bank | `probe_all`, `both/clean/csid` | complete reference follow-up |
| 2026-06-06 | CIFAR-100 | `all_rpc32` | `cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_all_rpc32_refseed0_minbank_pce_uniform_merged8` | 8 target shards, reduced fresh response bank | `probe_all`, `both/clean/csid` | complete reference follow-up |
| 2026-06-06 | CIFAR-100 | `correct_rpc16` | `cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_correct_rpc16_refseed0_minbank_pce_uniform_merged8` | 8 target shards, reduced fresh response bank | `probe_all`, `both/clean/csid` | complete reference follow-up |
| 2026-06-06 | CIFAR-100 | `highconf09_rpc8` | `cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_highconf09_rpc8_refseed0_minbank_pce_uniform_merged8` | 8 target shards, reduced fresh response bank | `probe_all`, `both/clean/csid` | complete reference follow-up |
| 2026-06-06 | CIFAR-100 | `highconf09_rpc16` | `cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_highconf09_rpc16_refseed0_minbank_pce_uniform_merged8` | 8 target shards, reduced fresh response bank | `probe_all`, `both/clean/csid` | complete reference follow-up |
| 2026-06-06 | CIFAR-100 | `highconf09_rpc32` | `cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_highconf09_rpc32_refseed0_minbank_pce_uniform_merged8` | 8 target shards, reduced fresh response bank | `probe_all`, `both/clean/csid` | complete reference follow-up |
| 2026-06-06 | CIFAR-100 | `correcthigh09_rpc8` | `cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_correcthigh09_rpc8_refseed0_minbank_pce_uniform_merged8` | 8 target shards, reduced fresh response bank | `probe_all`, `both/clean/csid` | complete reference follow-up |
| 2026-06-06 | CIFAR-100 | `correcthigh09_rpc16` | `cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_correcthigh09_rpc16_refseed0_minbank_pce_uniform_merged8` | 8 target shards, reduced fresh response bank | `probe_all`, `both/clean/csid` | complete reference follow-up |
| 2026-06-06 | CIFAR-100 | `correcthigh09_rpc32` | `cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_correcthigh09_rpc32_refseed0_minbank_pce_uniform_merged8` | 8 target shards, reduced fresh response bank | `probe_all`, `both/clean/csid` | complete reference follow-up |
| 2026-06-06 | CIFAR-100 | `strat_rpc8` | `cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_strat_rpc8_refseed0_minbank_pce_uniform_merged8` | 8 target shards, reduced fresh response bank | `probe_all`, `both/clean/csid` | complete reference follow-up |
| 2026-06-06 | CIFAR-100 | `strat_rpc16` | `cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_strat_rpc16_refseed0_minbank_pce_uniform_merged8` | 8 target shards, reduced fresh response bank | `probe_all`, `both/clean/csid` | complete reference follow-up |
| 2026-06-06 | CIFAR-100 | `strat_rpc32` | `cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_strat_rpc32_refseed0_minbank_pce_uniform_merged8` | 8 target shards, reduced fresh response bank | `probe_all`, `both/clean/csid` | complete reference follow-up |
| 2026-06-05 | ImageNet-200 | `correct_rpc32` | `imagenet200_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_correct_rpc32_refseed0_merged8` | 8 target shards, fresh response bank | `probe_all`, `both/clean/csid` | complete |
| 2026-06-05 | ImageNet-200 | `correcthigh09_rpc16` | `imagenet200_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_correcthigh09_rpc16_refseed0_minbank_pce_uniform_merged8` | 8 target shards, reduced fresh response bank | `probe_all`, `both/clean/csid` | complete reference follow-up |
| 2026-06-05 | ImageNet-200 | `correcthigh09_rpc16` | earlier broad-bank attempts | failed/partial Stage 3 | not scored | superseded by reduced-bank merged8 run |

Group 1 comparison:

| Dataset | Reference | ID side | Score | Step | Accept | Reject | Near | Far | Both avg | Group 1 best | Gap | Decision |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CIFAR-100 | `correct_rpc32` | both | `accept_abs_ref_efficiency` | 5 | `entropy_min` | - | 63.37 | 65.89 | 64.63 | RMDS 66.20 | -1.57 | improves old default, below Group 1 best |
| CIFAR-100 | `all_rpc16` | both | `reject_efficiency` | 30 | - | `uniform` | 63.76 | 69.81 | 66.78 | RMDS 66.20 | +0.58 | first fresh claim-compatible focused run above Group 1 best |
| CIFAR-100 | `correct_rpc8` | both | `accept_abs_ref_efficiency` | 10 | `predicted_label_ce` | - | 65.63 | 65.23 | 65.43 | RMDS 66.20 | -0.77 | reduced-bank reference follow-up; below `all_rpc16` |
| ImageNet-200 | `correct_rpc32` | both | `accept_efficiency` | 5 | `predicted_label_ce` | - | 63.16 | 74.68 | 68.92 | SCALE 65.545 | +3.38 | strong focused result above Group 1 best |
| ImageNet-200 | `correcthigh09_rpc16` | both | `accept_efficiency` | 5 | `predicted_label_ce` | - | 58.22 | 70.01 | 64.12 | SCALE 65.545 | -1.43 | reduced-bank far-biased reference follow-up; below `correct_rpc32` and Group 1 |

CIFAR-100 `all_rpc16` is the current focused candidate. The margin over RMDS is
modest, so the result needs reference-grid and refseed robustness before it can
support a final claim.

ImageNet-200 `correct_rpc32` is the current strongest fresh focused result. The
best score is accept-only rather than paired A/R: `accept_efficiency` with
`predicted_label_ce` at response step 5. This should be followed by
reference-grid and refseed robustness, while separately analyzing why paired
`ar_efficiency_contrast` is much weaker.

Reference follow-up status:

The focused result is not a sufficient stopping point. CIFAR-100 15-reference
follow-up is now complete. Additional references were run with the reduced bank
needed by the focused best score families. Completed fresh references are:

| Dataset | Reference | Best FSOOD score | Step | Branch | Near | Far | Both avg | Group 1 best | Gap | Status |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| CIFAR-100 | `all_rpc8` | `target_weighted_ref_loss_delta_contrast` | 30 | accept=`predicted_label_ce`, reject=`uniform` | 64.35 | 64.62 | 64.49 | RMDS 66.20 | -1.72 | below `all_rpc16` |
| CIFAR-100 | `all_rpc16` | `reject_efficiency` | 30 | reject=`uniform` | 63.76 | 69.81 | 66.78 | RMDS 66.20 | +0.58 | current CIFAR-100 reference follow-up best |
| CIFAR-100 | `all_rpc32` | `accept_abs_ref_efficiency` | 10 | accept=`predicted_label_ce` | 64.34 | 65.43 | 64.89 | RMDS 66.20 | -1.31 | below `all_rpc16`; weaker than `correct_rpc16` |
| CIFAR-100 | `correct_rpc8` | `accept_abs_ref_efficiency` | 10 | accept=`predicted_label_ce` | 65.63 | 65.23 | 65.43 | RMDS 66.20 | -0.77 | below `all_rpc16`; better than `all_rpc8`/`correct_rpc32` |
| CIFAR-100 | `correct_rpc16` | `reject_efficiency` | 30 | reject=`uniform` | 62.95 | 69.61 | 66.28 | RMDS 66.20 | +0.08 | valid follow-up; only marginally above Group 1 and below `all_rpc16` |
| CIFAR-100 | `correct_rpc32` | `accept_abs_ref_efficiency` | 5 | accept=`entropy_min` | 63.37 | 65.89 | 64.63 | RMDS 66.20 | -1.57 | below Group 1 |
| CIFAR-100 | `highconf09_rpc8` | `accept_abs_ref_efficiency` | 30 | accept=`predicted_label_ce` | 65.17 | 65.54 | 65.36 | RMDS 66.20 | -0.84 | below `all_rpc16` and Group 1 |
| CIFAR-100 | `highconf09_rpc16` | `accept_efficiency` | 10 | accept=`predicted_label_ce` | 64.44 | 67.58 | 66.01 | RMDS 66.20 | -0.19 | improves over `highconf09_rpc8`, but remains below `all_rpc16` and Group 1 |
| CIFAR-100 | `highconf09_rpc32` | `reject_efficiency` | 30 | reject=`uniform` | 64.16 | 65.59 | 64.88 | RMDS 66.20 | -1.32 | below `highconf09_rpc16`, `all_rpc16`, and Group 1 |
| CIFAR-100 | `correcthigh09_rpc8` | `accept_abs_ref_efficiency` | 30 | accept=`predicted_label_ce` | 65.17 | 65.54 | 65.36 | RMDS 66.20 | -0.84 | below `all_rpc16` and Group 1 |
| CIFAR-100 | `correcthigh09_rpc16` | `accept_efficiency` | 10 | accept=`predicted_label_ce` | 64.44 | 67.58 | 66.01 | RMDS 66.20 | -0.19 | matches `highconf09_rpc16`; below `all_rpc16` and Group 1 |
| CIFAR-100 | `correcthigh09_rpc32` | `reject_efficiency` | 30 | reject=`uniform` | 64.16 | 65.59 | 64.88 | RMDS 66.20 | -1.32 | matches `highconf09_rpc32`; below `all_rpc16` and Group 1 |
| CIFAR-100 | `strat_rpc8` | `reject_efficiency` | 30 | reject=`uniform` | 63.39 | 70.02 | 66.70 | RMDS 66.20 | +0.50 | above Group 1 but below `all_rpc16` |
| CIFAR-100 | `strat_rpc16` | `reject_efficiency` | 30 | reject=`uniform` | 64.41 | 66.09 | 65.25 | RMDS 66.20 | -0.95 | below `all_rpc16` and Group 1 |
| CIFAR-100 | `strat_rpc32` | `reject_efficiency` | 30 | reject=`uniform` | 64.89 | 64.34 | 64.62 | RMDS 66.20 | -1.58 | below `all_rpc16` and Group 1 |
| ImageNet-200 | `correct_rpc32` | `accept_efficiency` | 5 | accept=`predicted_label_ce` | 63.16 | 74.68 | 68.92 | SCALE 65.545 | +3.38 | current ImageNet-200 focused/reference seed 0 best |
| ImageNet-200 | `correcthigh09_rpc16` | `accept_efficiency` | 5 | accept=`predicted_label_ce` | 58.22 | 70.01 | 64.12 | SCALE 65.545 | -1.43 | far-biased reference did not improve the fresh reduced-bank result |

Clean-only OOD diagnostic for the same completed references:

| Dataset | Reference | Best clean-only score | Step | Branch | Near | Far | Avg | Group 1 clean avg | Gap |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-100 | `all_rpc8` | `target_weighted_ref_loss_delta_contrast` | 30 | accept=`view_consistency`, reject=`uniform` | 80.85 | 80.70 | 80.78 | 81.535 | -0.76 |
| CIFAR-100 | `all_rpc16` | `target_weighted_ref_loss_delta_contrast` | 30 | accept=`view_consistency`, reject=`uniform` | 81.11 | 81.70 | 81.41 | 81.535 | -0.13 |
| CIFAR-100 | `all_rpc32` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.17 | 80.22 | 80.69 | 81.535 | -0.85 |
| CIFAR-100 | `correct_rpc8` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 80.93 | 80.70 | 80.81 | 81.535 | -0.72 |
| CIFAR-100 | `correct_rpc16` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.18 | 81.81 | 81.50 | 81.535 | -0.04 |
| CIFAR-100 | `correct_rpc32` | `target_weighted_ref_loss_delta_contrast` | 30 | accept=`view_consistency`, reject=`uniform` | 81.20 | 80.14 | 80.67 | 81.535 | -0.87 |
| CIFAR-100 | `highconf09_rpc8` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.23 | 79.50 | 80.37 | 81.535 | -1.17 |
| CIFAR-100 | `highconf09_rpc16` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.22 | 80.68 | 80.95 | 81.535 | -0.59 |
| CIFAR-100 | `highconf09_rpc32` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.40 | 80.61 | 81.00 | 81.535 | -0.54 |
| CIFAR-100 | `correcthigh09_rpc8` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.23 | 79.50 | 80.37 | 81.535 | -1.17 |
| CIFAR-100 | `correcthigh09_rpc16` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.22 | 80.68 | 80.95 | 81.535 | -0.59 |
| CIFAR-100 | `correcthigh09_rpc32` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.40 | 80.61 | 81.00 | 81.535 | -0.54 |
| CIFAR-100 | `strat_rpc8` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.07 | 81.59 | 81.33 | 81.535 | -0.21 |
| CIFAR-100 | `strat_rpc16` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.24 | 80.45 | 80.84 | 81.535 | -0.70 |
| CIFAR-100 | `strat_rpc32` | `reject_pos_ref_loss_delta_ood` | 30 | reject=`uniform` | 81.36 | 80.07 | 80.72 | 81.535 | -0.82 |
| ImageNet-200 | `correct_rpc32` | `target_weighted_ref_loss_delta_contrast` | 10 | accept=`view_consistency`, reject=`uniform` | 84.91 | 91.84 | 88.38 | 89.41 | -1.03 |
| ImageNet-200 | `correcthigh09_rpc16` | `reject_pos_ref_loss_delta_ood` | 10 | reject=`uniform` | 83.99 | 92.47 | 88.23 | 89.41 | -1.18 |

Do not treat the older interrupted CIFAR-100 `all_rpc32` shard-only attempt or
the earlier failed/partial broad-bank ImageNet-200 `correcthigh09_rpc16`
attempts as valid results. The valid CIFAR-100 `all_rpc32` reduced-bank
merged8 run is now complete, and it is below `all_rpc16`. The valid
CIFAR-100 `highconf09_rpc16` reduced-bank merged8 run is also complete; it
improves over `highconf09_rpc8`, but remains below `all_rpc16` and Group 1. The valid
CIFAR-100 `highconf09_rpc32` reduced-bank merged8 run is complete; it drops below
`highconf09_rpc16` and remains below `all_rpc16` and Group 1. The valid
CIFAR-100 `correcthigh09_rpc8` reduced-bank merged8 run is complete and remains
below `all_rpc16` and Group 1. The valid
CIFAR-100 `correcthigh09_rpc16` reduced-bank merged8 run is complete; it matches
`highconf09_rpc16` and remains below `all_rpc16` and Group 1. The valid
CIFAR-100 `correcthigh09_rpc32` reduced-bank merged8 run is complete; it matches
`highconf09_rpc32` and remains below `all_rpc16` and Group 1. The valid
CIFAR-100 `strat_rpc8` reduced-bank merged8 run is complete; it exceeds Group 1
but remains below `all_rpc16`. The valid CIFAR-100 `strat_rpc16` and
`strat_rpc32` reduced-bank merged8 runs are complete and remain below
`all_rpc16` and Group 1. The valid
ImageNet-200 `correcthigh09_rpc16` reduced-bank run is also complete and is
below `correct_rpc32`.

Runtime correction:

The original broad response-bank setup
`accept=predicted_label_ce,entropy_min,view_consistency` and
`reject=entropy_max,uniform` is useful for focused discovery, but it is too
expensive for a full reference grid. On 2026-06-05, attached retries showed:

```text
ImageNet-200 correcthigh09_rpc16:
  broad bank on GPU1 was too slow for reference follow-up and was superseded.
  The valid reduced-bank merged8 follow-up later completed and scored.
  Its best FSOOD both avg is 64.12, below correct_rpc32 and below SCALE 65.545.

CIFAR-100 correct_rpc8:
  broad bank was slow, so the valid follow-up used the reduced bank
  accept=predicted_label_ce, reject=uniform.
  The reduced-bank merged8 run completed and scored successfully.
```

Therefore reference-grid follow-up should use score-specific reduced banks, not
the full discovery bank:

```bash
# For the current CIFAR-100 reject-efficiency candidate:
RUN_SUFFIX=minbank_pce_uniform \
ACCEPT_PROBE_TYPES=predicted_label_ce \
REJECT_PROBE_TYPES=uniform \
scripts_my/runners/tarr_ar_bank_sharded_focused.sh \
  cifar100 <ref_id> "<ref_spec>" 8

# For the current ImageNet-200 accept-efficiency candidate:
RUN_SUFFIX=minbank_pce_uniform \
ACCEPT_PROBE_TYPES=predicted_label_ce \
REJECT_PROBE_TYPES=uniform \
scripts_my/runners/tarr_ar_bank_sharded_focused.sh \
  imagenet200 <ref_id> "<ref_spec>" 8
```

When using the 15-reference wrapper, skip the slow full-tree collector during
the experiment run and rank the finished merged run directories separately:

```bash
RUN_SUFFIX=minbank_pce_uniform \
ACCEPT_PROBE_TYPES=predicted_label_ce \
REJECT_PROBE_TYPES=uniform \
SKIP_COLLECT_SCORE=1 \
scripts_my/runners/tarr_ar_bank_sharded_refgrid.sh \
  cifar100 8
```

This preserves the branch needed by the current best scores
(`accept_efficiency` with `predicted_label_ce` and `reject_efficiency` with
`uniform`) while avoiding unnecessary Stage 3 updates for branches that are not
being grid-expanded.

Post-hoc ordering/sign-combination sweep:

```bash
python scripts_my/tarr/analyze_probe_ordering.py \
  --output results_test/tarr/summary/tarr_probe_ordering_grid.csv \
  --top-k 15
```

This is a Stage 4-only diagnostic sweep over saved score artifacts. It evaluates
single-score sign flips and accept/reject pair combinations:

```text
+score, -score
+A+R, +A-R, -A+R, -A-R
max/min over all (+/-A, +/-R) sign pairs
```

Output:

```text
results_test/tarr/summary/tarr_probe_ordering_grid.csv
```

The sweep uses global FSOOD score leaves and excludes `id_side_clean` /
`id_side_csid` score leaves. Near/far group AUROC is sample-weighted after
concatenating all datasets in the group, not dataset-balanced.

The clean CSV contains `9210` candidates:

```text
ImageNet-200 correct_rpc32:                  2880
ImageNet-200 correcthigh09_rpc16 minbank:     570
CIFAR-100 all_rpc16:                         2880
CIFAR-100 correct_rpc32:                     2880
```

Main ordering results:

| Dataset | Reference | Candidate | Step | Accept | Reject | clean/csID | csID/near | near/far | ordering min | Both avg | Group 1 gap |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ImageNet-200 | `correct_rpc32` | `ref_abs:-A-R` | 30 | `predicted_label_ce` | `uniform` | 70.79 | 60.79 | 59.00 | 59.00 | 68.87 | +3.33 |
| ImageNet-200 | `correcthigh09_rpc16` | `reject_efficiency` | 30 | - | `uniform` | 46.47 | 56.78 | 54.84 | 46.47 | 62.70 | -2.85 |
| CIFAR-100 | `all_rpc16` | `accept_abs_ref_loss_delta_mean` | 30 | `view_consistency` | - | 64.86 | 56.11 | 55.75 | 55.75 | 64.23 | -1.97 |
| CIFAR-100 | `correct_rpc32` | `ref_signed:+A+R` | 30 | `predicted_label_ce` | `entropy_max` | 55.11 | 54.33 | 54.48 | 54.33 | 57.58 | -8.62 |

Main FSOOD results from the same sweep:

| Dataset | Reference | Candidate | Step | Accept | Reject | clean/csID | csID/near | near/far | ordering min | Both avg | Group 1 gap |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ImageNet-200 | `correct_rpc32` | `ref_abs:-A-R` | 10 | `predicted_label_ce` | `uniform` | 72.91 | 61.47 | 57.33 | 57.33 | 69.27 | +3.73 |
| ImageNet-200 | `correcthigh09_rpc16` | `eff_abs:+A+R` | 10 | `predicted_label_ce` | `uniform` | 44.92 | 53.67 | 56.23 | 44.92 | 64.70 | -0.84 |
| CIFAR-100 | `all_rpc16` | `-reject_signed_ref_loss_delta_mean` | 5 | - | `uniform` | 77.85 | 50.80 | 57.77 | 50.80 | 66.91 | +0.71 |
| CIFAR-100 | `correct_rpc32` | `target_objective:+A+R` | 10 | `entropy_min` | `uniform` | 72.61 | 54.96 | 51.24 | 51.24 | 65.21 | -0.99 |

Interpretation:

- No candidate in the four fresh score sets reaches pairwise AUROC `>=60` for all
  three boundaries `clean/csID`, `csID/near`, and `near/far`.
- ImageNet-200 has useful post-hoc combinations above Group 1. The best one is
  closer to a reference-stability score than a semantic A/R efficiency score.
- ImageNet-200 `correcthigh09_rpc16` does not improve the fresh `correct_rpc32`
  result. Even the best post-hoc FSOOD candidate remains below Group 1.
- CIFAR-100 `all_rpc16` can slightly exceed Group 1 in FSOOD both avg, but
  csID/near remains nearly random. This is the main remaining weakness for the
  intended four-way ordering.

Clean/csID diagnostics:

| Dataset | Reference | ID side | Score | Step | Accept | Reject | Near | Far | Avg |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| CIFAR-100 | `correct_rpc32` | clean | `target_weighted_ref_loss_delta_contrast` | 30 | `view_consistency` | `uniform` | 81.20 | 80.14 | 80.67 |
| CIFAR-100 | `correct_rpc32` | csid | `accept_pos_ref_loss_delta_mean` | 30 | `view_consistency` | - | 57.24 | 55.06 | 56.15 |
| CIFAR-100 | `all_rpc16` | clean | `target_weighted_ref_loss_delta_contrast` | 30 | `view_consistency` | `uniform` | 81.11 | 81.70 | 81.41 |
| CIFAR-100 | `all_rpc16` | csid | `reject_efficiency` | 5 | - | `uniform` | 53.87 | 62.56 | 58.22 |
| ImageNet-200 | `correct_rpc32` | clean | `target_weighted_ref_loss_delta_contrast` | 10 | `view_consistency` | `uniform` | 84.91 | 91.84 | 88.38 |
| ImageNet-200 | `correct_rpc32` | csid | `accept_efficiency` | 5 | `entropy_min` | - | 59.05 | 69.38 | 64.22 |

This runner executes shard-specific Stage 3 jobs, merges their `tta_response`
artifacts, validates the merged cache, and runs `probe_all` for FSOOD
`both/clean/csid`. Use it as the default entrypoint for the next fresh
ImageNet-200 and CIFAR-100 focused runs.
