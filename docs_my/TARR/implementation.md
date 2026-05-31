# TARR Implementation

TARR is implemented as a four-stage artifact pipeline:

```text
train_candidate_metadata -> reference_set -> tta_response -> score_result
```

Each stage has a manifest, an identity, and a single canonical output location. Inline scoring and offline scoring both execute Stage 4 and write the same `score_result` layout.

## Package Layout

```text
scripts_my/tarr/
  reference.py    Stage 1 train_candidate_metadata and Stage 2 reference_set builders
  eval.py         Stage 3 tta_response runner and one-command Stage 1-4 orchestration
  adaptation.py   TTA/runtime/update-policy config helpers
  scoring.py      score_rule definitions that produce score_result values
  cache.py        artifact validation and Stage 4 score_result CLI
  protocol.py     manifest/protocol identity helpers
  reports.py      diagnostics, collection, and Group 1 comparison CLI
  run_matrix.py   protocol/job matrix orchestration CLI
```

## Execution Flow

1. Stage 1 builds `train_candidate_metadata` from the ID train split. The artifact records candidate labels, model predictions, confidence/entropy/margin/energy diagnostics, correctness, CE loss, dataset indices, and the identity of the train imglist, checkpoint, model, preprocessing, and schema.
2. Stage 2 builds one or more `reference_set` artifacts from `train_candidate_metadata`. Each `reference_set` records the selected samples, selection hash, per-class counts, base reference response, and reference config.
3. Stage 3 runs target-only TTA once per target sample and writes `tta_response` for every target dataset/reference pair. It records target pre/post adaptation diagnostics, adapted reference response, response deltas, perturbation diagnostics when configured, and runtime.
4. Stage 4 applies a score rule to `tta_response` and writes a `score_result`. Metrics, OpenOOD score files, and score manifests live under `score_results/<score_rule>/`.
5. `reports.py diagnostics` summarizes `tta_response` and `score_result` artifacts, including clean-vs-csID alignment for FSOOD.

## Stage Identity

| Stage | Identity fields | Rebuild trigger |
| --- | --- | --- |
| `train_candidate_metadata` | Dataset, train imglist path/SHA256, checkpoint path/SHA256, model architecture, class count, preprocessing identity, metadata schema | Any identity change. |
| `reference_set` | Stage 1 identity, reference source, filter, per-class size, confidence threshold, seed, selected-sample hash | Reference config or Stage 1 change. |
| `tta_response` | Dataset, scheme, baseline protocol, target split, csID identity, checkpoint identity, TTA config, runtime mode, perturbation config, `reference_set_id`, full/subset status, response schema | Protocol, TTA, perturbation, target sampling, checkpoint, or reference_set change. |
| `score_result` | `tta_response_id`, score rule, score direction, FSOOD ID-side aggregation, scoring schema | Score-rule or scoring aggregation change. |

Scoring-only changes start at Stage 4. TTA or protocol changes start at Stage 3. Reference selection changes start at Stage 2. Dataset/checkpoint/preprocessing changes start at Stage 1.

## Canonical Layout

```text
results_test/tarr/
  train_candidate_metadata/
    <dataset>/<candidate_id>/
      manifest.json
      candidates.npz

  reference_sets/
    <dataset>/<reference_config_id>/seed<seed>/<reference_set_id>/
      manifest.json
      reference_set.npz
      selected_samples.csv
      preview/  # optional image copies for human inspection

  outputs/
    <dataset>/<baseline_protocol>/seed<seed>/<run_id>/
      run_info.md
      run_manifest.json
      <scheme>/
        scheme_manifest.json
        references/<reference_config_id>/
          tta_response/<target_dataset_name>/
            manifest.json
            part_*.npz
          score_results/<score_rule>/
            scores/<target_dataset_name>.npz
            ood.csv
          diagnostics/
```

## Main CLI Shape

One-command orchestration can run all four stages:

```bash
python scripts_my/tarr/eval.py \
  --dataset cifar10 \
  --baseline-protocol eval_api \
  --scheme fsood \
  --reference-config all_rpc16:per_class=16,filter=all,min_confidence=0.9,seed=0 \
  --objective entropy \
  --steps 5 \
  --lr 1e-2 \
  --update-scope classifier \
  --runtime-mode auto \
  --score-rule all \
  --save-tta-response \
  --tta-response-shard-size 0
```

Stage-specific CLIs follow the same artifact boundaries:

```bash
python scripts_my/tarr/reference.py build-train-metadata ...
python scripts_my/tarr/reference.py build-reference-set ...
python scripts_my/tarr/eval.py run-response --use-prebuilt-reference-set ...
python scripts_my/tarr/cache.py score ...
python scripts_my/tarr/reports.py diagnostics ...
```

Runnable dataset-specific commands are collected in [commands.md](commands.md).

`eval.py run-response --use-prebuilt-reference-set` is strict Stage 3 mode: it
loads prebuilt `reference_set` artifacts and fails if any requested reference
set is missing. `eval.py run-all` is the convenience orchestration mode for
running the stages together.

## Option Axes

| Axis | Examples | First affected stage |
| --- | --- | --- |
| Dataset/checkpoint/preprocessing | dataset, train imglist, checkpoint SHA256, model arch | Stage 1 |
| `reference_set` config | per-class size, filter, confidence threshold, seed | Stage 2 |
| Protocol config | OOD/FSOOD scheme, baseline protocol, csID identity, target split list | Stage 3 |
| TTA config | objective, steps, lr, update scope, runtime mode, freeze BN stats | Stage 3 |
| Perturbation/view config | view count, view seed, perturbation kind/epsilon/repeats | Stage 3 |
| `score_result` config | score rule, FSOOD clean/csID/both ID-side aggregation | Stage 4 |

Multiple `reference_set` configs can share the same Stage 3 target TTA work when they are bundled into the same run. Each reference still writes its own `tta_response` and `score_result`.

In current broad search runs, one Stage 3 run uses one reference seed and
15 reference configs: `all`, `correct`, `high_confidence`,
`correct_high_confidence`, and `correct_confidence_stratified`, each with
`per_class` values 8, 16, and 32. Reference seeds 0, 1, and 2 are separate
Stage 3 runs so the `reference_config_id` paths remain unambiguous.

## Runtime Modes

| Mode | Reference response path | Target TTA path | Valid update scope |
| --- | --- | --- | --- |
| `auto` | selected from update scope | selected from update scope | `classifier`, `all` |
| `full_forward` | reference images through full model | target image through full model | `classifier`, `all` |
| `classifier_feature_cache` | cached reference features through classifier | cached target feature through classifier | `classifier` only |

`auto` selects `classifier_feature_cache` for `update_scope=classifier` and `full_forward` for `update_scope=all`.

## TTA Objectives

Soft view-consistency uses multiple deterministic or stochastic views of the same target sample during target-only TTA. It penalizes soft distribution mismatch, not hard predicted-label matching. `reference_set` data is used only to measure response after target adaptation.

| Objective | Target views | Optimization signal | Role |
| --- | --- | --- | --- |
| `predicted_label_ce` | Original target view | CE to the pretrained predicted class | Baseline target-only objective. |
| `entropy` | Original target view | Entropy minimization on the current prediction | Tent-style single-view entropy objective. |
| `memo_marginal_entropy` | Augmentation views | Entropy of the average prediction over views | MEMO-style marginal entropy objective. |
| `view_consistency_js` | Original + augmentation views | Jensen-Shannon divergence among soft view predictions | Symmetric bounded soft consistency. |
| `view_consistency_kl` | Original + augmentation views | KL divergence from a stop-gradient mean or anchor distribution | Directional soft consistency. |
| `entropy_consistency` | Original + augmentation views | Average view entropy plus a soft view-consistency penalty | Combined confidence and consistency objective. |

## Reference Set Protocol

`reference_set` artifacts are always selected from ID train data. OOD, csID, and ID test target samples are not eligible for `reference_set` selection.

| Filter | Rule | Purpose |
| --- | --- | --- |
| `all` | Sample train examples without model-based filtering. | Default coverage-oriented protocol. |
| `correct` | Keep only examples correctly classified by the pretrained model. | Test whether model-correct references stabilize response. |
| `high_confidence` | Keep examples whose pretrained confidence is at least `reference_min_confidence`. | Test whether confident references improve response quality. |
| `correct_high_confidence` | Require both correct prediction and high confidence. | Test strict quality-vs-coverage tradeoff. |
| `correct_confidence_stratified` | Select correctly classified samples from low, middle, and high confidence bins within each class. | Test confidence diversity among correct references. |

For CIFAR-10, CIFAR-100, and ImageNet-200, high-confidence filters use
`min_confidence=0.9` and config ids such as `highconf09_rpc32` and
`correcthigh09_rpc32`. For ImageNet-1K, high-confidence coverage is lower, so
the current grid uses `high_confidence` at `0.8` (`highconf08_*`) and
`correct_high_confidence` at `0.75` (`correcthigh075_*`).

When candidate count exceeds `per_class`, `all`, `correct`,
`high_confidence`, and `correct_high_confidence` use class-wise seed-controlled
random selection. `correct_confidence_stratified` first keeps only correct
examples, sorts each class by pretrained confidence, splits the sorted list
into low/mid/high thirds, and allocates the remainder in `mid -> high -> low`
order before sampling within each stratum.

ImageNet-scale `reference_set` sizes must be predeclared and reported as resource-adjusted when they materially change runtime.

## TTA Response

`tta_response` is the Stage 3 artifact. For each target dataset and `reference_set`, it stores:

- Target labels, predictions, pretrained probabilities, confidence, entropy, margin, and energy.
- Adapted target prediction summaries.
- Base and adapted reference losses by class.
- `response_delta_c = L_c(theta_T) - L_c(theta_0)`.
- Reference class-wise confidence/entropy/margin/energy/correctness deltas.
- Perturbation diagnostic fields when configured.
- Runtime per target.

ImageNet-scale full runs write sharded `tta_response` directories with `manifest.json` and `part_*.npz`.

## Score Rules

Canonical primitive:

```text
response_delta_c = L_c(theta_T) - L_c(theta_0)
```

`response_delta_c > 0` means reference loss increased. `response_delta_c < 0` means reference loss decreased.

Claim score rules:

| Score rule | Calculation | Meaning |
| --- | --- | --- |
| `predicted_class_loss_increase` | `response_delta_y_hat` | Loss increase for the target's pretrained predicted class. |
| `predicted_class_loss_decrease` | `-response_delta_y_hat` | Loss decrease for the target's pretrained predicted class. |
| `target_weighted_loss_increase` | `sum_c p_0(c) * response_delta_c` | Pretrained-probability-weighted loss increase. |
| `target_weighted_loss_decrease` | `-sum_c p_0(c) * response_delta_c` | Pretrained-probability-weighted loss decrease. |
| `mean_loss_increase` | `mean_c response_delta_c` | Average class loss increase. |
| `mean_loss_decrease` | `-mean_c response_delta_c` | Average class loss decrease. |
| `positive_loss_increase_mean` | `mean_c max(response_delta_c, 0)` | Average of class losses that increased. |
| `positive_loss_decrease_mean` | `mean_c max(-response_delta_c, 0)` | Average of class losses that decreased. |

Here `y_hat` is the target sample's pretrained predicted class, and `p_0(c)` is the pretrained probability for class `c`.

Score convention:

```text
ood_score larger = more OOD-like
OpenOOD conf = -ood_score
```

## Diagnostic Score Results

Diagnostic score rules can also produce `score_result` artifacts, but they are not claim-bearing until their formula, sign convention, protocol selection rule, and promotion gate are fixed before the comparison run.

| Family | Inputs | Purpose |
| --- | --- | --- |
| Direction-aware summaries | `response_delta_c`, `y_hat`, target probabilities | Inspect whether class-wise response direction explains failures. |
| Magnitude-only summaries | Full `response_delta_c` vector | Test whether total response magnitude matters more than sign. |
| Clean-prototype vector summaries | Clean-ID fit from `tta_response` | Test distance from clean response shape without fitting on csID/OOD. |
| Perturbation-response summaries | Perturbation fields in `tta_response` | Test whether controlled target perturbation response preserves clean/csID alignment. |

## Protocol Handling

`baseline_protocol` controls which baseline table a run can be compared against.

| Dataset | `main_py` FSOOD csID | `eval_api` FSOOD csID |
| --- | --- | --- |
| CIFAR-10 | `cinic10` | `cifar10c` |
| CIFAR-100 | `cifar100c` | `cifar100c` |
| ImageNet-1K | `imagenetv2`, `imagenetc`, `imagenetr` | `imagenet_v2`, `imagenet_c`, `imagenet_r` |
| ImageNet-200 | `imagenetv2`, `imagenetc`, `imagenetr` | `imagenet_v2`, `imagenet_c`, `imagenet_r` |

FSOOD main metrics use clean ID + csID as the ID side. Clean-only and csID-only rows are diagnostic `score_result` views used to inspect alignment.

## Diagnostics

`reports.py diagnostics` reads canonical artifacts and writes:

- `score_summary.csv`: score-rule distributions per split/dataset.
- `target_summary.csv`: pre/post TTA target behavior.
- `delta_summary.csv`: response-delta summaries.
- `runtime_summary.csv`: runtime distribution.
- `reference_summary.csv`: base/adapted reference response summaries.
- `alignment_summary.csv`: clean-vs-csID score alignment and protocol identity.
- `perturbation_summary.csv`: perturbation field summaries when present.
- `perturbation_alignment_summary.csv`: perturbation clean/csID alignment when present.
- `score_direction.csv`: split-vs-ID score direction checks.
- `failure_cases.csv`: high-scoring ID/csID and low-scoring OOD examples.

## Runtime Reporting

Runtime rows should use canonical stage names:

| Column | Meaning |
| --- | --- |
| `run_id` | TARR run id or benchmark id. |
| `dataset` | Dataset argument. |
| `baseline_protocol` | Baseline protocol used by the run. |
| `scheme` | `ood` or `fsood`. |
| `reference_config_id` | Reference config measured. |
| `runtime_mode` | Resolved runtime mode. |
| `train_candidate_metadata_id` | Stage 1 artifact id. |
| `train_candidate_metadata_reused` | Whether Stage 1 artifact was reused. |
| `train_candidate_metadata_sec` | Stage 1 build/load seconds. |
| `reference_set_id` | Stage 2 artifact id. |
| `reference_set_reused` | Whether Stage 2 artifact was reused. |
| `reference_set_sec` | Stage 2 build/load seconds. |
| `tta_response_total_sec` | Stage 3 target TTA plus adapted reference response seconds. |
| `score_result_total_sec` | Stage 4 scoring seconds. |
| `processed_targets` | Number of target samples processed. |
| `runtime_per_target_sec` | Stage 3 seconds per target. |
| `batch_size` | Target dataloader batch size. |
| `train_candidate_batch_size` | Stage 1 candidate scan batch size. |
| `reference_set_batch_size` | Stage 2/3 selected reference response batch size. |
| `num_workers` | Dataloader worker count. |
| `cuda_device` | GPU name or `unavailable`. |
