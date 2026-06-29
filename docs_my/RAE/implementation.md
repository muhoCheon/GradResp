# RAE Implementation

RAE is a standalone pipeline under `scripts_my/rae`. It must not import from
`scripts_my/tarr`; shared behavior should be copied or factored into a neutral
module only after that boundary is made explicit.

## Pipeline Shape

RAE scores whether a target sample has any ID class hypothesis that is both
plausible under the pretrained classifier and validated by labeled ID
references in gradient space.

```text
train_candidate_metadata -> reference_set -> gradient_evidence -> score_result -> diagnostics
```

Main quantities:

- `q_c(x)`: pretrained target probability for candidate class `c`.
- `V_c(x)`: signed reference-validation support for accepting `x` as class `c`.
- `E_c(x) = q_c(x) * V_c(x)`: class-wise ID evidence.
- `eid = max_c E_c(x)`: best ID evidence over evaluated candidates.
- `ood_score`: larger means more OOD-like.
- OpenOOD `conf = -ood_score`.

## CLI Shape

Use `run-single` for one concrete RAE configuration: one reference size, one
reference seed, one gradient space, and one candidate mode.

```bash
conda run --no-capture-output -n openood python -m scripts_my.rae.eval run-single \
  --dataset cifar10 \
  --baseline-protocol eval_api \
  --scheme fsood \
  --gradient-space classifier \
  --reference-per-class 16 \
  --reference-filter correct \
  --reference-seed 0 \
  --candidate-mode all \
  --score-rules neglog_eid,neg_eid \
  --diagnostics all \
  --diagnostic-samples 256 \
  --output-root results_test/rae
```

`run-single` builds or reuses train candidate metadata, reference artifacts,
and gradient banks, then writes score files and child diagnostics for Gates 1,
2, 3, 4, 6, 7, and 10. Gate 3 is required only for `classifier` because it is
the FC-head factorization check.

Use `run-all` for the full experiment suite. It expands the grid options,
launches the corresponding child `run-single` jobs, aggregates all metrics, and
writes parent-level comparison diagnostics for Gate 8 and Gate 9.

```bash
conda run --no-capture-output -n openood python -m scripts_my.rae.eval run-all \
  --dataset cifar10 \
  --baseline-protocol eval_api \
  --scheme fsood \
  --reference-filter correct \
  --reference-per-class-grid 4,8,16,32,64 \
  --reference-seeds 0,1,2 \
  --gradient-spaces classifier,last_block,all \
  --candidate-modes all,pred \
  --score-rules neglog_eid,neg_eid \
  --diagnostics all \
  --diagnostic-samples 256 \
  --output-root results_test/rae
```

`run-all` has no separate gate selector for comparison studies. Multiple
reference sizes or seeds produce Gate 8; multiple gradient spaces produce Gate
9. When both `all` and `pred` candidate modes are requested with
`--rejection-rule off`, `pred` score artifacts are derived from the already saved
`all` artifacts using `eid_pred`/`v_pred`; the target splits are not scored a
second time. Gate 5 is still unimplemented and is recorded only as deferred
metadata, not as a skipped required result.

## Key Options

| Option | Values | Meaning |
| --- | --- | --- |
| `--gradient-space` | `classifier`, `last_block`, `all` | Parameter space used for acceptance/reference gradients. `classifier` is the fast FC-head control, `last_block` uses the last trainable non-classifier block, and `all` uses every trainable model parameter. |
| `--gradient-spaces` | comma list | `run-all` grid over gradient spaces. Defaults to `classifier,last_block`. |
| `--reference-per-class` | integer | Number of ID train references selected per class. |
| `--reference-per-class-grid` | comma list | `run-all` grid over reference sizes. Defaults to `4,8,16,32,64`. |
| `--reference-filter` | `all`, `correct`, implementation-defined stricter filters | Selects the eligible ID train pool before class-balanced sampling. |
| `--reference-seed` | integer | Seed for class-wise reference sampling; part of the reference identity. |
| `--reference-seeds` | comma list | `run-all` grid over reference seeds. Defaults to `--reference-seed`. |
| `--rebuild-train-metadata` | flag | Re-scan ID train candidate metadata even when the checkpoint/dataset identity matches an existing artifact. |
| `--candidate-mode` | `all`, `pred` | Candidate classes evaluated per target. `all` scores every class and matches the exact RAE definition; `pred` validates only the model argmax class as a lower-compute RAE variant. |
| `--candidate-modes` | comma list | `run-all` grid over candidate modes. Defaults to `all`. |
| `--validation-rule` | `pairwise_rank`, `pairwise_margin`, `same_mean`, `mean_margin`, `soft_margin` | Rule used for acceptance-direction validation `V_c`. `pairwise_rank` is the original hard pairwise definition; the others are acceptance-agreement alternatives for reducing `V_c` saturation. |
| `--validation-rules` | comma list | `run-all` grid over validation rules. Defaults to `pairwise_rank`. |
| `--validation-temperature` | float | Temperature used only by `soft_margin`. |
| `--rejection-rule` | `off`, `uniform` | Optional rejection-evidence multiplier. `off` is the default and preserves the acceptance-only RAE score. `uniform` is an experimental classifier-only variant that compares the target's confidence residual against same-class reference acceptance directions. |
| `--rejection-power` | float | Power for multiplying rejection-derived ID evidence into final class evidence. `0` is equivalent to no rejection effect; larger values strengthen the multiplier. |
| `--rejection-rules` | comma list | `run-all` grid over rejection rules. Defaults to `off`. |
| `--score-rules` | `neglog_eid`, `neg_eid`, comma list | Score rules materialized for Stage 4. |
| `--diagnostics` | `off`, `all` | Runs all currently implemented diagnostic gates and writes `diagnostics_manifest.json`; `all` is the default. |
| `--diagnostic-gates` | `all`, or numeric comma list | Selects debugging child gates such as `1,2,3,4,6,7,10`; claim-bearing runs should use `all`. Gate 8/9 are parent `run-all` comparisons. |
| `--diagnostic-samples` | integer | Number of small-batch samples for online gradient gates. |
| `--diagnostic-seed` | integer | Seed used for diagnostic label-shuffle artifacts. |

Score rules:

- `neglog_eid`: `ood_score = -log(max_c q_c(x)V_c(x) + eps)`.
- `neg_eid`: `ood_score = -max_c q_c(x)V_c(x)`.

Both rules follow the same direction convention: larger `ood_score` is more
OOD-like, and exported OpenOOD confidence is `conf = -ood_score`.

Candidate mode semantics:

- `all`: exact RAE candidate set, `E_ID(x) = max_c q_c(x)V_c(x)`.
- `pred`: predicted-class-only validation,
  `E_pred(x) = q_argmax(x)V_argmax(x)`. This is recorded as
  `resource_adjusted_candidate_set=true` and `claim_scope=predicted_class_rae`.
  It can be claim-bearing for the predicted-class RAE variant when the run is
  full-data and diagnostics pass, but it should not be described as the exact
  max-over-classes RAE estimator. In `run-all`, the `pred` artifact can be
  materialized from the matching `all` artifact because `all` scoring already
  saves `eid_pred` and `v_pred`.

Validation rule semantics:

- `pairwise_rank`: original rule,
  `V_c = mean[(K_same > 0) and (K_same > K_other)]` over all same/other
  reference pairs.
- `pairwise_margin`: positive pairwise excess margin,
  `V_c = mean[max(K_same - K_other, 0) * 1[K_same > 0]] / 2`.
- `same_mean`: positive mean same-class acceptance agreement,
  `V_c = max(mean K_same, 0)`.
- `mean_margin`: positive mean same-vs-other agreement margin,
  `V_c = max(mean K_same - mean K_other, 0) / 2`.
- `soft_margin`: smooth same-support and margin gate,
  `sigmoid(mean K_same / temperature) *
  sigmoid((mean K_same - mean K_other) / temperature)`.

Rejection rule semantics:

- `off`: final evidence is the acceptance-only RAE evidence,
  `E_c^{final}(x) = E_c^{accept}(x)`.
- `uniform`: experimental classifier-only rejection evidence. It builds a
  rejection residual from the target distribution against the uniform class
  distribution and compares it with same-class reference acceptance directions.
  For each candidate class, `R_c(x) = clamp(-mean_{r:y_r=c} K^R(x,r), 0, 1)`.
  The final evidence is
  `E_c^{final}(x) = E_c^{accept}(x) * (R_c(x) + eps)^beta`, where
  `beta = --rejection-power`. This keeps the RAE acceptance-evidence structure
  and adds a class-wise rejection multiplier, but it is not the default because
  current CIFAR full tests did not improve the best baseline.

## Output Layout

Canonical outputs live under `results_test/rae`:

```text
results_test/rae/
  train_candidate_metadata/
    <dataset>/<metadata_id>/
      manifest.json
      candidates.npz

  reference_sets/
    <dataset>/<reference_config_id>/seed<reference_seed>/
      manifest.json
      reference_set.npz
      selected_samples.csv

  gradient_banks/
    <dataset>/<gradient_config_id>/<reference_set_hash>/
      manifest.json
      gradient_bank.npz

  outputs/
    <dataset>/<baseline_protocol>/seed<reference_seed>/<run_id>/
      run_manifest.json
      <scheme>/
        <score_rule>/
          scores/<target_dataset_name>.npz
          ood.csv

  diagnostics/
    <dataset>/<run_id>/

  experiments/
    <dataset>/<experiment_id>/
      experiment_manifest.json
      diagnostics_manifest.json
      runs.csv
      metrics.csv
      gate08_reference_size_stability.csv
      gate08_reference_size_trends.csv
      gate09_gradient_space_ablation.csv
      gates/
```

Reference artifacts are built from `train_candidate_metadata`, not by directly
sampling the loader in-memory. The metadata records ID train labels, pretrained
predictions, confidence, correctness, dataset indices, and image names under a
checkpoint/dataset identity.

Class-wise quota is strict. If `--reference-per-class N` is requested and any
class has fewer than `N` eligible candidates after filtering, reference
construction fails with `Not enough RAE reference samples...`. Such a run is not
silently converted into a smaller reference set. This keeps Gate 8 reference
size comparisons interpretable.

Score `.npz` files retain `pred`, OpenOOD `conf`, `label`, `ood_score`, raw
`eid`, `best_class`, `q_best`, and `v_best`.

They also retain class-wise RAE diagnostics:

- `candidate_classes`
- `q_c`, `v_c`, `e_c`
- `rank_only_scores`
- `same_positive_rates`
- `q_max`, `v_pred`, `eid_pred`
- `accept_eid`, `reject_id_evidence`, `reject_ood_evidence`,
  `reject_k_mean` when rejection evidence is enabled
- `v_c_label_shuffle`, `e_c_label_shuffle`, `eid_label_shuffle` when
  diagnostics are enabled

The score files are therefore sufficient for post-hoc Gates 4, 6, 7, and 10.

Diagnostics are written as first-class artifacts:

```text
results_test/rae/diagnostics/<dataset>/<run_id>/
  diagnostics_manifest.json
  gates.csv
  gates/
    gate01_acceptance_delta.json
    gate02_reference_sign_delta.json
    gate03_classifier_fc_factorization.json
    gate04_<score_rule>_confidence_bins.csv
    gate04_confidence_matched_separation_<score_rule>.json
    gate06_signed_support_<score_rule>.json
    gate07_label_shuffle_<score_rule>.json
    gate10_score_ablation_<score_rule>.json
```

Saved runs can be summarized again without rebuilding references or rescoring:

```bash
conda run --no-capture-output -n openood python -m scripts_my.rae.eval run-diagnostics \
  --run-dir results_test/rae/outputs/cifar10/eval_api/seed0/<run_id> \
  --diagnostic-gates 4,6,7,10
```

Saved child runs are reused by `run-all` unless `--overwrite` is passed.

`run_manifest.json` embeds the diagnostics manifest and records
`claim_bearing=false` when target-limited smoke mode is used, diagnostics are disabled, or a
required gate is missing, skipped, or failed. Candidate-set semantics are
tracked separately through `claim_scope`, `exact_candidate_set`, and
`resource_adjusted_candidate_set`; therefore `--candidate-mode pred` is not
automatically non-claim-bearing.

`experiment_manifest.json` records the full grid, child run table, aggregated
metric table, Gate 8/9 diagnostics, and claim-bearing reasons for the parent
experiment.

## Diagnostic Gates v1

The v1 diagnostic report tracks these gates:

| Gate | Name | Check |
| --- | --- | --- |
| 1 | Acceptance direction sanity | A small step along the acceptance direction decreases the target candidate CE. |
| 2 | First-order sign prediction | Gradient compatibility sign predicts the first-order reference loss direction; finite-step deltas are recorded as auxiliary metrics. |
| 3 | FC factorization / triviality | For `classifier`, compact FC compatibility must match dense classifier-gradient compatibility. |
| 4 | Confidence-matched separation | Within confidence bins, RAE evidence separates clean ID, csID, near OOD, and far OOD. |
| 6 | Signed-support necessity | Compare unsigned ranking support against signed positive support. |
| 7 | Reference label shuffle | Shuffling reference labels should degrade evidence and OOD metrics. |
| 8 | Reference size and seed stability | Compare reference sizes/seeds for stable evidence and AUROC. |
| 9 | Gradient-space ablation | Compare classifier, last-block, and all-parameter gradient spaces when requested. |
| 10 | Final score ablation | Compare confidence only, validation only, joint evidence, and final OOD score. |

Gate 5 PASS-distance control is not implemented yet. It is recorded in
`diagnostics_manifest.json` under `deferred_gates`, not as a skipped required
gate. Gate 8 and Gate 9 are generated by the parent `run-all` experiment, not
by an individual `run-single` child.
