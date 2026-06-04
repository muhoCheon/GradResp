# TARR: Test-time Adaptation Reference Response

TARR measures how a short target-only test-time adaptation changes the model response on an ID train reference set.

```text
When one target sample adapts the model briefly,
does the ID reference response stay stable or shift in an OOD-like way?
```

The method is organized around four artifacts:

```text
train_candidate_metadata -> reference_set -> tta_response -> score_result
```

## Method Intuition

If a target sample is close to the ID distribution, adapting on that target should not strongly disrupt the model's response on ID train references. If the target is OOD, the same target-only adaptation can produce larger or differently directed changes in reference loss and related response statistics. TARR turns those changes into an OOD score.

`reference_set` data is never used as the adaptation loss. It is used only as a measurement surface before and after target adaptation.

## Claim Status

The current claim-bearing implementation is the `scripts_my/tarr/` package pipeline. Active scope:

- Target-only TTA objectives: `predicted_label_ce`, `entropy`, `memo_marginal_entropy`, `view_consistency_js`, `view_consistency_kl`, and `entropy_consistency`.
- ID train `train_candidate_metadata`.
- ID train `reference_set`.
- Canonical loss-response score rules that write `score_result`.
- Acceptance/Rejection response-bank runs with documented accept probes
  `predicted_label_ce`, `entropy_min`, `view_consistency` and semantic reject
  probes `entropy_max`, `uniform`.
- Runtime modes `auto`, `full_forward`, and `classifier_feature_cache`.

Planned extensions remain outside claim scope until separately specified: energy objective, soft pseudo-label CE, far-shift response branches, and OpenOOD postprocessor integration. `logit_suppression` is only an evidence/energy suppression probe, not semantic rejection. `topk_ce` and `allclass_ce` are not default/documented accept probes.

Every claim-bearing run must write manifests that identify the dataset, baseline protocol, csID datasets, checkpoint SHA256, imglist SHA256, full/subset status, `train_candidate_metadata_id`, `reference_set_id`, TTA config, `tta_response` config including `response_steps`, score rule, selected `response_step`, and `score_result_id`.

## Claims To Validate

### Reference Response Stability

- ID targets should produce relatively stable `reference_set` response after TTA.
- OOD targets should produce stronger or more OOD-like `tta_response` deltas.
- Clean ID, csID, nearOOD, and farOOD distributions should be compared separately.
- Response deltas should be inspected by mean, variance, quantiles, class-wise maxima, and predicted-class response.

### Score Ordering

- `ood_score` must increase in the OOD-like direction.
- OpenOOD `conf` is `-ood_score`, so larger `conf` remains ID-like.
- ID and csID should stay closer to each other than to OOD.
- FSOOD promotion requires clean-vs-csID alignment, not only clean-ID vs OOD separation.

### Reference-Only Evidence

- TARR must show that adaptation-induced `reference_set` response changes add evidence beyond target confidence scores such as MSP, MLS, and EBO.
- The TTA objective may use target data and target-derived context only.
- `score_result` values should remain interpretable from the corresponding `tta_response`.

### Transfer And Cost

- CIFAR-10 settings should transfer to CIFAR-100 before CIFAR-100-specific tuning.
- Runtime must be reported by artifact stage.
- Cost claims use matched protocol, hardware, target count, and artifact identities.

## Four-Stage Pipeline

### Stage 1: `train_candidate_metadata`

Build model-derived metadata over the ID train split:

- labels and dataset indices
- pretrained predictions and probabilities
- confidence, entropy, margin, energy
- correctness and CE loss
- train imglist, checkpoint, model, preprocessing, and schema identity

This artifact is the source of eligible candidates for `reference_set` selection.

### Stage 2: `reference_set`

Select class-balanced ID train references from `train_candidate_metadata`. The selection config includes source, filter, per-class size, confidence threshold, and seed. The artifact records selected samples, selected hash, per-class counts, base reference response, and manifest identity.

### Stage 3: `tta_response`

For each target sample:

1. Save pretrained target response.
2. Run target-only TTA for the configured objective, max `--steps`, learning rate, update scope, and runtime mode.
3. Measure adapted response on each `reference_set` at the saved `response_steps` selected by `--save-steps`.
4. Store target diagnostics, reference losses, response deltas, perturbation diagnostics when configured, and runtime. Step-wise class fields use `[N,S,C]`; scalar response fields use `[N,S]`. A/R response-bank fields add accept/reject branch axes, for example `[N,S,A,C]` and `[N,S,R,C]`.

### Stage 4: `score_result`

Apply a score rule to `tta_response`. For step-wise responses, Stage 4 selects one saved update count with `--response-step` before scoring. For A/R response-bank artifacts, Stage 4 also selects branches with `--accept-branch`, `--reject-branch`, and `--branch-combine`. Stage 4 writes OpenOOD score files, `ood.csv`, score direction metadata, and any ID-side aggregation used for FSOOD. Running Stage 4 immediately after Stage 3 and running it later from saved `tta_response` are equivalent when the inputs match.

## Score Convention

```text
ood_score larger -> OOD-like
conf = -ood_score
conf larger -> ID-like
plot score = ood_score
```

## Reference Set

Reference selection is ID train only. OOD, csID, and ID test target loaders are not reference sources.

Reference filters:

- `all`: class-balanced random subset from eligible train samples.
- `correct`: pretrained model must classify the train sample correctly.
- `high_confidence`: pretrained confidence must meet `reference_min_confidence`.
- `correct_high_confidence`: both correct and high confidence.
- `correct_confidence_stratified`: correctly classified samples selected from low/middle/high confidence bins within each class.

Primary default candidate: ID train source, `reference_filter=all`, predeclared seed or seed mean/variance reporting.

Per-class size candidates:

```text
CIFAR-10: 4, 8, 16, 32, 64
CIFAR-100: 2, 4, 8, 16, 32
ImageNet-scale: predeclared resource-adjusted sizes only
```

## TTA Objective

Active objectives:

- `predicted_label_ce`: cross entropy to the pretrained predicted class.
- `entropy`: single-view target entropy minimization.
- `memo_marginal_entropy`: entropy of the augmentation-view average prediction.
- `view_consistency_js`: Jensen-Shannon soft consistency across target views.
- `view_consistency_kl`: KL soft consistency to a fixed mean or anchor distribution.
- `entropy_consistency`: view entropy plus a soft consistency penalty.

Default optimizer family is SGD. Classifier-only update is the efficient default; all-parameter update is a separate TTA config and must report its cost.

## Response And Score Rules

Class `c` reference loss:

```text
L_c(theta) = mean_{(x_i, y_i=c) in reference_set[c]} CE(f_theta(x_i), c)
```

Response delta:

```text
response_delta_c = L_c(theta_T) - L_c(theta_0)
```

Canonical score rules:

- `predicted_class_loss_increase = response_delta_{y_hat}`
- `predicted_class_loss_decrease = -response_delta_{y_hat}`
- `target_weighted_loss_increase = sum_c p_0(c) * response_delta_c`
- `target_weighted_loss_decrease = -sum_c p_0(c) * response_delta_c`
- `mean_loss_increase = mean_c response_delta_c`
- `mean_loss_decrease = -mean_c response_delta_c`
- `positive_loss_increase_mean = mean_c max(response_delta_c, 0)`
- `positive_loss_decrease_mean = mean_c max(-response_delta_c, 0)`

A/R response-bank score rules are selected directly or with
`--score-rule probe_all`. `reject_efficiency` scores rejection branches,
`accept_efficiency` scores acceptance branches, and
`ar_efficiency_contrast` scores accept/reject branch pairs, usually with
`--accept-branch all --reject-branch all --branch-combine cross`.

Each score rule produces a `score_result`. New score rules must fix formula and direction before they are used for claim-bearing comparison.

## Output Shape

Canonical artifact roots:

```text
results_test/tarr/train_candidate_metadata/
results_test/tarr/reference_sets/
results_test/tarr/outputs/<dataset>/<baseline_protocol>/seed<seed>/<run_id>/<scheme>/references/<reference_config_id>/tta_response/
results_test/tarr/outputs/<dataset>/<baseline_protocol>/seed<seed>/<run_id>/<scheme>/references/<reference_config_id>/score_results/
```

Diagnostics live beside the reference-specific outputs under `diagnostics/`.
