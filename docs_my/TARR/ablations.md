# TARR Ablations

TARR ablations are expressed as controlled changes to the four artifact stages:

```text
train_candidate_metadata -> reference_set -> tta_response -> score_result
```

Only full dataset OOD/FSOOD rows can support performance claims. Smoke/subset runs are code checks.

## Interpretation Principles

- Claim-bearing rows must have validated `train_candidate_metadata`, `reference_set`, `tta_response`, and `score_result` artifacts.
- FSOOD main metrics use clean ID + csID as the ID side.
- Clean-only and csID-only `score_result` rows are required diagnostics for FSOOD promotion.
- CIFAR-10 FSOOD must use the csID dataset matching `baseline_protocol`: `main_py -> cinic10`, `eval_api -> cifar10c`.
- Runtime/cost claims compare full runs on the same hardware, protocol, target count, and artifact identities.
- Score direction is always `ood_score` larger = OOD-like.
- Promotion requires strict artifact validation, clean/csID alignment, clean-only/csID-only/both reporting, and a predeclared score formula/direction.

## Primary Axes

### Stage 1: `train_candidate_metadata`

Stage 1 is usually fixed within a dataset/checkpoint protocol. It should change only when the dataset, train imglist, checkpoint, model architecture, preprocessing identity, or metadata schema changes.

Required checks:

- Candidate count matches the train split.
- Candidate labels and dataset indices are present.
- Confidence, entropy, margin, energy, correctness, and CE loss are finite.
- Manifest identity matches downstream `reference_set` manifests.

### Stage 2: `reference_set`

| Axis | Candidates | Purpose |
| --- | --- | --- |
| Source | ID `train` only | Match post-hoc baselines that use train data statistics/features. |
| Per-class size | CIFAR-10: `8`, `16`, `32`, `64`; CIFAR-100: `2`, `4`, `8`, `16` | Measure quality/runtime tradeoff. |
| Per-class size, ImageNet-scale | Predeclared resource-adjusted sizes | Keep response cost explicit at high class counts. |
| Filter | `all`, `correct`, `high_confidence`, `correct_high_confidence`, `correct_confidence_stratified` | Test whether reference quality or confidence diversity improves response signal. |
| Seed | selected sizes with seeds `0`, `1`, `2` | Estimate reference-selection variance. |

`correct_confidence_stratified` is claim-eligible only if declared before the run, evaluated on full data, and validated under the same protocol gates as other filters.

### Stage 3: `tta_response`

| Axis | Candidates | Purpose |
| --- | --- | --- |
| Objective | `predicted_label_ce`, `entropy`, `memo_marginal_entropy`, `view_consistency_js`, `view_consistency_kl`, `entropy_consistency` | Compare target-only adaptation signals. |
| Max `steps` | `1`, `5`, `10`, optional `20/30` | Maximum update budget and response strength/runtime tradeoff. |
| Saved `response_steps` | final only, or predeclared lists such as `1,5,10` | Store step-wise response without rerunning Stage 3. |
| Learning rate | `1e-2`, `3e-2`, optional `1e-3` | Stability and over-adaptation check. |
| Update scope | `classifier`, optional `all` | Compare efficient classifier-only update against broader adaptation. |
| Runtime mode | `auto`, `full_forward`, `classifier_feature_cache` | Verify metric parity and cost. |
| View config | view count, augmentation family, view seed | Required for soft view-consistency objectives. |
| Perturbation config | kind, epsilon, repeats, seed | Required for perturbation-response diagnostics. |

BN running-stat updates remain deferred unless a TTA objective explicitly requires them.

### Stage 4: `score_result`

Score rules are compared from the same `tta_response` whenever possible. For
step-wise `tta_response`, each claim row must predeclare and record the selected
`response_step`.

Claim score rules:

- `predicted_class_loss_increase`
- `predicted_class_loss_decrease`
- `target_weighted_loss_increase`
- `target_weighted_loss_decrease`
- `mean_loss_increase`
- `mean_loss_decrease`
- `positive_loss_increase_mean`
- `positive_loss_decrease_mean`

Diagnostic score families:

| Family | Candidates | Purpose | Promotion gate |
| --- | --- | --- | --- |
| Direction-aware | `rest_positive_loss_increase_mean`, `pred_vs_rest_loss_increase_gap`, `target_centered_loss_increase` | Test whether class-wise response direction explains protocol failures. | Improve OOD/FSOOD without worsening clean-vs-csID alignment. |
| Magnitude-only | `delta_l2_norm` | Test whether total reference-response magnitude matters more than sign. | Beat the corresponding claim score under the same `tta_response`. |
| Vector-aware | `clean_delta_z_l2`, `clean_delta_cosine_distance` | Test whether distance from the clean-ID response prototype distinguishes clean ID, csID, and OOD. | Stable clean-vs-csID alignment and competitive OOD performance. |
| Perturbation-response | perturbation score rules | Test whether controlled target perturbation response separates OOD while preserving csID as ID. | Pass the common promotion gate; clean-only improvement alone is insufficient. |

## Protocol And Alignment

Required diagnostics for FSOOD promotion:

- `alignment_summary.csv` exists.
- Expected and resolved csID dataset names match the selected protocol.
- Clean-only, csID-only, and both-ID-side `score_result` metrics are reported.
- `csid_alignment_status=aligned`.
- `alignment_error` and `csid_tail_at_clean_q95` are low among candidates with similar OOD AUROC.
- `perturbation_alignment_summary.csv` exists when perturbation score rules are considered.

The main open problem is that strong clean-ID vs OOD separation can coexist with poor csID alignment. Ablations should therefore prioritize objective and scoring changes that keep csID near clean ID.

## Required Outputs

For each promoted full run:

- `run_manifest.json` and `scheme_manifest.json`
- `train_candidate_metadata` manifest and id
- `reference_set` manifest and id
- `tta_response` manifest, shard count, and target split list
- `score_result` manifest, score files, and `ood.csv`
- `score_summary.csv`
- `target_summary.csv`
- `delta_summary.csv`
- `runtime_summary.csv`
- `reference_summary.csv`
- `alignment_summary.csv`
- `perturbation_summary.csv` when perturbation fields exist
- `perturbation_alignment_summary.csv` when perturbation score rules exist
- `score_direction.csv`
- `failure_cases.csv`

Optional plots after tabular results are stable:

- score density plots
- delta density plots
- runtime Pareto plots
- MSP/MLS/EBO correlation plots

## Next Ablation Order

1. Run the CIFAR-10 `eval_api` full objective screen for `memo_marginal_entropy`, `view_consistency_js`, `view_consistency_kl`, and `entropy_consistency` through the four-stage pipeline.
2. Promote only candidates that improve OOD AUROC/FPR95 without worsening clean-vs-csID alignment.
3. Run narrow promoted refinement over learning rate, step count, view count, and one `reference_set` adjustment.
4. Transfer the promoted CIFAR-10 recipe to CIFAR-100 `eval_api` before CIFAR-100-specific tuning.
5. Run robustness checks over reference seed and view seed/count, reporting mean/variance rather than best seed.
