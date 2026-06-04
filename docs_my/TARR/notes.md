# TARR Notes

Canonical docs:

- Method overview: `overview.md`
- Implementation contract: `implementation.md`
- Current experiment plan: `experiments.md`
- Historical experiment results: `legacy_experiments.md`
- Ablation policy: `ablations.md`

## Current Notes

- Canonical artifact chain: `train_candidate_metadata -> reference_set -> tta_response -> score_result`.
- Target-only TTA uses target data and target-derived context only.
- `reference_set` data is used to measure pre/post adaptation response, not to optimize the target adaptation loss.
- Internal `ood_score` is OOD-like when larger.
- Stored OpenOOD `conf` is `-ood_score`, so larger `conf` is ID-like.
- Inline scoring and offline scoring are both Stage 4 and must produce the same `score_result` for the same `tta_response` and score config.
- Step-wise runs store `response_steps` in `tta_response`; Stage 4 claim rows must record the selected `response_step`.
- Claim-bearing FSOOD rows require clean-only, csID-only, and clean+csID `score_result` reporting.
- Soft view-consistency hypothesis: clean ID and csID should remain aligned under controlled target-view response while OOD separates.
- Runtime claims must report time by artifact stage.

## Resolved Constraints

- ID train is the only source for `train_candidate_metadata` and `reference_set`.
- `score_result` direction must be fixed before a claim-bearing comparison.
- Stage 4 scoring changes do not require Stage 3 TTA to run again when the `tta_response` identity matches.
- Diagnostic score rules are not claim-bearing until their formula, direction, protocol selection rule, and promotion gate are predeclared.
- Perturbation diagnostics are interpreted through clean/csID alignment before OOD AUROC.

## Open Questions

- Which soft view-consistency objective keeps csID close to clean ID while preserving OOD separation?
- Does `correct_confidence_stratified` improve OOD performance or clean-vs-csID alignment compared with `all`, `high_confidence`, and `correct_high_confidence`?
- Which `reference_set` size is the best quality/runtime tradeoff after the objective screen?
- Do perturbation score rules improve csID-only and clean+csID OOD separation, or only sharpen clean-only separation?
- What minimum robustness set over reference seed and view seed is enough before a claim row?

## Next Notes To Add

- CIFAR-10 `eval_api` four-stage objective screen results.
- Promoted refinement results.
- CIFAR-100 transfer results.
- Robustness results for promoted settings.
