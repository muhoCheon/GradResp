# Archived Planning Docs

These plans are no longer the primary source of truth. They are kept to explain
why earlier TARR implementation and experiment decisions were made. Current
behavior should be checked against `docs_my/TARR/` and the code under
`scripts_my/tarr/`.

## Index

| File | Scope | Notes |
| --- | --- | --- |
| [TARR_package_refactor_plan.md](TARR_package_refactor_plan.md) | Package structure and multi-reference cache | Historical plan for splitting TARR into package modules and sharing TTA work across reference configs. |
| [TARR_pipeline_artifact_refactor_plan.md](TARR_pipeline_artifact_refactor_plan.md) | Canonical artifact layout | Historical no-legacy-compatibility plan for `train_candidate_metadata`, `reference_set`, `tta_response`, and `score_result`. |
| [TARR_perturbation_response_plan.md](TARR_perturbation_response_plan.md) | Perturbation-response experiments | Historical experiment plan for perturbation and soft view-consistency objective screening. |
| [TARR_soft-view-consistency_plan.md](TARR_soft-view-consistency_plan.md) | Soft view-consistency objective | Historical implementation and experiment plan for soft-label view consistency. |
| [TARR_optimization_plan.md](TARR_optimization_plan.md) | Hot-path optimization and FSOOD operation | Historical speed and operating plan for full-run TARR experiments. |
| [TARR_ref_cache_opt_plan.md](TARR_ref_cache_opt_plan.md) | Reference cache optimization | Historical plan for reference cache and batch hot-path speed evaluation. |
| [TARR_sharded_npz_cache_plan.md](TARR_sharded_npz_cache_plan.md) | Large-dataset response cache | Historical plan for sharded NPZ response cache support and rerun commands. |
| [shard_ref_cache_plan.md](shard_ref_cache_plan.md) | Shared candidate cache | Historical plan for shared reference candidate cache and `correct_confidence_stratified`. |
| [calibrated_z-score_remove_plan.md](calibrated_z-score_remove_plan.md) | Diagnostic cleanup | Historical plan for removing calibrated z-score diagnostics. |
| [score_density_plot.md](score_density_plot.md) | Plot utility | Historical plan for score density plotting outputs. |
| [random_model_sanity_check_v1_plan.md](random_model_sanity_check_v1_plan.md) | Random-model sanity check | Historical implementation plan for a first sanity-check workflow. |

## Archive Rules

- Do not treat archived plans as current requirements.
- If an archived plan conflicts with `docs_my/TARR/`, the canonical TARR docs
  win.
- Preserve old operational commands only as historical references. Re-validate
  commands against current CLI docs before running them.
