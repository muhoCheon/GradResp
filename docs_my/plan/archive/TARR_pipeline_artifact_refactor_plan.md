# TARR Pipeline Artifact Refactor Plan: No Legacy Compatibility

> Archive note: This plan is historical. Use docs_my/TARR/ and scripts_my/tarr/ as the current source of truth unless this file is explicitly referenced for context.

## Summary
TARR artifact system을 네 단계로 단순화한다.

```text
train_candidate_metadata -> reference_set -> tta_response -> score_result
```

기존 `reference_candidates`, `reference_banks`, `response_cache`, direct `<score_rule>/ood.csv`, `active/rescore` 용어와 경로는 제거한다. Transition period는 두지 않는다. 잘못된 legacy CLI/path가 들어오면 명확한 error로 중단한다.

## Canonical Layout
새 구조만 지원한다.

```text
results_test/tarr/
  train_candidate_metadata/
    <dataset>/
      <candidate_id>/
        manifest.json
        candidates.npz

  reference_sets/
    <dataset>/
      <reference_config_id>/
        seed<seed>/
          <reference_set_id>/
            manifest.json
            reference_set.npz
            selected_samples.csv
            preview/

  outputs/
    <dataset>/
      <baseline_protocol>/
        seed<seed>/
          <run_id>/
            run_info.md
            run_manifest.json
            <scheme>/
              scheme_manifest.json
              references/
                <reference_config_id>/
                  tta_response/
                    <target_dataset_name>/
                      manifest.json
                      part_000000.npz

                  score_results/
                    <score_rule>/
                      scores/
                        <target_dataset_name>.npz
                      ood.csv
```

`score_result`는 실행 시점과 무관하게 동일한 artifact다.

```text
inline scoring  = response run 직후 orchestration이 Stage 4를 바로 실행
offline scoring = 저장된 tta_response에서 나중에 Stage 4를 실행
```

## Key Implementation Changes
- Remove legacy artifact names and path fallbacks:
  - remove `reference_candidates`
  - remove `reference_banks`
  - remove `response_cache`
  - remove direct `<reference>/<score_rule>/ood.csv`
  - remove `active/rescore` terminology from canonical code paths

- Stage 1: `train_candidate_metadata`
  - CLI:
    - `--train-candidate-metadata-root`
    - `--rebuild-train-candidate-metadata`
    - `--train-candidate-batch-size`
  - Output:
    - `train_candidate_metadata/<dataset>/<candidate_id>/manifest.json`
    - `train_candidate_metadata/<dataset>/<candidate_id>/candidates.npz`
  - Old candidate CLI flags should be rejected by argparse.

- Stage 2: `reference_set`
  - CLI:
    - `--reference-set-root`
    - `--rebuild-reference-set`
    - `--reference-set-batch-size`
  - Output:
    - `reference_sets/<dataset>/<reference_config_id>/seed<seed>/<reference_set_id>/reference_set.npz`
    - `manifest.json`
    - `selected_samples.csv`
    - optional `preview/`
  - Old bank/cache CLI flags should be rejected.

- Stage 3: `tta_response`
  - CLI:
    - `--save-tta-response`
    - `--tta-response-shard-size`
    - `--debug-output-mode`
  - Output path:
    - `references/<reference_config_id>/tta_response/<target_dataset_name>/`
  - ImageNet-scale full runs still require sharded output.
  - Old `--save-response-cache` and `--response-cache-shard-size` should be rejected.

- Stage 4: `score_result`
  - All scoring outputs go under:
    - `references/<reference_config_id>/score_results/<score_rule>/`
  - There is no canonical distinction between active score and rescore.
  - If scoring is run immediately after Stage 3, it still writes the same `score_result`.
  - If scoring is run later from saved `tta_response`, it also writes the same `score_result`.

- CLIs:
  - `reference.py build-train-metadata`
  - `reference.py build-reference-set`
  - `eval.py run-response`
  - `cache.py score`
  - `reports.py diagnostics`
  - Existing one-command `eval.py` orchestration may remain, but internally it executes Stage 1-4 in order.

- Error policy:
  - If user passes old CLI flags, argparse should fail.
  - If expected `tta_response/` is missing but old `response_cache/` exists, fail with a clear message.
  - If expected `score_results/` is missing but old score-rule directories exist, fail with a clear message.
  - Do not auto-migrate legacy results.

## Documentation Changes
- `implementation.md`
  - Rewrite around four-stage pipeline.
  - Remove `active score`, `rescore`, `response cache`, `reference bank cache`, `reference candidate cache` as canonical terms.
  - Explain that inline/offline scoring are execution modes of the same Stage 4.

- `overview.md`
  - Use conceptual names only:
    - `train_candidate_metadata`
    - `reference_set`
    - `tta_response`
    - `score_result`

- `experiments.md`
  - Rename current file to `legacy_experiments.md`.
  - Keep only important previous best runs and conclusions.
  - Create new `experiments.md` for new-structure experiments only.

- `ablations.md`
  - Express all ablations as changes to:
    - TTA config
    - reference_set config
    - score_rule producing score_result

- `notes.md`
  - Keep only open issues and pointers to canonical docs.

## Subagent Allocation
- Agent 1: Stage 1/2 reference artifacts
  - Own `reference.py`.
  - Implement new naming, paths, CLIs, selected sample CSV, and remove legacy candidate/bank APIs.

- Agent 2: Stage 3 orchestration
  - Own `eval.py`.
  - Replace response-cache naming with `tta_response`.
  - Update one-command orchestration to call Stage 1-4 with canonical artifacts only.

- Agent 3: Stage 4 scoring/reports
  - Own `cache.py`, `reports.py`, `protocol.py`, `run_matrix.py`.
  - Replace active/rescore terminology with `score_result`.
  - Remove legacy path fallback.
  - Ensure diagnostics read only canonical `tta_response` and `score_results`.

- Agent 4: docs
  - Own `docs_my/TARR/*.md`.
  - Create `legacy_experiments.md`.
  - Rewrite canonical docs around the new pipeline.

- Parent integrator:
  - Close unrelated subagents before implementation.
  - Resolve naming consistency.
  - Run static/smoke tests.
  - Do not delete old result directories unless separately requested.

## Test Plan
- Static/help:
  ```bash
  conda run -n openood python -m py_compile scripts_my/tarr/*.py
  conda run -n openood python scripts_my/tarr/reference.py --help
  conda run -n openood python scripts_my/tarr/eval.py --help
  conda run -n openood python scripts_my/tarr/cache.py --help
  conda run -n openood python scripts_my/tarr/reports.py --help
  ```

- Stage smoke:
  ```bash
  reference.py build-train-metadata
  reference.py build-reference-set
  eval.py run-response
  cache.py score
  reports.py diagnostics
  ```

- Layout acceptance:
  - New run writes only:
    - `train_candidate_metadata/`
    - `reference_sets/`
    - `tta_response/`
    - `score_results/`
  - New run does not write:
    - `reference_candidates/`
    - `reference_banks/`
    - `response_cache/`
    - direct `<score_rule>/ood.csv`

- Negative tests:
  - Old CLI flags fail.
  - Old `response_cache/` path fails with clear error.
  - Old direct score-rule output path fails with clear error.
  - Old candidate/bank root paths are not silently reused.

- Regression:
  - CIFAR-10 tiny full pipeline succeeds.
  - CIFAR-10 sharded `tta_response` succeeds.
  - ImageNet smoke with sharded `tta_response` succeeds.
  - Score values from new Stage 4 match previous formula behavior on a fresh tiny run.

## Assumptions
- No legacy compatibility is required.
- Existing result directories are not migrated in this task.
- Existing previous results are documented only in `legacy_experiments.md`.
- `active score` and `rescore` are removed as canonical concepts; both become `score_result`.
- Current response cache schema fields remain the same unless a separate schema refactor is requested.
