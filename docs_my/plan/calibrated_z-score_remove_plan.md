# Calibrated Z-Score Diagnostic 제거 계획

## Summary
목표는 TARR의 current code/documentation에서 `calibrated z-score` diagnostic score family를 제거하고, active/vector/perturbation score 경로만 남기는 것이다. Calibration/hybrid score는 claim protocol이 명확해지는 최종 단계에서 새로 설계한다.

기존 생성 결과 파일(`results_test/tarr/outputs/**/rescore/calibrated/**`)은 삭제하지 않는다. 대신 코드와 report collector에서 canonical 결과로 읽히지 않게 한다.

## Key Changes

- **Implementation 전 subagent 정리**
  - 현재 작업과 관련 없는 subagent가 남아 있으면 먼저 종료한다.
  - 새 implementation subagent는 아래 3개로 나눈다.
    - Agent 1: `scoring.py`, `cache.py` calibrated z-score code removal.
    - Agent 2: `reports.py` calibrated rescore collection/report removal.
    - Agent 3: TARR docs cleanup.
  - Parent integrator가 patch 통합, 충돌 해결, static/help tests를 담당한다.

- **Score/cache code cleanup**
  - `scripts_my/tarr/scoring.py`에서 calibrated-only API 제거:
    - `CALIBRATION_METHOD`
    - `CALIBRATED_SCORE_SUFFIXES`
    - score z-score fit/apply helpers
    - `z_plus_entropy`, `z_plus_non_msp`, `z_plus_entropy_non_msp` 관련 helper
  - `clean_delta_z_l2`, `clean_delta_cosine_distance`는 유지한다. 이것은 calibrated target score가 아니라 vector-aware diagnostic이다.
  - `CALIBRATION_EPS`는 vector code에서 쓰이므로 유지하거나 `NUMERIC_EPS` 같은 neutral name으로 rename한다. v1에서는 churn을 줄이기 위해 유지한다.

- **`cache.py rescore` cleanup**
  - `--calibrated-diagnostics` CLI 제거.
  - `rescore/calibrated/` output branch 제거.
  - `calibration.json` 생성 제거.
  - diagnostic mode validation은 `--vector-score-rule`과 `--perturbation-score-rule`만 상호 배타로 검사한다.
  - active rescore, vector rescore, perturbation rescore behavior는 바꾸지 않는다.

- **`reports.py` cleanup**
  - `calibrated`를 `RESCORE_KIND_CHOICES`에서 제거.
  - `collect-rescore --score-family calibrated` 선택지를 제거.
  - `read_rescore_manifest()`에서 `calibration.json` discovery 제거.
  - default `collect-rescore`가 기존 `rescore/calibrated/` artifact를 만나도 row로 수집하지 않도록 명시적으로 skip한다.
  - `active`, `vector`, `perturbation` report 경로는 유지한다.

- **Documentation cleanup**
  - `implementation.md`, `experiments.md`, `ablations.md`, `notes.md`에서 operational calibrated diagnostic 설명 제거:
    - `--calibrated-diagnostics`
    - `rescore/calibrated/`
    - `calibration.json`
    - z-score calibrated result tables/claims
  - 남길 표현은 한 줄 수준으로 제한한다:
    - “Hybrid/calibrated score는 future protocol에서 별도로 설계한다.”
  - vector-aware score, perturbation diagnostic, `standardized_mean_gap` 문서는 유지한다.

## Test Plan

- Static/help:
  ```bash
  conda run -n openood python -m py_compile scripts_my/tarr/*.py
  conda run -n openood python scripts_my/tarr/cache.py --help
  conda run -n openood python scripts_my/tarr/cache.py rescore --help
  conda run -n openood python scripts_my/tarr/reports.py --help
  conda run -n openood python scripts_my/tarr/reports.py rescore-diagnostics --help
  conda run -n openood python scripts_my/tarr/reports.py collect-rescore --help
  ```

- Removal checks:
  ```bash
  rg "calibrated-diagnostics|rescore/calibrated|z_plus_|CALIBRATED_SCORE|CALIBRATION_METHOD|calibration.json" scripts_my/tarr docs_my/TARR
  ```
  Expected: no current operational references. Future hybrid wording may mention “calibrated” only as non-implemented future work.

- Negative CLI checks:
  ```bash
  conda run -n openood python scripts_my/tarr/cache.py rescore --calibrated-diagnostics
  conda run -n openood python scripts_my/tarr/reports.py rescore-diagnostics --rescore-kind calibrated
  conda run -n openood python scripts_my/tarr/reports.py collect-rescore --score-family calibrated
  ```
  Expected: argparse failure.

- Regression checks:
  - `cache.py rescore --vector-score-rule all` still works on one existing cache.
  - `cache.py rescore --perturbation-score-rule all` still works on one existing schema v5 cache.
  - `reports.py collect-rescore` over `results_test/tarr/outputs` emits no `score_family=calibrated` rows even if old calibrated artifacts remain.

## Assumptions

- Existing calibrated result artifacts are not deleted in this task.
- External notebooks or ad hoc scripts importing calibrated helpers are out of scope.
- This task is cleanup/optimization only; it does not change active score definitions, vector score definitions, perturbation score definitions, or response cache schema.
- Hybrid/calibrated scoring can be reintroduced later under a separate predeclared protocol.
