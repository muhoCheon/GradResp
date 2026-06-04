# Shared Reference Candidate Cache + `correct_confidence_stratified` 구현 계획

> Archive note: This plan is historical. Use docs_my/TARR/ and scripts_my/tarr/ as the current source of truth unless this file is explicitly referenced for context.

## Summary
`correct_confidence_stratified`를 추가하면서, reference build를 config별 train full scan 방식에서 **dataset/checkpoint 단위 shared candidate metadata cache** 기반으로 바꾼다. 같은 dataset, train imglist, checkpoint, model/preprocessing identity가 같으면 여러 run과 여러 reference filter ablation에서 train prediction metadata를 재사용한다.

기본 동작은 shared cache 자동 사용이다. 저장 위치는 `--output-root` 기준 `results_test/tarr/reference_candidates/` 아래로 둔다.

구현할 때는 여러 subagent를 사용한다. 기존 작업과 관련 없는 subagent가 남아 있으면 종료하고, 아래 역할로 새 subagent를 생성한다.

## Subagent Allocation
- **Agent 1: Candidate Cache / Reference Selection**
  - `scripts_my/tarr/reference.py` 중심.
  - shared candidate cache identity, manifest, load/build helper 구현.
  - `correct_confidence_stratified` selection 구현.
  - 기존 `all`, `correct`, `high_confidence`, `correct_high_confidence`를 candidate metadata 기반 selection으로 정리.

- **Agent 2: Eval Integration**
  - `scripts_my/tarr/eval.py` 중심.
  - setup 단계에서 candidate cache load-or-build 호출.
  - reference config별 bank 생성이 candidate cache를 공유하도록 연결.
  - CLI 옵션 추가:
    - `--reference-candidate-cache-root`
    - `--rebuild-reference-candidate-cache`
  - run info / manifest에 candidate cache path와 identity 기록.

- **Agent 3: Validation / Matrix / Docs**
  - `scripts_my/tarr/cache.py`, `scripts_my/tarr/run_matrix.py`, TARR docs 담당.
  - candidate cache identity mismatch 검증 추가.
  - `run_matrix.py`에서 `REFERENCE_FILTERS` choices 사용.
  - `implementation.md`, `overview.md`, `ablations.md`, `experiments.md` 업데이트.
  - 실험 성능 해석은 추가하지 않음.

- **Parent Integrator**
  - subagent patch 통합 및 충돌 해결.
  - static/smoke test 실행.
  - 기존 score/cache semantics가 바뀌지 않았는지 확인.

## Key Changes
- Reference candidate cache
  - train split 전체를 한 번 forward해서 candidate metadata를 저장한다:
    - sample index / scan index
    - label
    - pred
    - confidence
    - entropy
    - margin
    - energy
    - correct
    - optional image path or imglist line identity
  - cache identity는 다음을 포함한다:
    - dataset
    - train imglist path + SHA256
    - checkpoint resolved path + SHA256
    - model arch
    - num classes
    - preprocessor identity string
    - candidate cache schema version
  - identity가 모두 일치하면 cache를 자동 재사용한다.
  - identity mismatch면 cache를 재생성한다.

- Reference selection
  - 기존 filters는 candidate cache에서 선택한다:
    - `all`: label/scan index만 사용
    - `correct`: `correct == true`
    - `high_confidence`: `confidence >= reference_min_confidence`
    - `correct_high_confidence`: `correct == true` and confidence threshold
    - `correct_confidence_stratified`: `correct == true` 후보를 class별 confidence quantile 위치에서 선택
  - `correct_confidence_stratified`에서는 `reference_min_confidence`를 threshold로 쓰지 않는다.
  - 모든 filter는 class별 `reference_per_class`개를 선택하고, 부족하면 `RuntimeError`를 낸다.
  - selected reference hash는 기존처럼 실제 선택된 reference tensor/label 기준으로 계산한다.

- Public CLI
  - 기본은 shared candidate cache 자동 사용.
  - 추가 옵션:
    - `--reference-candidate-cache-root`, default: `<output_root>/reference_candidates`
    - `--rebuild-reference-candidate-cache`, identity가 맞아도 강제 재생성
  - 기존 `--reference-config`와 단일 reference 옵션은 그대로 유지한다.

- Documentation
  - `implementation.md`
    - Reference Protocol에 shared candidate cache flow 추가.
    - Reference Filters 표에 `correct_confidence_stratified` 추가.
    - `correct_confidence_stratified`는 correct 후보의 confidence 분포 전반을 선택하며 `reference_min_confidence` threshold를 쓰지 않는다고 명시.
  - `overview.md`, `ablations.md`, `experiments.md`
    - 새 filter를 ablation candidate로만 추가.
    - 실험 성능 해석은 추가하지 않는다.

## Test Plan
- Static:
  ```bash
  conda run -n openood python -m py_compile scripts_my/tarr/*.py
  conda run -n openood python scripts_my/tarr/eval.py --help
  conda run -n openood python scripts_my/tarr/run_matrix.py --help
  ```

- Candidate cache behavior:
  - CIFAR-10 tiny run에서 candidate cache 생성 확인.
  - 같은 dataset/checkpoint/imglist/preprocessor로 재실행 시 cache 재사용 확인.
  - `--rebuild-reference-candidate-cache` 사용 시 재생성 확인.
  - checkpoint hash 또는 train imglist hash가 다르면 재사용하지 않는지 확인.

- Reference filter smoke:
  - 한 run에서 아래 configs를 함께 실행:
    - `all_rpc1`
    - `correct_rpc1`
    - `correcthigh_rpc1`
    - `strat_rpc1:filter=correct_confidence_stratified`
  - 확인:
    - train candidate cache build는 한 번만 수행
    - reference config별 selected hash가 기록됨
    - class별 reference count가 `per_class`를 만족
    - response cache와 `ood.csv` 생성
    - `cache.py validate --expect-reference-filter correct_confidence_stratified` 통과

- Regression:
  - 기존 `all`, `correct`, `high_confidence`, `correct_high_confidence`가 기존과 동일한 semantics로 동작하는지 확인.
  - score rules, response cache schema, FSOOD metric 계산은 변경하지 않는다.

## Assumptions
- Shared candidate cache는 TTA response cache가 아니라 train reference 후보 metadata cache다.
- Candidate cache는 claim-bearing score가 아니며, reference bank 선택을 빠르게 하기 위한 preprocessing artifact다.
- Preprocessor identity는 우선 `get_default_preprocessor(dataset)` 기반 문자열로 기록한다. 더 정밀한 transform hash는 필요 시 후속 개선으로 둔다.
- ImageNet에서도 사용할 수 있게 설계하되, v1 검증은 CIFAR-10/100 smoke를 우선한다.
