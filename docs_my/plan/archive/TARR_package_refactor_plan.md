# TARR Package Refactor + Multi-Reference Cache Plan

> Archive note: This plan is historical. Use docs_my/TARR/ and scripts_my/tarr/ as the current source of truth unless this file is explicitly referenced for context.

## Summary
TARR 코드를 `scripts_my/tarr/` package로 리팩터링하고, TTA/reference/scoring/protocol 축을 명확히 분리한다. TARR 실행/분석 entrypoint는 새 package 아래로 통합한다. 추가로 같은 TTA pass에서 여러 reference config를 평가할 수 있는 **multi-reference evaluation**을 지원해 reference ablation 비용을 줄인다.

```text
scripts_my/tarr/
  __init__.py
  eval.py
  reference.py
  adaptation.py
  scoring.py
  cache.py
  protocol.py
  reports.py
  run_matrix.py
```

핵심 원칙:

```text
TTA option 변경 -> rerun 필요
Reference option 변경 -> 같은 TTA pass 안에서 multi-reference response 계산 가능
Scoring option 변경 -> response cache에서 offline 재계산 가능
```

## Key Changes
- `eval.py`
  - main TARR 실행 CLI.
  - config를 `tta_config`, `reference_configs`, `scoring_config`, `protocol_config`로 정규화한다.
  - 하나의 TTA config와 protocol/data config에 대해 여러 reference config를 동시에 받을 수 있게 한다.
  - target sample별 TTA는 한 번만 수행하고, adaptation 후 모든 reference bank의 response를 계산한다.
  - `freeze_bn_stats`를 명시 option/config로 추가하고 기본값은 `true`.

- `reference.py`
  - 여러 `ReferenceConfig`를 받아 `reference_config_id -> ReferenceBank` dict를 만든다.
  - 각 reference bank는 selected sample hash, per-class counts, base reference loss, optional feature cache를 가진다.
  - reference response 계산은 reference bank별로 수행한다.
  - reference option 변경은 cache identity를 바꾸지만, 같은 TTA pass 안에서는 재사용 가능한 target adaptation 결과를 공유한다.

- `adaptation.py`
  - target-only TTA 담당.
  - reference data를 절대 adaptation loss에 사용하지 않는다.
  - target sample 하나에 대해 adapted model state를 만든 뒤, caller가 여러 reference response를 측정할 수 있게 한다.
  - `objective`, `steps`, `lr`, `update_scope`, runtime mode, optimizer policy, `freeze_bn_stats`는 TTA config에 속한다.

- `scoring.py`
  - response delta에서 OOD score를 계산한다.
  - score rule만 바뀌는 경우 TTA/reference response를 다시 계산하지 않는다.
  - canonical score rule, schema version, score direction, `conf = -ood_score` 변환을 중앙화한다.

- `cache.py`
  - run-local response cache 저장/로드/offline evaluation 담당.
  - reference config별 cache를 저장한다:
    ```text
    <run_dir>/<scheme>/references/<reference_config_id>/response_cache/*.npz
    <run_dir>/<scheme>/references/<reference_config_id>/<score_rule>/ood.csv
    ```
  - offline scoring과 FSOOD `both|clean|csid` diagnostic은 reference config별 cache에서 수행한다.
  - `validate_response_cache()`는 TTA, reference, protocol, dataset, schema mismatch를 strict validation한다.

- `protocol.py`
  - dataset/protocol/scheme/csID/near/far 정의와 manifest 생성/읽기 담당.
  - CIFAR-10:
    - `main_py -> cinic10`
    - `eval_api -> cifar10c`
  - CIFAR-100:
    - both protocols -> `cifar100c`
  - manifest에는 nested config blocks를 저장한다:
    ```text
    tta_config
    reference_configs
    scoring_config
    protocol_config
    cache_identity
    ```

- `reports.py`
  - diagnostics, protocol metric collection, Group1 baseline comparison 담당.
  - summary CSV는 `reference_config_id`, `tta_config_id`, `scoring_config_id`, `protocol_config_id`, `cache_run_id`, `score_result_id`를 포함한다.
  - `score_rule=all`은 같은 reference response cache에서 score rule별 row로 확장한다.

- `run_matrix.py`
  - TTA x protocol job을 만들고, 여러 reference config를 같은 job 안에 묶을 수 있게 한다.
  - matrix 표현은 축을 분리한다:
    ```text
    tta_configs
    reference_configs
    scoring_configs
    protocol_configs
    ```
  - run id는 axis 기반으로 생성한다:
    ```text
    <dataset>_<protocol>_<scheme>__tta-<tta_id>__refs-<ref_group_id>__score-<score_id>
    ```

## Cache Validation
- Strict error fields:
  - cache schema, score direction, delta definition
  - dataset, scheme, baseline protocol, csID identity
  - checkpoint identity, checkpoint SHA256, model arch, num classes, classifier layer
  - TTA config: objective, steps, lr, update scope, runtime mode, optimizer policy, `freeze_bn_stats`
  - reference config: source, per-class, filter, min-confidence, seed, selected reference hash
  - sample limits/full-run flag, processed count, array shapes
  - dataset manifest/imglist path and SHA256 identity

- Warning fields:
  - command string, output path, run id
  - batch size, num workers, CUDA device
  - runtime seconds
  - checkpoint path difference when hash matches
  - optional diagnostics missing

- Reuse rules:
  - score-only reuse is safe when response cache validation passes.
  - reference reuse is not pure offline unless the reference response cache already exists.
  - multi-reference run reduces TTA reruns by measuring several reference responses after the same target adaptation.
  - TTA option changes always require rerun.

## Documentation Updates
- `implementation.md`
  - 새 package 구조와 multi-reference execution flow를 명확히 설명한다:
    1. `eval.py` prepares model/protocol/config
    2. `reference.py` builds multiple reference banks
    3. `reference.py` computes base response for each bank
    4. `adaptation.py` runs target-only TTA once per target sample
    5. `reference.py` computes adapted response for every reference bank
    6. `scoring.py` computes OOD scores from each bank’s delta
    7. `cache.py` saves/validates/offline-rescores reference-specific response caches
    8. `reports.py` summarizes diagnostics and baseline gaps
    9. `run_matrix.py` orchestrates TTA/protocol/reference/scoring matrices
  - “TTA option”, “Reference option”, “Scoring option”, “Protocol option”을 별도 섹션으로 분리한다.
  - 어떤 option 변경이 rerun을 요구하는지 표로 정리한다.

- `overview.md`, `implementation.md`, `ablations.md`, `experiments.md`, `notes.md`
  - 모든 script path를 `scripts_my/tarr/*.py`로 교체한다.
  - reference ablation은 multi-reference evaluation으로 수행한다고 명시한다.
  - score ablation은 offline cache evaluation으로 수행한다고 명시한다.
  - old entrypoint나 compatibility wrapper 언급은 남기지 않는다.

## Subagent Implementation Allocation
- Agent 1: core execution
  - `eval.py`, `reference.py`, `adaptation.py`, `scoring.py`
  - target TTA 1회 후 multiple reference response 계산 구조 구현.

- Agent 2: cache/protocol
  - `cache.py`, `protocol.py`
  - reference-specific cache layout, manifest schema, cache validator, csID protocol resolution 구현.

- Agent 3: reports/run matrix
  - `reports.py`, `run_matrix.py`
  - reference config-aware diagnostics, summary, Group1 comparison, matrix execution 구현.

- Parent integrator
  - TARR 실행/분석 entrypoint를 `scripts_my/tarr/`로 통합.
  - import/path/doc 통합.
  - static/smoke/regression checks 수행.

## Test Plan
- Static
  - `python -m py_compile scripts_my/tarr/*.py`
  - `python scripts_my/tarr/eval.py --help`
  - `python scripts_my/tarr/cache.py --help`
  - `python scripts_my/tarr/reports.py --help`
  - `python scripts_my/tarr/run_matrix.py --help`

- Path cleanup
  - TARR 관련 문서와 code path가 `scripts_my/tarr/*.py` 기준으로 정리되어야 한다.

- Smoke
  - CIFAR-10 tiny OOD/FSOOD run with two reference configs in one run.
  - 확인:
    - target TTA는 sample당 한 번 수행
    - reference config별 response cache 생성
    - reference config별 `ood.csv` 생성
    - `run_manifest.json`, `scheme_manifest.json`에 reference config list 기록
  - `cache.py`로 reference config별 offline score 재계산 확인.
  - `reports.py` summary가 reference config별 row를 생성하는지 확인.
  - `run_matrix.py` dry-run이 TTA/protocol job 하나에 reference configs 여러 개를 묶는지 확인.

- Validation
  - score rule만 바꾼 offline rescore는 통과.
  - reference config mismatch는 해당 reference cache에서 error.
  - objective/steps/lr/update_scope/freeze_bn_stats mismatch는 error.
  - CIFAR-10 protocol csID mismatch는 error.
  - batch size/num worker/runtime mismatch는 warning.
  - FSOOD main metric은 항상 `both`.

## Assumptions
- 기존 TARR result compatibility는 고려하지 않는다.
- TARR 실행/분석 entrypoint는 `scripts_my/tarr/` package만 사용한다.
- cache 저장은 중앙 cache store로 옮기지 않고 run-local 구조를 유지한다.
- multi-reference evaluation은 같은 TTA config/protocol/data config 안에서만 reference response 계산을 공유한다.
- adapted model state를 저장하는 adapted-state cache는 구현하지 않는다.
- `freeze_bn_stats=true`가 기본 동작이며 manifest와 validation에 명시한다.
