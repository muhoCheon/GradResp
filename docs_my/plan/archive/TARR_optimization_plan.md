# TARR Hot-Path Optimization + FSOOD-Only Search 운영 계획

> Archive note: This plan is historical. Use docs_my/TARR/ and scripts_my/tarr/ as the current source of truth unless this file is explicitly referenced for context.

## Summary
목표는 TARR score 정의와 claim protocol을 바꾸지 않고 full-run 실험 속도를 줄이는 것이다. 이번 범위에는 이전에 합의한 최적화 항목 `1~6, 9`를 포함하고, `run_matrix.py` 2-GPU scheduler와 `--scheme both` 코드 변경은 제외한다.

Broad search 운영은 `--scheme fsood`만 사용한다. FSOOD cache에는 clean ID, csID, near/far semantic OOD가 모두 있으므로, `cache.py rescore --fsood-id-side clean|csid|both`로 clean/csID/semantic OOD separation을 분석한다.

Implementation 시 기존 작업과 관련 없는 subagent는 먼저 종료하고, 새 subagent를 아래 역할로 나눠 진행한다.

## Key Changes

### 1. `eval.py` hot-path 최적화
- Soft-view objective의 TTA step에서 불필요한 clean forward를 제거한다.
  - 대상: `memo_marginal_entropy`, `view_consistency_kl`, `view_consistency_js`, `entropy_consistency`
  - 제거 대상은 per-step `net(data)` 또는 `classifier(target_feature)` clean logits 계산뿐이다.
  - 초기 clean forward는 pseudo-label, target diagnostics, baseline probs에 필요하므로 유지한다.
  - `--no-freeze-bn-stats`에서는 BN/dropout side effect 가능성이 있으므로 기존 경로를 유지한다.
- `_restore_base()`는 per-sample `load_state_dict()` 대신 저장된 parameter/buffer tensor를 `copy_`로 복구한다.
  - classifier mode와 full-model mode 모두 parameter와 buffer를 복구한다.
  - optimizer는 현재 SGD no-momentum 기준으로 기존 semantics를 유지한다.
- CPU sync를 줄인다.
  - `score_one()` 내부의 `.item()`, `.cpu()`, `.numpy()`, `.tolist()` 호출을 가능한 한 늦춘다.
  - per-sample/per-reference 계산은 tensor로 유지하고, cache/debug row 조립 시점에만 CPU로 이동한다.
- `RUNTIME_IMPL_VERSION`은 새 hot-path 최적화 버전으로 bump한다.
  - response cache schema는 변경하지 않는다.

### 2. Reference response 최적화
- setup 시점에 reference bank별 static 값을 precompute한다.
  - `base_reference_loss` stats
  - `base_reference_diag`의 CPU/list export 형태
  - class count/index layout
- Reference candidate cache v2를 사용한다.
  - `candidate_cache_schema_version`은 candidate cache identity에 포함한다.
  - v2 candidate metadata는 CE loss를 포함하고 full logits는 저장하지 않는다.
  - response cache schema는 변경하지 않는다.
- classifier feature cache runtime에서는 reference bank cache를 별도로 둘 수 있다.
  - 위치: `<output_root>/reference_banks/<dataset>/<bank_cache_id_prefix>/bank.npz`
  - 저장: selected labels, selected candidate indices/hash, selected data hash, base diagnostics, feature tensor
  - 저장하지 않음: image tensor
- class-balanced reference bank는 class-wise mean을 Python class loop 대신 reshape/reduction 기반으로 계산한다.
- `classifier_feature_cache` mode에서는 reference feature에 대해 direct classifier linear path를 사용한다.
  - score/cache output의 `delta = adapted_reference_loss - base_reference_loss` 정의는 그대로 유지한다.
- full debug/cache compatibility를 우선 유지한다.
  - score-only fast path는 이번 v1에서는 별도 모드로 만들지 않는다.

### 3. Perturbation diagnostics 최적화
- Gaussian perturbation diagnostics는 `[repeats, batch, classes]` tensor를 유지한 채 vectorized 계산한다.
  - `logit_l2`
  - `prob_l1`
  - `conf_delta`
  - `entropy_delta`
- `sign_ce` diagnostics는 동일 perturbed input/feature를 repeat만큼 중복 forward하지 않는다.
  - `freeze_bn_stats=true` 경로에서만 1회 forward 결과를 repeat 평균으로 재사용한다.
  - `--no-freeze-bn-stats`에서는 기존 반복 forward를 유지한다.
- Perturbation RNG 호출 순서는 가능한 한 유지한다.
  - diagnostics 제거, 순서 변경, RNG stream 분리는 이번 최적화 범위에 포함하지 않는다.

### 4. DataLoader/transfer 최적화
- Batch size를 용도별로 분리한다.
  - `--batch-size`: ID/csID/OOD target loader
  - `--reference-candidate-batch-size`: train full-forward candidate metadata scan, `0`이면 `--batch-size`
  - `--reference-batch-size`: selected reference feature caching and adapted response
- `DataLoader` 생성 시 기본적으로 다음을 적용한다.
  - `pin_memory=torch.cuda.is_available()`
  - `persistent_workers=True` when `num_workers > 0`
  - `prefetch_factor=2` when `num_workers > 0`
- GPU transfer는 `data.cuda(non_blocking=True)`, `label.cuda(non_blocking=True)`로 바꾼다.
- `num_workers=0`에서도 정상 동작하도록 `persistent_workers`와 `prefetch_factor`는 조건부로만 설정한다.

### 5. `cache.py rescore` I/O 최적화
- `rescore`에서 ID/csID/near/far cache를 score rule loop 밖에서 한 번만 load한다.
- 이후 `score_rule=all`, `vector-score-rule=all`, `perturbation-score-rule=all`을 메모리 resident cache에서 계산한다.
- output layout은 유지한다.
  - active: `<reference_dir>/rescore/id_side_<side>/<score_rule>/...`
  - vector: `<reference_dir>/rescore/vector/id_side_<side>/<score_rule>/...`
  - perturbation: `<reference_dir>/rescore/perturbation/id_side_<side>/<score_rule>/...`
- 기존 artifact는 삭제하지 않는다.
- `validate`는 response cache schema를 바꾸지 않고 candidate cache identity만 추가 검증한다.
  - candidate manifest가 제공되면 `candidate_cache_schema_version`/legacy `schema_version`을 strict 비교한다.
  - `preprocessor_identity`가 양쪽에 있으면 strict 비교한다.
  - run/scheme manifest의 relative candidate manifest/cache path는 선언한 manifest directory 기준으로 resolve한다.

### 6. 실험 운영 규칙
- Broad search는 `eval.py --scheme fsood`만 사용한다.
  - `--scheme both`는 broad search에서 사용하지 않는다.
  - `--scheme both` 관련 코드는 이번 작업에서 수정하지 않는다.
- FSOOD full cache에서 다음 rescore를 수행한다.
  - claim main: `--fsood-id-side both`
  - OOD-style diagnostic: `--fsood-id-side clean`
  - csID diagnostic: `--fsood-id-side csid`
- 최종 claim에서 FSOOD metric은 `both`만 사용한다.
  - `clean`/`csid` rows는 diagnostic-only로 표시한다.
- Soft-view broad search는 `run_matrix.py`가 아니라 direct `eval.py` command 또는 subagent별 queue script로 실행한다.
  - `run_matrix.py` scheduler/CLI는 이번 작업에서 수정하지 않는다.
- GPU 실행은 GPU당 full TARR process 1개를 기본으로 한다.
  - CPU rescore/report는 GPU job과 동시에 많이 돌리지 않고 낮은 concurrency로 수행한다.
- `score_rule`과 reference configs는 가능한 한 한 run에 묶는다.
  - GPU rerun을 줄이고, score 변화는 response cache 기반 offline rescore로 처리한다.

## Subagent Implementation Allocation
- Agent 1: `eval.py` hot path
  - dead clean forward 제거
  - tensor-copy restore
  - CPU sync 축소
  - DataLoader/non-blocking transfer
- Agent 2: reference/perturbation optimization
  - reference static precompute
  - class-wise reduction vectorization
  - perturbation diagnostics vectorization
  - `sign_ce` repeat collapse
- Agent 3: `cache.py` rescore optimization + docs
  - one-load multi-rule rescore
  - candidate cache schema/preprocessor validation
  - relative candidate manifest path resolution
  - experiments/ablations 문서에 FSOOD-only broad search 운영 규칙 반영
- Parent integrator
  - subagent patch review
  - runtime version bump 확인
  - static/smoke/regression tests
  - old/new tiny output comparison

## Test Plan
- Static/help
  ```bash
  conda run -n openood python -m py_compile scripts_my/tarr/*.py
  conda run -n openood python scripts_my/tarr/eval.py --help
  conda run -n openood python scripts_my/tarr/cache.py --help
  conda run -n openood python scripts_my/tarr/reports.py --help
  ```
- Restore invariant
  - `score_one()` 후 parameter와 buffer가 base state와 일치하는지 확인한다.
  - classifier mode와 full-model mode를 모두 확인한다.
- Objective smoke
  - `predicted_label_ce`
  - `entropy`
  - `memo_marginal_entropy`
  - `view_consistency_js`
  - 각 objective에서 max-samples tiny run이 성공해야 한다.
- Perturbation smoke
  - `none`
  - `pixel gaussian`
  - `feature gaussian`
  - `pixel sign_ce`
  - `feature sign_ce`
  - `freeze_bn_stats=true`와 `num_workers=0/>0`를 포함한다.
- Cache/report validation
  - schema v5 response cache validate 통과
  - candidate cache v2 manifest가 있으면 candidate schema/preprocessor validation 통과
  - relative candidate manifest path가 run/scheme manifest 기준으로 resolve되는지 확인
  - `cache.py rescore --fsood-id-side clean --score-rule all`
  - `cache.py rescore --fsood-id-side csid --score-rule all`
  - `cache.py rescore --fsood-id-side both --score-rule all`
  - vector/perturbation rescore도 기존 output layout으로 생성되는지 확인
- Runtime benchmark CSV/schema
  - `results_test/tarr/summary/runtime_benchmark.csv`를 사용한다.
  - Required columns:
    `run_id,dataset,baseline_protocol,scheme,reference_config_id,runtime_mode,candidate_cache_schema_version,candidate_cache_reused,candidate_cache_sec,bank_cache_reused,bank_cache_sec,setup_total_sec,inference_total_sec,rescore_total_sec,processed_targets,runtime_per_target_sec,batch_size,reference_candidate_batch_size,reference_batch_size,num_workers,cuda_device`
  - before/after speedup 비교는 같은 protocol, reference config, hardware, target count, cache warm/cold condition에서만 수행한다.
- Regression
  - tiny fixed-seed run에서 old/new active score metric이 tolerance 내 일치해야 한다.
  - soft-view objective는 `freeze_bn_stats=true` 기준으로 비교한다.
  - `--scheme fsood` broad search 결과에서 clean/csID/both diagnostic rows가 분리되어 수집되는지 확인한다.

## Assumptions
- 이번 작업은 성능 최적화이며 TARR score 수식, FSOOD main protocol, response cache schema는 바꾸지 않는다.
- `run_matrix.py` parallel scheduler는 이번 범위에서 제외한다.
- `--scheme both` 동작은 바꾸지 않고, broad search 운영에서만 `--scheme fsood`를 사용한다.
- `--no-freeze-bn-stats`는 최적화 적용을 보수적으로 제한한다.
- 최종 claim은 full run, schema v5, cache validation 통과, FSOOD `both` metric 기준만 사용한다.
