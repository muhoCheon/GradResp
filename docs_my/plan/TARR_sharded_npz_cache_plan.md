# TARR Large-Dataset Sharded NPZ Response Cache Plan

## Summary
ImageNet-1K full run에서 split 전체 response cache를 메모리에 쌓다가 kill되는 문제를 해결한다. 기존 CIFAR/small-cache single `.npz` 경로는 유지하고, ImageNet-scale run에서는 response cache를 shard 단위로 streaming 저장한다. Score output은 기존 single `.npz`로 유지한다.

## Key Changes

### Eval / 저장 경로
- `eval.py`에 `--response-cache-shard-size` 추가:
  - `0`: 기존 single `.npz` 방식.
  - `>0`: sharded response cache 사용.
  - ImageNet/ImageNet-200 full run에서 `--save-response-cache` 또는 `--score-rule all`인데 shard size가 `0`이면 clear error로 중단.
- `ResponseCacheShardWriter` 추가:
  - 저장 위치: `response_cache/<dataset_name>/part_000000.npz`, `part_000001.npz`, ...
  - manifest: `response_cache/<dataset_name>/manifest.json`
  - shard manifest에는 `storage=sharded_npz`, `complete`, `num_samples`, `shard_size`, `shards`, `cache_schema_version`, `dataset_name`, `reference_config_id` 기록.
  - 각 shard는 기존 `save_response_cache()`와 동일한 keys/schema v5를 포함한다.
- `TARRPostprocessor.inference()`는 split 전체 cache list를 만들지 않는다.
  - metric 계산용 `pred`, `label`, active `ood_score`만 메모리에 유지.
  - response cache fields는 sample 또는 shard buffer 단위로 writer에 넘기고 flush 후 버린다.
- `score_rule`별 score output은 기존처럼 single `.npz` 유지:
  - `<score_rule>/scores/<dataset>.npz`
  - `<score_rule>/ood.csv`
- 대형 run 메모리 방지를 위해 `--debug-output-mode full|none` 추가:
  - default: `full`로 기존 동작 유지.
  - ImageNet-1K full command는 `--debug-output-mode none` 사용.
  - `none`이면 `sample_debug` list에 row를 쌓지 않고 `debug_samples*.csv` 생성을 건너뛴다.

### Cache / Report 로드 경로
- `cache.py`에 logical response cache loader 추가:
  - single: `response_cache/<name>.npz`
  - sharded: `response_cache/<name>/manifest.json` + parts
  - 기존 `load_cache()`는 새 loader의 compatibility alias로 유지.
- `cache.py validate`는 logical dataset 단위로 검사:
  - single `.npz`는 기존 검사 유지.
  - shard는 모든 part 존재, `complete=true`, required keys, scalar metadata 일치, sample-axis shape 연결 가능 여부 검사.
- `cache.py rescore`는 single/sharded 입력을 모두 지원한다.
  - rescore output은 기존 single score `.npz`와 `ood.csv` 유지.
- `reports.py diagnostics`는 shared loader를 사용해 single/sharded response cache를 모두 읽는다.
  - CSV 호환성을 위해 기존 `cache_path`는 유지하고, 가능하면 `cache_num_shards`, `cache_paths`를 추가한다.

### Manifest / Compatibility
- `scheme_manifest.json`의 `cache_files`는 기존 string path와 새 dict path를 모두 허용:
  - single: `"imagenet.npz"`
  - sharded: `{"storage": "sharded_npz", "manifest": ".../response_cache/imagenet/manifest.json"}`
- response cache schema version은 유지한다.
  - 저장 layout만 바뀌며 score 정의, delta 정의, protocol identity는 변경하지 않는다.
- incomplete shard manifest는 claim-invalid로 처리한다.
  - `--overwrite` 재실행 시 incomplete shard directory를 삭제하고 해당 split을 재생성한다.
  - true resume은 이번 v1 범위 밖으로 둔다.

## Subagent Implementation Allocation
- Agent 1: `eval.py`
  - `ResponseCacheShardWriter`, `--response-cache-shard-size`, `--debug-output-mode none`, inference streaming refactor.
- Agent 2: `cache.py`
  - logical loader, validate/rescore sharded support, manifest compatibility.
- Agent 3: `reports.py` / docs
  - diagnostics sharded loader 적용, CSV metadata 추가, `experiments.md`/implementation docs 업데이트.
- Parent integrator
  - static checks, smoke, single-vs-sharded regression, ImageNet cleanup/retry commands 검증.

## Test Plan
- Static:
  ```bash
  conda run -n openood python -m py_compile scripts_my/tarr/*.py
  conda run -n openood python scripts_my/tarr/eval.py --help
  conda run -n openood python scripts_my/tarr/cache.py validate --help
  conda run -n openood python scripts_my/tarr/cache.py rescore --help
  conda run -n openood python scripts_my/tarr/reports.py diagnostics --help
  ```
- Synthetic/small regression:
  - 기존 single response cache를 2 shards로 나눈 fixture 생성.
  - `cache.py validate`, `cache.py rescore`, `reports.py diagnostics` 결과가 single과 sharded에서 동일한지 확인.
- CIFAR regression:
  - `--response-cache-shard-size 0` 기존 single `.npz` 경로 통과.
  - `--response-cache-shard-size 16` sharded 경로도 validate/rescore 통과.
- ImageNet smoke:
  ```bash
  CUDA_VISIBLE_DEVICES=0 conda run -n openood python scripts_my/tarr/eval.py \
    --dataset imagenet --baseline-protocol eval_api --scheme fsood \
    --run-id smoke_imagenet_sharded_npz \
    --output-root results_test/tarr \
    --reference-config all_rpc8:per_class=8,filter=all,seed=0 \
    --objective predicted_label_ce --steps 1 --lr 1e-2 \
    --update-scope classifier --runtime-mode auto --score-rule all \
    --max-id-samples 128 --max-ood-samples 128 \
    --batch-size 64 --reference-candidate-batch-size 256 \
    --reference-batch-size 1024 --num-workers 8 \
    --response-cache-shard-size 32 \
    --debug-output-mode none \
    --save-response-cache --overwrite
  ```
  Then validate/rescore/diagnostics on the smoke run.

## Cleanup Commands For Failed ImageNet-1K Runs
먼저 경로 확인:

```bash
du -sh \
  results_test/tarr/outputs/imagenet/eval_api/seed0/imagenet_evalapi_fsood_hugeref_plce_s5_lr1e2_rpc8_16_32_feasible_seed0 \
  results_test/tarr/outputs/imagenet/eval_api/seed0/imagenet_evalapi_fsood_hugeref_entropy_s5_lr1e2_rpc8_16_32_feasible_seed0
```

잔여 partial output 삭제:

```bash
rm -rf \
  results_test/tarr/outputs/imagenet/eval_api/seed0/imagenet_evalapi_fsood_hugeref_plce_s5_lr1e2_rpc8_16_32_feasible_seed0 \
  results_test/tarr/outputs/imagenet/eval_api/seed0/imagenet_evalapi_fsood_hugeref_entropy_s5_lr1e2_rpc8_16_32_feasible_seed0
```

reference candidate cache와 reference bank cache는 삭제하지 않는다. 재사용해야 setup 시간을 줄일 수 있다.

## Re-Run Commands After Implementation

### GPU0: predicted_label_ce
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n openood python scripts_my/tarr/eval.py \
  --dataset imagenet \
  --baseline-protocol eval_api \
  --scheme fsood \
  --run-id imagenet_evalapi_fsood_hugeref_plce_s5_lr1e2_rpc8_16_32_feasible_seed0 \
  --output-root results_test/tarr \
  --reference-config all_rpc8:per_class=8,filter=all,seed=0 \
  --reference-config all_rpc16:per_class=16,filter=all,seed=0 \
  --reference-config all_rpc32:per_class=32,filter=all,seed=0 \
  --reference-config correct_rpc8:per_class=8,filter=correct,seed=0 \
  --reference-config correct_rpc16:per_class=16,filter=correct,seed=0 \
  --reference-config correct_rpc32:per_class=32,filter=correct,seed=0 \
  --reference-config strat_rpc8:per_class=8,filter=correct_confidence_stratified,seed=0 \
  --reference-config strat_rpc16:per_class=16,filter=correct_confidence_stratified,seed=0 \
  --reference-config strat_rpc32:per_class=32,filter=correct_confidence_stratified,seed=0 \
  --objective predicted_label_ce \
  --steps 5 \
  --lr 1e-2 \
  --update-scope classifier \
  --runtime-mode auto \
  --score-rule all \
  --batch-size 64 \
  --reference-candidate-batch-size 256 \
  --reference-batch-size 1024 \
  --num-workers 8 \
  --response-cache-shard-size 1024 \
  --debug-output-mode none \
  --save-response-cache \
  --overwrite
```

### GPU1: entropy
```bash
CUDA_VISIBLE_DEVICES=1 conda run -n openood python scripts_my/tarr/eval.py \
  --dataset imagenet \
  --baseline-protocol eval_api \
  --scheme fsood \
  --run-id imagenet_evalapi_fsood_hugeref_entropy_s5_lr1e2_rpc8_16_32_feasible_seed0 \
  --output-root results_test/tarr \
  --reference-config all_rpc8:per_class=8,filter=all,seed=0 \
  --reference-config all_rpc16:per_class=16,filter=all,seed=0 \
  --reference-config all_rpc32:per_class=32,filter=all,seed=0 \
  --reference-config correct_rpc8:per_class=8,filter=correct,seed=0 \
  --reference-config correct_rpc16:per_class=16,filter=correct,seed=0 \
  --reference-config correct_rpc32:per_class=32,filter=correct,seed=0 \
  --reference-config strat_rpc8:per_class=8,filter=correct_confidence_stratified,seed=0 \
  --reference-config strat_rpc16:per_class=16,filter=correct_confidence_stratified,seed=0 \
  --reference-config strat_rpc32:per_class=32,filter=correct_confidence_stratified,seed=0 \
  --objective entropy \
  --steps 5 \
  --lr 1e-2 \
  --update-scope classifier \
  --runtime-mode auto \
  --score-rule all \
  --batch-size 64 \
  --reference-candidate-batch-size 256 \
  --reference-batch-size 1024 \
  --num-workers 8 \
  --response-cache-shard-size 1024 \
  --debug-output-mode none \
  --save-response-cache \
  --overwrite
```

## Assumptions
- Sharded `.npz` is the canonical response cache for ImageNet-scale full runs; it is not merged back into one `.npz`.
- Score output `.npz` remains single-file because it stores scalar scores and is small.
- Debug sample CSV is not claim-bearing for ImageNet-1K; diagnostics/rescore outputs are the claim-relevant artifacts.
- Candidate cache and reference bank cache are valid and should be reused.
