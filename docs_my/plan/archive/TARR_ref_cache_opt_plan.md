# TARR Reference Cache / Batch Hot-Path Optimization + Speed Evaluation Plan

> Archive note: This plan is historical. Use docs_my/TARR/ and scripts_my/tarr/ as the current source of truth unless this file is explicitly referenced for context.

## Summary
기존 수정계획은 유지한다. 추가로 최적화 효과를 정량화하기 위해 **before/after runtime 비교**와 **batch-size sweep**을 별도 benchmark protocol로 추가한다. 목표는 score 정의와 response cache schema v5를 바꾸지 않고, reference candidate scan, reference bank setup, per-target adapted response의 속도 개선을 분리해서 측정하는 것이다.

## Key Changes
- **Reference candidate cache v2**
  - `REFERENCE_CANDIDATE_CACHE_SCHEMA_VERSION = 2`로 bump한다. response `CACHE_SCHEMA_VERSION`은 변경하지 않는다.
  - candidate cache에 `ce_loss`를 추가하고, selected reference의 base diagnostics는 candidate metadata에서 만든다.
  - full logits는 저장하지 않는다.
  - old v1 cache는 fallback으로 읽되, 새 run은 v2 cache를 생성한다.

- **Reference selection / bank reuse**
  - reference selection은 selected candidate indices와 selected metadata를 함께 반환한다.
  - selected sample 로딩은 `dataset[dataset_index]` direct access를 우선 사용하고, 실패 시 loader scan fallback.
  - classifier feature cache runtime에서 reference bank cache를 추가한다.
    - 저장 위치: `<output_root>/reference_banks/<bank_cache_id>/bank.npz`
    - 저장: labels, selected candidate indices/hash, selected data hash, base diagnostics, feature tensor.
    - image tensor는 저장하지 않는다.
  - `selected_reference_hash`는 기존 tensor/label 기반을 유지한다.

- **Batch-size separation**
  - 새 CLI:
    - `--reference-candidate-batch-size`, default `0`; `0`이면 기존 `--batch-size` 사용.
    - `--reference-bank-cache-root`, default `<output_root>/reference_banks`.
    - `--rebuild-reference-bank-cache`.
  - `--batch-size`: ID/csID/OOD target loader batch.
  - `--reference-candidate-batch-size`: train full forward candidate metadata build 전용.
  - `--reference-batch-size`: selected reference feature caching과 adapted response 전용.

## Speed Evaluation Plan
- **Benchmark outputs**
  - 각 run의 `run_info.md`와 manifest에 아래 값을 기록한다:
    - candidate cache build/reuse 여부와 seconds
    - reference bank cache build/reuse 여부와 seconds
    - setup total seconds
    - inference total seconds
    - processed target count
    - runtime per target
    - `batch_size`, `reference_candidate_batch_size`, `reference_batch_size`, `num_workers`
  - 가능하면 benchmark CSV를 별도로 저장한다:
    - `results_test/tarr/summary/runtime_benchmark.csv`

- **Before/after comparison**
  - 최적화 전 코드 기준 결과가 이미 있으면 기존 run의 `run_info.md`를 baseline으로 사용한다.
  - 없으면 implementation 직전 현재 코드로 small benchmark를 `/tmp/tarr_runtime_baseline`에 1회 실행한다.
  - 최적화 후 같은 command를 `/tmp/tarr_runtime_optimized`에 실행한다.
  - 비교는 claim-bearing 결과가 아니라 runtime-only diagnostic이다.

- **Measured stages**
  - `candidate_build`: train full scan으로 candidate cache 생성.
  - `candidate_reuse`: candidate cache hit 후 setup.
  - `bank_build`: selected reference feature/base metadata 생성.
  - `bank_reuse`: bank cache hit 후 setup.
  - `inference`: target sample TTA + adapted reference response.
  - `rescore`: response cache offline score 계산.

- **Dataset benchmark matrix**
  | Dataset | Runtime benchmark samples | Ref configs | Candidate batch sweep | Reference batch sweep |
  |---|---:|---|---|---|
  | `cifar10` | 500 ID / 500 OOD | `all_rpc32`, `correcthigh_rpc32` | 512, 1024, 2048 | 512, 1024, 2048 |
  | `cifar100` | 500 ID / 500 OOD | `all_rpc8`, `all_rpc32`, `correcthigh_rpc32` | 512, 1024, 2048 | 512, 1024, 2048 |
  | `imagenet200` | 100 ID / 100 OOD | `all_rpc1`, `all_rpc2` | 128, 256, 512 | 256, 512, 1024 |
  | `imagenet` | 50 ID / 50 OOD | `all_rpc1` | 128, 256, 384 | 256, 512, 768 |

- **Full-run recommended starting values**
  | Dataset | `--batch-size` | `--reference-candidate-batch-size` | `--reference-batch-size` | `--num-workers` |
  |---|---:|---:|---:|---:|
  | `cifar10` | 512 | 2048 | 2048 | 8 |
  | `cifar100` | 512 | 2048 | 2048 | 8 |
  | `imagenet200` | 64 | 512 | 1024 | 8 |
  | `imagenet` | 32 | 256 | 512 | 8 |

- **Success criteria**
  - Candidate cache v2 build adds no measurable slowdown versus v1 beyond CE loss compute noise.
  - Candidate cache reuse avoids train full scan completely.
  - Bank cache reuse reduces setup time for repeated reference configs.
  - ImageNet-200 smoke runtime is meaningfully lower on second run.
  - ImageNet-1K smoke becomes feasible enough to retry after optimization.
  - Active scores remain numerically unchanged within tolerance.

## Subagent Implementation Allocation
- **Agent 1: candidate cache**
  - `reference.py`: v2 cache, `ce_loss`, selected metadata, direct dataset indexing.
- **Agent 2: eval hot path**
  - `eval.py`: metadata-derived base diagnostics, bank cache, new CLI, timing instrumentation.
- **Agent 3: validation/docs**
  - `cache.py` validation, manifests, docs, benchmark CSV schema.
- **Agent 4: benchmark runner**
  - Run before/after and batch sweep benchmarks.
  - Summarize speedup by stage and dataset.

## Test Plan
- Static/help:
  - `conda run -n openood python -m py_compile scripts_my/tarr/*.py`
  - `conda run -n openood python scripts_my/tarr/eval.py --help`
  - `conda run -n openood python scripts_my/tarr/cache.py validate --help`

- Correctness:
  - Tiny CIFAR-10 run creates candidate cache v2 with `ce_loss`.
  - Metadata-derived base diagnostics match forward-derived base diagnostics within tolerance.
  - Existing reference filters select the same samples as before.
  - Response cache schema remains v5.
  - Active score outputs match pre-optimization tiny fixed-seed run within tolerance.

- Runtime:
  - Run each benchmark once with cold candidate/bank cache and once with warm cache.
  - Record runtime stage breakdown and GPU memory.
  - OOM fallback:
    - candidate build OOM: halve `--reference-candidate-batch-size`.
    - reference response OOM: halve `--reference-batch-size`.
    - target loader OOM: halve `--batch-size`.
    - host RAM pressure: reduce `--num-workers` to `4`.

## Assumptions
- This is runtime optimization only; no TTA objective, score rule, protocol, delta definition, or response cache schema changes.
- Benchmark runs are runtime diagnostics, not claim-bearing experiment results.
- ImageNet-1K full/smoke is retried only after ImageNet-200 optimized smoke and warm-cache benchmark pass.
