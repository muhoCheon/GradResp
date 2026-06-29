# RAE Claim-bearing Full Test 결과 - 2026-06-29

## 요약

이번 full test의 목표는 RAE의 핵심 아이디어, 즉 acceptance evidence와 confidence를 유지하면서 CIFAR-10/CIFAR-100의 OOD 또는 FSOOD에서 기존 OpenOOD baseline을 넘는 claim-bearing 결과를 찾는 것이었다.

결론부터 말하면, 현재 full 결과에서 SOTA는 달성하지 못했다. CIFAR-100 FSOOD nearOOD는 기존 MSP 65.05와 사실상 tie 수준까지 도달했지만, farOOD와 CIFAR-10에서는 baseline과 차이가 남는다.

## 실험 범위

- Full split만 사용했다. subset/smoke 결과는 비교에서 제외했다.
- Output root: `results_test/rae_claim_full_20260629`
- Dataset/scheme: CIFAR-10 FSOOD, CIFAR-10 OOD, CIFAR-100 FSOOD, CIFAR-100 OOD
- Gradient space: `classifier`
- Candidate mode: `all`
- Reference filter: `correct`
- Reference per class grid: `4,8,16,32,64`
- Validation rule grid: `pairwise_rank`, `pairwise_margin`, `same_mean`, `mean_margin`, `soft_margin_t0p1`
- Score rule: `neg_eid`, `neglog_eid`
- 생성 결과: 각 dataset/scheme별 50개 `ood.csv`, 총 200개

## Baseline 최고값

`docs_my/experiments/group1_validation.md` 기준이다.

| dataset | scheme | nearOOD best | farOOD best |
|---|---|---:|---:|
| CIFAR-10 | OOD | KNN 90.56 | KNN 92.89 |
| CIFAR-10 | FSOOD | KNN 79.46 | Gram 84.09 |
| CIFAR-100 | OOD | IODIN 81.09 | KNN 81.86 |
| CIFAR-100 | FSOOD | MSP 65.05 | RMDS 65.73 |

## RAE Full Grid 최고 결과

| dataset | scheme | nearOOD best | gap | farOOD best | gap |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | FSOOD | 75.86, `rpc32`, `soft_margin_t0p1`, `neg_eid` | -3.60 | 79.76, `rpc64`, `soft_margin_t0p1`, `neg_eid` | -4.33 |
| CIFAR-10 | OOD | 88.22, `rpc32`, `soft_margin_t0p1`, `neg_eid` | -2.34 | 91.28, `rpc64`, `soft_margin_t0p1`, `neg_eid` | -1.61 |
| CIFAR-100 | FSOOD | 65.05, `rpc64`, `pairwise_rank`, `neg_eid` | ~0.00 | 61.39, `rpc32`, `mean_margin`, `neg_eid` | -4.34 |
| CIFAR-100 | OOD | 80.41, `rpc16`, `pairwise_rank`, `neg_eid` | -0.68 | 77.57, `rpc16`, `pairwise_rank`, `neg_eid` | -4.29 |

`neg_eid`와 `neglog_eid`는 이번 grid에서 AUROC가 사실상 동일했다. 둘 다 `E_ID`에 대한 monotonic transform이므로 ranking 기반 metric에서는 큰 차이가 나지 않는다.

## 추가 후보 해석

`docs_my/RAE/gradient_signal_diagnostics_20260628.md`에서는 dense `last_block`/`all`의 all-candidate direction geometry가 confidence와 일부 분리된 정보를 준다는 점이 확인됐다. 특히 `candidate_direction_cos_to_pred_mean`, `candidate_effective_dim_mean`, `candidate_raw_grad_norm_mean`은 confidence residual에서도 group signal이 남았다.

그러나 이 신호를 단일 RAE score로 full test에 올렸을 때는 아직 baseline을 넘지 못했다.

- CIFAR-10 FSOOD `last_block/pred` geometry full: best near 75.92, far 80.14
- CIFAR-100 FSOOD classifier geometry/full: best near 65.04, far 61.35
- CIFAR-100 OOD classifier geometry/full: best near 80.41, far 77.57
- uniform rejection evidence sweep: CIFAR-100 FSOOD에서 기존 `E_ID`보다 개선되지 않음

따라서 현재 claim-bearing 아이디어로 가장 가까운 것은 CIFAR-100 FSOOD nearOOD의 classifier RAE tie 수준 결과와 dense direction geometry의 진단 신호다. 하지만 “기존 OpenOOD baseline보다 높다”는 claim은 아직 할 수 없다.

## Acceptance Geometry Prototype 추가 결과

2026-06-29에 `last_block` acceptance direction이 reference ID geometry와 얼마나 정렬되는지 보는 score를 추가했다. 각 class \(c\)에 대해 reference acceptance direction \(d_y(r)\)의 평균 prototype \(\mu_c\)를 만들고, target direction \(d_c(x)\)와의 cosine을 점수화한다.

- `geom_proto_cos_max`: \(-\max_c \langle d_c(x), \mu_c \rangle\)
- `geom_proto_eid`: \(-\max_c q_c(x)\max(\langle d_c(x), \mu_c \rangle, 0)\)

이 score는 feature distance가 아니라 acceptance direction 자체가 ID reference의 class-specific gradient geometry와 맞는지를 본다. 따라서 RAE의 acceptance-evidence 맥락에서 해석 가능하다.

### CIFAR-10 full 결과

- Output root: `results_test/rae_accept_geom_full_20260629`
- Gradient space: `last_block`
- Candidate mode: `pred`
- Reference: `correct`, `reference_per_class=16`, `reference_seed=0`
- Diagnostics: off

| scheme | score | nearOOD AUROC | farOOD AUROC | 해석 |
|---|---|---:|---:|---|
| FSOOD | `geom_proto_cos_max` | 76.40 | 82.40 | 기존 dense geometry보다 개선, baseline 미달 |
| FSOOD | `geom_proto_eid` | 76.96 | 82.12 | near는 가장 높지만 baseline 미달 |
| FSOOD | `geom_rawnorm_mean` | 75.92 | 80.14 | 이전 dense geometry 후보 |
| FSOOD | `neg_eid` | 75.28 | 79.12 | RAE acceptance evidence |
| OOD | `geom_proto_cos_max` | 90.16 | 93.64 | farOOD에서 KNN 92.89 초과 |
| OOD | `geom_proto_eid` | 90.35 | 93.55 | near는 KNN 90.56에 근접, farOOD 초과 |
| OOD | `geom_rawnorm_mean` | 88.26 | 91.56 | prototype보다 낮음 |
| OOD | `neg_eid` | 87.25 | 90.35 | RAE acceptance evidence |

현재 기준으로 claim-bearing에 가장 가까운 결과는 CIFAR-10 OOD의 `last_block/pred/geom_proto_eid`다. nearOOD는 KNN 90.56보다 0.21 낮지만, farOOD는 KNN 92.89보다 0.66 높다. 즉 “전체 SOTA”라고 말하기는 아직 이르지만, acceptance geometry가 기존 RAE score보다 더 강한 단일 score 후보임은 확인됐다.

### `all` gradient space 상태

`all + pred`는 `reference_per_class=2`, `max_target_samples=64` smoke에서 OOM 없이 실행됐다.

- Output root: `results_test/rae_accept_geom_smoke_20260629`
- `geom_proto_eid`: nearOOD 76.73, farOOD 86.76
- `geom_proto_cos_max`: nearOOD 74.37, farOOD 83.60

이는 subset 결과라 claim-bearing으로 사용하지 않는다. 다만 `all` gradient space도 prototype score 계산 자체는 가능함을 확인했다. full run은 비용이 크므로, 우선 CIFAR-10 `last_block` 결과를 기준으로 추가 sweep 여부를 결정하는 것이 적절하다.

## 결론

현재 RAE score는 CIFAR-100 nearOOD에서는 경쟁력이 있지만, farOOD 분리력이 부족하다. 반면 `last_block` acceptance prototype score는 CIFAR-10 OOD에서 기존 dense geometry와 RAE `E_ID`보다 뚜렷하게 개선됐고 farOOD baseline을 넘었다. 이는 target acceptance direction이 ID reference의 class-specific gradient geometry와 정렬되는지를 보는 방향이 유망하다는 근거다.

다음 개선은 full-test 통계를 쓰는 z-score 방식이 아니라, train/reference-only calibration이나 class-specific prototype/subspace score로 진행하는 것이 적절하다. 예를 들어 reference/train held-out에서 class-conditional prototype alignment의 정상 범위를 추정하거나, prototype 하나가 아니라 class별 low-rank acceptance subspace projection으로 확장해야 claim-bearing 조건을 유지할 수 있다.
