# RAE non-claim-bearing artifact cleanup - 2026-06-29

## 목적

RAE claim-bearing 실험 비교에서 사용하지 않을 smoke, subset, feasibility, feature-rule 결과를 제거했다. 이후 실험 탐색에서는 아래 artifact를 근거로 삼지 않는다.

## 삭제한 artifact

| artifact root | 제외 이유 |
|---|---|
| `results_test/rae_accept_geom_smoke_20260629` | smoke/subset 결과 |
| `results_test/rae_geometry_smoke_20260629` | smoke/subset 결과 |
| `results_test/rae_proto_smoke_20260629` | smoke/subset 결과 |
| `results_test/rae_dense_feasibility_20260628` | feasibility 결과 |
| `results_test/rae_dense_feasibility_bs4_20260628` | feasibility 결과 |
| `results_test/rae_dense_feasibility_bs8_20260628` | feasibility 결과 |
| `results_test/rae_dense_feasibility_cifar100_20260628` | feasibility 결과 |
| `results_test/rae_dense_feasibility_cifar100_all_20260628` | feasibility 결과 |
| `results_test/rae_ratio_full` | feature-rule 계열 결과 |

## 다시 탐색하지 않을 방향

- Subset/smoke 결과는 성능 비교나 claim-bearing 판단에 사용하지 않는다.
- Feasibility run은 구현 가능성 확인 용도였으므로 성능 근거로 사용하지 않는다.
- `feature_*` 또는 feature ratio 기반 rule은 RAE의 acceptance-evidence 핵심 claim에서 벗어나므로 claim-bearing 후보로 재탐색하지 않는다.

## 유지할 비교 기준

- Full split 결과만 claim-bearing 비교 대상으로 둔다.
- Baseline 비교가 가능한 `ood.csv`와 run manifest가 있어야 한다.
- RAE 핵심 claim은 acceptance direction, reference acceptance geometry, confidence 결합 범위 안에서 유지한다.
