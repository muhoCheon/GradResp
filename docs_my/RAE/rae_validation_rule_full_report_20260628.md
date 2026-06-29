# RAE validation-rule full 실험 보고서

작성일: 2026-06-28

## 범위

이 문서는 최적화 이후 현재 RAE 코드로 새로 실행한 full target split 결과만 정리한다. 기존 `feature_*` rule 결과와 pre-optimization 산출물은 비교 근거에서 제외했다.

- Artifact root: `results_test/rae_validation_full_20260628`
- 분석 요약: `results_test/rae_validation_full_20260628/analysis_summary.json`
- 대상 dataset/scheme: CIFAR-10 OOD, CIFAR-10 FSOOD, CIFAR-100 OOD, CIFAR-100 FSOOD
- Gradient space: `classifier`
- Reference-per-class grid: `4,8,16,32,64`
- Candidate mode: `all,pred`
- Validation rule: `pairwise_rank`, `pairwise_margin`, `same_mean`, `mean_margin`, `soft_margin`
- Score rule: `neglog_eid`, `neg_eid`
- Soft-margin temperature sweep: `0.03,0.1,0.3,1.0` at `reference_per_class=16`

`last_block`/`all` full run은 아래 resource 제약 때문에 claim-bearing 결과로 포함하지 않았다.

## Full grid 완료 여부

| Dataset | Scheme | 완료 run | Rule | Reference grid | Candidate |
|---|---:|---:|---|---|---|
| CIFAR-10 | OOD | 50/50 | 5개 | 4,8,16,32,64 | all,pred |
| CIFAR-10 | FSOOD | 50/50 | 5개 | 4,8,16,32,64 | all,pred |
| CIFAR-100 | OOD | 50/50 | 5개 | 4,8,16,32,64 | all,pred |
| CIFAR-100 | FSOOD | 50/50 | 5개 | 4,8,16,32,64 | all,pred |

## 주요 성능

`reference_per_class=16`, `candidate_mode=all`, `neglog_eid` 기준 mean AUROC:

| Dataset | Scheme | pairwise_rank | pairwise_margin | same_mean | mean_margin | soft_margin |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | OOD | 89.76 | 88.00 | 88.69 | 88.00 | 90.07 |
| CIFAR-10 | FSOOD | 78.08 | 76.13 | 76.77 | 76.13 | 78.26 |
| CIFAR-100 | OOD | 78.64 | 77.87 | 77.86 | 77.87 | 78.55 |
| CIFAR-100 | FSOOD | 62.73 | 62.30 | 62.29 | 62.30 | 62.69 |

Full grid 전체 best mean AUROC:

| Dataset | Scheme | Best rule | Ref/class | Candidate | Mean AUROC |
|---|---:|---|---:|---|---:|
| CIFAR-10 | OOD | soft_margin | 64 | all/pred tie | 90.13 |
| CIFAR-10 | FSOOD | soft_margin | 64 | all/pred tie | 78.30 |
| CIFAR-100 | OOD | pairwise_rank | all grid tie | all/pred tie | 78.64 |
| CIFAR-100 | FSOOD | pairwise_rank | 64 | all/pred tie | 62.73 |

현재 결과만 보면 `soft_margin`은 CIFAR-10에서 confidence-only에 가까운 `pairwise_rank`보다 소폭 좋지만, CIFAR-100에서는 `pairwise_rank`가 가장 높다. CIFAR-100의 `pairwise_rank` 우위는 acceptance evidence가 유효해서라기보다 `V_c`가 거의 1로 포화되어 confidence score와 사실상 같아졌기 때문이다.

## 질문별 결론

### 1. `pairwise_rank`는 왜 saturation을 일으키는가?

`pairwise_rank`는 같은 class reference의 signed support가 다른 class reference보다 얼마나 자주 큰지만 세는 rank/count rule이다. 즉 margin 크기를 보지 않고 inequality 성공 여부만 본다.

Classifier gradient space에서는 현재 학습된 CIFAR 모델과 clean/correct reference 조합에서 대부분의 target에 대해 같은 class reference가 다른 class reference보다 항상 위에 놓인다. 그 결과:

| Dataset | Scheme | ID `V_mean` | ID `V>=0.999` | OOD `V_mean` | OOD `V>=0.999` |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | OOD/FSOOD | 1.0000 | 100.00% | 1.0000 | 100.00% |
| CIFAR-100 | OOD/FSOOD | 0.99998 | 99.84% | 1.0000 | 100.00% |

따라서 `E_ID=q_c V_c`에서 `V_c≈1`이 되고, score는 거의 `q_c` confidence-only score와 같아진다. 실제 ablation에서도 CIFAR-10 `pairwise_rank`의 `V`-only AUC는 50.0%이고 `E_ID` AUC는 `q` AUC와 동일하다.

### 2. `pairwise_margin`이나 `mean_margin`은 `V_c`를 덜 포화시키는가?

그렇다. 두 rule은 binary rank 성공만 보지 않고 signed support gap의 크기를 반영한다. `reference_per_class=16`, `candidate_mode=all` 기준:

| Dataset | Rule | ID `V_mean` | ID `V>=0.999` | OOD `V_mean` | OOD `V>=0.999` |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | pairwise_margin | 0.445 | 0.00% | 0.374 | 0.00% |
| CIFAR-10 | mean_margin | 0.445 | 0.00% | 0.374 | 0.00% |
| CIFAR-100 | pairwise_margin | 0.337 | 0.00% | 0.294 | 0.00% |
| CIFAR-100 | mean_margin | 0.337 | 0.00% | 0.294 | 0.00% |

즉 saturation 문제는 확실히 줄어든다. 다만 최종 score `E_ID=q_c V_c` 기준 성능은 CIFAR-10/100 모두에서 단순 confidence보다 대체로 낮아졌다. `V_c`는 정보를 갖지만, 현재 곱셈 결합이 항상 성능 개선으로 이어지지는 않는다.

### 3. `soft_margin` temperature는 full run에서 제대로 sweep했는가?

그렇다. `reference_per_class=16`, classifier, full target split에서 `0.03,0.1,0.3,1.0`을 sweep했다.

| Dataset | Scheme | Temp | ID `V_mean` | OOD `V_mean` | ID `V>=0.999` | OOD `V>=0.999` | mean `E_ID` AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | OOD | 0.03 | 1.000 | 1.000 | 100.00% | 100.00% | 89.89 |
| CIFAR-10 | OOD | 0.1 | 0.999 | 0.998 | 85.16% | 28.92% | 90.20 |
| CIFAR-10 | OOD | 0.3 | 0.894 | 0.835 | 0.00% | 0.00% | 89.09 |
| CIFAR-10 | OOD | 1.0 | 0.496 | 0.452 | 0.00% | 0.00% | 88.88 |
| CIFAR-100 | OOD | 0.03 | 1.000 | 1.000 | 99.83% | 100.00% | 78.50 |
| CIFAR-100 | OOD | 0.1 | 0.996 | 0.993 | 22.39% | 0.84% | 78.43 |
| CIFAR-100 | OOD | 0.3 | 0.810 | 0.764 | 0.00% | 0.00% | 78.06 |
| CIFAR-100 | OOD | 1.0 | 0.438 | 0.413 | 0.00% | 0.00% | 78.08 |

FSOOD에서도 같은 sweep을 full target split으로 실행했다. mean `E_ID` AUC는 CIFAR-10에서 `0.03=87.84`, `0.1=88.17`, `0.3=87.17`, `1.0=86.98`이고, CIFAR-100에서 `0.03=78.46`, `0.1=78.38`, `0.3=78.01`, `1.0=78.03`이다.

`0.03`은 완전히 포화된다. `0.1`은 CIFAR-10에서 가장 좋은 편이지만 여전히 ID 쪽은 강하게 포화된다. `0.3`과 `1.0`은 포화를 풀지만 최종 detection AUC는 내려간다.

### 4. `classifier` 말고 `last_block/all`에서 acceptance agreement가 더 의미 있는가?

현재 full claim-bearing 결과로는 판단할 수 없다. 이유는 resource 비용이 매우 크기 때문이다.

- CIFAR-10 `reference_per_class=16`: 160 reference
- CIFAR-100 `reference_per_class=16`: 1600 reference
- `last_block`: 약 8.39M parameter, float32 기준 약 32 MB/reference
- `all`: 약 11.2M parameter, float32 기준 약 42-43 MB/reference

CIFAR-100 기준 reference bank만 `last_block≈51 GB`, `all≈68 GB` 수준이 된다. 여기에 full target split scoring의 per-sample/per-candidate gradient 계산이 추가된다. 따라서 현재 32 GB GPU 환경에서 `last_block/all + full grid + all candidate`는 claim-bearing full run으로 바로 넣기 어렵다.

결론적으로 `last_block/all`이 acceptance agreement를 더 의미 있게 만들 가능성은 열려 있지만, 이번 full 결과로 입증되지는 않았다. 먼저 classifier에서 rule/score 구조를 정리한 뒤, `candidate_mode=pred`, 작은 reference size, 선택 split부터 resource-adjusted full run을 따로 설계하는 것이 현실적이다.

### 5. Confidence `q_c`와 acceptance evidence `V_c`는 서로 다른 정보를 제공하는가?

부분적으로 그렇다.

`pairwise_rank`에서는 아니다. `V_c`가 거의 상수 1이므로 `E_ID=q_c V_c≈q_c`가 되고, `V`-only AUC는 chance 수준이다.

Margin 계열에서는 다르다. `V`-only AUC가 chance보다 높고, `q`와 `V`의 correlation도 완전하지 않다.

| Dataset | Rule | mean `q` AUC | mean `V` AUC | mean `E_ID` AUC |
|---|---|---:|---:|---:|
| CIFAR-10 OOD | pairwise_rank | 89.89 | 50.00 | 89.89 |
| CIFAR-10 OOD | pairwise_margin | 89.89 | 86.36 | 88.07 |
| CIFAR-10 OOD | same_mean | 89.89 | 87.42 | 88.77 |
| CIFAR-10 OOD | soft_margin | 89.89 | 87.10 | 90.20 |
| CIFAR-100 OOD | pairwise_rank | 78.53 | 49.92 | 78.52 |
| CIFAR-100 OOD | pairwise_margin | 78.52 | 74.37 | 77.79 |
| CIFAR-100 OOD | same_mean | 78.52 | 74.32 | 77.78 |
| CIFAR-100 OOD | soft_margin | 78.53 | 74.31 | 78.43 |

따라서 `V_c`는 별도 정보를 갖지만, 현재의 단순 곱셈 `E_ID=q_c V_c`가 그 정보를 항상 성능 개선으로 바꾸지는 못한다. CIFAR-10 `soft_margin`에서는 소폭 개선이 보이고, CIFAR-100에서는 confidence보다 낮다.

## 종합 판단

1. `pairwise_rank`는 현재 classifier gradient space에서 RAE mechanism을 검증하기에 부적절하다. 성능이 높아 보여도 acceptance evidence가 아니라 confidence-only에 가깝다.
2. `pairwise_margin`, `mean_margin`, `same_mean`은 `V_c` saturation을 줄이고 acceptance evidence를 실제 변수로 만든다.
3. 하지만 margin 계열의 `E_ID=q_c V_c`는 현재 full 결과에서 confidence baseline을 안정적으로 넘지 못한다.
4. `soft_margin`은 CIFAR-10에서 가장 promising하지만, temperature가 낮으면 다시 saturation되고 높이면 detection AUC가 떨어진다.
5. 다음 개선은 `V_c`를 더 잘 만드는 것보다 `q_c`와 `V_c`의 결합 방식을 재검토하는 쪽이 더 가능성이 높다. 예: `E_ID=q_c * g(V_c)`의 calibration, additive log-score, split-free unsupervised scaling.

## 다음 실험 제안

- Claim-bearing RAE score는 `pairwise_rank`를 제외하고 `soft_margin`, `same_mean`, `pairwise_margin` 중심으로 정리한다.
- CIFAR-10에서는 `soft_margin temperature=0.1`, `reference_per_class=32/64`를 우선 후보로 둔다.
- CIFAR-100에서는 confidence-only를 넘지 못했으므로 `V_c` 결합식을 바꾸기 전에는 SOTA claim 후보로 두기 어렵다.
- `last_block/all`은 full grid가 아니라 resource-adjusted 별도 실험으로 분리한다. 우선 `candidate_mode=pred`, 낮은 reference size, 대표 split에서 `V_c` saturation과 `q/V` ablation만 확인하는 것이 현실적이다.
