# CIFAR-10 RAE 실험 중간 보고

작성일: 2026-06-26

이 문서는 `docs_my/RAE/Reference-validated Acceptance Evidence.md` 구현 이후 현재까지 실행한 CIFAR-10 RAE 실험 결과를 정리한다. 이번 갱신에서는 Gate 1 online diagnostic의 기본 finite step size를 `1e-4`에서 `1e-2`로 바꾼 뒤 classifier full grid를 다시 실행했다.

현재 결과는 최종 claim으로 사용하기보다, RAE score가 실제 reference-validated evidence로 동작하는지 점검하기 위한 중간 보고로 해석해야 한다.

## 실행 범위

| 구분 | 경로 | 설정 | 상태 |
| --- | --- | --- | --- |
| classifier full grid, FSOOD | `results_test/rae/experiments/cifar10/cifar10_fsood_classifier_grid_step1e2_20260626` | reference-per-class `4,8,16,32,64`, seed `0,1,2`, candidate `all,pred` | 완료 |
| classifier full grid, OOD | `results_test/rae_ood/experiments/cifar10/cifar10_ood_classifier_grid_step1e2_20260626` | 동일 | 완료 |

이번 재실행은 classifier-only full grid다. 따라서 experiment-level Gate 9 gradient-space ablation은 `skip`이고, `last_block`/`all`까지 포함한 gradient-space ablation은 별도 full 실험으로만 비교한다.

## Classifier Full Grid 결과

`classifier` gradient space에서는 모든 reference size, reference seed, candidate mode, score rule 조합이 사실상 동일한 성능을 냈다.

| Scheme | Candidate | Score | near AUROC / FPR95 | far AUROC / FPR95 | mean AUROC / FPR95 |
| --- | --- | --- | --- | --- | --- |
| FSOOD | all | neglog_eid | 75.53 / 68.97 | 79.61 / 52.13 | 77.57 / 60.55 |
| FSOOD | pred | neglog_eid | 75.53 / 68.97 | 79.61 / 52.13 | 77.57 / 60.55 |
| OOD | all | neglog_eid | 87.69 / 53.54 | 91.00 / 31.43 | 89.34 / 42.49 |
| OOD | pred | neglog_eid | 87.69 / 53.54 | 91.00 / 31.43 | 89.34 / 42.49 |

`neg_eid`도 `neglog_eid`와 동일한 ranking을 만들기 때문에 AUROC/FPR95는 동일했다. reference-per-class와 reference seed를 바꿔도 mean AUROC가 변하지 않았다.

기존 OpenOOD baseline과 비교하면 다음과 같다.

| Scheme | Method | mean AUROC | mean FPR95 |
| --- | --- | ---: | ---: |
| FSOOD | KNN | 80.99 | 50.13 |
| FSOOD | MSP | 79.22 | 57.97 |
| FSOOD | RAE classifier | 77.57 | 60.55 |
| OOD | KNN | 91.73 | 30.34 |
| OOD | RMDS | 90.28 | 38.49 |
| OOD | MSP | 89.34 | 42.50 |
| OOD | RAE classifier | 89.34 | 42.49 |

classifier 결과는 OOD에서는 MSP와 사실상 같은 수준이고, FSOOD에서는 MSP보다 낮다.

## Classifier Diagnostics

Gate 1 실패는 해결됐다. 이전에는 step size `1e-4`가 너무 작아 float32에서 loss 변화가 0으로 관측되는 sample이 있었고, 이 때문에 Gate 1 pass rate가 1.0에 도달하지 못했다. `1e-2`로 바꾼 새 실험에서는 모든 child run에서 Gate 1이 pass했다.

child diagnostics 결과는 다음과 같다.

| Gate | 상태 | 관찰 |
| --- | --- | --- |
| Gate 1 acceptance_delta | pass | 예시 pass_rate `1.0`, mean_loss_delta 약 `-1.52e-4` |
| Gate 2 reference_sign_delta | pass | finite_delta_match_rate 예시 `0.875`, pass_rate `1.0` |
| Gate 3 classifier_fc_factorization | pass | compact classifier K와 dense K가 일치 |
| Gate 4 confidence_matched_separation | pass | confidence bin 기준 ID-side evidence 우세 |
| Gate 6 signed_support | warn | rank-only, same-positive, signed validation이 분리되지 않음 |
| Gate 7 label_shuffle | pass | label shuffle은 evidence를 낮춤 |
| Gate 10 score_ablation | pass | score ablation artifact 생성 및 평가 완료 |

30개 child run 모두 `diagnostics_status=warn`이다. 원인은 Gate 6 warn이다. `runs.csv`와 experiment manifest 기준으로는 `claim_bearing=True`로 기록되지만, 이 claim은 “실험 artifact가 full-data 조건과 required diagnostics를 만족한다”는 의미에 가깝다. mechanism 해석 측면에서는 Gate 6 warn과 score collapse 때문에 아직 강한 claim으로 쓰기 어렵다.

Experiment-level Gate는 다음과 같다.

| Gate | 상태 | 관찰 |
| --- | --- | --- |
| Gate 8 reference_size_stability | pass | 30/30 child run 완료, summary row 480 |
| Gate 9 gradient_space_ablation | skip | 이번 실험은 `classifier`만 포함 |

## 핵심 이상 징후

가장 중요한 문제는 classifier RAE evidence가 pretrained probability와 완전히 같은 ranking으로 붕괴한다는 점이다.

대표 artifact에서 확인한 내부값:

| Run / split | `best_class == pred` | corr(`eid`, `q_max`) | `v_best` 평균 / 표준편차 | `E_ID == q_max` |
| --- | ---: | ---: | --- | --- |
| classifier FSOOD rpc4 all, ID split | 1.0000 | 1.00000 | 1.00000 / 0.00000 | true |
| classifier FSOOD rpc4 all, CIFAR-100 split | 1.0000 | 1.00000 | 1.00000 / 0.00000 | true |
| classifier OOD rpc4 all, ID split | 1.0000 | 1.00000 | 1.00000 / 0.00000 | true |
| classifier OOD rpc4 all, CIFAR-100 split | 1.0000 | 1.00000 | 1.00000 / 0.00000 | true |

즉 현재 classifier setting에서는 `E_ID = max_c q_c V_c`에서 `V_c`가 1로 포화되어 있고, 최종 score는 사실상 `q_max`의 단조 변환이다. 이 상태에서는 RAE가 reference-validation을 이용한 새 detector라기보다 MSP와 같은 ranking으로 동작한다.

이 현상 때문에 다음 결과들이 동시에 나타난다.

- `candidate-mode all`과 `pred`가 같은 metric을 낸다.
- `reference-per-class`와 `reference-seed` 변화에 metric이 반응하지 않는다.
- `neglog_eid`와 `neg_eid`는 ranking이 같으므로 metric이 같다.
- Gate 6이 warn으로 남는다.

## Artifact 추적성

새 experiment artifact 자체는 개선되어 있다.

- `experiment_manifest.json`에는 experiment-level diagnostics manifest가 embed된다.
- `runs.csv`에는 child run별 `diagnostics_status=warn`, `claim_bearing=True`가 기록된다.
- score `.npz`에는 `candidate_classes`, `q_c`, `v_c`, `e_c`, `rank_only_scores`, `same_positive_rates` 등 diagnostics 재현에 필요한 중간값이 저장된다.

하지만 개별 `run_manifest.json`에는 아직 diagnostics 경로/status가 비어 있다.

- `diagnostics_status: null`
- `diagnostics_manifest_path: null`

실제 child diagnostics artifact는 `results_test/rae/diagnostics/cifar10/<run_id>/` 또는 `results_test/rae_ood/diagnostics/cifar10/<run_id>/`에 존재한다. 따라서 run-level artifact만 단독으로 열었을 때 diagnostics와 연결되지 않는 문제는 아직 남아 있다.

## 현재 결론

Gate 1 실패는 `diagnostic_step_size=1e-2` 기본값으로 해결됐다. 따라서 이전 보고서의 “classifier full grid가 Gate 1 실패 때문에 non-claim”이라는 결론은 더 이상 맞지 않다.

그러나 CIFAR-10 classifier 결과를 RAE mechanism claim으로 쓰기에는 여전히 이르다. 성능은 OOD에서 MSP와 사실상 같고, FSOOD에서는 MSP보다 낮으며, 내부값은 `E_ID == q_max`로 붕괴한다. 즉 현재 classifier RAE는 reference-validation이 score ranking에 실질적으로 기여하지 못한다.

## 다음 수정 우선순위

1. `V_c` 계산을 먼저 점검한다. 특히 classifier setting에서 validation이 왜 항상 1로 포화되는지 확인해야 한다.
2. Gate 6을 더 엄격하게 해석한다. 현재 warning은 RAE의 signed validation이 rank-only/same-positive ablation과 구분되지 않는다는 직접 신호다.
3. 개별 `run_manifest.json`에 diagnostics status/path를 연결한다. 현재 experiment-level 추적은 가능하지만 run-level 단독 추적은 불완전하다.
4. `last_block`/`all` gradient-space ablation을 full run으로 실행해 Gate 9를 실제로 평가한다.
5. score collapse가 해결된 뒤 CIFAR-10 full grid를 다시 실행한다.
