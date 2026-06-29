# RAE Gradient Signal Diagnostics - 2026-06-28

## 목적

`V_c(x)`를 보기 전에 acceptance direction 자체가 cleanID, csID, nearOOD, farOOD에서 다른 양상을 보이는지 확인했다. 이번 진단은 두 종류로 나눴다.

- `candidate-mode pred`: 모델 argmax class의 acceptance direction `d_pred(x)` 단독 특성
- `candidate-mode all`: 모든 class candidate의 `d_c(x)` 방향 분포와 reference alignment 특성

## 설정

- Dataset: CIFAR-10
- Scheme: FSOOD
- Reference: correct filter, 16 per class, seed 0
- Gradient spaces: `classifier`, `last_block`, `all`
- pred 진단 subset: split별 최대 512 samples
- all-candidate 진단 subset: split별 최대 256 samples
- Output root: `results_test/rae_dense_full_20260628/gradient_diagnostics/cifar10/`
- 추가 산출물: `sample_metrics.csv`, `confidence_independence.csv`, `plots/*.png`

## `d_pred(x)` 단독 특성

모델이 예측한 class 하나에 대한 acceptance direction만 보면, raw gradient norm은 cleanID에서 작고 csID/OOD에서 커진다. 이 패턴은 classifier, last_block, all 모두에서 보인다.

| space | group | pred conf | raw grad norm | effective dim | k margin |
|---|---:|---:|---:|---:|---:|
| classifier | id | 0.9796 | 0.1604 | - | 0.8673 |
| classifier | csid | 0.8897 | 0.8021 | - | 0.7957 |
| classifier | nearOOD | 0.8337 | 1.1717 | - | 0.7459 |
| classifier | farOOD | 0.7950 | 1.3629 | - | 0.7411 |
| last_block | id | 0.9796 | 3.0231 | 15961.9504 | 0.3363 |
| last_block | csid | 0.8896 | 16.6862 | 11521.7678 | 0.2621 |
| last_block | nearOOD | 0.8337 | 25.4666 | 11175.7658 | 0.2250 |
| last_block | farOOD | 0.7950 | 31.9029 | 8578.7677 | 0.2039 |
| all | id | 0.9796 | 8.3637 | 22778.4511 | 0.1936 |
| all | csid | 0.8897 | 46.7203 | 14187.3375 | 0.1266 |
| all | nearOOD | 0.8337 | 63.0667 | 14491.6910 | 0.1020 |
| all | farOOD | 0.7950 | 75.2967 | 10489.2086 | 0.0904 |

해석:

- `d_pred(x)` raw norm은 cleanID confidence saturation과 반대로 움직인다. cleanID는 이미 예측 class loss가 낮아서 acceptance 방향의 원 gradient가 작다.
- csID/OOD는 confidence가 낮아지고 loss residual이 커지므로 `d_pred(x)` raw norm이 커진다.
- dense gradient space에서는 effective dimension이 cleanID에서 더 크고 farOOD에서 낮아지는 경향이 있다. OOD 쪽 acceptance direction은 더 적은 차원에 집중되는 신호가 있다.
- reference alignment인 `k margin`은 cleanID가 가장 높고 OOD로 갈수록 낮아진다.

## 모든 `d_c(x)` 후보 방향 분포

모든 class candidate를 보면 `d_c(x)` 방향이 예측 class 방향과 얼마나 다른지, 그리고 reference same/other class와 어떤 alignment를 갖는지 볼 수 있다.

| space | group | raw grad norm | effective dim | cos to pred | k margin | same positive | other positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| classifier | id | 8.0982 | - | -0.5005 | 0.3575 | 1.0000 | 0.4460 |
| classifier | csid | 7.6117 | - | -0.4243 | 0.3722 | 1.0000 | 0.4310 |
| classifier | nearOOD | 7.2537 | - | -0.3792 | 0.3884 | 1.0000 | 0.4246 |
| classifier | farOOD | 6.7613 | - | -0.3606 | 0.4015 | 1.0000 | 0.4394 |
| last_block | id | 92.8825 | 12536.2090 | -0.5541 | 0.1367 | 0.3505 | 0.1110 |
| last_block | csid | 105.6737 | 10183.3561 | 0.4461 | 0.1402 | 0.6075 | 0.1362 |
| last_block | nearOOD | 112.7012 | 9624.0816 | 0.3953 | 0.1500 | 0.8083 | 0.1646 |
| last_block | farOOD | 118.8039 | 7151.1392 | 0.2962 | 0.1445 | 0.9059 | 0.1909 |
| all | id | 144.2730 | 20717.6845 | -0.5482 | 0.0842 | 0.4535 | 0.1580 |
| all | csid | 212.5682 | 15217.5470 | 0.4773 | 0.0718 | 0.6154 | 0.2450 |
| all | nearOOD | 231.0704 | 15784.1130 | 0.4448 | 0.0746 | 0.7556 | 0.2890 |
| all | farOOD | 237.5510 | 10874.3704 | 0.3365 | 0.0687 | 0.8247 | 0.3208 |

해석:

- classifier의 all-candidate 평균 raw norm은 `d_pred(x)`와 반대로 cleanID가 더 크게 보인다. 이는 cleanID에서 pred class 하나는 gradient가 거의 0이지만, 나머지 non-pred class 후보는 큰 loss residual을 갖기 때문이다. 따라서 classifier에서 candidate 평균 norm은 detection score로 직접 쓰기보다 candidate geometry 진단으로 보는 게 맞다.
- dense spaces에서는 all-candidate 평균에서도 raw norm이 cleanID < csID < near/farOOD 순서로 커진다.
- dense spaces에서 `cos(d_c, d_pred)` 평균은 cleanID에서 음수이고, csID/OOD에서는 양수로 바뀐다. cleanID에서는 class 후보 방향들이 pred 방향과 더 잘 분리되고, shift/OOD에서는 여러 candidate direction이 pred direction 쪽으로 뭉치는 신호로 해석된다.
- same positive와 other positive가 OOD에서 같이 증가한다. 이는 OOD에서 acceptance direction이 reference class를 구분해 지지하기보다 더 많은 reference에 광범위하게 양의 alignment를 만드는 현상이다.
- `k margin`은 dense all 공간에서 cleanID가 가장 높고 farOOD가 가장 낮다. 따라서 내적 기반 reference alignment는 여전히 유효하지만, raw direction 특성만으로도 별도 신호가 존재한다.

## 시각화 산출물

각 gradient space별로 네 종류의 그림을 생성했다.

- `confidence_vs_gradient_metrics.png`: 정사각형 `pred_conf`-metric scatter와 축별 group lane half-violin
- `confidence_vs_gradient_metrics_overlay.png`: 정사각형 `pred_conf`-metric scatter와 같은 축 위에 겹친 marginal density
- `group_boxplots.png`: cleanID, csID, nearOOD, farOOD group별 지표 분포
- `confidence_residual_group_signal.png`: raw group separation과 confidence 제거 후 group separation 비교

| space | lane half-violin scatter | overlay marginal scatter | group boxplot | confidence residual |
|---|---|---|---|---|
| classifier | [png](../../results_test/rae_dense_full_20260628/gradient_diagnostics/cifar10/c10_fsood_grad_signal_allcand_20260628_graddiag_classifier_correct_rpc16_all_refseed0_subset256/plots/confidence_vs_gradient_metrics.png) | [png](../../results_test/rae_dense_full_20260628/gradient_diagnostics/cifar10/c10_fsood_grad_signal_allcand_20260628_graddiag_classifier_correct_rpc16_all_refseed0_subset256/plots/confidence_vs_gradient_metrics_overlay.png) | [png](../../results_test/rae_dense_full_20260628/gradient_diagnostics/cifar10/c10_fsood_grad_signal_allcand_20260628_graddiag_classifier_correct_rpc16_all_refseed0_subset256/plots/group_boxplots.png) | [png](../../results_test/rae_dense_full_20260628/gradient_diagnostics/cifar10/c10_fsood_grad_signal_allcand_20260628_graddiag_classifier_correct_rpc16_all_refseed0_subset256/plots/confidence_residual_group_signal.png) |
| last_block | [png](../../results_test/rae_dense_full_20260628/gradient_diagnostics/cifar10/c10_fsood_grad_signal_allcand_20260628_graddiag_last_block_correct_rpc16_all_refseed0_subset256/plots/confidence_vs_gradient_metrics.png) | [png](../../results_test/rae_dense_full_20260628/gradient_diagnostics/cifar10/c10_fsood_grad_signal_allcand_20260628_graddiag_last_block_correct_rpc16_all_refseed0_subset256/plots/confidence_vs_gradient_metrics_overlay.png) | [png](../../results_test/rae_dense_full_20260628/gradient_diagnostics/cifar10/c10_fsood_grad_signal_allcand_20260628_graddiag_last_block_correct_rpc16_all_refseed0_subset256/plots/group_boxplots.png) | [png](../../results_test/rae_dense_full_20260628/gradient_diagnostics/cifar10/c10_fsood_grad_signal_allcand_20260628_graddiag_last_block_correct_rpc16_all_refseed0_subset256/plots/confidence_residual_group_signal.png) |
| all | [png](../../results_test/rae_dense_full_20260628/gradient_diagnostics/cifar10/c10_fsood_grad_signal_allcand_20260628_graddiag_all_correct_rpc16_all_refseed0_subset256/plots/confidence_vs_gradient_metrics.png) | [png](../../results_test/rae_dense_full_20260628/gradient_diagnostics/cifar10/c10_fsood_grad_signal_allcand_20260628_graddiag_all_correct_rpc16_all_refseed0_subset256/plots/confidence_vs_gradient_metrics_overlay.png) | [png](../../results_test/rae_dense_full_20260628/gradient_diagnostics/cifar10/c10_fsood_grad_signal_allcand_20260628_graddiag_all_correct_rpc16_all_refseed0_subset256/plots/group_boxplots.png) | [png](../../results_test/rae_dense_full_20260628/gradient_diagnostics/cifar10/c10_fsood_grad_signal_allcand_20260628_graddiag_all_correct_rpc16_all_refseed0_subset256/plots/confidence_residual_group_signal.png) |


## Confidence와 분리되는 정보인지 확인

분리성은 두 값으로 봤다.

- `spearman_with_conf`: 해당 지표와 `pred_conf`의 monotonic correlation
- `group_eta2_conf_residual`: 지표에서 `pred_conf`의 선형 효과를 제거한 residual이 cleanID/csID/nearOOD/farOOD group을 얼마나 설명하는지

`candidate_prob`, `candidate_is_pred`, `candidate_is_label`처럼 confidence 자체이거나 label-derived인 항목은 confidence 분리성 분석에서 제외했다.

| space | metric | spearman with conf | residual group eta2 | interpretation |
|---|---|---:|---:|---|
| classifier | `candidate_raw_grad_norm_pred` | -0.9943 | 0.0165 | 거의 confidence proxy |
| classifier | `candidate_raw_grad_norm_mean` | 0.8266 | 0.0821 | confidence 의존성이 큼 |
| classifier | `k_mean_margin_mean` | -0.8301 | 0.0855 | confidence와 강하게 얽힘 |
| last_block | `candidate_raw_grad_norm_pred` | -0.9951 | 0.0025 | 거의 confidence proxy |
| last_block | `candidate_raw_grad_norm_mean` | -0.0798 | 0.2768 | confidence와 분리된 group 정보가 있음 |
| last_block | `candidate_effective_dim_mean` | 0.5335 | 0.3142 | confidence 영향은 있으나 별도 정보도 큼 |
| last_block | `candidate_direction_cos_to_pred_mean` | 0.2621 | 0.4458 | confidence와 꽤 분리된 방향 geometry 정보 |
| last_block | `k_same_positive_rate_mean` | -0.7799 | 0.3234 | confidence와 관련 있지만 residual signal도 큼 |
| all | `candidate_raw_grad_norm_pred` | -0.9891 | 0.0098 | 거의 confidence proxy |
| all | `candidate_raw_grad_norm_mean` | -0.1898 | 0.2465 | confidence와 분리된 group 정보가 있음 |
| all | `candidate_effective_dim_mean` | 0.3503 | 0.2579 | confidence와 부분 독립 |
| all | `candidate_direction_cos_to_pred_mean` | 0.1739 | 0.3986 | confidence와 가장 잘 분리되는 방향 geometry 정보 중 하나 |
| all | `k_same_positive_rate_mean` | -0.8210 | 0.3131 | confidence와 관련 있지만 residual signal도 남음 |
| all | `k_other_positive_rate_mean` | -0.5839 | 0.2535 | confidence 제거 후에도 OOD broad alignment 신호가 남음 |

해석:

- `d_pred(x)` raw norm은 confidence와 거의 같은 정보를 준다. cleanID에서 confidence가 높으면 pred-class CE gradient가 작고, OOD에서 confidence가 낮아지면 gradient가 커지는 구조이기 때문이다.
- classifier 공간의 주요 지표는 confidence와 많이 얽혀 있다. RAE reference alignment 자체는 계산되지만, confidence와 분리된 추가 정보는 상대적으로 약하다.
- `last_block`과 `all`의 all-candidate 방향 지표는 confidence와 더 잘 분리된다. 특히 `candidate_direction_cos_to_pred_mean`, `candidate_effective_dim_mean`, `candidate_raw_grad_norm_mean`은 confidence residual에서도 group 차이가 남는다.
- `k_same_positive_rate_mean`, `k_other_positive_rate_mean`도 confidence와 상관은 있지만 residual group signal이 남는다. 이는 OOD에서 confidence만 낮아지는 것이 아니라 reference alignment의 class-specificity가 실제로 바뀐다는 뜻이다.

## 결론

`d_c(x)` 단독 특성은 RAE 진단에 추가할 가치가 있다.

- `d_pred(x)` raw norm: confidence saturation과 강하게 연결된 uncertainty 신호
- dense `d_c(x)` effective dimension: cleanID와 OOD의 gradient concentration 차이
- `cos(d_c, d_pred)`: cleanID에서는 class 후보 방향 분리가 크고, OOD에서는 후보 방향이 pred 방향 쪽으로 collapse되는 신호
- same/other positive rate: OOD에서 class-specific support가 약해지고 broad positive alignment가 늘어나는지 확인하는 보조 진단
- confidence와 가장 잘 분리되는 후보는 dense all-candidate 방향 geometry 지표이며, 단일 `d_pred(x)` norm은 confidence proxy에 가깝다.

다음 단계는 이 지표들을 RAE score로 바로 결합하기보다, 먼저 diagnostic gate 또는 score artifact summary에 기록해서 `V_c` 포화/실패 원인을 설명하는 데 사용하는 것이다.
