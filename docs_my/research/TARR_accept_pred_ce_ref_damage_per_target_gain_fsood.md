# TARR accept/predicted_label_ce Ref-Damage-per-Target-Gain FSOOD 분석

날짜: 2026-06-12.

이 문서는 ImageNet-200과 CIFAR-100을 같은 accept score 기준으로 비교한다.
파일명과 제목의 핵심은 다음이다.

```text
accept_pred_ce = accept / predicted_label_ce
ref_damage_per_target_gain = ref_damage / target_gain
```

두 dataset 모두 다음 branch만 사용한다.

```text
role = accept
branch = predicted_label_ce
```

Reject branch 결과는 target objective의 의미가 다르므로 이 문서에 섞지 않는다.

## Score 정의

Accept/predicted-label CE update에서 보고 싶은 질문은 다음이다.

```text
target x의 predicted-label CE를 줄이는 gain 1만큼 얻기 위해
ID reference surface를 얼마나 손상시키는가?
```

분자는 reference damage다.

```text
ref_damage =
  mean_c max(ref CE loss after update - ref CE loss before update, 0)
```

분모는 target gain이다. 여기서는 accept branch가 `predicted_label_ce`이므로
target loss는 pretrained top-1 pseudo label에 대한 CE loss다.

```text
target_gain =
  max(-(target CE loss after update - target CE loss before update), 0)
```

Raw ratio와 OOD score는 다음이다.

```text
raw_damage_per_gain =
  ref_damage / (eps + target_gain)

score =
  -raw_damage_per_gain

eps = 1e-12
```

Score 방향은 기존 convention에 맞춘다.

```text
higher score = more OOD-like
```

즉 `score`가 클수록 `raw_damage_per_gain`은 작다. 이 score는 target CE를
줄일 때 ID reference CE 손상이 적게 따라오는지를 본다. 따라서 단순히
"reference damage가 작아서 좋다"가 아니라, **target update와 ID reference
surface 반응이 약하게 결합되어 있는가**를 보는 post-hoc ratio로 해석한다.

## FSOOD 성능 요약

아래 성능은 저장된 raw `tta_response`에서
`score=-ref_damage/(eps+target_gain)`을 post-hoc으로 재계산한 것이다. Dataset별
AUROC를 먼저 계산한 뒤 near/far group 평균을 내는 공식 FSOOD 집계 방식으로
확인했다.

### ImageNet-200 `correct_rpc32`

설정:

```text
artifact:
  results_test/tarr/outputs/imagenet200/eval_api/seed0/
  imagenet200_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_correct_rpc32_refseed0_merged8

reference = correct_rpc32
role = accept
branch = predicted_label_ce
response_step = 5
```

| ID side | near AUROC | far AUROC | avg AUROC | Group 1 avg | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| both | 63.16 | 74.68 | 68.92 | 65.55 | +3.38 |
| clean-only | 83.74 | 90.12 | 86.93 | 89.41 | -2.48 |
| csID-only | 57.72 | 70.60 | 64.16 | 57.26 | +6.90 |

### CIFAR-100 `highconf09_rpc16`

설정:

```text
artifact:
  results_test/tarr/outputs/cifar100/eval_api/seed0/
  cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_highconf09_rpc16_refseed0_minbank_pce_uniform_merged8

reference = highconf09_rpc16
role = accept
branch = predicted_label_ce
response_step = 10
```

| ID side | near AUROC | far AUROC | avg AUROC | Group 1 avg | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| both | 64.21 | 67.35 | 65.78 | 66.20 | -0.42 |
| clean-only | 78.93 | 80.30 | 79.61 | 81.54 | -1.93 |
| csID-only | 50.97 | 55.69 | 53.33 | 65.13 | -11.80 |

## Split 분포 요약

분포는 mean보다 median과 IQR을 중심으로 읽는다. `target_gain`이 0에 가까운
sample은 ratio 평균을 크게 왜곡할 수 있기 때문이다.

표의 `score IQR`은 `[P25, P75]`다. Score가 0에 가까울수록
`raw_damage_per_gain`이 작다는 뜻이다.

### ImageNet-200

| Split | N | score median | score IQR | raw ratio median | ref_damage median | target_gain median | zero target_gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 9000 | -0.020254 | [-0.353244, -0.003109] | 0.020254 | 0.000023 | 0.001115 | 1282 |
| csID | 34000 | -0.001856 | [-0.003209, -0.001329] | 0.001856 | 0.001049 | 0.755610 | 542 |
| nearOOD | 54879 | -0.001640 | [-0.003079, -0.001070] | 0.001640 | 0.000655 | 0.541547 | 686 |
| farOOD | 31029 | -0.001208 | [-0.001780, -0.000839] | 0.001208 | 0.000977 | 1.005917 | 59 |

분자/분모 관점에서 보면 핵심은 다음이다.

| Split | ref_damage median | target_gain median | raw ratio median | 읽는 방법 |
| --- | ---: | ---: | ---: | --- |
| clean | 0.000023 | 0.001115 | 0.020254 | reference damage는 작지만 target gain이 거의 없어 cost/gain이 커진다 |
| csID | 0.001049 | 0.755610 | 0.001856 | target gain이 커지면서 ratio가 크게 낮아진다 |
| nearOOD | 0.000655 | 0.541547 | 0.001640 | reference damage가 csID보다 작아 ratio가 더 낮다 |
| farOOD | 0.000977 | 1.005917 | 0.001208 | target gain이 가장 커서 ratio가 가장 낮다 |

ImageNet-200에서는 median 기준으로 다음 ordering이 나온다.

```text
clean < csID < nearOOD < farOOD
```

Clean split에는 `target_gain ~= 0`인 sample이 많아서 mean score가 크게
왜곡된다. 따라서 이 ratio는 평균보다 rank/AUROC와 median/IQR로 보는 것이 맞다.

### CIFAR-100

| Split | N | score median | score IQR | raw ratio median | ref_damage median | target_gain median | zero target_gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 9000 | -0.001325 | [-0.022008, -0.000323] | 0.001325 | 0.000008 | 0.005990 | 805 |
| csID | 10000 | -0.000257 | [-0.000425, -0.000180] | 0.000257 | 0.000121 | 0.608266 | 85 |
| nearOOD | 16526 | -0.000256 | [-0.000388, -0.000185] | 0.000256 | 0.000125 | 0.598719 | 28 |
| farOOD | 135445 | -0.000245 | [-0.000425, -0.000175] | 0.000245 | 0.000090 | 0.444096 | 198 |

분자/분모 관점에서 보면 핵심은 다음이다.

| Split | ref_damage median | target_gain median | raw ratio median | 읽는 방법 |
| --- | ---: | ---: | ---: | --- |
| clean | 0.000008 | 0.005990 | 0.001325 | target gain이 거의 없어 ratio가 가장 높다 |
| csID | 0.000121 | 0.608266 | 0.000257 | target gain이 커지며 ratio가 낮아진다 |
| nearOOD | 0.000125 | 0.598719 | 0.000256 | csID와 거의 같은 영역이다 |
| farOOD | 0.000090 | 0.444096 | 0.000245 | ref_damage가 더 작아 ratio가 약간 낮다 |

CIFAR-100도 accept/predicted CE 기준으로 보면 ImageNet-200과 같은 큰 방향은
나온다.

```text
clean < csID ~= nearOOD < farOOD
```

하지만 csID, nearOOD, farOOD의 ratio 차이가 매우 작고 `csID-only` AUROC가 낮다.
따라서 CIFAR-100 accept score는 clean-vs-rest 성격은 있지만, csID-aware FSOOD
분리에는 약하다.

## 결론

이 문서의 두 dataset은 모두 동일한 score를 쓴다.

```text
role = accept
branch = predicted_label_ce
score = -ref_damage / (eps + target_gain)
```

ImageNet-200에서는 FSOOD both avg `68.92`로 Group 1 best를 넘고, split median도
`clean < csID < nearOOD < farOOD` 방향을 만든다. CIFAR-100에서는 FSOOD both avg
`65.78`로 Group 1 best보다 낮고, csID/nearOOD/farOOD 분리가 약하다.

따라서 accept ratio score는 ImageNet-200에서는 promising하지만, CIFAR-100까지
일관되게 강한 SOTA 후보라고 보기는 어렵다. Reject/uniform score는 target
objective 의미가 다르므로 별도 문서에서 따로 해석한다.
