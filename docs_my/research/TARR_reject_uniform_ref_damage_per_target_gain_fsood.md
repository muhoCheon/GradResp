# TARR reject/uniform Ref-Damage-per-Target-Gain FSOOD 분석

날짜: 2026-06-12.

이 문서는 reject branch에서 계산한 `reference damage / target gain` ratio를
별도로 정리한다. 파일명과 제목의 핵심은 다음이다.

```text
reject_uniform = reject / uniform
ref_damage_per_target_gain = ref_damage / target_gain
```

Accept/predicted-label CE와 target objective 의미가 다르므로 accept 분석 문서와
분리한다.

## Score 정의

이 문서의 예시는 다음 branch를 사용한다.

```text
role = reject
branch = uniform
```

분자는 accept 문서와 동일하게 reference damage다.

```text
ref_damage =
  mean_c max(ref CE loss after update - ref CE loss before update, 0)
```

분모는 target gain이다. 단, 여기서 target loss는 ordinary CE가 아니라
`uniform` rejection objective다.

```text
target_gain =
  max(-(uniform rejection objective after update
        - uniform rejection objective before update), 0)
```

Raw ratio와 OOD score는 다음이다.

```text
raw_damage_per_gain =
  ref_damage / (eps + target_gain)

score =
  -raw_damage_per_gain
```

## CIFAR-100 `all_rpc16`

설정:

```text
artifact:
  results_test/tarr/outputs/cifar100/eval_api/seed0/
  cifar100_eval_api_fsood_arbank_semantic_s30_5x10x30_lr1em2_all_rpc16_refseed0_merged8

reference = all_rpc16
role = reject
branch = uniform
response_step = 30
```

### FSOOD 성능

| ID side | near AUROC | far AUROC | avg AUROC | Group 1 avg | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| both | 63.76 | 69.81 | 66.79 | 66.20 | +0.59 |
| clean-only | 76.33 | 80.42 | 78.37 | 81.54 | -3.17 |
| csID-only | 52.46 | 60.25 | 56.36 | 65.13 | -8.77 |

### Split 분포 요약

| Split | N | score median | score IQR | raw ratio median | ref_damage median | target_gain median | zero target_gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 9000 | -0.001791 | [-0.002905, -0.000966] | 0.001791 | 0.010008 | 5.501493 | 0 |
| csID | 10000 | -0.000894 | [-0.001244, -0.000689] | 0.000894 | 0.001332 | 1.735421 | 0 |
| nearOOD | 16526 | -0.000872 | [-0.001193, -0.000683] | 0.000872 | 0.001400 | 1.824074 | 0 |
| farOOD | 135445 | -0.000843 | [-0.001171, -0.000630] | 0.000843 | 0.001536 | 2.091336 | 0 |

분자/분모 관점에서 보면 clean은 target gain도 크지만 reference damage가 훨씬
크다. OOD split들은 reference damage가 작고 target gain도 낮아지지만,
ratio는 clean보다 낮다. 다만 csID, nearOOD, farOOD 사이 차이가 작아서
csID-aware FSOOD 분리에는 약하다.

## 결론

Reject/uniform ratio는 CIFAR-100 FSOOD both에서 Group 1 best를 근소하게 넘는다.
하지만 target objective가 uniform rejection objective이므로 accept/predicted CE
ratio와 같은 분자/분모 해석으로 묶으면 안 된다.
