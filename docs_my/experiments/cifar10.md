# CIFAR-10 Reproduction

CIFAR-10 관련 데이터 다운로드와 MSP 평가 재현 메모다.

## 추가 데이터

### CINIC-10

CIFAR-10의 csID 데이터로 사용된다.

```bash
python ./scripts_my/download/download.py \
  --contents datasets \
  --datasets cinic10 \
  --dataset_mode dataset \
  --save_dir ./data ./results
```

### Misc Benchmark

```bash
python ./scripts_my/download/download.py \
  --contents datasets \
  --datasets misc \
  --dataset_mode benchmark \
  --save_dir ./data ./results
```

### CIFAR-100-C, ImageNet-O

```bash
python ./scripts_my/download/download.py \
  --contents datasets \
  --datasets cifar100c imagenet_o \
  --dataset_mode dataset \
  --save_dir ./data ./results
```

## MSP OOD 평가

```bash
sh scripts_my/ood/msp/cifar10_test_ood_msp.sh
```

이 스크립트는 두 단계를 실행한다.

1. `main.py` 기반 단일 run 평가
2. `scripts/eval_ood.py` 기반 unified evaluator 평가

`scripts/eval_ood.py` 경로는 `timm`, `statsmodels`, `foolbox==3.2.1`이 필요하다.

## 최근 확인한 MSP 결과

`main.py` 기반 평가:

```text
ID Accuracy: 95.22%

Near-OOD mean:
FPR@95 53.57, AUROC 87.68

Far-OOD mean:
FPR@95 31.43, AUROC 91.00
```

`scripts/eval_ood.py` 기반 평가에서는 root 아래 `s0`, `s1`, `s2`를 모두 평가한다.

```text
s0:
nearood FPR@95 53.55, AUROC 87.68
farood  FPR@95 31.43, AUROC 91.00
ACC 95.22

s1:
nearood FPR@95 44.30, AUROC 88.16
farood  FPR@95 29.62, AUROC 91.06
ACC 94.63

s2:
nearood FPR@95 46.67, AUROC 88.26
farood  FPR@95 34.09, AUROC 90.12
ACC 95.32
```
