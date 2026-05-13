# OOD Method Groups

이 문서는 새로 실험하려는 TTA response 기반 OOD detection 방법을 어떤 baseline과 먼저 비교할지 정리한 메모다.

## 목표 방법의 위치

현재 목표는 다음과 같다.

```text
test sample에 TTA를 수행한 뒤,
reference ID data에 대한 모델 반응과 비교해서 OOD를 탐지한다.
```

이 방법은 기존 classifier checkpoint를 그대로 사용한다.
따라서 별도 neural network를 새로 학습하는 방법이라기보다, reference data의 반응 분포를 이용하는 post-hoc 방식에 가깝다.

```text
기존 checkpoint 사용
reference ID data의 TTA response 계산
test sample의 TTA response와 비교
```

즉 1차 비교는 train-dependent 방법보다 post-hoc 방법들과 하는 것이 더 적절하다.

## Group 1. Post-hoc 방법

특징:

- backbone classifier를 새로 학습하지 않는다.
- 기존 checkpoint를 그대로 사용한다.
- score 계산 방식이나 reference 통계만 다르다.
- 방법에 따라 ID train/val feature 통계를 계산할 수는 있다.

확인된 방법:

```text
MSP
MLS
EBO
ODIN
IODIN
GradNorm
MDS
RMDS
KNN
ViM
Residual
ReAct
ASH
DICE
Gram
KLM
SHE
SCALE
AdaScale-A
AdaScale-L
```

주의할 점:

```text
통계 fitting이 필요하다고 해서 train-dependent는 아니다.
```

예를 들어 MDS, KNN, ViM, ReAct는 ID train/val data를 한 번 통과시켜 평균, covariance, threshold, index 등을 만든다.
하지만 classifier 자체를 새로 학습하지 않으므로 post-hoc 비교군에 들어간다.

## Group 2. Train-dependent 방법

특징:

- 방법 전용 학습 과정이 필요하다.
- 기존 backbone classifier만으로는 바로 비교하기 어렵다.
- loss, auxiliary task, extra head, outlier data, synthetic OOD 등이 추가될 수 있다.
- 성능 차이가 score function 때문인지 학습 방식 때문인지 분리하기 어렵다.

확인된 방법:

```text
CSI
RotPred
ConfBranch
GODIN
MOS
VOS
NPOS
CIDER
OE / MixOE
LogitNorm
```

이 방법들은 나중에 추가 비교 대상으로 둘 수 있다.
하지만 처음부터 섞으면 실험 해석이 복잡해진다.

## 권장 비교 순서

1차 비교:

```text
기존 checkpoint 고정
새 TTA response score vs 기존 post-hoc OOD methods
```

이 단계의 질문:

```text
별도 재학습 없이,
기존 classifier에 post-hoc TTA response score만 추가했을 때
기존 post-hoc OOD detection 방법들보다 나은가?
```

2차 ablation:

```text
feature response vs logit response vs softmax response
TTA 종류
TTA 횟수
reference set 크기
ID train vs ID val reference
KNN/MDS/energy 조합
```

3차 비교:

```text
train-dependent 방법들과 비교
```

이 단계의 질문:

```text
전용 학습이 필요한 방법들과 비교해도 어느 정도 경쟁력이 있는가?
```

## scripts_my 실행 정책

`scripts_my/ood/<method>/`의 Group 1 스크립트는 한 파일에서 OOD와 FSOOD를 모두 확인하는 방향으로 정리한다.

대상 dataset:

```text
MNIST
CIFAR-10
CIFAR-100
ImageNet
ImageNet-200
```

기본 구조:

```text
<dataset>_test_ood_<method>.sh

1. main.py로 OOD 평가
2. main.py로 FSOOD 평가
3. unified evaluator로 OOD 평가
4. unified evaluator로 FSOOD 평가
```

단, MNIST는 `scripts/eval_ood.py`의 `--id-data` 선택지에 없으므로 unified evaluator block을 두지 않고 `main.py` 평가만 둔다.

FSOOD main 평가에서는 일반 OOD 설정을 그대로 쓰지 않고 다음 조합을 사용한다.

```text
configs/datasets/<dataset>/<dataset>_fsood.yml
configs/pipelines/test/test_fsood.yml
```

ImageNet 계열은 원본 스크립트처럼 unified evaluator에서도 `--fsood`를 함께 실행한다.

```text
eval_ood_imagenet.py --fsood   # ImageNet
eval_ood.py --fsood            # ImageNet-200
```

이렇게 하면 한 스크립트를 실행했을 때 같은 method에 대한 일반 OOD와 full-spectrum OOD 결과를 함께 확인할 수 있다.
