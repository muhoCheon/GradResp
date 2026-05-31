# Random Model Sanity Check

이 실험은 학습되지 않은 random model에서 ID 대비 csID/nearOOD/farOOD score 분포와 AUROC가 어떻게 나오는지 확인하기 위한 sanity check다.
목적은 OOD method 성능 비교가 아니라, dataset shift 자체가 random feature에서도 분리되는지 확인하는 것이다.

## 범위

- Methods: `msp`, `mls`, `ebo`, `odin`, `iodin`, `gradnorm`, `mds`, `rmds`, `knn`, `vim`, `react`, `ash`, `dice`, `gram`, `klm`, `she`, `scale`
- Datasets: `cifar10`, `cifar100`, `imagenet`, `imagenet200`
- Seeds: `0,1,2,3,4`
- MNIST는 현재 `openood/evaluation_api/datasets.py`에 `mnist` 항목이 없어 제외한다.
- ImageNet-OOD 데이터셋 연결은 보류한다.
- `residual`, `adascale_a`, `adascale_l`은 기존 Group 1 validation의 known issue/skip 정책에 맞춰 random sanity 기본 지원 목록에서 제외한다.

## Dataset 정의 주의사항

Random sanity는 `Evaluator` API를 사용하므로 dataset 구성은 `openood/evaluation_api/datasets.py`를 따른다.
반면 Group 1 validation runner는 `configs/datasets/*/*_fsood.yml` 기반으로 실행한다.

두 경로의 dataset 정의가 완전히 동일하지 않다. 특히 CIFAR-10 csID 정의가 다르다.

```text
Group 1 config 기준:
  configs/datasets/cifar10/cifar10_fsood.yml
  csID = cinic10

Random sanity evaluation_api 기준:
  openood/evaluation_api/datasets.py
  csID = cifar10c
```

ImageNet 계열은 주로 이름 표기 방식이 다르다.

```text
Group 1 config 기준:
  imagenetv2, imagenetc, imagenetr

Random sanity evaluation_api 기준:
  imagenet_v2, imagenet_c, imagenet_r
```

따라서 결과는 "현재 `evaluation_api` dataset 정의 기준의 random sanity check"로 해석한다.
Group 1 validation과 완전히 같은 dataset 구성의 bias check가 필요하면, 추후에 dataset 정의를 통일하거나 random sanity 전용 dataloader mapping을 별도로 맞춰야 한다.
이 차이는 결과 해석과 후속 확장 전에 반드시 다시 고려해야 한다.

## Random Model

기존 dataset별 architecture를 사용하되 checkpoint를 로드하지 않는다.

```text
cifar10     -> ResNet18_32x32
cifar100    -> ResNet18_32x32
imagenet    -> ResNet50
imagenet200 -> ResNet18_224x224
```

초기화 옵션:

```text
--init kaiming
  Conv/Linear는 Kaiming normal, bias는 0, BatchNorm weight/bias는 1/0으로 명시 초기화한다.

--init default
  repo의 model 생성 기본 초기화를 그대로 사용한다.
```

기본값은 `kaiming`이다. checkpoint는 로드하지 않으며, random model 상태 그대로 평가한다.

각 seed에서 `parameter_checksum`을 기록한다. 이 값은 모델 `state_dict` tensor 전체를 SHA256으로 해시한 값이며, seed별 random model weight가 실제로 달라졌는지 확인하기 위한 audit 값이다.

## 실행

smoke test:

```bash
bash scripts_my/runners/random_sanity.sh \
  --methods msp \
  --datasets cifar10 \
  --seeds 0 \
  --gpus 0 \
  --jobs-per-gpu 1 \
  --init kaiming
```

전체 실행:

```bash
bash scripts_my/runners/random_sanity.sh \
  --methods all \
  --datasets all \
  --seeds 0,1,2,3,4 \
  --gpus 0,1 \
  --jobs-per-gpu 2 \
  --init kaiming
```

`--jobs-per-gpu` 기본값은 `2`다. ImageNet 계열에서 VRAM, RAM, CPU, disk I/O 병목이 생기면 `1`로 낮춘다.
여러 method를 쉼표로 지정할 수도 있다.

```bash
bash scripts_my/runners/random_sanity.sh \
  --methods msp,mls,ebo \
  --datasets cifar10 \
  --seeds 0,1,2,3,4 \
  --gpus 0 \
  --jobs-per-gpu 1 \
  --init kaiming
```

## 출력

seed별 실행 기록:

```text
results_test/random_sanity/runs/<method>/<dataset>/seed_<seed>/
```

seed별 score:

```text
results_test/random_sanity/outputs/<method>/<dataset>/seed_<seed>/scores/*.npz
```

`eval_random_ood.py`는 FSOOD dataloader path를 사용해 ID, csID, nearOOD, farOOD inference를 한 번에 수행한다. 단, OpenOOD의 원래 OOD/FSOOD metric CSV는 random sanity 목적과 맞지 않으므로 저장하지 않는다.

summary:

```text
results_test/random_sanity/summary/<method>_classification_acc_by_seed.csv
results_test/random_sanity/summary/<method>_classification_acc_summary.csv
results_test/random_sanity/summary/<method>_id_vs_dataset_auroc_by_seed.csv
results_test/random_sanity/summary/<method>_id_vs_dataset_auroc_summary.csv
```

`*_classification_acc_*.csv`는 ID와 csID의 classification accuracy만 기록한다. 이는 random model 자체가 특정 covariate-shift dataset에서 다른 정확도 bias를 보이는지 확인하기 위한 값이다.

`*_id_vs_dataset_auroc_*.csv`는 OpenOOD의 OOD/FSOOD protocol metric을 사용하지 않고, 항상 ID test score를 기준으로 각 csID/nearOOD/farOOD dataset과의 AUROC만 계산한다. 따라서 `scheme` 컬럼은 두지 않는다.
AUROC 계산 방향은 OpenOOD metric convention과 동일하게 target dataset을 OOD-positive로 두고 `-conf`를 score로 사용한다. 저장된 `conf`는 클수록 ID-like라고 가정하므로, 이 AUROC는 "target dataset이 ID보다 더 OOD-like score를 받는가"를 측정한다.

Random sanity의 핵심 질문은 "ID와 비교했을 때 나머지 dataset들이 random model에서도 분리되는가"이므로, OpenOOD 원래 `ID` 또는 `ID+csID` 기준 detection metric summary는 생성하지 않는다.

Plot은 runner에서 생성하지 않는다. 후속으로 `plot_score_density.py`의 `random_sanity` source를 실행한다.

단일 dataset/method에 대해 seed aggregate plot을 생성한다.

```bash
python scripts_my/tools/plot_score_density.py \
  --source random_sanity \
  --dataset cifar10 \
  --method msp \
  --seed all
```

단일 dataset에 대해 random sanity에서 지원하는 모든 method plot을 생성한다.

```bash
python scripts_my/tools/plot_score_density.py \
  --source random_sanity \
  --dataset cifar10 \
  --method all \
  --seed all
```

지원하는 모든 dataset/method plot을 생성한다.

```bash
python scripts_my/tools/plot_score_density.py \
  --source random_sanity \
  --dataset all \
  --method all \
  --seed all
```

특정 seed만 확인할 수도 있다.

```bash
python scripts_my/tools/plot_score_density.py \
  --source random_sanity \
  --dataset cifar10 \
  --method msp \
  --seed 0
```

출력:

```text
results_test/random_sanity/plots/score_density/<dataset>/<method>/seed_<seed_or_all>/per_dataset.png
results_test/random_sanity/plots/score_density/<dataset>/<method>/seed_<seed_or_all>/combined_id_groups.png
```

Plot도 OpenOOD convention에 맞춰 `ood_score = -conf`를 사용한다. x축 값이 클수록 더 OOD-like다.
Group 1 / OpenOOD 일반 validation 결과용 plot은 `docs_my/experiments/group1_validation.md`를 따른다.
