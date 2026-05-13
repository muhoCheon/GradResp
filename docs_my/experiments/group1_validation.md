# Group 1 Validation Checklist

Group 1 post-hoc OOD 방법들의 `scripts_my` 실행 스크립트가 정상 동작하는지 확인하기 위한 체크리스트다.

## 검증 범위

```text
20 post-hoc methods x 5 datasets = 100 scripts

Methods:
- MSP
- MLS
- EBO
- ODIN
- IODIN
- GradNorm
- MDS
- RMDS
- KNN
- ViM
- Residual
- ReAct
- ASH
- DICE
- Gram
- KLM
- SHE
- SCALE
- AdaScale-A
- AdaScale-L

Datasets:
- MNIST
- CIFAR-10
- CIFAR-100
- ImageNet
- ImageNet-200
```

## 실행 정책

- 모든 Group 1 `scripts_my/ood/<method>/*test_ood*.sh` 스크립트는 `set -e`를 사용한다.
- 중간 block이 실패하면 이후 block을 실행하지 않고 즉시 종료한다.
- 한 스크립트는 가능한 경우 다음 순서로 실행한다.

```text
1. main.py OOD
2. main.py FSOOD
3. unified evaluator OOD
4. unified evaluator FSOOD
```

- MNIST는 `scripts/eval_ood.py --id-data`에서 지원하지 않으므로 unified evaluator block이 없다.
- GPU는 실행 시점에 `CUDA_VISIBLE_DEVICES=<id>`로 지정한다.
- 같은 script를 동시에 중복 실행하지 않는다. 같은 output csv에 append/충돌이 생길 수 있다.

권장 경로:

```text
OpenOOD output: results_test/outputs/
Run records:    results_test/runs/<method>/<dataset>/run.log
```

실행 템플릿:

```bash
METHOD=knn
DATASET=cifar100
GPU=0
SCRIPT=scripts_my/ood/${METHOD}/${DATASET}_test_ood_${METHOD}.sh
LOG_DIR=results_test/runs/${METHOD}/${DATASET}

mkdir -p ${LOG_DIR}
/usr/bin/time -f "RUNTIME=%E" \
bash -c "CUDA_VISIBLE_DEVICES=${GPU} conda run -n openood sh ${SCRIPT}" \
2>&1 | tee ${LOG_DIR}/run.log

python scripts_my/tools/make_run_metadata.py \
  --method ${METHOD} \
  --dataset ${DATASET} \
  --gpu ${GPU}
```

`run.log` 마지막의 `RUNTIME` 값을 보고 체크리스트의 `Runtime` 컬럼에 기록한다.
`metadata.md`는 `make_run_metadata.py`로 생성한다.
각 method/dataset 조합은 일단 1회 수행 시간을 기준으로 기록한다.

여러 실험을 병렬로 실행하는 경우, 개별 `command.sh`에서는 `metadata.md`까지만 생성한다.
모든 실행이 끝난 뒤 아래 명령으로 체크리스트를 한 번에 갱신한다.

전체 small/medium dataset 검증 실행:

```bash
bash scripts_my/runners/group1_small_medium.sh
```

`MNIST`, `CIFAR-10`, `CIFAR-100`에 대해 Group 1 post-hoc method 스크립트를 순서대로 실행한다.
각 method/dataset 실행 로그와 metadata는 `results_test/runs/<method>/<dataset>/` 아래에 저장된다.

전체 ImageNet 계열 dataset 검증 실행:

```bash
bash scripts_my/runners/group1_imagenet.sh
```

`ImageNet`, `ImageNet-200`에 대해 Group 1 post-hoc method 스크립트를 순서대로 실행한다.
small/medium runner와 동일하게 각 method/dataset 실행 로그와 metadata는 `results_test/runs/<method>/<dataset>/` 아래에 저장된다.
현재 로컬 ImageNet train split 준비 상태에 따라 train feature 의존 방법 일부는 skip 처리될 수 있으며, 자세한 사유는 아래 `Skip 사유` 섹션을 따른다.

metadata를 기준으로 체크리스트 갱신:

```bash
python scripts_my/tools/update_group1_validation.py
```

`results_test/runs/*/*/metadata.md`를 읽어서 아래 체크리스트 표의 status, runtime, AUROC 값을 갱신한다.

체크리스트 표 초기화:

```bash
python scripts_my/tools/update_group1_validation.py --reset
```

아래 체크리스트 표를 초기 `pending` 상태로 되돌린다.
`results_test/runs/`의 로그나 metadata 파일은 삭제하지 않는다.

주의: 일부 파일명은 method 이름과 script suffix가 다르다.
예를 들어 `mls`는 `*_maxlogit.sh`, `kl_matching`은 `*_kl_matching.sh`, `adascale_*`는 `*_scale.sh`를 사용한다.
이 경우 `scripts_my/ood/<method>/` 아래의 실제 파일명을 확인해서 실행한다.

## 상태 값

```text
pending : 아직 실행 전
running : 실행 중
pass    : 모든 기대 block 성공
fail    : 실행 실패
skip    : 의도적으로 건너뜀
```

## Skip 사유

현재 `skip`은 방법 자체를 영구 제외한다는 의미가 아니라, 이번 Group 1 스크립트 동작 검증에서 known issue로 인해 전체 진행을 막지 않기 위해 pass 취급으로 넘긴 항목이다.
각 skip 기록은 `results_test/runs/<method>/<dataset>/metadata.md`의 `notes`에도 남긴다.

### ViM

- MNIST: LeNet feature dimension이 256인데 현재 ViM sweep의 `dim=256` 조합에서 null-space가 비어 `alpha=inf/NaN`이 발생한다.
- CIFAR-10 / CIFAR-100: ResNet18 feature dimension이 512인데 현재 ViM sweep에 `dim=1000`이 포함되어 `alpha=inf/NaN`이 발생한다.
- ImageNet: 로컬 `data/benchmark_imglist/imagenet/train_imagenet.txt`가 실제 `data/images_largescale/imagenet_1k/train/` 파일과 맞지 않는다. ViM은 ID train feature setup을 수행하므로 missing file에서 실패한다.
- ImageNet-200: 위 train imglist 누락 문제에 더해 현재 ViM sweep에 `dim=1000`이 포함되어 ResNet18 feature dimension 512와 맞지 않는다.

### Residual

- 현재 공통 config `configs/postprocessors/residual.yml`의 `dim=512`를 모든 dataset에 그대로 사용한다.
- Residual은 feature covariance의 상위 `dim`개 direction을 제거하고 남은 residual subspace에서 score를 계산한다. 따라서 `dim`이 feature dimension과 같거나 크면 residual subspace가 비어 score가 상수화된다.
- MNIST LeNet은 feature dimension 256이라 `dim=512`가 feature dimension보다 크다.
- CIFAR-10 / CIFAR-100 / ImageNet-200의 ResNet18 계열은 feature dimension 512라 `dim=512`와 같고, AUROC가 50.00으로 고정된다.
- ImageNet ResNet50은 feature dimension 2048이라 `dim=512`에서 residual subspace가 남으므로 skip하지 않는다.

### AdaScale-A / AdaScale-L

- 모든 dataset에서 skip한다.
- AdaScale 계열은 `AdaScaleANet` / `AdaScaleLNet` wrapper가 필요하지만, 현재 `main.py` 기반 `test_ood` pipeline에서는 이 wrapper가 적용되지 않는다.
- 명확한 AdaScale 비교가 필요하면 `evaluation_api` 쪽 구현처럼 wrapper를 적용하는 별도 실행 경로를 추가해야 한다.

### ImageNet Train Feature 의존 방법

- ImageNet / ImageNet-200의 `MDS`, `RMDS`, `KNN`, `DICE`, `Gram`, `SHE`는 ID train loader를 사용해 feature statistics 또는 activation prototype을 만든다.
- 현재 로컬 ImageNet train imglist가 실제 파일과 맞지 않아 setup 초반에 `FileNotFoundError`가 발생한다.
- 확인된 예시는 `./data/images_largescale/imagenet_1k/train/n02110958/n02110958_6259.JPEG`, `./data/images_largescale/imagenet_1k/train/n02206856/n02206856_22203.JPEG` 등이다.
- 이 항목들을 실행하려면 ImageNet train 데이터와 `train_imagenet.txt`, `train_imagenet200.txt`의 경로 일치를 먼저 복구하거나, 존재하는 파일만 포함하도록 train imglist를 재생성해야 한다.

## 체크리스트

| Dataset | Method | GPU | Runtime | main OOD | main FSOOD | eval OOD | eval FSOOD | OOD Near AUROC | OOD Far AUROC | FSOOD Near AUROC | FSOOD Far AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | MSP | 0 | 0:42.04 | pass | pass | - | - | 93.07 | 99.06 | 31.27 | 48.96 |
| MNIST | MLS | 1 | 0:42.12 | pass | pass | - | - | 93.63 | 99.44 | 29.07 | 47.26 |
| MNIST | EBO | 0 | 0:42.93 | pass | pass | - | - | 93.62 | 99.46 | 28.44 | 45.87 |
| MNIST | ODIN | 1 | 0:51.81 | pass | pass | - | - | 93.63 | 99.41 | 29.19 | 47.32 |
| MNIST | IODIN | 0 | 0:51.88 | pass | pass | - | - | 93.71 | 99.44 | 29.07 | 47.26 |
| MNIST | GradNorm | 1 | 1:57.59 | pass | pass | - | - | 90.12 | 99.45 | 26.66 | 45.67 |
| MNIST | MDS | 0 | 0:59.38 | pass | pass | - | - | 71.23 | 46.66 | 76.31 | 41.79 |
| MNIST | RMDS | 1 | 0:58.76 | pass | pass | - | - | 95.96 | 98.28 | 43.52 | 45.70 |
| MNIST | KNN | 0 | 2:05.94 | pass | pass | - | - | 95.07 | 97.10 | 64.83 | 68.38 |
| MNIST | ViM | - | - | skip | skip | - | - | - | - | - | - |
| MNIST | Residual | - | - | skip | skip | - | - | - | - | - | - |
| MNIST | ReAct | 1 | 0:48.63 | pass | pass | - | - | 96.54 | 99.42 | 29.49 | 45.74 |
| MNIST | ASH | 0 | 0:47.77 | pass | pass | - | - | 93.61 | 99.46 | 28.44 | 45.87 |
| MNIST | DICE | 1 | 0:49.89 | pass | pass | - | - | 87.04 | 98.57 | 26.33 | 46.95 |
| MNIST | Gram | 0 | 1:38.86 | pass | pass | - | - | 99.20 | 100.00 | 25.87 | 47.63 |
| MNIST | KLM | 1 | 1:08.14 | pass | pass | - | - | 87.85 | 98.60 | 31.95 | 47.45 |
| MNIST | SHE | 0 | 0:47.89 | pass | pass | - | - | 86.63 | 99.19 | 27.60 | 50.71 |
| MNIST | SCALE | 1 | 0:49.00 | pass | pass | - | - | 93.62 | 99.46 | 28.44 | 45.87 |
| MNIST | AdaScale-A | - | - | skip | skip | - | - | - | - | - | - |
| MNIST | AdaScale-L | - | - | skip | skip | - | - | - | - | - | - |
| CIFAR-10 | MSP | 0 | 1:06.13 | pass | pass | pass | pass | 87.68 | 91.00 | 77.24 | 81.20 |
| CIFAR-10 | MLS | 1 | 1:04.87 | pass | pass | pass | pass | 86.86 | 91.61 | 77.26 | 83.05 |
| CIFAR-10 | EBO | 0 | 1:08.01 | pass | pass | pass | pass | 86.93 | 91.74 | 77.39 | 83.34 |
| CIFAR-10 | ODIN | 1 | 2:02.01 | pass | pass | pass | pass | 80.26 | 87.21 | 68.44 | 77.91 |
| CIFAR-10 | IODIN | 0 | 2:00.46 | pass | pass | pass | pass | 87.67 | 91.01 | 77.24 | 83.04 |
| CIFAR-10 | GradNorm | 1 | 3:44.55 | pass | pass | pass | pass | 53.77 | 58.56 | 51.93 | 56.59 |
| CIFAR-10 | MDS | 0 | 1:57.00 | pass | pass | pass | pass | 82.84 | 83.67 | 72.67 | 73.43 |
| CIFAR-10 | RMDS | 1 | 1:58.12 | pass | pass | pass | pass | 88.77 | 91.78 | 77.80 | 81.00 |
| CIFAR-10 | KNN | 0 | 12:52.14 | pass | pass | pass | pass | 90.56 | 92.89 | 79.46 | 82.52 |
| CIFAR-10 | ViM | - | - | skip | skip | - | - | - | - | - | - |
| CIFAR-10 | Residual | - | - | skip | skip | - | - | - | - | - | - |
| CIFAR-10 | ReAct | 1 | 3:32.38 | pass | pass | pass | pass | 86.47 | 91.02 | 71.26 | 74.31 |
| CIFAR-10 | ASH | 0 | 2:26.43 | pass | pass | pass | pass | 86.93 | 91.74 | 77.39 | 83.34 |
| CIFAR-10 | DICE | 1 | 2:23.82 | pass | pass | pass | pass | 77.82 | 85.57 | 72.06 | 80.83 |
| CIFAR-10 | Gram | 0 | 15:52.03 | pass | pass | pass | pass | 80.47 | 92.23 | 64.99 | 84.09 |
| CIFAR-10 | KLM | 1 | 4:24.36 | pass | pass | pass | pass | 78.80 | 82.76 | 71.57 | 75.51 |
| CIFAR-10 | SHE | 0 | 2:23.08 | pass | pass | pass | pass | 81.18 | 86.91 | 73.25 | 79.72 |
| CIFAR-10 | SCALE | 1 | 2:27.37 | pass | pass | pass | pass | 86.93 | 91.74 | 77.39 | 83.34 |
| CIFAR-10 | AdaScale-A | - | - | skip | skip | - | - | - | - | - | - |
| CIFAR-10 | AdaScale-L | - | - | skip | skip | - | - | - | - | - | - |
| CIFAR-100 | MSP | 0 | 1:02.16 | pass | pass | pass | pass | 80.42 | 77.58 | 65.05 | 61.35 |
| CIFAR-100 | MLS | 1 | 1:02.31 | pass | pass | pass | pass | 81.04 | 79.60 | 64.64 | 62.36 |
| CIFAR-100 | EBO | 0 | 2:00.29 | pass | pass | pass | pass | 80.83 | 79.71 | 64.28 | 62.46 |
| CIFAR-100 | ODIN | 1 | 4:00.20 | pass | pass | pass | pass | 79.80 | 79.44 | 63.08 | 62.55 |
| CIFAR-100 | IODIN | 0 | 4:11.64 | pass | pass | pass | pass | 81.09 | 79.60 | 64.64 | 62.35 |
| CIFAR-100 | GradNorm | 1 | 8:00.95 | pass | pass | pass | pass | 69.73 | 68.82 | 60.15 | 59.33 |
| CIFAR-100 | MDS | 0 | 9:58.25 | pass | pass | pass | pass | 55.57 | 67.04 | 49.07 | 60.50 |
| CIFAR-100 | RMDS | 1 | 10:02.91 | pass | pass | pass | pass | 80.14 | 81.11 | 63.83 | 65.73 |
| CIFAR-100 | KNN | 0 | 25:37.70 | pass | pass | pass | pass | 80.26 | 81.86 | 62.18 | 64.12 |
| CIFAR-100 | ViM | - | - | skip | skip | - | - | - | - | - | - |
| CIFAR-100 | Residual | - | - | skip | skip | - | - | - | - | - | - |
| CIFAR-100 | ReAct | 1 | 3:51.95 | pass | pass | pass | pass | 80.70 | 79.84 | 59.76 | 61.02 |
| CIFAR-100 | ASH | 0 | 2:33.28 | pass | pass | pass | pass | 80.83 | 79.71 | 64.28 | 62.46 |
| CIFAR-100 | DICE | 1 | 2:29.14 | pass | pass | pass | pass | 79.17 | 79.82 | 64.33 | 65.31 |
| CIFAR-100 | Gram | 0 | 57:32.80 | pass | pass | pass | pass | 69.16 | 78.47 | 45.49 | 58.35 |
| CIFAR-100 | KLM | 1 | 23:42.47 | pass | pass | pass | pass | 76.90 | 76.03 | 62.90 | 62.46 |
| CIFAR-100 | SHE | 0 | 2:20.57 | pass | pass | pass | pass | 78.64 | 78.28 | 64.26 | 64.13 |
| CIFAR-100 | SCALE | 1 | 2:31.07 | pass | pass | pass | pass | 80.83 | 79.71 | 64.28 | 62.46 |
| CIFAR-100 | AdaScale-A | - | - | skip | skip | - | - | - | - | - | - |
| CIFAR-100 | AdaScale-L | - | - | skip | skip | - | - | - | - | - | - |
| ImageNet | MSP | 0 | 17:23.50 | pass | pass | pass | pass | 76.02 | 85.23 | 60.56 | 72.07 |
| ImageNet | MLS | 1 | 17:19.50 | pass | pass | pass | pass | 76.45 | 89.57 | 57.24 | 73.60 |
| ImageNet | EBO | 0 | 17:42.95 | pass | pass | pass | pass | 75.89 | 89.46 | 56.35 | 73.07 |
| ImageNet | ODIN | 1 | 37:31.95 | pass | pass | pass | pass | 74.75 | 89.46 | 57.18 | 75.44 |
| ImageNet | IODIN | 0 | 37:30.02 | pass | pass | pass | pass | 76.53 | 89.62 | 57.24 | 73.61 |
| ImageNet | GradNorm | 1 | 20:03.05 | pass | pass | pass | pass | 72.95 | 90.25 | 62.18 | 83.01 |
| ImageNet | MDS | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet | RMDS | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet | KNN | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet | ViM | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet | Residual | 0 | 29:08.06 | pass | pass | pass | pass | 48.06 | 66.40 | 42.84 | 62.54 |
| ImageNet | ReAct | 1 | 25:01.34 | pass | pass | pass | pass | 77.37 | 93.66 | 60.46 | 82.90 |
| ImageNet | ASH | 0 | 20:56.65 | pass | pass | pass | pass | 75.89 | 89.46 | 56.35 | 73.07 |
| ImageNet | DICE | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet | Gram | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet | KLM | 1 | 4:33:17 | pass | pass | pass | pass | 76.64 | 87.60 | 61.91 | 75.31 |
| ImageNet | SHE | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet | SCALE | 1 | 20:51.52 | pass | pass | pass | pass | 75.89 | 89.46 | 56.35 | 73.07 |
| ImageNet | AdaScale-A | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet | AdaScale-L | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet-200 | MSP | 0 | 15:27.92 | pass | pass | pass | pass | 83.30 | 90.20 | 53.94 | 66.00 |
| ImageNet-200 | MLS | 1 | 15:26.97 | pass | pass | pass | pass | 82.96 | 91.34 | 51.07 | 65.57 |
| ImageNet-200 | EBO | 0 | 15:54.44 | pass | pass | pass | pass | 82.57 | 91.12 | 50.38 | 65.08 |
| ImageNet-200 | ODIN | 1 | 21:26.18 | pass | pass | pass | pass | 80.32 | 91.89 | 48.18 | 66.62 |
| ImageNet-200 | IODIN | 0 | 21:48.98 | pass | pass | pass | pass | 83.05 | 91.38 | 51.07 | 65.57 |
| ImageNet-200 | GradNorm | 1 | 17:02.83 | pass | pass | pass | pass | 73.33 | 85.29 | 54.11 | 69.52 |
| ImageNet-200 | MDS | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet-200 | RMDS | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet-200 | KNN | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet-200 | ViM | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet-200 | Residual | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet-200 | ReAct | 1 | 19:14.30 | pass | pass | pass | pass | 80.48 | 93.10 | 51.08 | 73.68 |
| ImageNet-200 | ASH | 0 | 18:09.14 | pass | pass | pass | pass | 82.57 | 91.12 | 50.38 | 65.08 |
| ImageNet-200 | DICE | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet-200 | Gram | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet-200 | KLM | 1 | 43:54.54 | pass | pass | pass | pass | 80.69 | 88.41 | 56.88 | 67.85 |
| ImageNet-200 | SHE | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet-200 | SCALE | 1 | 18:15.06 | pass | pass | pass | pass | 82.57 | 91.12 | 50.38 | 65.08 |
| ImageNet-200 | AdaScale-A | - | - | skip | skip | skip | skip | - | - | - | - |
| ImageNet-200 | AdaScale-L | - | - | skip | skip | skip | skip | - | - | - | - |
