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
ImageNet train imglist/path mismatch warning is resolved for the completed validation rows in this document. Train-feature methods that still show `pending` are pending because they have not been scheduled or completed in this validation batch, not because of the old train imglist mismatch.
For ImageNet/ImageNet-200 FSOOD, csID consists of ImageNet-V2, ImageNet-C, and ImageNet-R. `main.py` config names use `imagenetv2`, `imagenetc`, `imagenetr`; `eval_api` result names use `imagenet_v2`, `imagenet_c`, `imagenet_r`.

metadata를 기준으로 체크리스트 갱신:

```bash
python scripts_my/tools/update_group1_validation.py
```

`results_test/runs/*/*/metadata.md`를 읽어서 아래 marker 구간의 checklist, main.py 결과 표, eval_api 결과 표를 갱신한다.

체크리스트 표 초기화:

```bash
python scripts_my/tools/update_group1_validation.py --reset
```

아래 체크리스트 표를 초기 `pending` 상태로 되돌린다.
`results_test/runs/`의 로그나 metadata 파일은 삭제하지 않는다.

Group 1 score density plot 생성:

```bash
python scripts_my/tools/plot_score_density.py \
  --source openood \
  --dataset cifar10 \
  --method msp \
  --scheme all
```

특정 scheme만 확인할 수도 있다.

```bash
python scripts_my/tools/plot_score_density.py \
  --source openood \
  --dataset cifar10 \
  --method msp \
  --scheme fsood
```

score가 저장된 모든 Group 1 조합에 대해 plot 생성을 시도하려면 다음 명령을 사용한다.

```bash
python scripts_my/tools/plot_score_density.py \
  --source openood \
  --dataset all \
  --method all \
  --scheme all
```

출력:

```text
results_test/plots/score_density/<dataset>/<method>/<scheme>/per_dataset.png
results_test/plots/score_density/<dataset>/<method>/<scheme>/combined_eval_groups.png
results_test/plots/score_density/<dataset>/<method>/<scheme>/combined_id_groups.png
```

Plot은 OpenOOD convention에 맞춰 `ood_score = -conf`를 사용한다. x축 값이 클수록 더 OOD-like다.
score가 없는 조합은 warning을 출력하고 건너뛴다.
random model sanity check 결과용 plot은 `docs_my/experiments/random_sanity.md`를 따른다.

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
- ImageNet: train imglist/path issue is resolved for the completed validation row; ViM is no longer skipped for ImageNet in the current table.
- ImageNet-200: skip is due to the ViM sweep containing `dim=1000`, which does not match the ResNet18 feature dimension 512. It is not attributed to the old train imglist mismatch.

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
- The earlier local ImageNet train imglist/path mismatch has been resolved for the completed rows. Do not carry forward the old `FileNotFoundError` warning as the reason for pending ImageNet-scale rows.
- Current pending rows should be treated as normal execution backlog unless a fresh run log records a new data/setup failure.
- If a new failure appears, record the exact missing path and update the relevant method/dataset metadata rather than reinstating the old blanket mismatch warning.

## 체크리스트

<!-- GROUP1_CHECKLIST:BEGIN -->
| Dataset | Method | GPU | Runtime | main OOD | main FSOOD | eval_api OOD | eval_api FSOOD |
|---|---|---|---|---|---|---|---|
| MNIST | MSP | 0 | 0:42.04 | pass | pass | - | - |
| MNIST | MLS | 1 | 0:42.12 | pass | pass | - | - |
| MNIST | EBO | 0 | 0:42.93 | pass | pass | - | - |
| MNIST | ODIN | 1 | 0:51.81 | pass | pass | - | - |
| MNIST | IODIN | 0 | 0:51.88 | pass | pass | - | - |
| MNIST | GradNorm | 1 | 1:57.59 | pass | pass | - | - |
| MNIST | MDS | 0 | 0:59.38 | pass | pass | - | - |
| MNIST | RMDS | 1 | 0:58.76 | pass | pass | - | - |
| MNIST | KNN | 0 | 2:05.94 | pass | pass | - | - |
| MNIST | ViM | - | - | skip | skip | - | - |
| MNIST | Residual | - | - | skip | skip | - | - |
| MNIST | ReAct | 1 | 0:48.63 | pass | pass | - | - |
| MNIST | ASH | 0 | 0:47.77 | pass | pass | - | - |
| MNIST | DICE | 1 | 0:49.89 | pass | pass | - | - |
| MNIST | Gram | 0 | 1:38.86 | pass | pass | - | - |
| MNIST | KLM | 1 | 1:08.14 | pass | pass | - | - |
| MNIST | SHE | 0 | 0:47.89 | pass | pass | - | - |
| MNIST | SCALE | 1 | 0:49.00 | pass | pass | - | - |
| MNIST | AdaScale-A | - | - | skip | skip | - | - |
| MNIST | AdaScale-L | - | - | skip | skip | - | - |
| CIFAR-10 | MSP | 0 | 1:06.13 | pass | pass | pass | pass |
| CIFAR-10 | MLS | 1 | 1:04.87 | pass | pass | pass | pass |
| CIFAR-10 | EBO | 0 | 1:08.01 | pass | pass | pass | pass |
| CIFAR-10 | ODIN | 1 | 2:02.01 | pass | pass | pass | pass |
| CIFAR-10 | IODIN | 0 | 2:00.46 | pass | pass | pass | pass |
| CIFAR-10 | GradNorm | 1 | 3:44.55 | pass | pass | pass | pass |
| CIFAR-10 | MDS | 0 | 1:57.00 | pass | pass | pass | pass |
| CIFAR-10 | RMDS | 1 | 1:58.12 | pass | pass | pass | pass |
| CIFAR-10 | KNN | 0 | 12:52.14 | pass | pass | pass | pass |
| CIFAR-10 | ViM | - | - | skip | skip | - | - |
| CIFAR-10 | Residual | - | - | skip | skip | - | - |
| CIFAR-10 | ReAct | 1 | 3:32.38 | pass | pass | pass | pass |
| CIFAR-10 | ASH | 0 | 2:26.43 | pass | pass | pass | pass |
| CIFAR-10 | DICE | 1 | 2:23.82 | pass | pass | pass | pass |
| CIFAR-10 | Gram | 0 | 15:52.03 | pass | pass | pass | pass |
| CIFAR-10 | KLM | 1 | 4:24.36 | pass | pass | pass | pass |
| CIFAR-10 | SHE | 0 | 2:23.08 | pass | pass | pass | pass |
| CIFAR-10 | SCALE | 1 | 2:27.37 | pass | pass | pass | pass |
| CIFAR-10 | AdaScale-A | - | - | skip | skip | - | - |
| CIFAR-10 | AdaScale-L | - | - | skip | skip | - | - |
| CIFAR-100 | MSP | 0 | 1:02.16 | pass | pass | pass | pass |
| CIFAR-100 | MLS | 1 | 1:02.31 | pass | pass | pass | pass |
| CIFAR-100 | EBO | 0 | 2:00.29 | pass | pass | pass | pass |
| CIFAR-100 | ODIN | 1 | 4:00.20 | pass | pass | pass | pass |
| CIFAR-100 | IODIN | 0 | 4:11.64 | pass | pass | pass | pass |
| CIFAR-100 | GradNorm | 1 | 8:00.95 | pass | pass | pass | pass |
| CIFAR-100 | MDS | 0 | 9:58.25 | pass | pass | pass | pass |
| CIFAR-100 | RMDS | 1 | 10:02.91 | pass | pass | pass | pass |
| CIFAR-100 | KNN | 0 | 25:37.70 | pass | pass | pass | pass |
| CIFAR-100 | ViM | - | - | skip | skip | - | - |
| CIFAR-100 | Residual | - | - | skip | skip | - | - |
| CIFAR-100 | ReAct | 1 | 3:51.95 | pass | pass | pass | pass |
| CIFAR-100 | ASH | 0 | 2:33.28 | pass | pass | pass | pass |
| CIFAR-100 | DICE | 1 | 2:29.14 | pass | pass | pass | pass |
| CIFAR-100 | Gram | 0 | 57:32.80 | pass | pass | pass | pass |
| CIFAR-100 | KLM | 1 | 23:42.47 | pass | pass | pass | pass |
| CIFAR-100 | SHE | 0 | 2:20.57 | pass | pass | pass | pass |
| CIFAR-100 | SCALE | 1 | 2:31.07 | pass | pass | pass | pass |
| CIFAR-100 | AdaScale-A | - | - | skip | skip | - | - |
| CIFAR-100 | AdaScale-L | - | - | skip | skip | - | - |
| ImageNet | MSP | 0 | 17:23.50 | pass | pass | pass | pass |
| ImageNet | MLS | 1 | 17:19.50 | pass | pass | pass | pass |
| ImageNet | EBO | 0 | 17:42.95 | pass | pass | pass | pass |
| ImageNet | ODIN | 1 | 37:31.95 | pass | pass | pass | pass |
| ImageNet | IODIN | 0 | 37:30.02 | pass | pass | pass | pass |
| ImageNet | GradNorm | 1 | 20:03.05 | pass | pass | pass | pass |
| ImageNet | MDS | 0 | 4:26:17 | pass | pass | pass | pass |
| ImageNet | RMDS | 1 | 4:22:04 | pass | pass | pass | pass |
| ImageNet | KNN | - | - | pending | pending | pending | pending |
| ImageNet | ViM | 1 | 2:12:46 | pass | pass | pass | pass |
| ImageNet | Residual | 0 | 29:08.06 | pass | pass | pass | pass |
| ImageNet | ReAct | 1 | 25:01.34 | pass | pass | pass | pass |
| ImageNet | ASH | 0 | 20:56.65 | pass | pass | pass | pass |
| ImageNet | DICE | - | - | pending | pending | pending | pending |
| ImageNet | Gram | - | - | pending | pending | pending | pending |
| ImageNet | KLM | 1 | 4:33:17 | pass | pass | pass | pass |
| ImageNet | SHE | - | - | pending | pending | pending | pending |
| ImageNet | SCALE | 1 | 20:51.52 | pass | pass | pass | pass |
| ImageNet | AdaScale-A | - | - | skip | skip | skip | skip |
| ImageNet | AdaScale-L | - | - | skip | skip | skip | skip |
| ImageNet-200 | MSP | 0 | 15:27.92 | pass | pass | pass | pass |
| ImageNet-200 | MLS | 1 | 15:26.97 | pass | pass | pass | pass |
| ImageNet-200 | EBO | 0 | 15:54.44 | pass | pass | pass | pass |
| ImageNet-200 | ODIN | 1 | 21:26.18 | pass | pass | pass | pass |
| ImageNet-200 | IODIN | 0 | 21:48.98 | pass | pass | pass | pass |
| ImageNet-200 | GradNorm | 1 | 17:02.83 | pass | pass | pass | pass |
| ImageNet-200 | MDS | - | - | pending | pending | pending | pending |
| ImageNet-200 | RMDS | - | - | pending | pending | pending | pending |
| ImageNet-200 | KNN | - | - | pending | pending | pending | pending |
| ImageNet-200 | ViM | - | - | skip | skip | skip | skip |
| ImageNet-200 | Residual | - | - | skip | skip | skip | skip |
| ImageNet-200 | ReAct | 1 | 19:14.30 | pass | pass | pass | pass |
| ImageNet-200 | ASH | 0 | 18:09.14 | pass | pass | pass | pass |
| ImageNet-200 | DICE | - | - | pending | pending | pending | pending |
| ImageNet-200 | Gram | - | - | pending | pending | pending | pending |
| ImageNet-200 | KLM | 1 | 43:54.54 | pass | pass | pass | pass |
| ImageNet-200 | SHE | - | - | pending | pending | pending | pending |
| ImageNet-200 | SCALE | 1 | 18:15.06 | pass | pass | pass | pass |
| ImageNet-200 | AdaScale-A | - | - | skip | skip | skip | skip |
| ImageNet-200 | AdaScale-L | - | - | skip | skip | skip | skip |
<!-- GROUP1_CHECKLIST:END -->

## main.py 결과

<!-- GROUP1_MAIN_RESULTS:BEGIN -->
| Dataset | Method | OOD Near AUROC | OOD Near FPR95 | OOD Far AUROC | OOD Far FPR95 | FSOOD Near AUROC | FSOOD Near FPR95 | FSOOD Far AUROC | FSOOD Far FPR95 |
|---|---|---|---|---|---|---|---|---|---|
| MNIST | MSP | 93.07 | - | 99.06 | - | 31.27 | - | 48.96 | - |
| MNIST | MLS | 93.63 | - | 99.44 | - | 29.07 | - | 47.26 | - |
| MNIST | EBO | 93.62 | - | 99.46 | - | 28.44 | - | 45.87 | - |
| MNIST | ODIN | 93.63 | - | 99.41 | - | 29.19 | - | 47.32 | - |
| MNIST | IODIN | 93.71 | - | 99.44 | - | 29.07 | - | 47.26 | - |
| MNIST | GradNorm | 90.12 | - | 99.45 | - | 26.66 | - | 45.67 | - |
| MNIST | MDS | 71.23 | - | 46.66 | - | 76.31 | - | 41.79 | - |
| MNIST | RMDS | 95.96 | - | 98.28 | - | 43.52 | - | 45.70 | - |
| MNIST | KNN | 95.07 | - | 97.10 | - | 64.83 | - | 68.38 | - |
| MNIST | ViM | - | - | - | - | - | - | - | - |
| MNIST | Residual | - | - | - | - | - | - | - | - |
| MNIST | ReAct | 96.54 | - | 99.42 | - | 29.49 | - | 45.74 | - |
| MNIST | ASH | 93.61 | - | 99.46 | - | 28.44 | - | 45.87 | - |
| MNIST | DICE | 87.04 | - | 98.57 | - | 26.33 | - | 46.95 | - |
| MNIST | Gram | 99.20 | - | 100.00 | - | 25.87 | - | 47.63 | - |
| MNIST | KLM | 87.85 | - | 98.60 | - | 31.95 | - | 47.45 | - |
| MNIST | SHE | 86.63 | - | 99.19 | - | 27.60 | - | 50.71 | - |
| MNIST | SCALE | 93.62 | - | 99.46 | - | 28.44 | - | 45.87 | - |
| MNIST | AdaScale-A | - | - | - | - | - | - | - | - |
| MNIST | AdaScale-L | - | - | - | - | - | - | - | - |
| CIFAR-10 | MSP | 87.68 | - | 91.00 | - | 77.24 | - | 81.20 | - |
| CIFAR-10 | MLS | 86.86 | - | 91.61 | - | 77.26 | - | 83.05 | - |
| CIFAR-10 | EBO | 86.93 | - | 91.74 | - | 77.39 | - | 83.34 | - |
| CIFAR-10 | ODIN | 80.26 | - | 87.21 | - | 68.44 | - | 77.91 | - |
| CIFAR-10 | IODIN | 87.67 | - | 91.01 | - | 77.24 | - | 83.04 | - |
| CIFAR-10 | GradNorm | 53.77 | - | 58.56 | - | 51.93 | - | 56.59 | - |
| CIFAR-10 | MDS | 82.84 | - | 83.67 | - | 72.67 | - | 73.43 | - |
| CIFAR-10 | RMDS | 88.77 | - | 91.78 | - | 77.80 | - | 81.00 | - |
| CIFAR-10 | KNN | 90.56 | - | 92.89 | - | 79.46 | - | 82.52 | - |
| CIFAR-10 | ViM | - | - | - | - | - | - | - | - |
| CIFAR-10 | Residual | - | - | - | - | - | - | - | - |
| CIFAR-10 | ReAct | 86.47 | - | 91.02 | - | 71.26 | - | 74.31 | - |
| CIFAR-10 | ASH | 86.93 | - | 91.74 | - | 77.39 | - | 83.34 | - |
| CIFAR-10 | DICE | 77.82 | - | 85.57 | - | 72.06 | - | 80.83 | - |
| CIFAR-10 | Gram | 80.47 | - | 92.23 | - | 64.99 | - | 84.09 | - |
| CIFAR-10 | KLM | 78.80 | - | 82.76 | - | 71.57 | - | 75.51 | - |
| CIFAR-10 | SHE | 81.18 | - | 86.91 | - | 73.25 | - | 79.72 | - |
| CIFAR-10 | SCALE | 86.93 | - | 91.74 | - | 77.39 | - | 83.34 | - |
| CIFAR-10 | AdaScale-A | - | - | - | - | - | - | - | - |
| CIFAR-10 | AdaScale-L | - | - | - | - | - | - | - | - |
| CIFAR-100 | MSP | 80.42 | - | 77.58 | - | 65.05 | - | 61.35 | - |
| CIFAR-100 | MLS | 81.04 | - | 79.60 | - | 64.64 | - | 62.36 | - |
| CIFAR-100 | EBO | 80.83 | - | 79.71 | - | 64.28 | - | 62.46 | - |
| CIFAR-100 | ODIN | 79.80 | - | 79.44 | - | 63.08 | - | 62.55 | - |
| CIFAR-100 | IODIN | 81.09 | - | 79.60 | - | 64.64 | - | 62.35 | - |
| CIFAR-100 | GradNorm | 69.73 | - | 68.82 | - | 60.15 | - | 59.33 | - |
| CIFAR-100 | MDS | 55.57 | - | 67.04 | - | 49.07 | - | 60.50 | - |
| CIFAR-100 | RMDS | 80.14 | - | 81.11 | - | 63.83 | - | 65.73 | - |
| CIFAR-100 | KNN | 80.26 | - | 81.86 | - | 62.18 | - | 64.12 | - |
| CIFAR-100 | ViM | - | - | - | - | - | - | - | - |
| CIFAR-100 | Residual | - | - | - | - | - | - | - | - |
| CIFAR-100 | ReAct | 80.70 | - | 79.84 | - | 59.76 | - | 61.02 | - |
| CIFAR-100 | ASH | 80.83 | - | 79.71 | - | 64.28 | - | 62.46 | - |
| CIFAR-100 | DICE | 79.17 | - | 79.82 | - | 64.33 | - | 65.31 | - |
| CIFAR-100 | Gram | 69.16 | - | 78.47 | - | 45.49 | - | 58.35 | - |
| CIFAR-100 | KLM | 76.90 | - | 76.03 | - | 62.90 | - | 62.46 | - |
| CIFAR-100 | SHE | 78.64 | - | 78.28 | - | 64.26 | - | 64.13 | - |
| CIFAR-100 | SCALE | 80.83 | - | 79.71 | - | 64.28 | - | 62.46 | - |
| CIFAR-100 | AdaScale-A | - | - | - | - | - | - | - | - |
| CIFAR-100 | AdaScale-L | - | - | - | - | - | - | - | - |
| ImageNet | MSP | 76.02 | - | 85.23 | - | 60.56 | - | 72.07 | - |
| ImageNet | MLS | 76.45 | - | 89.57 | - | 57.24 | - | 73.60 | - |
| ImageNet | EBO | 75.89 | - | 89.46 | - | 56.35 | - | 73.07 | - |
| ImageNet | ODIN | 74.75 | - | 89.46 | - | 57.18 | - | 75.44 | - |
| ImageNet | IODIN | 76.53 | - | 89.62 | - | 57.24 | - | 73.61 | - |
| ImageNet | GradNorm | 72.95 | - | 90.25 | - | 62.18 | - | 83.01 | - |
| ImageNet | MDS | 50.42 | - | 67.56 | - | 43.66 | - | 63.08 | - |
| ImageNet | RMDS | 77.27 | - | 87.16 | - | 61.69 | - | 74.13 | - |
| ImageNet | KNN | - | - | - | - | - | - | - | - |
| ImageNet | ViM | 71.73 | - | 92.00 | - | 51.60 | - | 75.37 | - |
| ImageNet | Residual | 48.06 | - | 66.40 | - | 42.84 | - | 62.54 | - |
| ImageNet | ReAct | 77.37 | - | 93.66 | - | 60.46 | - | 82.90 | - |
| ImageNet | ASH | 75.89 | - | 89.46 | - | 56.35 | - | 73.07 | - |
| ImageNet | DICE | - | - | - | - | - | - | - | - |
| ImageNet | Gram | - | - | - | - | - | - | - | - |
| ImageNet | KLM | 76.64 | - | 87.60 | - | 61.91 | - | 75.31 | - |
| ImageNet | SHE | - | - | - | - | - | - | - | - |
| ImageNet | SCALE | 75.89 | - | 89.46 | - | 56.35 | - | 73.07 | - |
| ImageNet | AdaScale-A | - | - | - | - | - | - | - | - |
| ImageNet | AdaScale-L | - | - | - | - | - | - | - | - |
| ImageNet-200 | MSP | 83.30 | - | 90.20 | - | 53.94 | - | 66.00 | - |
| ImageNet-200 | MLS | 82.96 | - | 91.34 | - | 51.07 | - | 65.57 | - |
| ImageNet-200 | EBO | 82.57 | - | 91.12 | - | 50.38 | - | 65.08 | - |
| ImageNet-200 | ODIN | 80.32 | - | 91.89 | - | 48.18 | - | 66.62 | - |
| ImageNet-200 | IODIN | 83.05 | - | 91.38 | - | 51.07 | - | 65.57 | - |
| ImageNet-200 | GradNorm | 73.33 | - | 85.29 | - | 54.11 | - | 69.52 | - |
| ImageNet-200 | MDS | - | - | - | - | - | - | - | - |
| ImageNet-200 | RMDS | - | - | - | - | - | - | - | - |
| ImageNet-200 | KNN | - | - | - | - | - | - | - | - |
| ImageNet-200 | ViM | - | - | - | - | - | - | - | - |
| ImageNet-200 | Residual | - | - | - | - | - | - | - | - |
| ImageNet-200 | ReAct | 80.48 | - | 93.10 | - | 51.08 | - | 73.68 | - |
| ImageNet-200 | ASH | 82.57 | - | 91.12 | - | 50.38 | - | 65.08 | - |
| ImageNet-200 | DICE | - | - | - | - | - | - | - | - |
| ImageNet-200 | Gram | - | - | - | - | - | - | - | - |
| ImageNet-200 | KLM | 80.69 | - | 88.41 | - | 56.88 | - | 67.85 | - |
| ImageNet-200 | SHE | - | - | - | - | - | - | - | - |
| ImageNet-200 | SCALE | 82.57 | - | 91.12 | - | 50.38 | - | 65.08 | - |
| ImageNet-200 | AdaScale-A | - | - | - | - | - | - | - | - |
| ImageNet-200 | AdaScale-L | - | - | - | - | - | - | - | - |
<!-- GROUP1_MAIN_RESULTS:END -->

## eval_api 결과

<!-- GROUP1_EVAL_API_RESULTS:BEGIN -->
| Dataset | Method | OOD Near AUROC | OOD Near FPR95 | OOD Far AUROC | OOD Far FPR95 | FSOOD Near AUROC | FSOOD Near FPR95 | FSOOD Far AUROC | FSOOD Far FPR95 |
|---|---|---|---|---|---|---|---|---|---|
| MNIST | MSP | - | - | - | - | - | - | - | - |
| MNIST | MLS | - | - | - | - | - | - | - | - |
| MNIST | EBO | - | - | - | - | - | - | - | - |
| MNIST | ODIN | - | - | - | - | - | - | - | - |
| MNIST | IODIN | - | - | - | - | - | - | - | - |
| MNIST | GradNorm | - | - | - | - | - | - | - | - |
| MNIST | MDS | - | - | - | - | - | - | - | - |
| MNIST | RMDS | - | - | - | - | - | - | - | - |
| MNIST | KNN | - | - | - | - | - | - | - | - |
| MNIST | ViM | - | - | - | - | - | - | - | - |
| MNIST | Residual | - | - | - | - | - | - | - | - |
| MNIST | ReAct | - | - | - | - | - | - | - | - |
| MNIST | ASH | - | - | - | - | - | - | - | - |
| MNIST | DICE | - | - | - | - | - | - | - | - |
| MNIST | Gram | - | - | - | - | - | - | - | - |
| MNIST | KLM | - | - | - | - | - | - | - | - |
| MNIST | SHE | - | - | - | - | - | - | - | - |
| MNIST | SCALE | - | - | - | - | - | - | - | - |
| MNIST | AdaScale-A | - | - | - | - | - | - | - | - |
| MNIST | AdaScale-L | - | - | - | - | - | - | - | - |
| CIFAR-10 | MSP | 88.03 ± 0.25 | 48.17 ± 3.92 | 90.73 ± 0.43 | 31.71 ± 1.83 | 75.72 ± 0.16 | 65.95 ± 2.30 | 79.23 ± 0.90 | 53.16 ± 2.02 |
| CIFAR-10 | MLS | 87.52 ± 0.47 | 61.33 ± 4.64 | 91.10 ± 0.89 | 41.67 ± 5.27 | 75.44 ± 0.24 | 74.19 ± 2.67 | 80.19 ± 1.77 | 59.38 ± 4.68 |
| CIFAR-10 | EBO | 87.58 ± 0.46 | 61.34 ± 4.64 | 91.21 ± 0.92 | 41.71 ± 5.34 | 75.50 ± 0.21 | 74.20 ± 2.67 | 80.38 ± 1.87 | 59.40 ± 4.73 |
| CIFAR-10 | ODIN | 82.87 ± 1.85 | 76.18 ± 6.04 | 87.96 ± 0.61 | 57.63 ± 4.25 | 72.36 ± 1.24 | 83.13 ± 3.78 | 78.90 ± 0.69 | 68.38 ± 3.32 |
| CIFAR-10 | IODIN | 87.92 ± 0.17 | 55.76 ± 1.78 | 90.99 ± 0.76 | 37.54 ± 6.44 | 75.64 ± 0.20 | 70.74 ± 1.49 | 79.68 ± 1.52 | 56.91 ± 5.35 |
| CIFAR-10 | GradNorm | 54.90 ± 0.98 | 94.72 ± 0.81 | 57.55 ± 3.22 | 91.90 ± 2.23 | 53.68 ± 0.95 | 95.29 ± 0.72 | 56.33 ± 3.65 | 92.40 ± 2.40 |
| CIFAR-10 | MDS | 84.20 ± 2.40 | 49.91 ± 3.98 | 89.72 ± 1.36 | 32.21 ± 3.39 | 70.27 ± 1.91 | 67.61 ± 2.83 | 77.69 ± 1.46 | 52.59 ± 2.38 |
| CIFAR-10 | RMDS | 89.80 ± 0.28 | 38.88 ± 2.38 | 92.20 ± 0.21 | 25.35 ± 0.73 | 76.81 ± 0.06 | 60.46 ± 1.26 | 79.77 ± 0.68 | 49.06 ± 1.31 |
| CIFAR-10 | KNN | 90.64 ± 0.20 | 34.00 ± 0.39 | 92.96 ± 0.14 | 24.28 ± 0.40 | 77.12 ± 0.06 | 57.71 ± 0.11 | 80.32 ± 0.61 | 48.49 ± 0.79 |
| CIFAR-10 | ViM | - | - | - | - | - | - | - | - |
| CIFAR-10 | Residual | 50.00 ± 0.00 | 100.00 ± 0.00 | 50.00 ± 0.00 | 100.00 ± 0.00 | 50.00 ± 0.00 | 100.00 ± 0.00 | 50.00 ± 0.00 | 100.00 ± 0.00 |
| CIFAR-10 | ReAct | 87.11 ± 0.61 | 63.56 ± 7.32 | 90.42 ± 1.41 | 44.90 ± 8.37 | 74.95 ± 0.57 | 75.72 ± 4.61 | 79.21 ± 2.46 | 62.09 ± 6.39 |
| CIFAR-10 | ASH | 75.27 ± 1.04 | 86.79 ± 1.81 | 78.49 ± 2.58 | 79.02 ± 4.22 | 66.95 ± 1.04 | 90.00 ± 1.46 | 70.31 ± 3.06 | 83.61 ± 3.70 |
| CIFAR-10 | DICE | 78.40 ± 0.73 | 70.11 ± 7.74 | 84.29 ± 1.92 | 51.71 ± 4.40 | 69.70 ± 0.44 | 78.90 ± 5.08 | 76.98 ± 1.94 | 62.84 ± 3.30 |
| CIFAR-10 | Gram | 85.61 ± 0.76 | 56.94 ± 1.73 | 95.64 ± 0.29 | 18.16 ± 1.00 | 65.79 ± 1.53 | 76.12 ± 1.01 | 84.44 ± 0.90 | 44.16 ± 0.97 |
| CIFAR-10 | KLM | 79.19 ± 0.80 | 87.84 ± 6.40 | 82.68 ± 0.21 | 78.31 ± 4.83 | 70.27 ± 0.55 | 89.16 ± 4.67 | 73.77 ± 0.59 | 81.79 ± 3.52 |
| CIFAR-10 | SHE | 81.54 ± 0.50 | 79.63 ± 3.47 | 85.32 ± 1.43 | 66.48 ± 5.98 | 70.98 ± 0.13 | 85.39 ± 2.00 | 75.31 ± 2.34 | 75.34 ± 4.92 |
| CIFAR-10 | SCALE | 82.55 ± 0.36 | 80.45 ± 4.02 | 86.39 ± 1.86 | 67.53 ± 7.51 | 71.69 ± 0.60 | 86.10 ± 2.92 | 75.96 ± 2.48 | 76.34 ± 5.89 |
| CIFAR-10 | AdaScale-A | - | - | - | - | - | - | - | - |
| CIFAR-10 | AdaScale-L | - | - | - | - | - | - | - | - |
| CIFAR-100 | MSP | 80.27 ± 0.11 | 54.80 ± 0.33 | 77.76 ± 0.44 | 58.70 ± 1.06 | 65.13 ± 0.07 | 73.79 ± 0.25 | 61.84 ± 0.55 | 76.47 ± 0.70 |
| CIFAR-100 | MLS | 81.05 ± 0.07 | 55.47 ± 0.66 | 79.67 ± 0.57 | 56.72 ± 1.33 | 64.84 ± 0.17 | 74.55 ± 0.41 | 62.66 ± 0.73 | 75.34 ± 0.87 |
| CIFAR-100 | EBO | 80.91 ± 0.08 | 55.61 ± 0.60 | 79.77 ± 0.61 | 56.58 ± 1.38 | 64.51 ± 0.20 | 74.66 ± 0.38 | 62.70 ± 0.82 | 75.25 ± 0.90 |
| CIFAR-100 | ODIN | 79.90 ± 0.11 | 57.92 ± 0.48 | 79.28 ± 0.21 | 58.87 ± 0.80 | 63.30 ± 0.20 | 75.65 ± 0.26 | 62.63 ± 0.25 | 76.04 ± 0.55 |
| CIFAR-100 | IODIN | 81.10 ± 0.07 | 55.33 ± 0.64 | 79.69 ± 0.56 | 56.63 ± 1.30 | 64.87 ± 0.17 | 74.47 ± 0.41 | 62.65 ± 0.72 | 75.29 ± 0.84 |
| CIFAR-100 | GradNorm | 70.13 ± 0.47 | 85.58 ± 0.47 | 69.14 ± 1.05 | 83.68 ± 1.92 | 60.06 ± 0.37 | 89.87 ± 0.51 | 59.25 ± 1.03 | 88.35 ± 1.71 |
| CIFAR-100 | MDS | 58.69 ± 0.09 | 83.52 ± 0.60 | 69.39 ± 1.39 | 72.26 ± 1.56 | 50.50 ± 0.46 | 90.03 ± 0.31 | 62.11 ± 1.96 | 81.44 ± 1.33 |
| CIFAR-100 | RMDS | 80.15 ± 0.11 | 55.46 ± 0.40 | 82.92 ± 0.42 | 52.81 ± 0.64 | 64.01 ± 0.46 | 74.54 ± 0.33 | 68.39 ± 0.97 | 72.83 ± 0.37 |
| CIFAR-100 | KNN | 80.18 ± 0.15 | 61.23 ± 0.13 | 82.40 ± 0.17 | 53.64 ± 0.28 | 62.16 ± 0.38 | 77.86 ± 0.06 | 64.97 ± 0.56 | 73.17 ± 0.25 |
| CIFAR-100 | ViM | - | - | - | - | - | - | - | - |
| CIFAR-100 | Residual | 50.00 ± 0.00 | 100.00 ± 0.00 | 50.00 ± 0.00 | 100.00 ± 0.00 | 50.00 ± 0.00 | 100.00 ± 0.00 | 50.00 ± 0.00 | 100.00 ± 0.00 |
| CIFAR-100 | ReAct | 80.77 ± 0.05 | 56.39 ± 0.33 | 80.39 ± 0.49 | 54.20 ± 1.58 | 64.27 ± 0.21 | 75.27 ± 0.20 | 63.16 ± 0.72 | 73.88 ± 0.97 |
| CIFAR-100 | ASH | 78.20 ± 0.16 | 65.72 ± 0.29 | 80.58 ± 0.66 | 59.20 ± 2.47 | 62.67 ± 0.58 | 80.02 ± 0.19 | 65.38 ± 0.41 | 75.51 ± 1.35 |
| CIFAR-100 | DICE | 79.38 ± 0.23 | 57.95 ± 0.54 | 80.01 ± 0.18 | 56.25 ± 0.59 | 64.08 ± 0.40 | 75.99 ± 0.40 | 65.06 ± 0.33 | 74.73 ± 0.54 |
| CIFAR-100 | Gram | 74.04 ± 0.44 | 72.47 ± 0.91 | 86.15 ± 0.27 | 47.40 ± 1.55 | 51.41 ± 0.40 | 85.68 ± 0.49 | 67.61 ± 0.78 | 69.99 ± 1.10 |
| CIFAR-100 | KLM | 76.57 ± 0.24 | 77.88 ± 1.43 | 76.23 ± 0.53 | 71.65 ± 2.03 | 62.91 ± 0.24 | 85.92 ± 0.86 | 62.81 ± 0.81 | 81.78 ± 1.28 |
| CIFAR-100 | SHE | 78.95 ± 0.18 | 59.07 ± 0.26 | 76.92 ± 1.15 | 64.12 ± 2.70 | 64.36 ± 0.13 | 76.02 ± 0.16 | 62.33 ± 1.36 | 79.19 ± 1.73 |
| CIFAR-100 | SCALE | 80.99 ± 0.12 | 55.67 ± 0.69 | 81.42 ± 0.43 | 54.08 ± 1.08 | 64.20 ± 0.14 | 74.74 ± 0.40 | 64.66 ± 0.52 | 73.68 ± 0.73 |
| CIFAR-100 | AdaScale-A | - | - | - | - | - | - | - | - |
| CIFAR-100 | AdaScale-L | - | - | - | - | - | - | - | - |
| ImageNet | MSP | 76.02 | 65.66 | 85.23 | 51.45 | 60.79 | 78.40 | 72.32 | 68.11 |
| ImageNet | MLS | 76.46 | 67.84 | 89.57 | 38.20 | 57.49 | 80.72 | 73.90 | 60.43 |
| ImageNet | EBO | 75.89 | 68.57 | 89.46 | 38.39 | 56.61 | 81.18 | 73.37 | 60.60 |
| ImageNet | ODIN | 74.76 | 72.48 | 89.47 | 43.95 | 57.41 | 83.19 | 75.62 | 63.28 |
| ImageNet | IODIN | 76.53 | 67.47 | 89.63 | 37.94 | 57.55 | 80.48 | 73.96 | 60.21 |
| ImageNet | GradNorm | 72.96 | 78.87 | 90.25 | 47.90 | 62.70 | 84.93 | 83.49 | 59.39 |
| ImageNet | MDS | 55.44 | 85.46 | 74.25 | 62.90 | 46.02 | 91.16 | 67.03 | 73.82 |
| ImageNet | RMDS | 76.99 | 65.04 | 86.38 | 40.90 | 62.06 | 78.68 | 73.29 | 61.59 |
| ImageNet | KNN | - | - | - | - | - | - | - | - |
| ImageNet | ViM | 72.08 | 71.35 | 92.68 | 24.66 | 52.50 | 83.35 | 80.10 | 48.81 |
| ImageNet | Residual | 48.06 | 93.77 | 66.39 | 75.68 | 42.70 | 96.01 | 62.34 | 81.11 |
| ImageNet | ReAct | 77.38 | 66.74 | 93.67 | 26.31 | 59.93 | 79.97 | 82.78 | 50.00 |
| ImageNet | ASH | 78.17 | 63.37 | 95.74 | 19.54 | 60.53 | 77.66 | 86.75 | 42.55 |
| ImageNet | DICE | - | - | - | - | - | - | - | - |
| ImageNet | Gram | - | - | - | - | - | - | - | - |
| ImageNet | KLM | 76.64 | 72.50 | 87.60 | 46.61 | 62.06 | 82.14 | 75.45 | 64.50 |
| ImageNet | SHE | - | - | - | - | - | - | - | - |
| ImageNet | SCALE | 81.36 | 59.76 | 96.53 | 16.53 | 64.67 | 75.24 | 88.85 | 37.93 |
| ImageNet | AdaScale-A | - | - | - | - | - | - | - | - |
| ImageNet | AdaScale-L | - | - | - | - | - | - | - | - |
| ImageNet-200 | MSP | 83.34 ± 0.06 | 54.83 ± 0.35 | 90.13 ± 0.09 | 35.43 ± 0.39 | 54.35 ± 0.35 | 84.19 ± 0.21 | 66.11 ± 0.37 | 73.75 ± 0.27 |
| ImageNet-200 | MLS | 82.90 ± 0.04 | 59.75 ± 0.59 | 91.11 ± 0.19 | 34.04 ± 1.21 | 51.30 ± 0.30 | 86.97 ± 0.28 | 65.24 ± 0.61 | 75.15 ± 0.78 |
| ImageNet-200 | EBO | 82.50 ± 0.05 | 60.20 ± 0.58 | 90.86 ± 0.21 | 34.85 ± 1.30 | 50.51 ± 0.30 | 87.15 ± 0.27 | 64.59 ± 0.76 | 75.71 ± 0.84 |
| ImageNet-200 | ODIN | 80.27 ± 0.08 | 66.72 ± 0.29 | 91.71 ± 0.19 | 34.26 ± 1.06 | 48.33 ± 0.05 | 89.17 ± 0.17 | 66.25 ± 0.42 | 73.57 ± 0.64 |
| ImageNet-200 | IODIN | 82.99 ± 0.04 | 59.42 ± 0.48 | 91.17 ± 0.18 | 33.76 ± 1.21 | 51.37 ± 0.30 | 86.86 ± 0.23 | 65.31 ± 0.59 | 74.96 ± 0.76 |
| ImageNet-200 | GradNorm | 72.75 ± 0.48 | 82.69 ± 0.30 | 84.26 ± 0.87 | 66.47 ± 0.22 | 54.41 ± 0.94 | 91.42 ± 0.31 | 69.01 ± 1.40 | 81.87 ± 0.44 |
| ImageNet-200 | MDS | - | - | - | - | - | - | - | - |
| ImageNet-200 | RMDS | - | - | - | - | - | - | - | - |
| ImageNet-200 | KNN | - | - | - | - | - | - | - | - |
| ImageNet-200 | ViM | - | - | - | - | - | - | - | - |
| ImageNet-200 | Residual | 50.00 ± 0.00 | 100.00 ± 0.00 | 50.00 ± 0.00 | 100.00 ± 0.00 | 50.00 ± 0.00 | 100.00 ± 0.00 | 50.00 ± 0.00 | 100.00 ± 0.00 |
| ImageNet-200 | ReAct | 81.87 ± 0.99 | 62.46 ± 2.20 | 92.31 ± 0.56 | 28.50 ± 0.95 | 50.55 ± 0.19 | 87.90 ± 0.61 | 67.73 ± 2.77 | 71.71 ± 1.54 |
| ImageNet-200 | ASH | 82.38 ± 0.19 | 64.88 ± 0.90 | 93.90 ± 0.27 | 27.29 ± 1.14 | 54.74 ± 0.73 | 87.25 ± 0.40 | 75.48 ± 0.95 | 66.96 ± 0.65 |
| ImageNet-200 | DICE | - | - | - | - | - | - | - | - |
| ImageNet-200 | Gram | - | - | - | - | - | - | - | - |
| ImageNet-200 | KLM | 80.77 ± 0.10 | 70.19 ± 0.62 | 88.52 ± 0.10 | 40.95 ± 1.09 | 57.26 ± 0.43 | 88.61 ± 0.35 | 68.54 ± 0.52 | 75.00 ± 0.54 |
| ImageNet-200 | SHE | - | - | - | - | - | - | - | - |
| ImageNet-200 | SCALE | 84.84 ± 0.28 | 57.29 ± 0.91 | 93.98 ± 0.25 | 26.46 ± 0.82 | 56.27 ± 1.20 | 85.13 ± 0.63 | 74.82 ± 1.71 | 67.95 ± 1.26 |
| ImageNet-200 | AdaScale-A | - | - | - | - | - | - | - | - |
| ImageNet-200 | AdaScale-L | - | - | - | - | - | - | - | - |
<!-- GROUP1_EVAL_API_RESULTS:END -->
