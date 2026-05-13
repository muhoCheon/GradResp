# MNIST Reproduction

MNIST 학습, ID 테스트, OOD/FSOOD 평가를 로컬에서 재현하기 위한 명령 모음이다.

참고: https://github.com/Jingkang50/OpenOOD/wiki/0.-Get-Started

기존 OpenOOD shell script는 HPC/클러스터 실행을 위한 Slurm/srun 기반으로 작성된 경우가 있다.
개인 재현용 OOD 평가 스크립트는 `scripts_my/` 아래에 별도로 둔다.

## 학습

```bash
sh scripts/basics/mnist/train_mnist_local.sh
```

## ID 테스트

```bash
sh scripts/basics/mnist/test_mnist_local.sh
```

## Classic OOD Benchmark: MSP

먼저 `notmnist`, `fashionmnist`가 없다면 다운로드한다.

```bash
python ./scripts_my/download/download.py \
  --contents datasets \
  --datasets notmnist fashionmnist \
  --dataset_mode dataset \
  --save_dir ./data ./results
```

MSP 평가:

```bash
sh scripts_my/ood/msp/mnist_test_ood_msp.sh
```

## Full-Spectrum OOD Benchmark: MSP

csID 데이터는 `svhn` 하나다.

```bash
sh scripts_my/ood/msp/mnist_test_fsood_msp.sh
```

기존 output directory가 있으면 프롬프트가 뜰 수 있으므로, 필요하면 스크립트에 다음 옵션을 추가한다.

```bash
--merge_option merge
```

## MDS on MNIST

MNIST에서 MDS를 실행하려면 `num_classes_dict`에 `mnist`가 있어야 한다.
관련 이슈는 [../notes/known_issues.md](../notes/known_issues.md)를 참고한다.

```bash
sh scripts_my/ood/mds/mnist_test_ood_mds.sh
sh scripts_my/ood/mds/mnist_test_fsood_mds.sh
```
