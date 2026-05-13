# Local Setup Guide

이 파일은 개인 재현 문서의 입구 역할만 한다.
세부 메모는 [docs_my](docs_my/README.md) 아래로 역할별로 분리한다.

## 관리 원칙

- `scripts/` 아래 원본 스크립트와 코드는 가능한 그대로 둔다.
- 개인 재현용 실행 스크립트와 원본에서 수정한 코드는 `scripts_my/` 아래에 둔다.
- 문서와 실험 메모는 `docs_my/` 아래에 둔다.

## 빠른 시작

```bash
conda create -n openood python=3.8 -y
conda activate openood

cd /home/dmlab/DataDrift/GradResp
pip install -e .
pip install libmr statsmodels timm foolbox==3.2.1
```

## 주요 문서

- 환경 세팅: [docs_my/setup/environment.md](docs_my/setup/environment.md)
- 데이터 다운로드: [docs_my/setup/data_download.md](docs_my/setup/data_download.md)
- MNIST 재현: [docs_my/experiments/mnist.md](docs_my/experiments/mnist.md)
- CIFAR-10 재현: [docs_my/experiments/cifar10.md](docs_my/experiments/cifar10.md)
- score 방향 convention: [docs_my/notes/score_convention.md](docs_my/notes/score_convention.md)
- known issues: [docs_my/notes/known_issues.md](docs_my/notes/known_issues.md)

## 자주 쓰는 실행

MNIST MSP OOD:

```bash
sh scripts_my/ood/msp/mnist_test_ood_msp.sh
```

MNIST MSP FSOOD:

```bash
sh scripts_my/ood/msp/mnist_test_fsood_msp.sh
```

CIFAR-10 MSP OOD:

```bash
sh scripts_my/ood/msp/cifar10_test_ood_msp.sh
```
