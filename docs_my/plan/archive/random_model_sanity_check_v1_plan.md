# Random Model Sanity Check v1 구현 계획

> Archive note: This plan is historical. Use docs_my/TARR/ and scripts_my/tarr/ as the current source of truth unless this file is explicitly referenced for context.

## Summary
- `scripts_my/tools/eval_random_ood.py`와 `scripts_my/runners/random_sanity.sh`를 추가해, checkpoint를 로드하지 않은 random model의 OOD/FSOOD AUROC와 score를 저장한다.
- v1 범위는 `MSP`만 지원하고, dataset은 `cifar10,cifar100,imagenet,imagenet200`만 지원한다. `mnist`는 현재 `evaluation_api`에 없으므로 v1에서는 보류한다.
- runner는 병렬 job queue를 사용하고 기본값은 `--jobs-per-gpu 2`로 둔다. 모든 job 완료 후 summary CSV만 생성하며 plot은 별도 후속 명령으로 실행한다.

## Key Changes
- `eval_random_ood.py`
  - CLI: `--dataset`, `--method msp`, `--seed`, `--init {kaiming,default}`, `--output-root`, `--batch-size`, `--num-workers`.
  - dataset별 기존 architecture 사용: `cifar10/cifar100 -> ResNet18_32x32`, `imagenet -> ResNet50`, `imagenet200 -> ResNet18_224x224`.
  - seed 설정 후 model 생성, checkpoint load 없음.
  - `--init kaiming` 기본값: Conv/Linear는 Kaiming normal, bias는 0, BatchNorm weight/bias는 1/0. `--init default`는 repo 기본 init 그대로 사용.
  - `Evaluator.eval_ood(fsood=False)`와 `Evaluator.eval_ood(fsood=True)`를 모두 실행.
  - metrics 저장: `outputs/<method>/<dataset>/seed_<seed>/ood.csv`, `fsood.csv`.
  - score 저장: `outputs/<method>/<dataset>/seed_<seed>/scores/<scheme>/*.npz` 형식으로 `pred/conf/label` 저장해 이후 density plot script가 읽을 수 있게 한다.

- `random_sanity.sh`
  - CLI:
    ```bash
    --methods msp
    --datasets cifar10,cifar100,imagenet,imagenet200 또는 all
    --seeds 0,1,2,3,4
    --gpus 0,1
    --jobs-per-gpu 2
    --init kaiming 또는 default
    ```
  - `--datasets all`은 v1에서 `cifar10,cifar100,imagenet,imagenet200`로 확장한다.
  - `mnist`가 지정되면 v1 unsupported로 명확히 실패시킨다.
  - GPU slot queue 방식으로 `gpus * jobs-per-gpu`만큼 병렬 실행한다.
  - 각 job 로그/metadata는 `results_test/random_sanity/runs/<method>/<dataset>/seed_<seed>/`에 저장한다.
  - 모든 job 완료 후 summary CSV 생성:
    - `summary/random_sanity_msp_per_seed.csv`
    - `summary/random_sanity_msp_summary.csv`

- 문서
  - `docs_my/experiments/random_sanity.md` 추가.
  - 목적, 논문 실험과의 차이, v1 범위, init 정책, runner 사용 예시, 출력 구조, MNIST 보류 사유를 기록한다.

## Test Plan
- 정적 확인:
  ```bash
  python -m py_compile scripts_my/tools/eval_random_ood.py
  bash -n scripts_my/runners/random_sanity.sh
  ```
- smoke test:
  ```bash
  bash scripts_my/runners/random_sanity.sh \
    --methods msp \
    --datasets cifar10 \
    --seeds 0 \
    --gpus 0 \
    --jobs-per-gpu 1 \
    --init kaiming
  ```
- 확인 항목:
  - `ood.csv`, `fsood.csv`, score npz, metadata 생성.
  - random model checkpoint load가 전혀 없음.
  - `seed=0`과 `seed=1`의 parameter checksum이 달라짐.
  - `--init default`와 `--init kaiming`이 둘 다 실행 가능.
  - summary CSV가 per-seed 결과와 mean/std 결과를 생성.

## Assumptions
- v1은 MSP only다. `--methods`에 `mls`, `ebo`, `all`, `pass` 등이 들어오면 unsupported로 실패시킨다.
- v1은 MNIST를 실행하지 않는다. MNIST는 `evaluation_api`에 dataset 정의를 추가하는 별도 작업으로 다룬다.
- runner는 plot을 생성하지 않는다. score density plot은 random sanity output 구조를 지원하도록 별도 후속 script 또는 기존 `plot_score_density.py` 확장으로 처리한다.
- 기본 병렬값은 `--jobs-per-gpu 2`이며, ImageNet에서 I/O/VRAM 병목이 생기면 사용자가 `--jobs-per-gpu 1`로 낮춘다.
