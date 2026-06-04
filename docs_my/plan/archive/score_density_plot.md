# Score Density Plot Script 구현 계획

> Archive note: This plan is historical. Use docs_my/TARR/ and scripts_my/tarr/ as the current source of truth unless this file is explicitly referenced for context.

## Summary
`scripts_my/tools/plot_score_density.py`를 새로 추가해, 저장된 `scores/*.npz`의 `conf` 값을 읽고 `ood_score = -conf` 기준으로 horizontal violin plot을 생성한다. 기본은 전체 데이터를 사용하고, 필요할 때만 `--max-samples`로 subsampling한다.

## Key Changes
- CLI 인터페이스:
  - `--dataset {mnist,cifar10,cifar100,imagenet,imagenet200,all}`
  - `--method {msp,mls,...,scale,all}`
  - `--scheme {ood,fsood,all}`
  - `--max-samples 0` 기본값: 전체 데이터 사용
  - `--dpi`, `--output-dir`, `--no-box` 정도의 최소 옵션 제공
- 입력 score 경로:
  - OOD: `results_test/outputs/<prefix>_test_ood_ood_<method>_0/s0/ood/scores/*.npz`
  - FSOOD: `results_test/outputs/<prefix>_test_ood_fsood_<method>_0/scores/*.npz`
- score 정의:
  - 저장된 `conf`는 클수록 ID라고 가정
  - plot에는 항상 `ood_score = -conf` 사용
  - `dsvdd`, `rts`, `rts_var` 계열 method가 들어오면 warning만 출력하고 동일 규칙 적용
- plotting backend:
  - `matplotlib` 사용
  - `MPLCONFIGDIR`가 없으면 `/tmp/matplotlib-cache` 같은 writable 경로를 script 안에서 설정해 font/cache warning을 줄임

## Plot Outputs
각 `dataset/method/scheme` 조합마다 아래 3개 저장:

```text
  results_test/plots/score_density/<dataset>/<method>/<scheme>/
  per_dataset.png
  combined_eval_groups.png
  combined_id_groups.png
```

- `per_dataset.png`
  - ID, csID, nearOOD, farOOD의 각 dataset을 개별 horizontal violin으로 표시
  - 내부 box plot + median line 포함
  - y-label 예: `ID / cifar10`, `csID / cinic10`, `nearOOD / cifar100`
- `combined_eval_groups.png`
  - `ID + csID` vs `nearOOD` vs `farOOD`
  - OOD scheme처럼 csID가 없으면 `ID` vs `nearOOD` vs `farOOD`
- `combined_id_groups.png`
  - `ID` vs `csID` vs `nearOOD` vs `farOOD`
  - OOD scheme에서 csID가 없으면 `ID`, `nearOOD`, `farOOD`만 표시

## Data Handling
- `.npz`에서 `conf`만 필수로 읽고, `pred`/`label`은 사용하지 않는다.
- `all` 옵션은 실제 score directory가 존재하는 조합만 처리한다.
- skip되어 score가 없는 조합은 warning을 출력하고 넘어간다.
- `--max-samples N`이 0보다 크면 distribution별로 최대 N개만 random subsample한다.
- split mapping은 dataset별 config에 맞춰 script 내부 상수로 둔다:
  - `id`, `csid`, `nearood`, `farood`
  - 알 수 없는 score file은 `unknown`으로 표시하되 plot에는 포함한다.

## Test Plan
- 문법 확인:
  ```bash
  python -m py_compile scripts_my/tools/plot_score_density.py
  ```
- 단일 조합 smoke test:
  ```bash
  python scripts_my/tools/plot_score_density.py --dataset cifar10 --method msp --scheme fsood
  ```
- 생성물 확인:
  ```text
  results_test/plots/score_density/cifar10/msp/fsood/per_dataset.png
  results_test/plots/score_density/cifar10/msp/fsood/combined_eval_groups.png
  results_test/plots/score_density/cifar10/msp/fsood/combined_id_groups.png
  ```
- all 옵션 dry smoke:
  ```bash
  python scripts_my/tools/plot_score_density.py --dataset cifar10 --method all --scheme ood --max-samples 1000
  ```

## Assumptions
- 이번 버전은 histogram/KDE/ECDF를 구현하지 않고 horizontal violin plot만 구현한다.
- box plot은 기본 ON, outlier/scatter point는 기본 OFF.
- 기본은 전체 score 사용이며, 속도 문제가 생기면 사용자가 `--max-samples`를 지정한다.
