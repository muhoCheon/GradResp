# Environment Setup

Slurm/srun 없이 로컬 또는 Remote SSH 환경에서 OpenOOD 스크립트를 실행하기 위한 환경 메모다.

## Conda 환경 생성

```bash
conda create -n openood python=3.8 -y
conda activate openood
```

## 패키지 설치

로컬 레포를 수정하면서 실행할 것이므로 editable 모드로 설치한다.

```bash
cd /home/dmlab/DataDrift/GradResp
pip install -e .
pip install libmr statsmodels timm
```

`scripts/eval_ood.py` 경로는 attack 관련 모듈까지 import하므로 `foolbox`도 필요하다.
Python 3.8 환경에서는 최신 `foolbox 3.3.4`가 import 에러를 낼 수 있으므로 `3.2.1`을 사용한다.

```bash
pip install foolbox==3.2.1
```

설치 확인:

```bash
python -c "import timm, statsmodels, foolbox; print(timm.__version__, statsmodels.__version__, foolbox.__version__)"
```

## GPU 실행 확인

Codex나 비대화형 실행 환경에서는 sandbox 때문에 기본 실행에서 GPU device가 안 보일 수 있다.
직접 터미널에서는 아래 명령으로 확인한다.

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

정상 예시는 다음과 같다.

```text
torch.cuda.is_available() == True
torch.cuda.device_count() > 0
```

## 기존 output directory 프롬프트 회피

OpenOOD는 결과 디렉토리가 이미 있으면 기본적으로 입력을 요구한다.

```text
Exp dir already exists, merge it? (y/n)
```

비대화형 실행에서는 `EOFError`가 날 수 있으므로 실험 명령에 아래 옵션을 붙인다.

```bash
--merge_option merge
```

