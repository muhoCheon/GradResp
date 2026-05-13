# Known Issues

로컬 재현 중 발견한 코드/설정 이슈를 정리한다.

## MDS가 MNIST를 지원하지 않는 문제

증상:

- `self.config.dataset.name`이 `mnist`
- `num_classes_dict`에 `mnist` 키가 없음
- 그래서 `KeyError` 발생

해결:

[openood/postprocessors/info.py](/home/dmlab/DataDrift/GradResp/openood/postprocessors/info.py)에 `mnist`를 추가한다.

```python
num_classes_dict = {
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'imagenet200': 200,
    'imagenet': 1000
}
```

## ImageNet-O 다운로드 경로 누락

`imagenet_o`는 기존 `scripts/download/download.py`의 `dir_dict` 경로 목록에 없어서 다운로드 실패 가능성이 있었다.

해결:

원본 `scripts/download/download.py`는 그대로 두고, 개인 수정본인 `scripts_my/download/download.py`의 `images_largescale` 목록에 `imagenet_o`를 추가한다.

```text
scripts_my/download/download.py
```

## `torch.load(..., weights_only=True)`와 CPU-only 실행

CUDA tensor로 저장된 checkpoint를 CPU-only 환경에서 `weights_only=True`로 로드하면 실패할 수 있다.

예시:

```text
Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False
```

이 레포의 평가 코드는 내부에서 `.cuda()`를 직접 호출하는 부분이 많으므로, CPU-only 실행을 목표로 하기보다는 GPU가 보이는 환경에서 실행하는 편이 현실적이다.

## `scripts/eval_ood.py` 의존성

`scripts/eval_ood.py`는 단순 MSP 평가에도 `openood.evaluation_api`를 import하고, 이 과정에서 attack 관련 모듈까지 import된다.
따라서 다음 패키지가 필요하다.

```bash
pip install timm statsmodels foolbox==3.2.1
```

주의:

- Python 3.8 환경에서 `foolbox 3.3.4`는 import 에러가 날 수 있다.
- `foolbox==3.2.1`은 현재 환경에서 import 확인됨.
