# Score Convention

OpenOOD evaluator의 metric 계산은 기본적으로 postprocessor의 `conf`가 클수록 ID라고 가정한다.

## 기본 convention

Postprocessor:

```text
conf 큼 -> ID 같음
conf 작음 -> OOD 같음
```

Metric:

OOD를 positive class로 두고 AUROC/FPR을 계산하므로 metric 쪽에서는 `-conf`를 사용한다.

```text
conf 큼  -> ID
-conf 큼 -> OOD
```

관련 코드:

- [openood/evaluators/metrics.py](/home/dmlab/DataDrift/GradResp/openood/evaluators/metrics.py)

## 예외 가능성이 있는 방법

DSVDD, RTS의 일부 설정은 현재 코드 구조상 score 방향 불일치 가능성이 있다.

### DSVDD

`DSVDDPostprocessor`는 `conf = distance/reconstruction error` 계열 값을 그대로 반환한다.
이 값은 보통 클수록 OOD에 가깝다.

관련 코드:

- [openood/postprocessors/dsvdd_postprocessor.py](/home/dmlab/DataDrift/GradResp/openood/postprocessors/dsvdd_postprocessor.py)

### RTS variance score

`RTSPostprocessor`에서 `ood_score == 'var'`일 때 `conf = mean(variance)`를 반환한다.
이 값도 의미상 클수록 OOD에 가까운 score일 수 있다.

관련 코드:

- [openood/postprocessors/rts_postprocessor.py](/home/dmlab/DataDrift/GradResp/openood/postprocessors/rts_postprocessor.py)

따라서 이 둘은 평가 방향을 별도로 확인해야 한다.

