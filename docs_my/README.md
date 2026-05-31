# Personal OpenOOD Notes

이 디렉토리는 OpenOOD를 개인적으로 재현하고 분석하면서 남긴 문서를 역할별로 분리한 공간이다.

## 관리 원칙

- `scripts/` 아래 원본 스크립트와 코드는 가능한 그대로 둔다.
- 개인 재현용 실행 스크립트와 원본에서 수정한 코드는 `scripts_my/` 아래에 둔다.
- 문서와 실험 메모는 `docs_my/` 아래에 둔다.

## 문서 구조

- [setup/environment.md](setup/environment.md): conda 환경, 패키지 설치, GPU 실행 관련 메모
- [setup/data_download.md](setup/data_download.md): 데이터셋 다운로드와 데이터별 주의점
- [experiments/mnist.md](experiments/mnist.md): MNIST 학습, 테스트, OOD/FSOOD 재현 명령
- [experiments/cifar10.md](experiments/cifar10.md): CIFAR-10 관련 데이터 및 평가 메모
- [experiments/group1_validation.md](experiments/group1_validation.md): Group 1 post-hoc 방법 100개 스크립트 실행 검증 체크리스트
- [notes/score_convention.md](notes/score_convention.md): OpenOOD score 방향 convention과 예외 가능성
- [notes/ood_method_groups.md](notes/ood_method_groups.md): post-hoc 방법과 train-dependent 방법 구분, TTA response 방법의 비교군
- [notes/known_issues.md](notes/known_issues.md): 로컬 재현 중 발견한 코드/설정 이슈
- [TARR/implementation.md](TARR/implementation.md): TARR four-stage artifact pipeline 설명
- [TARR/commands.md](TARR/commands.md): TARR stage별 실행 명령어
