# Data Download Notes

OpenOOD 데이터 다운로드와 로컬 재현 중 발견한 데이터별 주의점이다.

## MNIST Classic OOD Benchmark

`sh ./scripts/download/download.sh`로 데이터셋을 받으면 `notmnist`, `fashionmnist`가 안 받아지는 경우가 있어 직접 받아야 한다.

개인 재현에서는 원본 `scripts/`를 직접 수정하지 않고, 필요한 수정이 반영된 `scripts_my/download/download.py`를 사용한다.

```bash
python ./scripts_my/download/download.py \
  --contents datasets \
  --datasets notmnist fashionmnist \
  --dataset_mode dataset \
  --save_dir ./data ./results
```

## CIFAR-10 csID: CINIC-10

```bash
python ./scripts_my/download/download.py \
  --contents datasets \
  --datasets cinic10 \
  --dataset_mode dataset \
  --save_dir ./data ./results
```

FSOOD의 csID 평가에는 CINIC-10 imglist가 필요하다.
현재 로컬 데이터 구조는 `data/images_classic/cinic10/{train,val,test}/<class>/...`이다.

생성 대상:

```text
data/benchmark_imglist/cifar10/train_cinic10.txt
data/benchmark_imglist/cifar10/val_cinic10.txt
data/benchmark_imglist/cifar10/test_cinic10.txt
```

재현 명령:

```bash
for split in train val test; do
  out="data/benchmark_imglist/cifar10/${split}_cinic10.txt"
  : > "${out}"
  for item in \
    "airplane 0" \
    "automobile 1" \
    "bird 2" \
    "cat 3" \
    "deer 4" \
    "dog 5" \
    "frog 6" \
    "horse 7" \
    "ship 8" \
    "truck 9"; do
    class="${item% *}"
    label="${item##* }"
    find "data/images_classic/cinic10/${split}/${class}" \
      -maxdepth 1 -type f -printf "cinic10/${split}/${class}/%f ${label}\n" \
      | sort >> "${out}"
  done
done
```

확인:

```bash
wc -l data/benchmark_imglist/cifar10/*_cinic10.txt
```

각 split은 90,000줄이어야 한다.

## Misc Benchmark Data

`misc`에는 `cifar10c`, `fractals_and_fvis`, `usps`, `imagenet10`, `hannover` 등이 포함된다.

```bash
python ./scripts_my/download/download.py \
  --contents datasets \
  --datasets misc \
  --dataset_mode benchmark \
  --save_dir ./data ./results
```

## CIFAR-100-C, ImageNet-O

```bash
python ./scripts_my/download/download.py \
  --contents datasets \
  --datasets cifar100c imagenet_o \
  --dataset_mode dataset \
  --save_dir ./data ./results
```

주의:

- `imagenet_o`는 기존 `scripts/download/download.py`의 `dir_dict` 경로 목록에 없어서 다운로드 실패 가능성이 있었다.
- `download_dataset()`는 먼저 `dir_dict`에서 대상 데이터셋의 저장 위치를 찾는다.
- 원본은 그대로 두고, `scripts_my/download/download.py`의 `images_largescale` 목록에 `imagenet_o`를 추가한 수정본을 사용한다.

CIFAR-100-C 다운로드 후 FSOOD의 csID 평가를 위해 imglist를 별도로 만든다.
현재 로컬 파일명은 `<image_index>_<class_label>.png` 형식이다.

생성 대상:

```text
data/benchmark_imglist/cifar100/test_cifar100c.txt
```

재현 명령:

```bash
find data/images_classic/cifar100c -maxdepth 1 -type f -printf '%f\n' \
  | sort \
  | awk -F'[_\\.]' '{print "cifar100c/" $0 " " $2}' \
  > data/benchmark_imglist/cifar100/test_cifar100c.txt
```

확인:

```bash
wc -l data/benchmark_imglist/cifar100/test_cifar100c.txt
head data/benchmark_imglist/cifar100/test_cifar100c.txt
tail data/benchmark_imglist/cifar100/test_cifar100c.txt
```

`test_cifar100c.txt`는 10,000줄이어야 하고 label 범위는 0부터 99까지여야 한다.

## ImageNet FSOOD csID

OpenOOD v1.5 공식 benchmark 기준의 ImageNet-1K FSOOD csID는 다음 3개로 맞춘다.

```text
ImageNet-V2
ImageNet-C
ImageNet-R
```

로컬 코드에는 `imagenet_es` 관련 imglist와 download id가 남아 있을 수 있지만, 공식 v1.5 benchmark overview와 leaderboard 기준에서는 ImageNet-ES가 ImageNet-1K FSOOD csID에 포함되지 않는다.
따라서 공식 기준 재현에서는 아래 두 설정에서 ImageNet-ES를 제외한다.

```text
configs/datasets/imagenet/imagenet_fsood.yml
openood/evaluation_api/datasets.py
```

구체적으로 `imagenet_fsood.yml`에서는 `csid.datasets`에서 `imagenetes`를 제거하고, `openood/evaluation_api/datasets.py`에서는 ImageNet `csid.datasets`에서 `imagenet_es`를 제거한다.
`data/benchmark_imglist/imagenet/test_imagenet_es.txt`가 있어도 공식 기준 실행 경로에서는 참조하지 않는다.

## TODO

`cifar10-c`는 원래 직접 데이터를 변환해야 하는데, 여기서는 이미 변환된 데이터 1만장을 받는 것으로 보인다.
직접 변환한 것인지, severity 5만 가져온 것인지 확인이 필요하다.
