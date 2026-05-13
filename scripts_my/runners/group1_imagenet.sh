#!/bin/bash
set -euo pipefail

# Group 1 post-hoc validation on ImageNet-scale datasets.
# Order is fixed by dataset:
#   1. ImageNet
#   2. ImageNet-200
#
# A run is skipped when results_test/runs/<method>/<dataset>/metadata.md
# already contains "- status: pass" or "- status: skip".

check_required_paths() {
  local missing=0

  for path in "$@"; do
    if [ ! -e "${path}" ]; then
      echo "[MISSING] ${path}" >&2
      missing=1
    fi
  done

  if [ "${missing}" -ne 0 ]; then
    echo "[FAIL] Required ImageNet data/checkpoints are missing." >&2
    exit 1
  fi
}

preflight() {
  echo "========== Preflight: ImageNet data/checkpoints =========="

  check_required_paths \
    results/pretrained_weights/resnet50_imagenet1k_v1.pth \
    results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt \
    data/images_largescale/imagenet_1k \
    data/images_largescale/ssb_hard \
    data/images_largescale/ninco \
    data/images_largescale/inaturalist \
    data/images_largescale/openimage_o \
    data/images_largescale/imagenet_v2 \
    data/images_largescale/imagenet_c \
    data/images_largescale/imagenet_r \
    data/images_classic/texture \
    data/benchmark_imglist/imagenet/train_imagenet.txt \
    data/benchmark_imglist/imagenet/val_imagenet.txt \
    data/benchmark_imglist/imagenet/test_imagenet.txt \
    data/benchmark_imglist/imagenet/val_openimage_o.txt \
    data/benchmark_imglist/imagenet/test_ssb_hard.txt \
    data/benchmark_imglist/imagenet/test_ninco.txt \
    data/benchmark_imglist/imagenet/test_textures.txt \
    data/benchmark_imglist/imagenet/test_inaturalist.txt \
    data/benchmark_imglist/imagenet/test_openimage_o.txt \
    data/benchmark_imglist/imagenet/test_imagenet_v2.txt \
    data/benchmark_imglist/imagenet/test_imagenet_c.txt \
    data/benchmark_imglist/imagenet/test_imagenet_r.txt \
    data/benchmark_imglist/imagenet200/train_imagenet200.txt \
    data/benchmark_imglist/imagenet200/val_imagenet200.txt \
    data/benchmark_imglist/imagenet200/test_imagenet200.txt \
    data/benchmark_imglist/imagenet200/val_openimage_o.txt \
    data/benchmark_imglist/imagenet200/test_ssb_hard.txt \
    data/benchmark_imglist/imagenet200/test_ninco.txt \
    data/benchmark_imglist/imagenet200/test_textures.txt \
    data/benchmark_imglist/imagenet200/test_inaturalist.txt \
    data/benchmark_imglist/imagenet200/test_openimage_o.txt \
    data/benchmark_imglist/imagenet200/test_imagenet200_v2.txt \
    data/benchmark_imglist/imagenet200/test_imagenet200_c.txt \
    data/benchmark_imglist/imagenet200/test_imagenet200_r.txt

  echo "[PASS] Preflight complete"
}

run_one() {
  local gpu="$1"
  local method="$2"
  local dataset="$3"
  local script="$4"
  local log_dir="results_test/runs/${method}/${dataset}"
  local command_path="${log_dir}/command.sh"
  local metadata_path="${log_dir}/metadata.md"

  if [ -f "${metadata_path}" ] && grep -Eq '^- status: (pass|skip)$' "${metadata_path}"; then
    echo "[SKIP] ${dataset}/${method} already completed"
    return 0
  fi

  mkdir -p "${log_dir}"

  cat > "${command_path}" <<EOF
#!/bin/bash
set -euo pipefail

METHOD=${method}
DATASET=${dataset}
GPU=${gpu}
SCRIPT=${script}
LOG_DIR=results_test/runs/\${METHOD}/\${DATASET}

mkdir -p \${LOG_DIR}
/usr/bin/time -f "RUNTIME=%E" \\
bash -c "CUDA_VISIBLE_DEVICES=\${GPU} conda run -n openood sh \${SCRIPT}" \\
> \${LOG_DIR}/run.log 2>&1

python scripts_my/tools/make_run_metadata.py \\
  --method \${METHOD} \\
  --dataset \${DATASET} \\
  --gpu \${GPU}
EOF

  chmod +x "${command_path}"

  echo "[RUN] GPU ${gpu}: ${dataset}/${method}"
  if "${command_path}"; then
    echo "[PASS] ${dataset}/${method}"
  else
    echo "[FAIL] ${dataset}/${method}"
    if [ -f "${log_dir}/run.log" ]; then
      echo "----- Last 80 log lines: ${log_dir}/run.log -----"
      tail -n 80 "${log_dir}/run.log"
      echo "--------------------------------------------------"
    fi
    return 1
  fi
}

mark_skip() {
  local method="$1"
  local dataset="$2"
  local reason="$3"
  local log_dir="results_test/runs/${method}/${dataset}"
  local metadata_path="${log_dir}/metadata.md"
  local dataset_label="${dataset}"
  local method_label="${method}"
  local eval_status="skip"

  case "${dataset}" in
    imagenet) dataset_label="ImageNet" ;;
    imagenet200) dataset_label="ImageNet-200" ;;
  esac

  case "${method}" in
    msp) method_label="MSP" ;;
    mls) method_label="MLS" ;;
    ebo) method_label="EBO" ;;
    odin) method_label="ODIN" ;;
    iodin) method_label="IODIN" ;;
    gradnorm) method_label="GradNorm" ;;
    mds) method_label="MDS" ;;
    rmds) method_label="RMDS" ;;
    knn) method_label="KNN" ;;
    vim) method_label="ViM" ;;
    residual) method_label="Residual" ;;
    react) method_label="ReAct" ;;
    ash) method_label="ASH" ;;
    dice) method_label="DICE" ;;
    gram) method_label="Gram" ;;
    klm) method_label="KLM" ;;
    she) method_label="SHE" ;;
    scale) method_label="SCALE" ;;
    adascale_a) method_label="AdaScale-A" ;;
    adascale_l) method_label="AdaScale-L" ;;
  esac

  mkdir -p "${log_dir}"

  cat > "${metadata_path}" <<EOF
# Run Metadata

- dataset: ${dataset_label}
- method: ${method_label}
- gpu: -
- status: skip
- runtime: -
- main OOD: skip
- main FSOOD: skip
- eval OOD: ${eval_status}
- eval FSOOD: ${eval_status}
- OOD Near AUROC: -
- OOD Far AUROC: -
- FSOOD Near AUROC: -
- FSOOD Far AUROC: -
- notes: ${reason}
EOF
}

run_pair() {
  local gpu_a="$1"
  local method_a="$2"
  local dataset_a="$3"
  local script_a="$4"
  local gpu_b="${5:-}"
  local method_b="${6:-}"
  local dataset_b="${7:-}"
  local script_b="${8:-}"

  run_one "${gpu_a}" "${method_a}" "${dataset_a}" "${script_a}" &
  local pid_a=$!

  local pid_b=""
  if [ -n "${gpu_b}" ]; then
    run_one "${gpu_b}" "${method_b}" "${dataset_b}" "${script_b}" &
    pid_b=$!
  fi

  set +e
  wait "${pid_a}"
  local status_a=$?
  local status_b=0
  if [ -n "${pid_b}" ]; then
    wait "${pid_b}"
    status_b=$?
  fi
  set -e

  if [ "${status_a}" -ne 0 ] || [ "${status_b}" -ne 0 ]; then
    echo "[FAIL] One or more jobs failed in this pair"
    exit 1
  fi
}

run_dataset() {
  local dataset="$1"

  echo "========== Group 1: ${dataset} =========="

  case "${dataset}" in
    imagenet)
      run_pair 0 msp imagenet scripts_my/ood/msp/imagenet_test_ood_msp.sh 1 mls imagenet scripts_my/ood/mls/imagenet_test_ood_maxlogit.sh
      run_pair 0 ebo imagenet scripts_my/ood/ebo/imagenet_test_ood_ebo.sh 1 odin imagenet scripts_my/ood/odin/imagenet_test_ood_odin.sh
      run_pair 0 iodin imagenet scripts_my/ood/iodin/imagenet_test_ood_iodin.sh 1 gradnorm imagenet scripts_my/ood/gradnorm/imagenet_test_ood_gradnorm.sh
      mark_skip mds imagenet "skipped because the local ImageNet train imglist references missing files, which breaks MDS train-feature statistics setup"
      mark_skip rmds imagenet "skipped because the local ImageNet train imglist references missing files, which breaks RMDS train-feature statistics setup"
      mark_skip knn imagenet "skipped because the local ImageNet train imglist references missing files, which breaks KNN train-feature setup"
      mark_skip vim imagenet "skipped because the local ImageNet train imglist references missing files, which breaks ViM train-feature setup"
      run_pair 0 residual imagenet scripts_my/ood/residual/imagenet_test_ood_residual.sh 1 react imagenet scripts_my/ood/react/imagenet_test_ood_react.sh
      run_pair 0 ash imagenet scripts_my/ood/ash/imagenet_test_ood_ash.sh
      mark_skip dice imagenet "skipped because the local ImageNet train imglist references missing files, which breaks DICE activation setup"
      mark_skip gram imagenet "skipped because the local ImageNet train imglist references missing files, which breaks GRAM feature statistics setup"
      run_pair 1 klm imagenet scripts_my/ood/kl_matching/imagenet_test_ood_kl_matching.sh
      mark_skip she imagenet "skipped because the local ImageNet train imglist references missing files, which breaks SHE activation setup"
      run_pair 1 scale imagenet scripts_my/ood/scale/imagenet_test_ood_scale.sh
      mark_skip adascale_a imagenet "skipped because AdaScale requires AdaScaleANet wrapping, which is not applied in the main.py test_ood pipeline"
      mark_skip adascale_l imagenet "skipped because AdaScale requires AdaScaleLNet wrapping, which is not applied in the main.py test_ood pipeline"
      ;;
    imagenet200)
      run_pair 0 msp imagenet200 scripts_my/ood/msp/imagenet200_test_ood_msp.sh 1 mls imagenet200 scripts_my/ood/mls/imagenet200_test_ood_maxlogit.sh
      run_pair 0 ebo imagenet200 scripts_my/ood/ebo/imagenet200_test_ood_ebo.sh 1 odin imagenet200 scripts_my/ood/odin/imagenet200_test_ood_odin.sh
      run_pair 0 iodin imagenet200 scripts_my/ood/iodin/imagenet200_test_ood_iodin.sh 1 gradnorm imagenet200 scripts_my/ood/gradnorm/imagenet200_test_ood_gradnorm.sh
      mark_skip mds imagenet200 "skipped because the local ImageNet-200 train imglist references missing ImageNet train files, which breaks MDS train-feature statistics setup"
      mark_skip rmds imagenet200 "skipped because the local ImageNet-200 train imglist references missing ImageNet train files, which breaks RMDS train-feature statistics setup"
      mark_skip knn imagenet200 "skipped because the local ImageNet-200 train imglist references missing ImageNet train files, which breaks KNN train-feature setup"
      mark_skip vim imagenet200 "skipped because the current ViM dim sweep includes 1000 and the local ImageNet-200 train imglist references missing ImageNet train files"
      mark_skip residual imagenet200 "skipped because the shared Residual dim=512 equals ImageNet-200 ResNet18 feature_dim=512 and produces a degenerate residual subspace"
      run_pair 1 react imagenet200 scripts_my/ood/react/imagenet200_test_ood_react.sh
      run_pair 0 ash imagenet200 scripts_my/ood/ash/imagenet200_test_ood_ash.sh
      mark_skip dice imagenet200 "skipped because the local ImageNet-200 train imglist references missing ImageNet train files, which breaks DICE activation setup"
      mark_skip gram imagenet200 "skipped because the local ImageNet-200 train imglist references missing ImageNet train files, which breaks GRAM feature statistics setup"
      run_pair 1 klm imagenet200 scripts_my/ood/kl_matching/imagenet200_test_ood_kl_matching.sh
      mark_skip she imagenet200 "skipped because the local ImageNet-200 train imglist references missing ImageNet train files, which breaks SHE activation setup"
      run_pair 1 scale imagenet200 scripts_my/ood/scale/imagenet200_test_ood_scale.sh
      mark_skip adascale_a imagenet200 "skipped because AdaScale requires AdaScaleANet wrapping, which is not applied in the main.py test_ood pipeline"
      mark_skip adascale_l imagenet200 "skipped because AdaScale requires AdaScaleLNet wrapping, which is not applied in the main.py test_ood pipeline"
      ;;
    *)
      echo "Unknown dataset: ${dataset}" >&2
      exit 1
      ;;
  esac
}

preflight
run_dataset imagenet
run_dataset imagenet200

echo "========== Updating validation checklist =========="
python scripts_my/tools/update_group1_validation.py

echo "Group 1 ImageNet validation completed."
