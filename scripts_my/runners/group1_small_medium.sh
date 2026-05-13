#!/bin/bash
set -euo pipefail

# Group 1 post-hoc validation on small/medium datasets.
# Order is fixed by dataset:
#   1. MNIST
#   2. CIFAR-10
#   3. CIFAR-100
#
# A run is skipped when results_test/runs/<method>/<dataset>/metadata.md
# already contains "- status: pass" or "- status: skip".

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

  case "${dataset}" in
    mnist) dataset_label="MNIST" ;;
    cifar10) dataset_label="CIFAR-10" ;;
    cifar100) dataset_label="CIFAR-100" ;;
  esac

  case "${method}" in
    residual) method_label="Residual" ;;
    vim) method_label="ViM" ;;
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
- eval OOD: -
- eval FSOOD: -
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
    mnist)
      run_pair 0 msp mnist scripts_my/ood/msp/mnist_test_ood_msp.sh 1 mls mnist scripts_my/ood/mls/mnist_test_ood_maxlogit.sh
      run_pair 0 ebo mnist scripts_my/ood/ebo/mnist_test_ood_ebo.sh 1 odin mnist scripts_my/ood/odin/mnist_test_ood_odin.sh
      run_pair 0 iodin mnist scripts_my/ood/iodin/mnist_test_ood_iodin.sh 1 gradnorm mnist scripts_my/ood/gradnorm/mnist_test_ood_gradnorm.sh
      run_pair 0 mds mnist scripts_my/ood/mds/mnist_test_ood_mds.sh 1 rmds mnist scripts_my/ood/rmds/mnist_test_ood_rmds.sh
      run_pair 0 knn mnist scripts_my/ood/knn/mnist_test_ood_knn.sh
      mark_skip vim mnist "skipped because MNIST LeNet feature_dim=256 makes the current ViM dim sweep produce alpha=inf/NaN"
      mark_skip residual mnist "skipped because the shared Residual dim=512 is incompatible with MNIST LeNet feature_dim=256 and produces a degenerate residual subspace"
      run_pair 1 react mnist scripts_my/ood/react/mnist_test_ood_react.sh
      run_pair 0 ash mnist scripts_my/ood/ash/mnist_test_ood_ash.sh 1 dice mnist scripts_my/ood/dice/mnist_test_ood_dice.sh
      run_pair 0 gram mnist scripts_my/ood/gram/mnist_test_ood_gram.sh 1 klm mnist scripts_my/ood/kl_matching/mnist_test_ood_kl_matching.sh
      run_pair 0 she mnist scripts_my/ood/she/mnist_test_ood_she.sh 1 scale mnist scripts_my/ood/scale/mnist_test_ood_scale.sh
      mark_skip adascale_a mnist "skipped because AdaScale requires AdaScaleANet wrapping, which is not applied in the main.py test_ood pipeline"
      mark_skip adascale_l mnist "skipped because AdaScale requires AdaScaleLNet wrapping, which is not applied in the main.py test_ood pipeline"
      ;;
    cifar10)
      run_pair 0 msp cifar10 scripts_my/ood/msp/cifar10_test_ood_msp.sh 1 mls cifar10 scripts_my/ood/mls/cifar10_test_ood_maxlogit.sh
      run_pair 0 ebo cifar10 scripts_my/ood/ebo/cifar10_test_ood_ebo.sh 1 odin cifar10 scripts_my/ood/odin/cifar10_test_ood_odin.sh
      run_pair 0 iodin cifar10 scripts_my/ood/iodin/cifar10_test_ood_iodin.sh 1 gradnorm cifar10 scripts_my/ood/gradnorm/cifar10_test_ood_gradnorm.sh
      run_pair 0 mds cifar10 scripts_my/ood/mds/cifar10_test_ood_mds.sh 1 rmds cifar10 scripts_my/ood/rmds/cifar10_test_ood_rmds.sh
      run_pair 0 knn cifar10 scripts_my/ood/knn/cifar10_test_ood_knn.sh
      mark_skip vim cifar10 "skipped because the current ViM dim sweep includes 1000, which exceeds CIFAR-10 ResNet18 feature_dim=512 and produces alpha=inf/NaN"
      mark_skip residual cifar10 "skipped because the shared Residual dim=512 equals CIFAR-10 ResNet18 feature_dim=512 and produces a degenerate residual subspace"
      run_pair 1 react cifar10 scripts_my/ood/react/cifar10_test_ood_react.sh
      run_pair 0 ash cifar10 scripts_my/ood/ash/cifar10_test_ood_ash.sh 1 dice cifar10 scripts_my/ood/dice/cifar10_test_ood_dice.sh
      run_pair 0 gram cifar10 scripts_my/ood/gram/cifar10_test_ood_gram.sh 1 klm cifar10 scripts_my/ood/kl_matching/cifar10_test_ood_kl_matching.sh
      run_pair 0 she cifar10 scripts_my/ood/she/cifar10_test_ood_she.sh 1 scale cifar10 scripts_my/ood/scale/cifar10_test_ood_scale.sh
      mark_skip adascale_a cifar10 "skipped because AdaScale requires AdaScaleANet wrapping, which is not applied in the main.py test_ood pipeline"
      mark_skip adascale_l cifar10 "skipped because AdaScale requires AdaScaleLNet wrapping, which is not applied in the main.py test_ood pipeline"
      ;;
    cifar100)
      run_pair 0 msp cifar100 scripts_my/ood/msp/cifar100_test_ood_msp.sh 1 mls cifar100 scripts_my/ood/mls/cifar100_test_ood_maxlogit.sh
      run_pair 0 ebo cifar100 scripts_my/ood/ebo/cifar100_test_ood_ebo.sh 1 odin cifar100 scripts_my/ood/odin/cifar100_test_ood_odin.sh
      run_pair 0 iodin cifar100 scripts_my/ood/iodin/cifar100_test_ood_iodin.sh 1 gradnorm cifar100 scripts_my/ood/gradnorm/cifar100_test_ood_gradnorm.sh
      run_pair 0 mds cifar100 scripts_my/ood/mds/cifar100_test_ood_mds.sh 1 rmds cifar100 scripts_my/ood/rmds/cifar100_test_ood_rmds.sh
      run_pair 0 knn cifar100 scripts_my/ood/knn/cifar100_test_ood_knn.sh
      mark_skip vim cifar100 "skipped because the current ViM dim sweep includes 1000, which exceeds CIFAR-100 ResNet18 feature_dim=512 and produces alpha=inf/NaN"
      mark_skip residual cifar100 "skipped because the shared Residual dim=512 equals CIFAR-100 ResNet18 feature_dim=512 and produces a degenerate residual subspace"
      run_pair 1 react cifar100 scripts_my/ood/react/cifar100_test_ood_react.sh
      run_pair 0 ash cifar100 scripts_my/ood/ash/cifar100_test_ood_ash.sh 1 dice cifar100 scripts_my/ood/dice/cifar100_test_ood_dice.sh
      run_pair 0 gram cifar100 scripts_my/ood/gram/cifar100_test_ood_gram.sh 1 klm cifar100 scripts_my/ood/kl_matching/cifar100_test_ood_kl_matching.sh
      run_pair 0 she cifar100 scripts_my/ood/she/cifar100_test_ood_she.sh 1 scale cifar100 scripts_my/ood/scale/cifar100_test_ood_scale.sh
      mark_skip adascale_a cifar100 "skipped because AdaScale requires AdaScaleANet wrapping, which is not applied in the main.py test_ood pipeline"
      mark_skip adascale_l cifar100 "skipped because AdaScale requires AdaScaleLNet wrapping, which is not applied in the main.py test_ood pipeline"
      ;;
    *)
      echo "Unknown dataset: ${dataset}" >&2
      exit 1
      ;;
  esac
}

run_dataset mnist
run_dataset cifar10
run_dataset cifar100

echo "========== Updating validation checklist =========="
python scripts_my/tools/update_group1_validation.py

echo "Group 1 small/medium validation completed."
