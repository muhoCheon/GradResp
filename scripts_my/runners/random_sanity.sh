#!/bin/bash
set -euo pipefail

METHODS="msp"
DATASETS="all"
SEEDS="0,1,2,3,4"
GPUS="0,1"
JOBS_PER_GPU=2
INIT="kaiming"
OUTPUT_ROOT="results_test/random_sanity"
BATCH_SIZE=200
NUM_WORKERS=8
SUPPORTED_METHODS=(
  msp
  mls
  ebo
  odin
  iodin
  gradnorm
  mds
  rmds
  knn
  vim
  react
  ash
  dice
  gram
  klm
  she
  scale
)

usage() {
  cat <<EOF
Usage: bash scripts_my/runners/random_sanity.sh [options]

Options:
  --methods msp|all|msp,mls,...         Default: msp
  --datasets all|cifar10,cifar100,...   Default: all
  --seeds 0,1,2,3,4                     Default: 0,1,2,3,4
  --gpus 0,1                            Default: 0,1
  --jobs-per-gpu N                      Default: 2
  --init kaiming|default                Default: kaiming
  --output-root PATH                    Default: results_test/random_sanity
  --batch-size N                        Default: 200
  --num-workers N                       Default: 8
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --methods) METHODS="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
    --init) INIT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

csv_to_array() {
  local input="$1"
  local -n output="$2"
  IFS=',' read -r -a output <<< "${input}"
}

csv_to_array "${METHODS}" METHODS_ARR
csv_to_array "${DATASETS}" DATASETS_ARR
csv_to_array "${SEEDS}" SEEDS_ARR
csv_to_array "${GPUS}" GPUS_ARR

if [ "${METHODS}" = "all" ]; then
  METHODS_ARR=("${SUPPORTED_METHODS[@]}")
fi

for method in "${METHODS_ARR[@]}"; do
  found=0
  for supported in "${SUPPORTED_METHODS[@]}"; do
    if [ "${method}" = "${supported}" ]; then
      found=1
      break
    fi
  done
  if [ "${found}" -ne 1 ]; then
    echo "[FAIL] Unknown or unsupported method for random_sanity: ${method}" >&2
    echo "[INFO] Supported methods: ${SUPPORTED_METHODS[*]}" >&2
    echo "[INFO] Known excluded methods: residual, adascale_a, adascale_l" >&2
    exit 1
  fi
done

case "${INIT}" in
  kaiming|default) ;;
  *) echo "[FAIL] --init must be kaiming or default." >&2; exit 1 ;;
esac

if [ "${DATASETS}" = "all" ]; then
  DATASETS_ARR=(cifar10 cifar100 imagenet imagenet200)
fi

for dataset in "${DATASETS_ARR[@]}"; do
  case "${dataset}" in
    cifar10|cifar100|imagenet|imagenet200) ;;
    mnist)
      echo "[FAIL] random_sanity v1 does not support mnist because evaluation_api has no MNIST DATA_INFO entry." >&2
      exit 1
      ;;
    *)
      echo "[FAIL] Unknown dataset: ${dataset}" >&2
      exit 1
      ;;
  esac
done

SLOTS=()
for gpu in "${GPUS_ARR[@]}"; do
  for _ in $(seq 1 "${JOBS_PER_GPU}"); do
    SLOTS+=("${gpu}")
  done
done

if [ "${#SLOTS[@]}" -eq 0 ]; then
  echo "[FAIL] No GPU slots configured." >&2
  exit 1
fi

run_job() {
  local gpu="$1"
  local method="$2"
  local dataset="$3"
  local seed="$4"
  local run_dir="${OUTPUT_ROOT}/runs/${method}/${dataset}/seed_${seed}"
  local metadata_path="${run_dir}/metadata.md"
  local command_path="${run_dir}/command.sh"

  if [ -f "${metadata_path}" ] && grep -Eq '^- status: pass$' "${metadata_path}"; then
    echo "[SKIP] ${method}/${dataset}/seed_${seed} already completed"
    return 0
  fi

  mkdir -p "${run_dir}"

  cat > "${command_path}" <<EOF
#!/bin/bash
set -euo pipefail

METHOD=${method}
DATASET=${dataset}
SEED=${seed}
GPU=${gpu}
INIT=${INIT}
OUTPUT_ROOT=${OUTPUT_ROOT}
BATCH_SIZE=${BATCH_SIZE}
NUM_WORKERS=${NUM_WORKERS}
RUN_DIR=${run_dir}

mkdir -p \${RUN_DIR}
/usr/bin/time -f "RUNTIME=%E" \\
bash -c "CUDA_VISIBLE_DEVICES=\${GPU} conda run -n openood python scripts_my/tools/eval_random_ood.py --dataset \${DATASET} --method \${METHOD} --seed \${SEED} --init \${INIT} --output-root \${OUTPUT_ROOT} --batch-size \${BATCH_SIZE} --num-workers \${NUM_WORKERS}" \\
> \${RUN_DIR}/run.log 2>&1

RUNTIME=\$(grep '^RUNTIME=' \${RUN_DIR}/run.log | tail -n 1 | cut -d= -f2-)
CHECKSUM=\$(grep '^parameter_checksum:' \${RUN_DIR}/run.log | tail -n 1 | awk '{print \$2}')

cat > \${RUN_DIR}/metadata.md <<META
# Random Sanity Metadata

- dataset: \${DATASET}
- method: \${METHOD}
- seed: \${SEED}
- gpu: \${GPU}
- init: \${INIT}
- status: pass
- runtime: \${RUNTIME}
- parameter_checksum: \${CHECKSUM}
- output: \${OUTPUT_ROOT}/outputs/\${METHOD}/\${DATASET}/seed_\${SEED}
- command: \${RUN_DIR}/command.sh
- log: \${RUN_DIR}/run.log
META
EOF

  chmod +x "${command_path}"

  echo "[RUN] GPU ${gpu}: ${method}/${dataset}/seed_${seed}"
  if "${command_path}"; then
    echo "[PASS] ${method}/${dataset}/seed_${seed}"
  else
    echo "[FAIL] ${method}/${dataset}/seed_${seed}"
    if [ -f "${run_dir}/run.log" ]; then
      echo "----- Last 80 log lines: ${run_dir}/run.log -----"
      tail -n 80 "${run_dir}/run.log"
      echo "--------------------------------------------------"
    fi
    return 1
  fi
}

PIDS=()
PID_LABELS=()
NEXT_SLOT=0
FAILURES=0

wait_for_one() {
  local pid="${PIDS[0]}"
  local label="${PID_LABELS[0]}"
  set +e
  wait "${pid}"
  local status=$?
  set -e
  if [ "${status}" -ne 0 ]; then
    echo "[FAIL] ${label}"
    FAILURES=$((FAILURES + 1))
  fi
  PIDS=("${PIDS[@]:1}")
  PID_LABELS=("${PID_LABELS[@]:1}")
}

for method in "${METHODS_ARR[@]}"; do
  for dataset in "${DATASETS_ARR[@]}"; do
    for seed in "${SEEDS_ARR[@]}"; do
      while [ "${#PIDS[@]}" -ge "${#SLOTS[@]}" ]; do
        wait_for_one
      done

      gpu="${SLOTS[${NEXT_SLOT}]}"
      NEXT_SLOT=$(((NEXT_SLOT + 1) % ${#SLOTS[@]}))
      run_job "${gpu}" "${method}" "${dataset}" "${seed}" &
      PIDS+=("$!")
      PID_LABELS+=("${method}/${dataset}/seed_${seed}")
    done
  done
done

while [ "${#PIDS[@]}" -gt 0 ]; do
  wait_for_one
done

if [ "${FAILURES}" -ne 0 ]; then
  echo "[FAIL] ${FAILURES} random sanity job(s) failed." >&2
  exit 1
fi

python scripts_my/tools/summarize_random_sanity.py \
  --output-root "${OUTPUT_ROOT}" \
  --methods "$(IFS=','; echo "${METHODS_ARR[*]}")" \
  --datasets "$(IFS=','; echo "${DATASETS_ARR[*]}")" \
  --seeds "${SEEDS}"

echo "Random sanity completed."
