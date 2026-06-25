#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat >&2 <<'EOF'
usage:
  tarr_ar_bank_sharded_focused.sh <cifar100|imagenet200> <ref_id> <ref_spec> <shard_count>

example:
  scripts_my/runners/tarr_ar_bank_sharded_focused.sh \
    imagenet200 correct_rpc32 "per_class=32,filter=correct,seed=0" 8

Environment overrides:
  LR=1e-2
  STEPS=30
  SAVE_STEPS=5,10,30
  GPUS=0,1
  MAX_PARALLEL=<number of concurrent shard jobs>
  SHARD_FILTER=<optional comma-separated shard indexes>
  SKIP_MERGE_SCORE=0
  MAX_ID_SAMPLES=<optional>
  MAX_OOD_SAMPLES=<optional>
  RUN_SUFFIX=<optional>
EOF
}

if [[ $# -ne 4 ]]; then
  usage
  exit 2
fi

DATASET="$1"
REF_ID="$2"
REF_SPEC="$3"
SHARD_COUNT="$4"

PROTOCOL="${PROTOCOL:-eval_api}"
SCHEME="${SCHEME:-fsood}"
RUN_SEED="${RUN_SEED:-0}"
REFSEED="${REFSEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results_test/tarr}"
STEPS="${STEPS:-30}"
SAVE_STEPS="${SAVE_STEPS:-5,10,30}"
LR="${LR:-1e-2}"
UPDATE_SCOPE="${UPDATE_SCOPE:-classifier}"
ACCEPT_PROBE_TYPES="${ACCEPT_PROBE_TYPES:-predicted_label_ce,entropy_min,view_consistency}"
REJECT_PROBE_TYPES="${REJECT_PROBE_TYPES:-entropy_max,uniform}"
PERTURBATION_RESPONSE="${PERTURBATION_RESPONSE:-pixel}"
PERTURBATION_KIND="${PERTURBATION_KIND:-gaussian}"
PERTURBATION_EPS="${PERTURBATION_EPS:-0.01}"
PERTURBATION_REPEATS="${PERTURBATION_REPEATS:-4}"
PERTURBATION_SEED="${PERTURBATION_SEED:-0}"
GPUS="${GPUS:-0,1}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_LIST[@]}}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
SHARD_FILTER="${SHARD_FILTER:-}"
SKIP_MERGE_SCORE="${SKIP_MERGE_SCORE:-0}"

case "${DATASET}" in
  cifar100)
    BATCH_SIZE="${BATCH_SIZE:-512}"
    REFERENCE_SET_BATCH_SIZE="${REFERENCE_SET_BATCH_SIZE:-2048}"
    TTA_RESPONSE_SHARD_SIZE="${TTA_RESPONSE_SHARD_SIZE:-1024}"
    NUM_WORKERS="${NUM_WORKERS:-0}"
    ;;
  imagenet200)
    BATCH_SIZE="${BATCH_SIZE:-64}"
    REFERENCE_SET_BATCH_SIZE="${REFERENCE_SET_BATCH_SIZE:-1024}"
    TTA_RESPONSE_SHARD_SIZE="${TTA_RESPONSE_SHARD_SIZE:-1024}"
    NUM_WORKERS="${NUM_WORKERS:-2}"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

if [[ "${SHARD_COUNT}" -lt 1 ]]; then
  echo "shard_count must be positive: ${SHARD_COUNT}" >&2
  exit 2
fi
if [[ "${MAX_PARALLEL}" -lt 1 ]]; then
  echo "MAX_PARALLEL must be positive: ${MAX_PARALLEL}" >&2
  exit 2
fi

should_run_shard() {
  local shard_index="$1"
  if [[ -z "${SHARD_FILTER}" ]]; then
    return 0
  fi
  local item
  IFS=',' read -r -a shard_filter_items <<< "${SHARD_FILTER}"
  for item in "${shard_filter_items[@]}"; do
    if [[ "${item}" == "${shard_index}" ]]; then
      return 0
    fi
  done
  return 1
}

LR_ID="${LR//./p}"
LR_ID="${LR_ID//-/m}"
STEP_ID="${STEPS}_${SAVE_STEPS//,/x}"
BASE_RUN_ID="${DATASET}_${PROTOCOL}_${SCHEME}_arbank_semantic_s${STEP_ID}_lr${LR_ID}_${REF_ID}_refseed${REFSEED}"
if [[ -n "${RUN_SUFFIX}" ]]; then
  BASE_RUN_ID="${BASE_RUN_ID}_${RUN_SUFFIX}"
fi
MERGED_RUN_ID="${BASE_RUN_ID}_merged${SHARD_COUNT}"
MERGED_RUN_DIR="${OUTPUT_ROOT}/outputs/${DATASET}/${PROTOCOL}/seed${RUN_SEED}/${MERGED_RUN_ID}"
LOG_DIR="${OUTPUT_ROOT}/job_logs/${BASE_RUN_ID}"
mkdir -p "${LOG_DIR}"

MAX_SAMPLE_ARGS=()
if [[ -n "${MAX_ID_SAMPLES:-}" ]]; then
  MAX_SAMPLE_ARGS+=(--max-id-samples "${MAX_ID_SAMPLES}")
fi
if [[ -n "${MAX_OOD_SAMPLES:-}" ]]; then
  MAX_SAMPLE_ARGS+=(--max-ood-samples "${MAX_OOD_SAMPLES}")
fi

run_shard() {
  local shard_index="$1"
  local gpu_id="$2"
  local run_id="${BASE_RUN_ID}_shard${shard_index}of${SHARD_COUNT}"
  local log_path="${LOG_DIR}/shard${shard_index}of${SHARD_COUNT}.log"
  echo "[tarr-ar-bank-shard] START shard=${shard_index}/${SHARD_COUNT} gpu=${gpu_id} run_id=${run_id} $(date -Iseconds)" | tee "${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" conda run --no-capture-output -n openood python scripts_my/tarr/eval.py run-response \
    --dataset "${DATASET}" \
    --baseline-protocol "${PROTOCOL}" \
    --scheme "${SCHEME}" \
    --run-id "${run_id}" \
    --experiment-tag ar_bank_sharded_focused \
    --ablation-type accept_reject_sharded_focused \
    --output-root "${OUTPUT_ROOT}" \
    --reference-config "${REF_ID}:${REF_SPEC}" \
    --use-prebuilt-reference-set \
    --tta-mode ar_bank \
    --accept-probe-types "${ACCEPT_PROBE_TYPES}" \
    --reject-probe-types "${REJECT_PROBE_TYPES}" \
    --perturbation-response "${PERTURBATION_RESPONSE}" \
    --perturbation-kind "${PERTURBATION_KIND}" \
    --perturbation-eps "${PERTURBATION_EPS}" \
    --perturbation-repeats "${PERTURBATION_REPEATS}" \
    --perturbation-seed "${PERTURBATION_SEED}" \
    --steps "${STEPS}" \
    --save-steps "${SAVE_STEPS}" \
    --lr "${LR}" \
    --update-scope "${UPDATE_SCOPE}" \
    --runtime-mode auto \
    --score-rule probe_all \
    --batch-size "${BATCH_SIZE}" \
    --reference-set-batch-size "${REFERENCE_SET_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${RUN_SEED}" \
    --save-tta-response \
    --tta-response-shard-size "${TTA_RESPONSE_SHARD_SIZE}" \
    --target-shard-count "${SHARD_COUNT}" \
    --target-shard-index "${shard_index}" \
    --debug-output-mode none \
    "${MAX_SAMPLE_ARGS[@]}" \
    --overwrite 2>&1 | tee -a "${log_path}"
  echo "[tarr-ar-bank-shard] DONE shard=${shard_index}/${SHARD_COUNT} run_id=${run_id} $(date -Iseconds)" | tee -a "${log_path}"
}

pids=()
for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
  if ! should_run_shard "${shard_index}"; then
    echo "[tarr-ar-bank-shard] SKIP shard=${shard_index}/${SHARD_COUNT} by SHARD_FILTER"
    continue
  fi
  gpu_index=$((shard_index % ${#GPU_LIST[@]}))
  run_shard "${shard_index}" "${GPU_LIST[${gpu_index}]}" &
  pids+=("$!")
  while [[ "${#pids[@]}" -ge "${MAX_PARALLEL}" ]]; do
    wait -n
    live=()
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        live+=("${pid}")
      fi
    done
    pids=("${live[@]}")
  done
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

if [[ "${SKIP_MERGE_SCORE}" == "1" ]]; then
  echo "[tarr-ar-bank-shard] SKIP merge/validate/score merged_run_dir=${MERGED_RUN_DIR} $(date -Iseconds)"
  exit 0
fi

MERGE_ARGS=()
for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
  MERGE_ARGS+=(
    --input-run-dir
    "${OUTPUT_ROOT}/outputs/${DATASET}/${PROTOCOL}/seed${RUN_SEED}/${BASE_RUN_ID}_shard${shard_index}of${SHARD_COUNT}"
  )
done

conda run --no-capture-output -n openood python scripts_my/tarr/merge_tta_response_shards.py \
  --dataset "${DATASET}" \
  --scheme "${SCHEME}" \
  --reference-config-id "${REF_ID}" \
  --output-run-dir "${MERGED_RUN_DIR}" \
  "${MERGE_ARGS[@]}" \
  --overwrite

conda run --no-capture-output -n openood python scripts_my/tarr/cache.py validate \
  --run-dir "${MERGED_RUN_DIR}" \
  --scheme "${SCHEME}" \
  --reference-config-id "${REF_ID}" \
  --expect-dataset "${DATASET}"

for side in both clean csid; do
  conda run --no-capture-output -n openood python scripts_my/tarr/cache.py score \
    --run-dir "${MERGED_RUN_DIR}" \
    --scheme "${SCHEME}" \
    --reference-config-id "${REF_ID}" \
    --dataset "${DATASET}" \
    --fsood-id-side "${side}" \
    --response-step all \
    --score-rule probe_all \
    --overwrite
done

echo "[tarr-ar-bank-shard] DONE merged_run_dir=${MERGED_RUN_DIR} $(date -Iseconds)"
