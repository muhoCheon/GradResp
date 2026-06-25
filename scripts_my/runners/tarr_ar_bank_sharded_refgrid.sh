#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat >&2 <<'EOF'
usage:
  tarr_ar_bank_sharded_refgrid.sh <cifar100|imagenet200> <shard_count>

Runs the fresh semantic A/R response-bank setup across the 15-reference grid by
calling scripts_my/runners/tarr_ar_bank_sharded_focused.sh once per reference.

Environment overrides:
  LR=1e-2
  STEPS=30
  SAVE_STEPS=5,10,30
  GPUS=0,1
  MAX_PARALLEL=<per-reference shard concurrency>
  RUN_SUFFIX=<optional>
  REFSEED=0
  FORCE=1             rerun references even if merged output exists
  REFERENCE_FILTER=   comma-separated reference ids to run
  SKIP_COLLECT_SCORE=0
EOF
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

DATASET="$1"
SHARD_COUNT="$2"

case "${DATASET}" in
  cifar100|imagenet200) ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

REFSEED="${REFSEED:-0}"
RUN_SEED="${RUN_SEED:-0}"
PROTOCOL="${PROTOCOL:-eval_api}"
SCHEME="${SCHEME:-fsood}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results_test/tarr}"
LR="${LR:-1e-2}"
STEPS="${STEPS:-30}"
SAVE_STEPS="${SAVE_STEPS:-5,10,30}"
FORCE="${FORCE:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
REFERENCE_FILTER="${REFERENCE_FILTER:-}"
SKIP_COLLECT_SCORE="${SKIP_COLLECT_SCORE:-0}"
SUMMARY_DIR="${OUTPUT_ROOT}/summary"
mkdir -p "${SUMMARY_DIR}" "${OUTPUT_ROOT}/job_logs"

REFERENCE_CONFIGS=(
  "all_rpc8:per_class=8,filter=all,seed=${REFSEED}"
  "all_rpc16:per_class=16,filter=all,seed=${REFSEED}"
  "all_rpc32:per_class=32,filter=all,seed=${REFSEED}"
  "correct_rpc8:per_class=8,filter=correct,seed=${REFSEED}"
  "correct_rpc16:per_class=16,filter=correct,seed=${REFSEED}"
  "correct_rpc32:per_class=32,filter=correct,seed=${REFSEED}"
  "highconf09_rpc8:per_class=8,filter=high_confidence,min_confidence=0.9,seed=${REFSEED}"
  "highconf09_rpc16:per_class=16,filter=high_confidence,min_confidence=0.9,seed=${REFSEED}"
  "highconf09_rpc32:per_class=32,filter=high_confidence,min_confidence=0.9,seed=${REFSEED}"
  "correcthigh09_rpc8:per_class=8,filter=correct_high_confidence,min_confidence=0.9,seed=${REFSEED}"
  "correcthigh09_rpc16:per_class=16,filter=correct_high_confidence,min_confidence=0.9,seed=${REFSEED}"
  "correcthigh09_rpc32:per_class=32,filter=correct_high_confidence,min_confidence=0.9,seed=${REFSEED}"
  "strat_rpc8:per_class=8,filter=correct_confidence_stratified,seed=${REFSEED}"
  "strat_rpc16:per_class=16,filter=correct_confidence_stratified,seed=${REFSEED}"
  "strat_rpc32:per_class=32,filter=correct_confidence_stratified,seed=${REFSEED}"
)

should_run_ref() {
  local ref_id="$1"
  if [[ -z "${REFERENCE_FILTER}" ]]; then
    return 0
  fi
  local item
  IFS=',' read -r -a filter_items <<< "${REFERENCE_FILTER}"
  for item in "${filter_items[@]}"; do
    if [[ "${item}" == "${ref_id}" ]]; then
      return 0
    fi
  done
  return 1
}

lr_id="${LR//./p}"
lr_id="${lr_id//-/m}"
step_id="${STEPS}_${SAVE_STEPS//,/x}"
base_prefix="${DATASET}_${PROTOCOL}_${SCHEME}_arbank_semantic_s${step_id}_lr${lr_id}"
log_path="${OUTPUT_ROOT}/job_logs/${base_prefix}_refgrid_refseed${REFSEED}.log"

echo "[tarr-ar-bank-refgrid] START dataset=${DATASET} shard_count=${SHARD_COUNT} refseed=${REFSEED} $(date -Iseconds)" | tee "${log_path}"

for cfg in "${REFERENCE_CONFIGS[@]}"; do
  ref_id="${cfg%%:*}"
  ref_spec="${cfg#*:}"
  if ! should_run_ref "${ref_id}"; then
    echo "[tarr-ar-bank-refgrid] SKIP filter ref=${ref_id}" | tee -a "${log_path}"
    continue
  fi

  run_id="${base_prefix}_${ref_id}_refseed${REFSEED}"
  if [[ -n "${RUN_SUFFIX}" ]]; then
    run_id="${run_id}_${RUN_SUFFIX}"
  fi
  merged_run_dir="${OUTPUT_ROOT}/outputs/${DATASET}/${PROTOCOL}/seed${RUN_SEED}/${run_id}_merged${SHARD_COUNT}"

  if [[ "${FORCE}" != "1" && -f "${merged_run_dir}/${SCHEME}/references/${ref_id}/score_results/score.json" ]]; then
    echo "[tarr-ar-bank-refgrid] SKIP existing ref=${ref_id} merged_run_dir=${merged_run_dir}" | tee -a "${log_path}"
    continue
  fi

  echo "[tarr-ar-bank-refgrid] RUN ref=${ref_id} spec=${ref_spec} $(date -Iseconds)" | tee -a "${log_path}"
  REFSEED="${REFSEED}" \
  RUN_SEED="${RUN_SEED}" \
  PROTOCOL="${PROTOCOL}" \
  SCHEME="${SCHEME}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  LR="${LR}" \
  STEPS="${STEPS}" \
  SAVE_STEPS="${SAVE_STEPS}" \
  scripts_my/runners/tarr_ar_bank_sharded_focused.sh \
    "${DATASET}" \
    "${ref_id}" \
    "${ref_spec}" \
    "${SHARD_COUNT}" 2>&1 | tee -a "${log_path}"
done

if [[ "${SKIP_COLLECT_SCORE}" == "1" ]]; then
  echo "[tarr-ar-bank-refgrid] SKIP collect-score" | tee -a "${log_path}"
else
  conda run --no-capture-output -n openood python scripts_my/tarr/reports.py collect-score \
    --dataset "${DATASET}" \
    --baseline-protocol "${PROTOCOL}" \
    --runs-root "${OUTPUT_ROOT}/outputs" \
    --output-csv "${SUMMARY_DIR}/${DATASET}_${PROTOCOL}_score_results.csv" 2>&1 | tee -a "${log_path}"
fi

echo "[tarr-ar-bank-refgrid] DONE dataset=${DATASET} $(date -Iseconds)" | tee -a "${log_path}"
