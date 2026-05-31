# TARR Commands

This file collects runnable commands for the canonical TARR artifact pipeline.

```text
train_candidate_metadata -> reference_set -> tta_response -> score_result
```

All examples assume the `openood` conda environment and the repository root as the working directory.

Stage 1 and Stage 2 commands print a one-line summary by default. Add
`--print-json` only when you want the full machine-readable summary in stdout.

## Stage 1: Build `train_candidate_metadata`

Stage 1 scans the ID train split once with the pretrained model and writes metadata that can be reused across reference filters, reference seeds, `reference_per_class`, TTA configs, and score rules.

Output:

```text
results_test/tarr/train_candidate_metadata/<dataset>/<candidate_id>/
  manifest.json
  candidates.npz
```

Add `--rebuild-train-candidate-metadata` only when you want to ignore an existing matching artifact and rebuild it.

### CIFAR-10

```bash
conda run --no-capture-output -n openood python scripts_my/tarr/reference.py build-train-metadata \
  --dataset cifar10 \
  --output-root results_test/tarr \
  --train-candidate-batch-size 2048 \
  --num-workers 8
```

### CIFAR-100

```bash
conda run --no-capture-output -n openood python scripts_my/tarr/reference.py build-train-metadata \
  --dataset cifar100 \
  --output-root results_test/tarr \
  --train-candidate-batch-size 2048 \
  --num-workers 8
```

### ImageNet-200

```bash
conda run --no-capture-output -n openood python scripts_my/tarr/reference.py build-train-metadata \
  --dataset imagenet200 \
  --output-root results_test/tarr \
  --train-candidate-batch-size 512 \
  --num-workers 8
```

### ImageNet-1K

ImageNet-1K uses the CLI dataset name `imagenet`.

```bash
conda run --no-capture-output -n openood python scripts_my/tarr/reference.py build-train-metadata \
  --dataset imagenet \
  --output-root results_test/tarr \
  --train-candidate-batch-size 256 \
  --num-workers 8
```

## Stage 2: Build `reference_set`

Stage 2 selects a concrete reference set from an existing `train_candidate_metadata` artifact.

Output:

```text
results_test/tarr/reference_sets/<dataset>/<reference_config_id>/seed<seed>/<reference_set_id>/
  manifest.json
  reference_set.npz
  selected_samples.csv
  preview/                 # optional, only with --write-preview
```

### Generic Template

Replace `<candidate_id>` with the directory created by Stage 1.

```bash
conda run --no-capture-output -n openood python scripts_my/tarr/reference.py build-reference-set \
  --dataset <dataset> \
  --metadata results_test/tarr/train_candidate_metadata/<dataset>/<candidate_id>/candidates.npz \
  --reference-config <reference_config_id>:per_class=<k>,filter=<filter>,seed=0 \
  --output-root results_test/tarr \
  --reference-set-batch-size <batch_size> \
  --num-workers 8
```

To copy a small image preview of the selected references, add:

```bash
  --write-preview \
  --preview-per-class 8
```

### Reference Config Policy

Use the same reference grid for CIFAR-10, CIFAR-100, ImageNet-200,
and ImageNet-1K:

```text
all_rpc8, all_rpc16, all_rpc32
correct_rpc8, correct_rpc16, correct_rpc32
high-confidence rpc8/16/32
correct-high-confidence rpc8/16/32
strat_rpc8, strat_rpc16, strat_rpc32
```

Use three reference seeds:

```text
seed=0, seed=1, seed=2
```

Confidence thresholds are dataset-specific and are included in the config id:

| Dataset CLI name | High-confidence id | High-confidence threshold | Correct-high id | Correct-high threshold |
|---|---|---:|---|---:|
| `cifar10` | `highconf09_*` | `0.9` | `correcthigh09_*` | `0.9` |
| `cifar100` | `highconf09_*` | `0.9` | `correcthigh09_*` | `0.9` |
| `imagenet200` | `highconf09_*` | `0.9` | `correcthigh09_*` | `0.9` |
| `imagenet` | `highconf08_*` | `0.8` | `correcthigh075_*` | `0.75` |

The commands below copy a human-inspection preview by default:

```bash
--write-preview --preview-per-class 8
```

Use `--preview-per-class 0` only for final inspection of a small number of
chosen reference sets, because it copies every selected image.

### Current Local Commands

### CIFAR-10

```bash
DATASET=cifar10
METADATA=results_test/tarr/train_candidate_metadata/cifar10/fae2630ca80585a073bd6990f116129faa3b95960dc4dccfc4a95f10829893a6/candidates.npz
REFERENCE_SET_BATCH_SIZE=2048

CONFIGS=(
  "all_rpc8:per_class=8,filter=all"
  "all_rpc16:per_class=16,filter=all"
  "all_rpc32:per_class=32,filter=all"
  "correct_rpc8:per_class=8,filter=correct"
  "correct_rpc16:per_class=16,filter=correct"
  "correct_rpc32:per_class=32,filter=correct"
  "highconf09_rpc8:per_class=8,filter=high_confidence,min_confidence=0.9"
  "highconf09_rpc16:per_class=16,filter=high_confidence,min_confidence=0.9"
  "highconf09_rpc32:per_class=32,filter=high_confidence,min_confidence=0.9"
  "correcthigh09_rpc8:per_class=8,filter=correct_high_confidence,min_confidence=0.9"
  "correcthigh09_rpc16:per_class=16,filter=correct_high_confidence,min_confidence=0.9"
  "correcthigh09_rpc32:per_class=32,filter=correct_high_confidence,min_confidence=0.9"
  "strat_rpc8:per_class=8,filter=correct_confidence_stratified"
  "strat_rpc16:per_class=16,filter=correct_confidence_stratified"
  "strat_rpc32:per_class=32,filter=correct_confidence_stratified"
)

for SEED in 0 1 2; do
  for CFG in "${CONFIGS[@]}"; do
    conda run --no-capture-output -n openood python scripts_my/tarr/reference.py build-reference-set \
      --dataset "${DATASET}" \
      --metadata "${METADATA}" \
      --reference-config "${CFG},seed=${SEED}" \
      --output-root results_test/tarr \
      --reference-set-batch-size "${REFERENCE_SET_BATCH_SIZE}" \
      --num-workers 8 \
      --write-preview \
      --preview-per-class 8
  done
done
```

### CIFAR-100

```bash
DATASET=cifar100
METADATA=results_test/tarr/train_candidate_metadata/cifar100/f13cf11ae61765816c3fd1781df6f20bca665e1bd28fae5899541359b3c400d9/candidates.npz
REFERENCE_SET_BATCH_SIZE=2048

CONFIGS=(
  "all_rpc8:per_class=8,filter=all"
  "all_rpc16:per_class=16,filter=all"
  "all_rpc32:per_class=32,filter=all"
  "correct_rpc8:per_class=8,filter=correct"
  "correct_rpc16:per_class=16,filter=correct"
  "correct_rpc32:per_class=32,filter=correct"
  "highconf09_rpc8:per_class=8,filter=high_confidence,min_confidence=0.9"
  "highconf09_rpc16:per_class=16,filter=high_confidence,min_confidence=0.9"
  "highconf09_rpc32:per_class=32,filter=high_confidence,min_confidence=0.9"
  "correcthigh09_rpc8:per_class=8,filter=correct_high_confidence,min_confidence=0.9"
  "correcthigh09_rpc16:per_class=16,filter=correct_high_confidence,min_confidence=0.9"
  "correcthigh09_rpc32:per_class=32,filter=correct_high_confidence,min_confidence=0.9"
  "strat_rpc8:per_class=8,filter=correct_confidence_stratified"
  "strat_rpc16:per_class=16,filter=correct_confidence_stratified"
  "strat_rpc32:per_class=32,filter=correct_confidence_stratified"
)

for SEED in 0 1 2; do
  for CFG in "${CONFIGS[@]}"; do
    conda run --no-capture-output -n openood python scripts_my/tarr/reference.py build-reference-set \
      --dataset "${DATASET}" \
      --metadata "${METADATA}" \
      --reference-config "${CFG},seed=${SEED}" \
      --output-root results_test/tarr \
      --reference-set-batch-size "${REFERENCE_SET_BATCH_SIZE}" \
      --num-workers 8 \
      --write-preview \
      --preview-per-class 8
  done
done
```

### ImageNet-200

```bash
DATASET=imagenet200
METADATA=results_test/tarr/train_candidate_metadata/imagenet200/a3604f8c99b3d6a9604a4c760d7d58788c39790ea8ce6594a8ca3daf79ec82a0/candidates.npz
REFERENCE_SET_BATCH_SIZE=512

CONFIGS=(
  "all_rpc8:per_class=8,filter=all"
  "all_rpc16:per_class=16,filter=all"
  "all_rpc32:per_class=32,filter=all"
  "correct_rpc8:per_class=8,filter=correct"
  "correct_rpc16:per_class=16,filter=correct"
  "correct_rpc32:per_class=32,filter=correct"
  "highconf09_rpc8:per_class=8,filter=high_confidence,min_confidence=0.9"
  "highconf09_rpc16:per_class=16,filter=high_confidence,min_confidence=0.9"
  "highconf09_rpc32:per_class=32,filter=high_confidence,min_confidence=0.9"
  "correcthigh09_rpc8:per_class=8,filter=correct_high_confidence,min_confidence=0.9"
  "correcthigh09_rpc16:per_class=16,filter=correct_high_confidence,min_confidence=0.9"
  "correcthigh09_rpc32:per_class=32,filter=correct_high_confidence,min_confidence=0.9"
  "strat_rpc8:per_class=8,filter=correct_confidence_stratified"
  "strat_rpc16:per_class=16,filter=correct_confidence_stratified"
  "strat_rpc32:per_class=32,filter=correct_confidence_stratified"
)

for SEED in 0 1 2; do
  for CFG in "${CONFIGS[@]}"; do
    conda run --no-capture-output -n openood python scripts_my/tarr/reference.py build-reference-set \
      --dataset "${DATASET}" \
      --metadata "${METADATA}" \
      --reference-config "${CFG},seed=${SEED}" \
      --output-root results_test/tarr \
      --reference-set-batch-size "${REFERENCE_SET_BATCH_SIZE}" \
      --num-workers 8 \
      --write-preview \
      --preview-per-class 8
  done
done
```

### ImageNet-1K

ImageNet-1K uses the CLI dataset name `imagenet`.

```bash
DATASET=imagenet
METADATA=results_test/tarr/train_candidate_metadata/imagenet/efde2be04820a489a46bf993fe028dc89409c81b1d2e4f087fbc00714d24f8e5/candidates.npz
REFERENCE_SET_BATCH_SIZE=256

CONFIGS=(
  "all_rpc8:per_class=8,filter=all"
  "all_rpc16:per_class=16,filter=all"
  "all_rpc32:per_class=32,filter=all"
  "correct_rpc8:per_class=8,filter=correct"
  "correct_rpc16:per_class=16,filter=correct"
  "correct_rpc32:per_class=32,filter=correct"
  "highconf08_rpc8:per_class=8,filter=high_confidence,min_confidence=0.8"
  "highconf08_rpc16:per_class=16,filter=high_confidence,min_confidence=0.8"
  "highconf08_rpc32:per_class=32,filter=high_confidence,min_confidence=0.8"
  "correcthigh075_rpc8:per_class=8,filter=correct_high_confidence,min_confidence=0.75"
  "correcthigh075_rpc16:per_class=16,filter=correct_high_confidence,min_confidence=0.75"
  "correcthigh075_rpc32:per_class=32,filter=correct_high_confidence,min_confidence=0.75"
  "strat_rpc8:per_class=8,filter=correct_confidence_stratified"
  "strat_rpc16:per_class=16,filter=correct_confidence_stratified"
  "strat_rpc32:per_class=32,filter=correct_confidence_stratified"
)

for SEED in 0 1 2; do
  for CFG in "${CONFIGS[@]}"; do
    conda run --no-capture-output -n openood python scripts_my/tarr/reference.py build-reference-set \
      --dataset "${DATASET}" \
      --metadata "${METADATA}" \
      --reference-config "${CFG},seed=${SEED}" \
      --output-root results_test/tarr \
      --reference-set-batch-size "${REFERENCE_SET_BATCH_SIZE}" \
      --num-workers 8 \
      --write-preview \
      --preview-per-class 8
  done
done
```

## Stage 3: Run `tta_response`

Stage 3 runs target-only TTA and measures how each `reference_set` responds.

### Stage 3 Reference Grid

Each Stage 3 run uses one reference seed and 15 prebuilt reference configs.
Use a separate `run_id` for each reference seed:

```text
<dataset>_<protocol>_<scheme>_<tta_id>_refseed0
<dataset>_<protocol>_<scheme>_<tta_id>_refseed1
<dataset>_<protocol>_<scheme>_<tta_id>_refseed2
```

For CIFAR-10, CIFAR-100, and ImageNet-200:

```bash
REFSEED=0

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
```

For ImageNet-1K, use the CLI dataset name `imagenet` and lower
high-confidence thresholds:

```bash
REFSEED=0

REFERENCE_CONFIGS=(
  "all_rpc8:per_class=8,filter=all,seed=${REFSEED}"
  "all_rpc16:per_class=16,filter=all,seed=${REFSEED}"
  "all_rpc32:per_class=32,filter=all,seed=${REFSEED}"
  "correct_rpc8:per_class=8,filter=correct,seed=${REFSEED}"
  "correct_rpc16:per_class=16,filter=correct,seed=${REFSEED}"
  "correct_rpc32:per_class=32,filter=correct,seed=${REFSEED}"
  "highconf08_rpc8:per_class=8,filter=high_confidence,min_confidence=0.8,seed=${REFSEED}"
  "highconf08_rpc16:per_class=16,filter=high_confidence,min_confidence=0.8,seed=${REFSEED}"
  "highconf08_rpc32:per_class=32,filter=high_confidence,min_confidence=0.8,seed=${REFSEED}"
  "correcthigh075_rpc8:per_class=8,filter=correct_high_confidence,min_confidence=0.75,seed=${REFSEED}"
  "correcthigh075_rpc16:per_class=16,filter=correct_high_confidence,min_confidence=0.75,seed=${REFSEED}"
  "correcthigh075_rpc32:per_class=32,filter=correct_high_confidence,min_confidence=0.75,seed=${REFSEED}"
  "strat_rpc8:per_class=8,filter=correct_confidence_stratified,seed=${REFSEED}"
  "strat_rpc16:per_class=16,filter=correct_confidence_stratified,seed=${REFSEED}"
  "strat_rpc32:per_class=32,filter=correct_confidence_stratified,seed=${REFSEED}"
)
```

When writing the final `eval.py run-response` command, expand the array as:

```bash
REF_ARGS=()
for CFG in "${REFERENCE_CONFIGS[@]}"; do
  REF_ARGS+=(--reference-config "${CFG}")
done
```

Strict Stage 3 mode requires prebuilt `reference_set` artifacts:

```bash
conda run --no-capture-output -n openood python scripts_my/tarr/eval.py run-response \
  --dataset cifar100 \
  --baseline-protocol eval_api \
  --scheme fsood \
  --run-id <run_id> \
  --output-root results_test/tarr \
  "${REF_ARGS[@]}" \
  --use-prebuilt-reference-set \
  --objective predicted_label_ce \
  --steps 5 \
  --lr 1e-2 \
  --update-scope classifier \
  --runtime-mode auto \
  --score-rule all \
  --save-tta-response \
  --tta-response-shard-size 1024 \
  --debug-output-mode none
```

Convenience orchestration can still use `run-all`, which may build missing Stage 1/2 artifacts before running Stage 3/4:

```bash
conda run --no-capture-output -n openood python scripts_my/tarr/eval.py run-all \
  --dataset cifar10 \
  --baseline-protocol eval_api \
  --scheme fsood \
  --run-id <run_id> \
  --output-root results_test/tarr \
  --reference-config all_rpc32:per_class=32,filter=all,seed=0 \
  --objective predicted_label_ce \
  --steps 5 \
  --lr 1e-2 \
  --update-scope classifier \
  --runtime-mode auto \
  --score-rule all \
  --save-tta-response
```

ImageNet-scale full runs and CIFAR-100 full broad-search runs should use
sharded `tta_response` and usually skip debug CSV rows:

```bash
  --tta-response-shard-size 1024 \
  --debug-output-mode none
```

### Dataset-Wise Broad Search Template

Use this template for full broad-search runs. It assumes Stage 2 has already
built all 15 `reference_set` artifacts for `REFSEED`.

Run only one dataset at a time. Use both GPUs to split TTA candidates for the
active dataset, finish analysis, then move to the next dataset:

```text
cifar10 -> cifar100 -> imagenet200 -> imagenet
```

Set dataset/runtime variables first:

```bash
# Choose one: cifar10, cifar100, imagenet200, imagenet
DATASET=cifar10
PROTOCOL=eval_api
SCHEME=fsood
REFSEED=0

# CIFAR-10 broad-search default. Resource smoke supports 4 Stage 3 processes
# per GPU with this setting.
BATCH_SIZE=512
REFERENCE_SET_BATCH_SIZE=2048
TTA_RESPONSE_SHARD_SIZE=1024
DEBUG_OUTPUT_MODE=none
NUM_WORKERS=0

# CIFAR-100 default. CIFAR-100 must use sharded tta_response for full broad
# search to avoid high host RAM usage. Resource smoke supports up to five
# Stage 3 processes per GPU with num_workers=0.
# BATCH_SIZE=512
# REFERENCE_SET_BATCH_SIZE=2048
# TTA_RESPONSE_SHARD_SIZE=1024
# DEBUG_OUTPUT_MODE=none
# NUM_WORKERS=0

# ImageNet-200 default. Resource smoke supports up to five Stage 3 processes
# per GPU with num_workers=2.
# BATCH_SIZE=64
# REFERENCE_SET_BATCH_SIZE=1024
# TTA_RESPONSE_SHARD_SIZE=1024
# DEBUG_OUTPUT_MODE=none
# NUM_WORKERS=2

# ImageNet-1K conservative default.
# BATCH_SIZE=64
# REFERENCE_SET_BATCH_SIZE=1024
# TTA_RESPONSE_SHARD_SIZE=1024
# DEBUG_OUTPUT_MODE=none
# NUM_WORKERS=4
```

Recommended Stage 3 process concurrency:

| Dataset | Processes per GPU | Notes |
|---|---:|---|
| `cifar10` | 4 | Measured default. Five can improve raw Stage 3 throughput slightly but leaves less scheduling margin. |
| `cifar100` | 5 | Measured with sharded `tta_response`, `debug_output_mode=none`, `num_workers=0`; reduce to 4 if Stage 4 overlap or soft-view memory pressure appears. |
| `imagenet200` | 5 | Measured with sharded `tta_response`, `debug_output_mode=none`, `num_workers=2`; reduce to 4 if Stage 4 overlap, host IO pressure, or soft-view memory pressure appears. |
| `imagenet` | 1 | ImageNet-1K runs should stay conservative until runtime/memory is remeasured. |

Choose the dataset-specific reference grid. For `cifar10`, `cifar100`, and
`imagenet200`:

```bash
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
```

For ImageNet-1K (`DATASET=imagenet`):

```bash
REFERENCE_CONFIGS=(
  "all_rpc8:per_class=8,filter=all,seed=${REFSEED}"
  "all_rpc16:per_class=16,filter=all,seed=${REFSEED}"
  "all_rpc32:per_class=32,filter=all,seed=${REFSEED}"
  "correct_rpc8:per_class=8,filter=correct,seed=${REFSEED}"
  "correct_rpc16:per_class=16,filter=correct,seed=${REFSEED}"
  "correct_rpc32:per_class=32,filter=correct,seed=${REFSEED}"
  "highconf08_rpc8:per_class=8,filter=high_confidence,min_confidence=0.8,seed=${REFSEED}"
  "highconf08_rpc16:per_class=16,filter=high_confidence,min_confidence=0.8,seed=${REFSEED}"
  "highconf08_rpc32:per_class=32,filter=high_confidence,min_confidence=0.8,seed=${REFSEED}"
  "correcthigh075_rpc8:per_class=8,filter=correct_high_confidence,min_confidence=0.75,seed=${REFSEED}"
  "correcthigh075_rpc16:per_class=16,filter=correct_high_confidence,min_confidence=0.75,seed=${REFSEED}"
  "correcthigh075_rpc32:per_class=32,filter=correct_high_confidence,min_confidence=0.75,seed=${REFSEED}"
  "strat_rpc8:per_class=8,filter=correct_confidence_stratified,seed=${REFSEED}"
  "strat_rpc16:per_class=16,filter=correct_confidence_stratified,seed=${REFSEED}"
  "strat_rpc32:per_class=32,filter=correct_confidence_stratified,seed=${REFSEED}"
)
```

Build reusable CLI argument arrays:

```bash
REF_ARGS=()
REF_IDS=()
for CFG in "${REFERENCE_CONFIGS[@]}"; do
  REF_ARGS+=(--reference-config "${CFG}")
  REF_IDS+=("${CFG%%:*}")
done
```

Choose one TTA candidate:

```bash
# predicted_label_ce, steps=5, lr=1e-2
TTA_ID=plce_s5_lr1e2
OBJECTIVE=predicted_label_ce
STEPS=5
LR=1e-2
PERT_ARGS=(--perturbation-response none)

# entropy, steps=5, lr=1e-2
# TTA_ID=ent_s5_lr1e2
# OBJECTIVE=entropy
# STEPS=5
# LR=1e-2
# PERT_ARGS=(--perturbation-response none)

# predicted_label_ce, steps=10, lr=3e-2
# TTA_ID=plce_s10_lr3e2
# OBJECTIVE=predicted_label_ce
# STEPS=10
# LR=3e-2
# PERT_ARGS=(--perturbation-response none)

# predicted_label_ce, steps=30, lr=1e-2
# TTA_ID=plce_s30_lr1e2
# OBJECTIVE=predicted_label_ce
# STEPS=30
# LR=1e-2
# PERT_ARGS=(--perturbation-response none)

# predicted_label_ce, steps=20, lr=1e-2
# TTA_ID=plce_s20_lr1e2
# OBJECTIVE=predicted_label_ce
# STEPS=20
# LR=1e-2
# PERT_ARGS=(--perturbation-response none)

# entropy, steps=10, lr=1e-2
# TTA_ID=ent_s10_lr1e2
# OBJECTIVE=entropy
# STEPS=10
# LR=1e-2
# PERT_ARGS=(--perturbation-response none)

# memo_marginal_entropy, pixel gaussian eps=0.01 repeats=4
# TTA_ID=memo_s5_lr1e2_pixgauss_eps1e2_r4
# OBJECTIVE=memo_marginal_entropy
# STEPS=5
# LR=1e-2
# PERT_ARGS=(--perturbation-response pixel --perturbation-kind gaussian --perturbation-eps 0.01 --perturbation-repeats 4 --perturbation-seed 0)

# view_consistency_js, pixel gaussian eps=0.01 repeats=4
# TTA_ID=vcjs_s5_lr1e2_pixgauss_eps1e2_r4
# OBJECTIVE=view_consistency_js
# STEPS=5
# LR=1e-2
# PERT_ARGS=(--perturbation-response pixel --perturbation-kind gaussian --perturbation-eps 0.01 --perturbation-repeats 4 --perturbation-seed 0)

# view_consistency_kl, pixel gaussian eps=0.01 repeats=4
# TTA_ID=vckl_s5_lr1e2_pixgauss_eps1e2_r4
# OBJECTIVE=view_consistency_kl
# STEPS=5
# LR=1e-2
# PERT_ARGS=(--perturbation-response pixel --perturbation-kind gaussian --perturbation-eps 0.01 --perturbation-repeats 4 --perturbation-seed 0)

# entropy_consistency, pixel gaussian eps=0.01 repeats=4
# TTA_ID=hcons_s5_lr1e2_pixgauss_eps1e2_r4
# OBJECTIVE=entropy_consistency
# STEPS=5
# LR=1e-2
# PERT_ARGS=(--perturbation-response pixel --perturbation-kind gaussian --perturbation-eps 0.01 --perturbation-repeats 4 --perturbation-seed 0)
```

Run Stage 3:

```bash
RUN_ID=${DATASET}_${PROTOCOL}_${SCHEME}_${TTA_ID}_refseed${REFSEED}
RUN_DIR=results_test/tarr/outputs/${DATASET}/${PROTOCOL}/seed0/${RUN_ID}

conda run --no-capture-output -n openood python scripts_my/tarr/eval.py run-response \
  --dataset "${DATASET}" \
  --baseline-protocol "${PROTOCOL}" \
  --scheme "${SCHEME}" \
  --run-id "${RUN_ID}" \
  --output-root results_test/tarr \
  "${REF_ARGS[@]}" \
  --use-prebuilt-reference-set \
  --objective "${OBJECTIVE}" \
  --steps "${STEPS}" \
  --lr "${LR}" \
  "${PERT_ARGS[@]}" \
  --update-scope classifier \
  --runtime-mode auto \
  --score-rule all \
  --batch-size "${BATCH_SIZE}" \
  --reference-set-batch-size "${REFERENCE_SET_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --save-tta-response \
  --tta-response-shard-size "${TTA_RESPONSE_SHARD_SIZE}" \
  --debug-output-mode "${DEBUG_OUTPUT_MODE}" \
  --overwrite
```

Run Stage 4 and diagnostics for every reference config:

```bash
for REF_ID in "${REF_IDS[@]}"; do
  conda run --no-capture-output -n openood python scripts_my/tarr/cache.py validate \
    --run-dir "${RUN_DIR}" \
    --scheme "${SCHEME}" \
    --reference-config-id "${REF_ID}" \
    --expect-dataset "${DATASET}"

  for SIDE in both clean csid; do
    conda run --no-capture-output -n openood python scripts_my/tarr/cache.py score \
      --run-dir "${RUN_DIR}" \
      --scheme "${SCHEME}" \
      --reference-config-id "${REF_ID}" \
      --dataset "${DATASET}" \
      --fsood-id-side "${SIDE}" \
      --score-rule all \
      --overwrite
  done

  for SIDE in both clean csid; do
    conda run --no-capture-output -n openood python scripts_my/tarr/cache.py score \
      --run-dir "${RUN_DIR}" \
      --scheme "${SCHEME}" \
      --reference-config-id "${REF_ID}" \
      --dataset "${DATASET}" \
      --fsood-id-side "${SIDE}" \
      --vector-score-rule all \
      --overwrite
  done

  if [[ "${PERT_ARGS[*]}" != "--perturbation-response none" ]]; then
    for SIDE in both clean csid; do
      conda run --no-capture-output -n openood python scripts_my/tarr/cache.py score \
        --run-dir "${RUN_DIR}" \
        --scheme "${SCHEME}" \
        --reference-config-id "${REF_ID}" \
        --dataset "${DATASET}" \
        --fsood-id-side "${SIDE}" \
        --perturbation-score-rule all \
        --overwrite
    done
  fi

  conda run --no-capture-output -n openood python scripts_my/tarr/reports.py diagnostics \
    --dataset "${DATASET}" \
    --run-dir "${RUN_DIR}" \
    --scheme "${SCHEME}" \
    --reference-config-id "${REF_ID}" \
    --score-rule all

  for SIDE in both clean csid; do
    conda run --no-capture-output -n openood python scripts_my/tarr/reports.py score-diagnostics \
      --dataset "${DATASET}" \
      --run-dir "${RUN_DIR}" \
      --scheme "${SCHEME}" \
      --reference-config-id "${REF_ID}" \
      --score-kind standard \
      --fsood-id-side "${SIDE}" \
      --score-rule all
  done
done

conda run --no-capture-output -n openood python scripts_my/tarr/reports.py collect-score \
  --dataset "${DATASET}" \
  --baseline-protocol "${PROTOCOL}" \
  --runs-root results_test/tarr/outputs \
  --output-csv results_test/tarr/summary/${DATASET}_${PROTOCOL}_score_results.csv

# compare-group1 expects one exact active score rule, not `all`.
# Set SELECTED_SCORE_RULE after collect-score identifies the best TARR row.
SELECTED_SCORE_RULE=predicted_class_loss_increase
conda run --no-capture-output -n openood python scripts_my/tarr/reports.py compare-group1 \
  --dataset "${DATASET}" \
  --baseline-protocol "${PROTOCOL}" \
  --score-rule "${SELECTED_SCORE_RULE}" \
  --output-csv results_test/tarr/summary/${DATASET}_${PROTOCOL}_${SELECTED_SCORE_RULE}_group1_compare.csv
```

## Stage 4: Build `score_result`

Stage 4 converts saved `tta_response` artifacts into scalar OOD scores.
During broad search, run Stage 4 immediately after each Stage 3 run and treat
the run as complete only after `both`, `clean`, and `csid` score results exist.

Command shape:

```bash
conda run --no-capture-output -n openood python scripts_my/tarr/cache.py score \
  --run-dir results_test/tarr/outputs/<dataset>/<protocol>/seed0/<run_id> \
  --scheme fsood \
  --reference-config-id all_rpc32 \
  --dataset <dataset> \
  --fsood-id-side both \
  --score-rule all \
  --overwrite
```

For FSOOD diagnostics, run the same command with:

```bash
--fsood-id-side clean
--fsood-id-side csid
```

## Diagnostics

After Stage 4, summarize response and score artifacts:

```bash
conda run --no-capture-output -n openood python scripts_my/tarr/reports.py diagnostics \
  --dataset <dataset> \
  --run-dir results_test/tarr/outputs/<dataset>/<protocol>/seed0/<run_id> \
  --scheme fsood \
  --reference-config-id all_rpc32 \
  --score-rule all
```
