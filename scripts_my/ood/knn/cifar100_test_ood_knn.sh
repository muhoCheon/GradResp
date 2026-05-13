#!/bin/bash
set -e

# OOD
PYTHONPATH='.':$PYTHONPATH \
python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/knn.yml \
    --num_workers 8 \
    --network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' \
    --output_dir ./results_test/outputs/ \
    --mark 0 \
    --merge_option merge

# Full-spectrum OOD
PYTHONPATH='.':$PYTHONPATH \
python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/datasets/cifar100/cifar100_fsood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_fsood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/knn.yml \
    --num_workers 8 \
    --network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' \
    --output_dir ./results_test/outputs/ \
    --mark 0 \
    --merge_option merge


# Unified evaluator: OOD
python scripts/eval_ood.py \
    --id-data cifar100 \
    --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor knn \
    --save-score --save-csv

# Unified evaluator: FSOOD
python scripts/eval_ood.py \
    --id-data cifar100 \
    --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor knn \
    --save-score --save-csv --fsood
