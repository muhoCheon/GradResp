#!/bin/bash
set -e

# OOD
PYTHONPATH='.':$PYTHONPATH \
python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/knn.yml \
    --num_workers 4 \
    --ood_dataset.image_size 256 \
    --dataset.test.batch_size 256 \
    --dataset.val.batch_size 256 \
    --network.checkpoint 'results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt' \
    --output_dir ./results_test/outputs/ \
    --mark 0 \
    --merge_option merge

# Full-spectrum OOD
PYTHONPATH='.':$PYTHONPATH \
python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_fsood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_fsood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/knn.yml \
    --num_workers 4 \
    --ood_dataset.image_size 256 \
    --dataset.test.batch_size 256 \
    --dataset.val.batch_size 256 \
    --network.checkpoint 'results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt' \
    --output_dir ./results_test/outputs/ \
    --mark 0 \
    --merge_option merge


# Unified evaluator: OOD
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --postprocessor knn \
    --save-score --save-csv

# Unified evaluator: FSOOD
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --postprocessor knn \
    --save-score --save-csv --fsood
