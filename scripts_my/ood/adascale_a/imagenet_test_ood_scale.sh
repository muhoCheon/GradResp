#!/bin/bash
set -e

# OOD
PYTHONPATH='.':$PYTHONPATH \
python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/adascale_a.yml \
    --num_workers 4 \
    --ood_dataset.image_size 256 \
    --dataset.test.batch_size 256 \
    --dataset.val.batch_size 256 \
    --network.pretrained True \
    --network.checkpoint 'results/pretrained_weights/resnet50_imagenet1k_v1.pth' \
    --output_dir ./results_test/outputs/ \
    --mark 0 \
    --merge_option merge

# Full-spectrum OOD
PYTHONPATH='.':$PYTHONPATH \
python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_fsood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_fsood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/adascale_a.yml \
    --num_workers 4 \
    --ood_dataset.image_size 256 \
    --dataset.test.batch_size 256 \
    --dataset.val.batch_size 256 \
    --network.pretrained True \
    --network.checkpoint 'results/pretrained_weights/resnet50_imagenet1k_v1.pth' \
    --output_dir ./results_test/outputs/ \
    --mark 0 \
    --merge_option merge


# Unified evaluator: OOD
python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor adascale_a \
    --save-score --save-csv

# Unified evaluator: FSOOD
python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor adascale_a \
    --save-score --save-csv --fsood
