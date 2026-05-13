#!/bin/bash
set -e

# OOD
PYTHONPATH='.':$PYTHONPATH \
python main.py \
    --config configs/datasets/mnist/mnist.yml \
    configs/datasets/mnist/mnist_ood.yml \
    configs/networks/lenet.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --num_workers 8 \
    --network.checkpoint 'results/mnist_lenet_base_e100_lr0.1_default/s0/best.ckpt' \
    --output_dir ./results_test/outputs/ \
    --mark 0 \
    --merge_option merge

# Full-spectrum OOD
PYTHONPATH='.':$PYTHONPATH \
python main.py \
    --config configs/datasets/mnist/mnist.yml \
    configs/datasets/mnist/mnist_fsood.yml \
    configs/networks/lenet.yml \
    configs/pipelines/test/test_fsood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --num_workers 8 \
    --network.checkpoint 'results/mnist_lenet_base_e100_lr0.1_default/s0/best.ckpt' \
    --output_dir ./results_test/outputs/ \
    --mark 0 \
    --merge_option merge
