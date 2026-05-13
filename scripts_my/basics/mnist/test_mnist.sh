#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \
python main.py \
    --config configs/datasets/mnist/mnist.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/lenet.yml \
    configs/pipelines/test/test_acc.yml \
    --network.checkpoint ./results/mnist_lenet_base_e100_lr0.1_default/s0/best.ckpt
