#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \
python main.py \
    --config configs/datasets/mnist/mnist.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/lenet.yml \
    configs/pipelines/train/baseline.yml
