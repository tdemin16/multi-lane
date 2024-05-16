#!/bin/bash

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate multilane

WANDB_API_KEY=ADD_YOUR_API_KEY_HERE \
CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=1 \
         --rdzv-backend=c10d \
         --rdzv-endpoint=localhost:0 \
         --nnodes=1 \
        main.py imr \
        "$@"