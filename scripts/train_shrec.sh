#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py --log_dir /mnt/Data/Cryo-ET/3D-UCaps/logs \
                --gpus 1 \
                --accelerator ddp \
                --check_val_every_n_epoch 10 \
                --max_epochs 100 \
                --dataset shrec \
                --model_name ucaps \
                --root_dir /mnt/Data/Cryo-ET/3D-UCaps/data/shrec2 \
                --cache_rate 1.0 \
                --train_patch_size 32 32 32 \
                --num_workers 4 \
                --batch_size 1 \
                --num_samples 8 \
                --in_channels 1 \
                --out_channels 13 \
                --val_patch_size 32 32 32 \
                --val_frequency 10 \
                --sw_batch_size 4 \
                --overlap 0.50