#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python evaluate.py --root_dir /mnt/Data/Cryo-ET/3D-UCaps/data/shrec2 \
                    --gpus 1 \
                    --save_image 0 \
                    --model_name ucaps \
                    --dataset shrecc2 \
                    --checkpoint_path /mnt/Data/Cryo-ET/3D-UCaps/logs/ucaps_shrec_0/version_4/checkpoints/epoch=8-val_dice=0.3822.ckpt \
                    --val_patch_size 64 64 64 \
                    --sw_batch_size 8 \
                    --overlap 0.0