#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py --log_dir /mnt/Data/Cryo-ET/3D-UCaps/logs \
                --gpus 1 \
                --check_val_every_n_epoch 1 \
                --max_epochs 200 \
                --dataset invitro \
                --model_name ucaps \
                --root_dir /mnt/Data/Cryo-ET/3D-UCaps/data/invitro \
                --fold 0 \
                --train_patch_size 32 32 32 \
                --num_workers 4 \
                --batch_size 1 \
                --share_weight 0 \
                --num_samples 1 \
                --in_channels 1 \
                --out_channels 3 \
                --val_patch_size 32 32 32 \
                --val_frequency 1 \
                --sw_batch_size 2 \
                --overlap 0.75


#python train.py --log_dir /mnt/Data/Cryo-ET/3D-UCaps/logs \
#                --gpus 1 \
#                --accelerator ddp \
#                --check_val_every_n_epoch 10 \
#                --max_epochs 2000 \
#                --dataset invitro \
#                --model_name ucaps \
#                --root_dir /mnt/Data/Cryo-ET/3D-UCaps/data/invitro \
#                --fold 0 \
#                --cache_rate 1.0 \
#                --train_patch_size 32 32 32 \
#                --num_workers 4 \
#                --batch_size 1 \
#                --share_weight 0 \
#                --num_samples 8 \
#                --in_channels 1 \
#                --out_channels 3 \
#                --val_patch_size 32 32 32 \
#                --val_frequency 1 \
#                --sw_batch_size 4 \
#                --overlap 0.75
#
#python train.py --log_dir /mnt/Data/Cryo-ET/3D-UCaps/logs \
#                --gpus 1 \
#                --accelerator ddp \
#                --check_val_every_n_epoch 10 \
#                --max_epochs 2000 \
#                --dataset invitro \
#                --model_name ucaps \
#                --root_dir /mnt/Data/Cryo-ET/3D-UCaps/data/invitro \
#                --fold 1 \
#                --cache_rate 1.0 \
#                --train_patch_size 32 32 32 \
#                --num_workers 4 \
#                --batch_size 10 \
#                --share_weight 0 \
#                --num_samples 8 \
#                --in_channels 1 \
#                --out_channels 3 \
#                --val_patch_size 32 32 32 \
#                --val_frequency 1 \
#                --sw_batch_size 4 \
#                --overlap 0.75
#
#python train.py --log_dir /mnt/Data/Cryo-ET/3D-UCaps/logs \
#                --gpus 1 \
#                --accelerator ddp \
#                --check_val_every_n_epoch 10 \
#                --max_epochs 2000 \
#                --dataset invitro \
#                --model_name ucaps \
#                --root_dir /mnt/Data/Cryo-ET/3D-UCaps/data/invitro \
#                --fold 2 \
#                --cache_rate 1.0 \
#                --train_patch_size 32 32 32 \
#                --num_workers 4 \
#                --batch_size 1 \
#                --share_weight 0 \
#                --num_samples 8 \
#                --in_channels 1 \
#                --out_channels 3 \
#                --val_patch_size 32 32 32 \
#                --val_frequency 1 \
#                --sw_batch_size 4 \
#                --overlap 0.75
#
#python train.py --log_dir /mnt/Data/Cryo-ET/3D-UCaps/logs \
#                --gpus 1 \
#                --accelerator ddp \
#                --check_val_every_n_epoch 1 \
#                --max_epochs 2000 \
#                --dataset invitro \
#                --model_name ucaps \
#                --root_dir /mnt/Data/Cryo-ET/3D-UCaps/data/invitro \
#                --fold 3 \
#                --cache_rate 1.0 \
#                --train_patch_size 32 32 32 \
#                --num_workers 4 \
#                --batch_size 1 \
#                --share_weight 0 \
#                --num_samples 8 \
#                --in_channels 1 \
#                --out_channels 3 \
#                --val_patch_size 32 32 32 \
#                --val_frequency 1 \
#                --sw_batch_size 4 \
#                --overlap 0.75
#
