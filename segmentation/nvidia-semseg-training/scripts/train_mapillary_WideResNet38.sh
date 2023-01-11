#!/usr/bin/env bash

    # Example on Mapillary
    
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py \
        --dataset mapillary \
        --apex \
        --syncbn \
        --arch network.deepv3.DeepWV3Plus \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --sgd \
        --lr 2e-2 \
        --lr_schedule poly \
        --poly_exp 1.0 \
        --crop_size 896 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --color_aug 0.25 \
        --gblur \
        --max_epoch 175 \
        --img_wt_loss \
        --wt_bound 6.0 \
        --bs_mult 1 \
        --exp mapillary_pretrain \
        --ckpt ./logs/ \
        --tb_path ./logs/ \
        --temperature_model learned \
        --fp16 \


