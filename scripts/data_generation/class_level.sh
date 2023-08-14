#!/bin/bash

dataset=$1    # choices=["imagenet100", "imagenette", "imagefruit", "imageyellow", "imagesquawk", "domainnet"]
exp_name=$2
gpu_id=$3

CUDA_VISIBLE_DEVICES=$gpu_id python class_level_generation.py \
    --data_type $dataset \
    --per_class_nums 5 \
    --exp_name $exp_name \
    --start_number 0 
