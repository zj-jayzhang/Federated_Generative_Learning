#!/bin/bash

net=$1  # choices = ['holocron_resnet18', 'holocron_resnet34', 'holocron_resnet50']
dataset=$2    # choices=["imagenet100", "imagenette", "imagefruit", "imageyellow", "imagesquawk", "domainnet"]
exp_name=$3
syn=$4
blip=$5
data_path_train=$6
data_path_test=$7
gpu_id=$8

CUDA_VISIBLE_DEVICES=$gpu_id python one_shot.py \
                        --wandb 0 \
                        --net $net \
                        --data_type $dataset \
                        --exp_name $exp_name \
                        --syn $syn \
                        --if_blip $blip \
                        --data_path_train $data_path_train \
                        --data_path_test $data_path_test 

