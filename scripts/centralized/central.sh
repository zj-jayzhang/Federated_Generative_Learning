#!/bin/bash

net=$1  # choices = ['holocron_resnet18', 'holocron_resnet34', 'holocron_resnet50']
dataset=$2    # choices=["imagenet100", "imagenette", "imagefruit", "imageyellow", "imagesquawk", "domainnet"]
exp_name=$3
data_path_train=$4 
data_path_test=$5 
gpu_id=$6

CUDA_VISIBLE_DEVICES=$gpu_id python one_shot.py \
                        --wandb 0 \
                        --net $net \
                        --data_type $dataset \
                        --exp_name $exp_name \
                        --syn 0 \
                        --if_blip 0 \
                        --data_path_train $data_path_train \
                        --data_path_test $data_path_test 

