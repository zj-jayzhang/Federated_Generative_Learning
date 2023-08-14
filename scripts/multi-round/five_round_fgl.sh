#!/bin/bash

net=$1  # choices = ['holocron_resnet18', 'holocron_resnet34', 'holocron_resnet50', 'resnet50']
dataset=$2    # choices=["imagenet100", "imagenette", "imagefruit", "imageyellow", "imagesquawk", "domainnet"]
exp_name=$3
data_path_train=$4 
data_path_test=$5
net_path=$6
partition=$7
beta=$8
gpu_id=$9

CUDA_VISIBLE_DEVICES=$gpu_id python multi_round.py \
    --net $net \
    --data_type=$dataset \
    --exp_name $exp_name  \
    --data_path_train $data_path_train \
    --data_path_test $data_path_test \
    --net_path $net_path \
    --partition $partition \
    --beta $beta \
    --wandb 1 \


