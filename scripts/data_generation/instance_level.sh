#!/bin/bash

# captions for all data in imagenet1k
raw_data_dir=$1
syn_data_dir=$2
save_prompt_dir=$3
gpu_id=$4

#todo: blip  #  \ --prompt_dir $path3 
CUDA_VISIBLE_DEVICES=$gpu_id python instance_level_generation.py \
    --raw_data_dir $raw_data_dir \
    --syn_data_dir $syn_data_dir \
    --save_prompt_dir $save_prompt_dir 
    

