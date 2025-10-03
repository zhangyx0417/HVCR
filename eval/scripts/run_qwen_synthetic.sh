#!/bin/bash
set -eux

python ./run_model_all_tasks_synthetic.py \
    --model "qwen2_5_vl" \
    --pretrained "Qwen/Qwen2.5-VL-7B-Instruct"

python ./run_model_all_tasks_synthetic.py \
    --model "qwen2_5_vl" \
    --pretrained "Qwen/Qwen2.5-VL-32B-Instruct"
