#!/bin/bash
set -eux

python ./run_model_all_tasks_realistic.py \
    --model "llava_onevision" \
    --pretrained "lmms-lab/llava-onevision-qwen2-7b-ov"