#!/bin/bash
set -eux

python ./run_model_all_tasks_realistic.py \
    --model "internvideo2_5" \
    --pretrained "OpenGVLab/InternVL2_5-8B"