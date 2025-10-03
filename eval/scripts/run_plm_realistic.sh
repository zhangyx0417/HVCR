#!/bin/bash
set -eux

python ./run_model_all_tasks_realistic.py \
    --model "plm" \
    --pretrained "facebook/Perception-LM-8B"