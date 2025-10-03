#!/bin/bash
set -eux

export API_TYPE=openai
export OPENAI_API_KEY=<your_api_key>

python ./run_model_all_tasks_realistic.py \
    --model "gemini" \
    --pretrained gemini