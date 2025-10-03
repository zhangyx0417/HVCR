# HVCR: Causal Evaluation of Large Multimodal Models for Human-like Video Reasoning

The HVCR benchmark and its evaluation toolkit.

## Repository Layout

### Data (`data/`)

`synthetic/`: 6 scenarios × 5 settings. Each leaf contains `questions/` and `videos/` (and for synthetic, `simulations/`).
- Scenarios: `overdetermination`, `switch`, `late`, `early`, `double`, `bogus`.
- Settings: `basic`, `add_one_static`, `add_two_static`, `add_one_moving`, `add_two_moving`.
- `synthetic/{scenario}/{setting}/simulations/simulation_*.json`: Simulation file for each video.

`realistic/`: 3 scenarios. Each leaf contains `questions/` and `videos/`.
- Scenarios: `switch`, `late`, `bogus`


`{synthetic|realistic}/{scenario}/{setting}/videos/video_*.mp4`: Video file.

`{synthetic|realistic}/{scenario}/{setting}/questions/questions_*.json`: Question file for each video, containing QA pairs, causal graphs, and twin networks.


### Evaluation (`eval/`)
`models/`: Model adapters/configs used by `lmms-eval` (e.g., `gpt4v.py`, `llava_onevision.py`, `internvideo2_5.py`, `plm.py`).

`tasks/`: Evaluation tasks for the benchmark. See **Task Generation** for task generation.

`scripts/`: Utilities for dataset conversion, task generation, and one-command evaluation runners.

## Data Preparation

If you want to generate synthetic data on your own, use `data/synthetic/scripts/simulation.py` for simulation, `data/synthetic/scripts/render.py` for rendering frames, and `data/synthetic/scripts/video.py` for converting frames to videos. Then, use `data/synthetic/scripts/generator.py` to generate QA pairs, causal graphs, and twin networks.

```bash
python data/synthetic/scripts/simulation.py
python data/synthetic/scripts/render.py
python data/synthetic/scripts/video.py
python data/synthetic/scripts/generator.py
```

The evaluation uses the Hugging Face Dataset on-disk format. Use `eval/scripts/convert_to_hf_dataset.py` to convert each leaf to `*_hf`:

```bash
python eval/scripts/convert_to_hf_dataset.py
```

## Task Generation

Generate `lmms-eval` task YAMLs pointing to your local HF datasets with `eval/scripts/generate_tasks.py`:

```bash
python eval/scripts/generate_tasks.py
```

This creates YAMLs under `eval/scripts/lmms-eval/lmms_eval/tasks/hvcr/`.

## Evaluation with lmms-eval

We build on the `EvolvingLMMs-Lab/lmms-eval` repository. You can run all synthetic/realistic tasks for a model with `eval/scripts/run_model_all_tasks_<synthetic|realistic>.py`:

```bash
python eval/scripts/run_model_all_tasks_<synthetic|realistic>.py \
  --model <gpt4o|gemini|internvideo2_5|llava_onevision|plm|qwen2_5_vl> \
  --pretrained <local_path_or_repo_id> \
  --only <OPTIONAL scenario-setting filter>
```

Or use our provided shell scripts:

```bash
./run_<model>_<synthetic|realistic>.sh
```

Notes:

- The script expects your converted datasets under `.../data/<synthetic|realistic>/<scenario>/<setting>_hf`. Edit `DATA_ROOT` in the script to your absolute path.
- Output directories are created under `eval/scripts/lmms-eval/outputs/<model>/<task>/`.


## Task Definition (Internals)

- `eval/tasks/videoac_task.py` registers the task (e.g., `videoac_switch`) for `lmms-eval`, providing:
  - `doc_to_visual`, `doc_to_text`, `doc_to_target`
  - `process_results` and `aggregation` (accuracy)
  - A minimal `construct_requests` for generate-until style models
- `_default_template_yaml` defines the common prompt template, including strict output format like:

```
Answer: A END
Answer: A B C END
```

Ensure your model adapter returns answers matching the expected format so that accuracy is computed correctly.

## Installation & Requirements

Minimum for evaluation:

- Python 3.9+
- `lmms-eval` (install from source recommended)
- Hugging Face `datasets`
- `accelerate` (for multi-process launches)

Example environment setup:

```bash
conda create -n hvcr python=3.10 -y
conda activate hvcr
pip install datasets accelerate
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval && pip install -e .
```

If you plan to regenerate synthetic videos from simulations, you will additionally need physics/rendering tools such as Bullet and Blender. This repository already includes curated simulations and videos, so regeneration is optional.

## Quick Start

1) Convert data to HF datasets:

```bash
python eval/scripts/convert_to_hf_dataset.py
```

2) Generate task YAMLs (update paths as needed in the script):

```bash
python eval/scripts/generate_tasks.py
```

3) Run evaluation over synthetic tasks (example: LLaVA-OneVision):

```bash
python eval/scripts/run_model_all_tasks_synthetic.py \
  --model llava_onevision \
  --pretrained <ABS_OR_REPO_ID> \
  --only switch:basic
```

4) Inspect results under `eval/scripts/lmms-eval/outputs/...` and the `lmms-eval` logs for accuracy.

## Data Format Details

The converter normalizes QA into a uniform multiple-choice representation:

- `answer_type` is one of `yes_no` or `multi_choice`
- `choices` is a list of `["A", "Yes"]`-style pairs
- `answer` is a list-of-list of letter labels, e.g., `[["A"]]` for single-answer or `[ ["A"], ["B", "C"] ]` for multiple answers across sub-questions

This enables pyarrow-safe storage and standardized scoring downstream..