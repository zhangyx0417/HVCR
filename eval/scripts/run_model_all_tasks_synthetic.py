#!/usr/bin/env python3
import os
import subprocess
import shlex

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(BASE_DIR, "lmms-eval", "lmms_eval", "tasks", "hvcr")
SCENARIOS = ["overdetermination", "switch", "late", "early", "double", "bogus"]
SETTINGS = ["basic", "add_one_static", "add_two_static", "add_one_moving", "add_two_moving"]

DATA_ROOT = ".../data/synthetic"


def dataset_exists(scenario: str, setting: str) -> bool:
    path = os.path.join(DATA_ROOT, scenario, f"{setting}_hf")
    if not os.path.isdir(path):
        return False
    has_dict = os.path.exists(os.path.join(path, "dataset_dict.json"))
    has_info = os.path.exists(os.path.join(path, "dataset_info.json"))
    has_valid_info = os.path.exists(os.path.join(path, "valid", "dataset_info.json"))
    return has_dict or has_info or has_valid_info


def model_name_sanitized(pretrained: str) -> str:
    name = pretrained.rstrip("/").split("/")[-1]
    return name.replace(" ", "_")


def run_eval(model: str, pretrained: str, scenario: str, setting: str):
    task = f"synthetic_{scenario}_{setting}"
    out_dir = os.path.join(BASE_DIR, "lmms-eval", "outputs", model_name_sanitized(pretrained), f"{task}")
    os.makedirs(out_dir, exist_ok=True)

    if model == "internvideo2_5":
        cmd = (
            f"accelerate launch --num_processes=8 -m lmms_eval --model internvideo2_5 --model_args pretrained={pretrained},max_frames_num=16,modality=video --tasks {task} --batch_size 1 --log_samples --log_samples_suffix internvl2_5 --output_path {shlex.quote(out_dir)} --verbosity DEBUG"
        )
    elif model == "llava_onevision":
        cmd = (
            f"accelerate launch --num_processes=8 -m lmms_eval --model llava_onevision --model_args pretrained={pretrained},max_frames_num=16,device_map=auto,model_name=llava_qwen --tasks {task} --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path {shlex.quote(out_dir)} --verbosity DEBUG"
        )
    elif model == "qwen2_5_vl":
        cmd = (
            f"accelerate launch --num_processes=8 -m lmms_eval --model vllm --model_args pretrained={pretrained},max_num_frames=16,force_sample=True,tensor_parallel_size=1 --tasks {task} --batch_size 1 --log_samples --log_samples_suffix qwen2_5_vl --output_path {shlex.quote(out_dir)} --verbosity DEBUG"
        )
    elif model == "plm":
        cmd = (
            f"accelerate launch --num_processes=8 -m lmms_eval --model plm --model_args pretrained={pretrained} --tasks {task} --batch_size 1 --log_samples --log_samples_suffix plm --output_path {shlex.quote(out_dir)} --verbosity DEBUG"
        )
    elif model == "gemini":
        cmd = (
            f"accelerate launch --num_processes=1 -m lmms_eval --model gpt4v --model_args model_version=gemini-2.5-flash-preview-04-17,timeout=300,continual_mode=False --tasks {task} --batch_size 1 --log_samples --log_samples_suffix gemini2_5_flash --output_path {shlex.quote(out_dir)} --verbosity DEBUG"
        )
    elif model == "gpt4o":
        cmd = (
            f"accelerate launch --num_processes=1 -m lmms_eval --model gpt4v --model_args model_version=gpt-4o-2024-11-20,max_frames_num=10,timeout=360 --tasks {task} --batch_size 1 --log_samples --log_samples_suffix gpt_4o --output_path {shlex.quote(out_dir)} --verbosity DEBUG"
        )

    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model name in lmms-eval, e.g., qwen_vl")
    parser.add_argument("--pretrained", required=True, help="absolute path to local model or HF repo id")
    parser.add_argument("--only", default="", help="optional filter like scenario:setting or scenario:*")
    parser.add_argument("--limit", type=int, default=0, help="optional limit for quick smoke test")
    args = parser.parse_args()

    # Optional quick test by setting environment LIMIT (used if YAML respects --limit)
    extra_limit = f" --limit {args.limit}" if args.limit and args.limit > 0 else ""

    # Iterate tasks
    for sc in SCENARIOS:
        for st in SETTINGS:
            if args.only:
                try:
                    s, sel = args.only.split(":", 1)
                    if sc != s:
                        continue
                    if sel != "*" and st != sel:
                        continue
                except Exception:
                    pass
            if not dataset_exists(sc, st):
                print(f"Skip {sc}/{st}: dataset not ready")
                continue
            run_eval(args.model, args.pretrained, sc, st)
