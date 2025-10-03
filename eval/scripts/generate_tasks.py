import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(BASE_DIR, "lmms-eval", "lmms_eval", "tasks", "hvcr")

def make_yaml(task_name: str, dataset_path: str) -> str:
    return (
        "include: _default_template_yaml\n"
        f"dataset_path: {dataset_path}\n"
        "dataset_kwargs:\n"
        "  load_from_disk: true\n"
        "\n"
        f"task: {task_name}\n"
        f"dataset_name: {task_name}\n"
        "test_split: valid\n"
        "metric_list:\n"
        "  - metric: accuracy\n"
        "    aggregation: !function utils.hvcr_accuracy\n"
        "    higher_is_better: true\n"
    )

if __name__ == "__main__":
    os.makedirs(TASKS_DIR, exist_ok=True)
    created = []

    # Synthetic
    scenarios = ["overdetermination", "switch", "late", "early", "double", "bogus"]
    settings = ["basic", "add_one_static", "add_two_static", "add_one_moving", "add_two_moving"]
    for sc in scenarios:
        for st in settings:
            task_key = f"synthetic_{sc}_{st}"
            hf_path = f".../data/synthetic/{sc}/{st}_hf/"
            yaml_path = os.path.join(TASKS_DIR, f"{task_key}.yaml")
            content = make_yaml(task_key, hf_path)
            with open(yaml_path, "w") as f:
                f.write(content)
            created.append(yaml_path)
    print(f"Generated {len(created)} synthetic YAMLs under {TASKS_DIR}")

    # Realistic
    scenarios = ["switch", "late", "bogus"]
    settings = ["basic"]
    for sc in scenarios:
        for st in settings:
            task_key = f"realistic_{sc}_{st}"
            hf_path = f".../data/realistic/{sc}/{st}_hf/"
            yaml_path = os.path.join(TASKS_DIR, f"{task_key}.yaml")
            content = make_yaml(task_key, hf_path)
            with open(yaml_path, "w") as f:
                f.write(content)
            created.append(yaml_path)
    print(f"Generated {len(created)} realistic YAMLs under {TASKS_DIR}")
