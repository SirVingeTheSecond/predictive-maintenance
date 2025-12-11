import os
import json
from datetime import datetime

import torch

import config


def get_experiment_dir(study_name: str, model: str, config_name: str, seed: int) -> str:
    """
    Get experiment directory path.
    
    Structure: results/{study_name}/{model}_{config_name}_seed{seed}/
    """
    exp_name = f"{model}_{config_name}_seed{seed}"
    return os.path.join(config.RESULTS_DIR, study_name, exp_name)


def experiment_exists(exp_dir: str) -> bool:
    """Check if experiment results already exist."""
    return os.path.exists(os.path.join(exp_dir, "results.json"))


def save_experiment(
    exp_dir: str,
    model_state: dict,
    results: dict,
    exp_config: dict,
    history: dict,
):
    """
    Save all experiments.
    
    Creates:
        exp_dir/
            config.json     - Experiment configuration
            results.json    - Test metrics
            history.json    - Training history
            model.pth       - Model weights
    """
    os.makedirs(exp_dir, exist_ok=True)

    # Config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(exp_config, f, indent=2)

    # Results
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # History
    with open(os.path.join(exp_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Model
    torch.save(model_state, os.path.join(exp_dir, "model.pth"))


def load_experiment(exp_dir: str) -> dict:
    """
    Load experiment artifacts.
    
    Returns:
        Dictionary with config, results, history
    """
    data = {}

    config_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            data["config"] = json.load(f)

    results_path = os.path.join(exp_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            data["results"] = json.load(f)

    history_path = os.path.join(exp_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            data["history"] = json.load(f)

    return data


def list_experiments(study_name: str) -> list:
    """
    List all experiments in a study.
    
    Returns:
        List of experiment directory paths
    """
    study_dir = os.path.join(config.RESULTS_DIR, study_name)
    if not os.path.exists(study_dir):
        return []

    experiments = []
    for name in sorted(os.listdir(study_dir)):
        exp_dir = os.path.join(study_dir, name)
        if os.path.isdir(exp_dir) and experiment_exists(exp_dir):
            experiments.append(exp_dir)

    return experiments


def parse_experiment_name(exp_dir: str) -> dict:
    """
    Parse experiment name to extract components.

    Example: "cnn_4class_fault_size_seed42" -> {"model": "cnn", "config": "4class_fault_size", "seed": 42}
    """
    name = os.path.basename(exp_dir)
    parts = name.rsplit("_seed", 1)

    if len(parts) == 2:
        prefix, seed_str = parts
        seed = int(seed_str)
    else:
        prefix = name
        seed = None

    # Find model prefix
    for model_name in ["cnnlstm", "lstm", "cnn"]:  # Order matters (longest first)
        if prefix.startswith(model_name + "_"):
            model = model_name
            config_name = prefix[len(model_name) + 1:]
            break
    else:
        model = None
        config_name = prefix

    return {
        "model": model,
        "config": config_name,
        "seed": seed,
    }
