import os
import json
import itertools
from datetime import datetime

import numpy as np
import torch

import config
from data import load_data, create_dataloaders
from models import create_model, count_parameters
from training import train, get_criterion, train_epoch, evaluate_epoch, EarlyStopping
from evaluation import evaluate
from utils import set_seed, get_device


def get_sweep_dir(sweep_name: str) -> str:
    """Get search results directory."""
    return os.path.join(config.RESULTS_DIR, "sweeps", sweep_name)


def get_config_name(params: dict) -> str:
    """Generate name for parameter combination."""
    parts = []
    for key in sorted(params.keys()):
        value = params[key]
        if isinstance(value, float):
            parts.append(f"{key}{value:.0e}")
        else:
            parts.append(f"{key}{value}")
    return "_".join(parts)


def generate_combinations(sweep_name: str) -> list:
    """Generate all hyperparameter combinations for a search."""
    if sweep_name not in config.SWEEPS:
        raise ValueError(f"Unknown search: {sweep_name}")

    sweep = config.SWEEPS[sweep_name]
    param_grid = sweep["param_grid"]

    # Generate all combinations
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    combinations = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        combinations.append(params)

    return combinations


def run_screening(
    model,
    train_loader,
    val_loader,
    mode: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> tuple:
    """
    Run screening phase (short training to filter bad configs).
    
    Returns:
        (best_val_acc, passed_screening)
    """
    criterion = get_criterion(mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, criterion, optimizer, device, mode)
        _, val_acc = evaluate_epoch(model, val_loader, criterion, device, mode)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc


def run_sweep_experiment(
    sweep_name: str,
    params: dict,
    seed: int = None,
    verbose: bool = True,
) -> dict:
    """
    Run a single search experiment with screening.
    
    Args:
        sweep_name: Name of the search
        params: Hyperparameter dictionary
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Result dictionary or None if screening failed
    """
    if seed is None:
        seed = config.DEFAULT_SEED

    sweep = config.SWEEPS[sweep_name]
    base_config = sweep["base_config"]
    screening_epochs = sweep.get("screening_epochs", 20)
    screening_threshold = sweep.get("screening_threshold", 0.4)
    full_epochs = sweep.get("full_epochs", 100)

    mode = base_config["mode"]
    split = base_config["split"]

    model_name = params["model"]
    lr = params["lr"]
    dropout = params["dropout"]
    weight_decay = params["weight_decay"]

    config_name = get_config_name(params)
    sweep_dir = get_sweep_dir(sweep_name)
    result_path = os.path.join(sweep_dir, config_name, "results.json")

    # Check if already done
    if os.path.exists(result_path):
        if verbose:
            print(f"SKIP: {config_name} (exists)")
        with open(result_path) as f:
            return json.load(f)

    set_seed(seed)
    device = get_device()

    if verbose:
        print(f"Search: {config_name}")

    # Load data
    data = load_data(mode=mode, split=split, seed=seed, verbose=verbose)
    train_loader, val_loader, test_loader = create_dataloaders(data, mode)

    # Create model
    num_classes = config.NUM_CLASSES[mode]
    model = create_model(model_name, num_classes=num_classes, dropout=dropout)
    model = model.to(device)

    if verbose:
        print(f"Parameters: {count_parameters(model):,}")
        print(f"Screening: {screening_epochs} epochs, threshold={screening_threshold}")

    # Screening phase
    screening_acc = run_screening(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode=mode,
        epochs=screening_epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    if verbose:
        print(f"Screening accuracy: {screening_acc:.4f}")

    if screening_acc < screening_threshold:
        if verbose:
            print(f"FAILED screening (< {screening_threshold})")

        result = {
            "params": params,
            "screening_acc": float(screening_acc),
            "passed_screening": False,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        }

        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    # Full training (reset model)
    if verbose:
        print(f"PASSED screening, running full training ({full_epochs} epochs)")

    set_seed(seed)
    model = create_model(model_name, num_classes=num_classes, dropout=dropout)

    train_result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode=mode,
        epochs=full_epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        verbose=verbose,
    )

    # Evaluate
    test_metrics = evaluate(model, test_loader, device, mode)

    if verbose:
        if mode == "multilabel":
            print(f"\nTest Macro AUROC: {test_metrics['auroc']['macro']:.4f}")
        else:
            print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")

    result = {
        "params": params,
        "screening_acc": float(screening_acc),
        "passed_screening": True,
        "best_val_acc": train_result["best_val_acc"],
        "best_epoch": train_result["best_epoch"],
        "epochs_trained": train_result["epochs_trained"],
        "test_metrics": test_metrics,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save model
    torch.save(model.state_dict(), os.path.join(sweep_dir, config_name, "model.pth"))

    return result


def run_sweep(
    sweep_name: str,
    seed: int = None,
    verbose: bool = True,
) -> list:
    """
    Run all combinations in a search.
    
    Args:
        sweep_name: Name of the search
        seed: Random seed
        verbose: Print progress
        
    Returns:
        List of results
    """
    if sweep_name not in config.SWEEPS:
        raise ValueError(f"Unknown search: {sweep_name}. Available: {list(config.SWEEPS.keys())}")

    combinations = generate_combinations(sweep_name)

    if verbose:
        print(f"SEARCH: {sweep_name}")
        print(f"Total combinations: {len(combinations)}")

    results = []

    for i, params in enumerate(combinations, 1):
        if verbose:
            print(f"\n[{i}/{len(combinations)}]")

        result = run_sweep_experiment(
            sweep_name=sweep_name,
            params=params,
            seed=seed,
            verbose=verbose,
        )
        results.append(result)

    return results


def load_sweep_results(sweep_name: str) -> list:
    """Load all results from a search."""
    sweep_dir = get_sweep_dir(sweep_name)
    if not os.path.exists(sweep_dir):
        return []

    results = []
    for name in os.listdir(sweep_dir):
        result_path = os.path.join(sweep_dir, name, "results.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                results.append(json.load(f))

    return results


def print_sweep_leaderboard(sweep_name: str, top_n: int = 10):
    """Print leaderboard of best configurations."""
    results = load_sweep_results(sweep_name)

    if not results:
        print(f"No results found for search: {sweep_name}")
        return

    # Filter to passed screening
    passed = [r for r in results if r.get("passed_screening", False)]
    failed = [r for r in results if not r.get("passed_screening", False)]

    # Sort by test metric
    def get_score(r):
        metrics = r.get("test_metrics", {})
        if "accuracy" in metrics:
            return metrics["accuracy"]
        elif "auroc" in metrics:
            return metrics["auroc"].get("macro", 0)
        return 0

    passed.sort(key=get_score, reverse=True)

    print(f"SEARCH RANKINGS: {sweep_name}")
    print(f"Passed: {len(passed)}/{len(results)}, Failed screening: {len(failed)}")

    if passed:
        print(f"\n{'Rank':<5} {'Model':<10} {'LR':<10} {'Dropout':<8} {'WD':<10} {'Score':<10}")
        print("-" * 60)

        for i, r in enumerate(passed[:top_n], 1):
            params = r["params"]
            score = get_score(r)
            print(f"{i:<5} {params['model']:<10} {params['lr']:<10.0e} "
                  f"{params['dropout']:<8} {params['weight_decay']:<10.0e} {score:<10.4f}")

    # THE BEST CONFIG
    if passed:
        best = passed[0]
        print(f"\nBest configuration:")
        print(f"  {best['params']}")
        print(f"  Score: {get_score(best):.4f}")
