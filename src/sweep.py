"""
Hyperparameter sweep with two-phase screening.

Automatically handles both:
- Splits WITH validation sets (random): uses validation accuracy
- Splits WITHOUT validation sets (fault_size): uses K-fold CV on training data

This is consistent with how runner.py handles the same scenarios.

Usage:
    python -m src.main sweep hyperparameter_search
    python -m src.main sweep-results hyperparameter_search
"""

import os
import json
import itertools
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from . import config
from .data import load_data, create_dataloaders, SignalDataset
from .models import create_model
from .utils import count_parameters, print_separator, set_seed, get_device
from .training import train, get_criterion, train_epoch, evaluate_epoch
from .evaluation import evaluate


# =============================================================================
# Directory and naming utilities
# =============================================================================

def get_sweep_dir(sweep_name: str) -> str:
    """Get sweep results directory."""
    return os.path.join(config.RESULTS_DIR, "sweeps", sweep_name)


def get_config_name(params: dict, seed: int = None) -> str:
    """Generate name for parameter combination."""
    parts = []
    for key in sorted(params.keys()):
        value = params[key]
        if isinstance(value, float):
            parts.append(f"{key}{value:.0e}")
        else:
            parts.append(f"{key}{value}")
    name = "_".join(parts)
    if seed is not None:
        name += f"_seed{seed}"
    return name


def generate_combinations(sweep_name: str) -> list:
    """Generate all hyperparameter combinations for a sweep."""
    if sweep_name not in config.SWEEPS:
        raise ValueError(f"Unknown sweep: {sweep_name}")

    sweep = config.SWEEPS[sweep_name]
    param_grid = sweep["param_grid"]

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    combinations = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        combinations.append(params)

    return combinations


# =============================================================================
# Screening functions
# =============================================================================

def run_screening_with_val(
        model,
        train_loader,
        val_loader,
        mode: str,
        epochs: int,
        lr: float,
        weight_decay: float,
        device: str,
) -> float:
    """
    Screening phase using validation set.

    Used when split has a validation set (e.g., random split).

    Returns:
        best_val_acc
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


def run_screening_with_kfold(
        model_name: str,
        num_classes: int,
        dropout: float,
        train_dataset,
        mode: str,
        epochs: int,
        lr: float,
        weight_decay: float,
        device: str,
        n_folds: int = 3,
        seed: int = 42,
) -> float:
    """
    Screening phase using K-fold CV on training data.

    Used when split has NO validation set (e.g., fault_size splits).
    Consistent with runner.py / train_kfold_cv() methodology.

    Returns:
        mean_val_acc across folds
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(train_dataset)))):
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)

        model = create_model(model_name, num_classes=num_classes, dropout=dropout)
        model = model.to(device)

        criterion = get_criterion(mode)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_acc = 0.0
        for epoch in range(1, epochs + 1):
            train_epoch(model, train_loader, criterion, optimizer, device, mode)
            _, val_acc = evaluate_epoch(model, val_loader, criterion, device, mode)
            if val_acc > best_val_acc:
                best_val_acc = val_acc

        fold_accs.append(best_val_acc)

    return float(np.mean(fold_accs))


# =============================================================================
# Full training functions
# =============================================================================

def run_full_training_with_kfold(
        model_name: str,
        num_classes: int,
        dropout: float,
        train_dataset,
        mode: str,
        epochs: int,
        lr: float,
        weight_decay: float,
        device: str,
        n_folds: int = 3,
        seed: int = 42,
        verbose: bool = False,
) -> dict:
    """
    K-fold CV to determine optimal epochs, then train on full data.

    Used when split has NO validation set.
    Consistent with runner.py / train_kfold_cv() methodology.

    Returns:
        Dictionary with trained model and metrics
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(train_dataset)))):
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)

        model = create_model(model_name, num_classes=num_classes, dropout=dropout)
        model = model.to(device)

        result = train(
            model, train_loader, val_loader, mode,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            device=device, verbose=False
        )

        fold_results.append({
            "best_val_acc": result["best_val_acc"],
            "best_epoch": result["best_epoch"],
        })

    # Determine optimal epochs from CV (median, at least 10)
    best_epochs = [r["best_epoch"] for r in fold_results]
    optimal_epochs = max(int(np.median(best_epochs)), 10)
    mean_val_acc = float(np.mean([r["best_val_acc"] for r in fold_results]))

    if verbose:
        print(f"  K-fold CV: mean_val_acc={mean_val_acc:.4f}, optimal_epochs={optimal_epochs}")

    # Retrain on full training data for optimal epochs
    set_seed(seed)
    full_train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = create_model(model_name, num_classes=num_classes, dropout=dropout)
    model = model.to(device)

    result = train(
        model, full_train_loader, None, mode,
        epochs=optimal_epochs, lr=lr, weight_decay=weight_decay,
        device=device, verbose=verbose
    )

    return {
        "model": model,
        "kfold_mean_val_acc": mean_val_acc,
        "kfold_optimal_epochs": optimal_epochs,
        "epochs_trained": optimal_epochs,
    }


# =============================================================================
# Main experiment runner
# =============================================================================

def run_sweep_experiment(
        sweep_name: str,
        params: dict,
        seed: int,
        verbose: bool = True,
) -> dict:
    """
    Run a single sweep experiment with screening.

    Automatically handles splits with and without validation sets:
    - With validation: standard screening and training with early stopping
    - Without validation: K-fold CV for screening and epoch selection

    This is consistent with runner.py behavior.
    """
    sweep = config.SWEEPS[sweep_name]
    base_config = sweep["base_config"]
    screening_epochs = sweep.get("screening_epochs", 20)
    screening_threshold = sweep.get("screening_threshold", 0.4)
    full_epochs = sweep.get("full_epochs", config.EPOCHS)
    n_folds = sweep.get("n_folds", 3)

    mode = base_config["mode"]
    split = base_config["split"]

    model_name = params["model"]
    lr = params["lr"]
    dropout = params["dropout"]
    weight_decay = params["weight_decay"]

    config_name = get_config_name(params, seed)
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
        print(f"Config: {get_config_name(params)} | Seed: {seed}")

    # Load data and check for validation set
    data = load_data(mode=mode, split=split, seed=seed, verbose=False)
    has_val = len(data["X_val"]) > 0
    num_classes = config.NUM_CLASSES[mode]

    if verbose:
        method = "validation set" if has_val else f"{n_folds}-fold CV"
        print(f"  Parameters: {count_parameters(create_model(model_name, num_classes, dropout)):,}")
        print(f"  Screening: {screening_epochs} epochs, threshold={screening_threshold} ({method})")

    # =========================================================================
    # SCREENING PHASE
    # =========================================================================

    if has_val:
        # Standard screening with validation set
        train_loader, val_loader, test_loader = create_dataloaders(data, mode)

        model = create_model(model_name, num_classes=num_classes, dropout=dropout)
        model = model.to(device)

        screening_acc = run_screening_with_val(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            mode=mode,
            epochs=screening_epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )
    else:
        # K-fold CV screening (no validation set)
        train_dataset = SignalDataset(data["X_train"], data["y_train"], mode)
        test_dataset = SignalDataset(data["X_test"], data["y_test"], mode)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        screening_acc = run_screening_with_kfold(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
            train_dataset=train_dataset,
            mode=mode,
            epochs=screening_epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            n_folds=n_folds,
            seed=seed,
        )

    if verbose:
        print(f"  Screening accuracy: {screening_acc:.4f}")

    # Check screening threshold
    if screening_acc < screening_threshold:
        if verbose:
            print(f"  FAILED screening (< {screening_threshold})")

        result = {
            "params": params,
            "screening_acc": float(screening_acc),
            "passed_screening": False,
            "has_validation": has_val,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        }

        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    # =========================================================================
    # FULL TRAINING PHASE
    # =========================================================================

    if verbose:
        print(f"  PASSED screening, full training ({full_epochs} epochs)")

    set_seed(seed)

    if has_val:
        # Standard training with validation set
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

        best_val_acc = train_result["best_val_acc"]
        best_epoch = train_result["best_epoch"]
        epochs_trained = train_result["epochs_trained"]
    else:
        # K-fold CV training (no validation set)
        train_dataset = SignalDataset(data["X_train"], data["y_train"], mode)

        kfold_result = run_full_training_with_kfold(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
            train_dataset=train_dataset,
            mode=mode,
            epochs=full_epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            n_folds=n_folds,
            seed=seed,
            verbose=verbose,
        )

        model = kfold_result["model"]
        best_val_acc = kfold_result["kfold_mean_val_acc"]
        best_epoch = kfold_result["kfold_optimal_epochs"]
        epochs_trained = kfold_result["epochs_trained"]

    # =========================================================================
    # EVALUATION
    # =========================================================================

    test_metrics = evaluate(model, test_loader, device, mode)

    if verbose:
        if mode == "multilabel":
            print(f"  Test Macro AUROC: {test_metrics['macro_auroc']:.4f}")
        else:
            print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")

    result = {
        "params": params,
        "screening_acc": float(screening_acc),
        "passed_screening": True,
        "has_validation": has_val,
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(epochs_trained),
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


# =============================================================================
# Sweep orchestration
# =============================================================================

def run_sweep(
        sweep_name: str,
        seed: int = None,
        seeds: list = None,
        verbose: bool = True,
) -> list:
    """
    Run all combinations in a sweep with multiple seeds.

    Args:
        sweep_name: Name of the sweep (from config.SWEEPS)
        seed: Single seed (for backward compatibility with CLI)
        seeds: List of seeds (overrides seed and config)
        verbose: Print progress

    Returns:
        List of result dictionaries
    """
    if sweep_name not in config.SWEEPS:
        raise ValueError(f"Unknown sweep: {sweep_name}. Available: {list(config.SWEEPS.keys())}")

    sweep = config.SWEEPS[sweep_name]

    # Determine seeds to use (priority: seeds arg > seed arg > config)
    if seeds is not None:
        pass
    elif seed is not None:
        seeds = [seed]
    else:
        seeds = sweep.get("seeds", config.EXPERIMENT_SEEDS)

    combinations = generate_combinations(sweep_name)
    total = len(combinations) * len(seeds)

    if verbose:
        print(f"SWEEP: {sweep_name}")
        print(f"Combinations: {len(combinations)}")
        print(f"Seeds: {seeds}")
        print(f"Total experiments: {total}")
        print_separator()

    results = []
    current = 0

    for params in combinations:
        for s in seeds:
            current += 1
            if verbose:
                print(f"\n[{current}/{total}]")

            result = run_sweep_experiment(
                sweep_name=sweep_name,
                params=params,
                seed=s,
                verbose=verbose,
            )
            results.append(result)

    return results


# =============================================================================
# Results loading and analysis
# =============================================================================

def load_sweep_results(sweep_name: str) -> list:
    """Load all results from a sweep."""
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


def aggregate_sweep_results(results: list) -> dict:
    """Aggregate results across seeds for each configuration."""
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in results:
        if r.get("passed_screening", False):
            key = tuple(sorted(r["params"].items()))
            grouped[key].append(r)

    aggregated = {}
    for key, runs in grouped.items():
        params = dict(key)
        accuracies = [r["test_metrics"]["accuracy"] for r in runs]

        aggregated[key] = {
            "params": params,
            "n_seeds": len(runs),
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "seeds": [r["seed"] for r in runs],
        }

    return aggregated


def print_sweep_leaderboard(sweep_name: str, top_n: int = 10):
    """Print leaderboard of best configurations (aggregated across seeds)."""
    results = load_sweep_results(sweep_name)

    if not results:
        print(f"No results found for sweep: {sweep_name}")
        return

    passed = [r for r in results if r.get("passed_screening", False)]
    failed = [r for r in results if not r.get("passed_screening", False)]

    print(f"SWEEP RESULTS: {sweep_name}")
    print(f"Total runs: {len(results)}")
    print(f"Passed screening: {len(passed)}")
    print(f"Failed screening: {len(failed)}")
    print_separator()

    if not passed:
        print("No configurations passed screening.")
        return

    aggregated = aggregate_sweep_results(results)

    sorted_configs = sorted(
        aggregated.values(),
        key=lambda x: x["accuracy_mean"],
        reverse=True
    )

    print(f"\n{'Rank':<5} {'Model':<8} {'LR':<8} {'Drop':<6} {'WD':<8} {'Acc Mean':<10} {'Std':<8} {'N':<3}")
    print_separator(char="-", width=70)

    for i, cfg in enumerate(sorted_configs[:top_n], 1):
        p = cfg["params"]
        print(f"{i:<5} {p['model']:<8} {p['lr']:<8.0e} {p['dropout']:<6} "
              f"{p['weight_decay']:<8.0e} {cfg['accuracy_mean'] * 100:<10.2f} "
              f"{cfg['accuracy_std'] * 100:<8.2f} {cfg['n_seeds']:<3}")

    best = sorted_configs[0]
    print(f"\nBest configuration:")
    print(f"  {best['params']}")
    print(f"  Accuracy: {best['accuracy_mean'] * 100:.2f}% ± {best['accuracy_std'] * 100:.2f}%")
    print(f"  Seeds tested: {best['n_seeds']}")
