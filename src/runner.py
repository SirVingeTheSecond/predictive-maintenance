import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data import load_data, create_dataloaders, SignalDataset
from models import create_model, count_parameters
from training import train, train_kfold_cv
from evaluation import evaluate, evaluate_noise_robustness
from experiment import (
    get_experiment_dir,
    experiment_exists,
    save_experiment,
)
from utils import set_seed, get_device


def run_single_experiment(
    study_name: str,
    model_name: str,
    mode: str,
    split: str,
    config_name: str,
    seed: int,
    epochs: int = None,
    data_dir: str = None,
    force: bool = False,
    verbose: bool = True,
    test_noise: bool = False,
    use_kfold_cv: bool = None,
    n_folds: int = 3,
    fold_verbose: bool = False,
) -> dict:
    """
    Run a single experiment.
    
    Args:
        study_name: Name of the study
        model_name: Model architecture
        mode: Classification mode
        split: Data split strategy
        config_name: Configuration name (for directory naming)
        seed: Random seed
        epochs: Number of epochs
        data_dir: Data directory
        force: Overwrite existing results
        verbose: Print progress
        test_noise: Run noise testing
        use_kfold_cv: Use k-fold CV for epoch selection (auto-enabled for fault-size split)
        n_folds: Number of folds for CV
        fold_verbose: Print per-epoch details during k-fold CV (default False for cleaner output)
        
    Returns:
        Dictionary with results
    """
    exp_dir = get_experiment_dir(study_name, model_name, config_name, seed)

    # Check if already done
    if experiment_exists(exp_dir) and not force:
        if verbose:
            print(f"SKIP: {exp_dir} (already exists)")
        return None

    if epochs is None:
        epochs = config.EPOCHS
    if data_dir is None:
        data_dir = config.DATA_DIR

    # Set seed
    set_seed(seed)

    if verbose:
        print(f"Experiment: {model_name} | {mode} | {split} | seed={seed}")

    # Load data
    data = load_data(mode=mode, split=split, seed=seed, data_dir=data_dir, verbose=verbose)
    train_loader, val_loader, test_loader = create_dataloaders(data, mode)
    
    # Check if validation set is empty (fault-size split)
    has_val = len(data["X_val"]) > 0
    
    # Auto-enable k-fold CV for fault-size splits if not explicitly set
    if use_kfold_cv is None:
        use_kfold_cv = not has_val
    
    # Create model
    num_classes = config.NUM_CLASSES[mode]
    model = create_model(model_name, num_classes=num_classes)
    params = count_parameters(model)

    if verbose:
        print(f"Parameters: {params:,}")

    device = get_device()
    
    # K-fold CV for epoch selection (fault-size split)
    kfold_result = None
    actual_epochs = epochs
    
    if use_kfold_cv and not has_val:
        if verbose:
            print(f"\n--- K-Fold CV for Epoch Selection ({n_folds} folds) ---")
        
        # Create training dataset for k-fold CV
        X_train = torch.FloatTensor(data["X_train"])
        if mode == "multilabel":
            y_train = torch.FloatTensor(data["y_train"])
        else:
            y_train = torch.LongTensor(data["y_train"])
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

        def make_model():
            return create_model(model_name, num_classes=num_classes)
        
        # Run k-fold CV to find optimal epochs
        kfold_result = train_kfold_cv(
            model_class=make_model,
            model_kwargs={},
            train_dataset=train_dataset,
            mode=mode,
            n_folds=n_folds,
            epochs=epochs,
            device=device,
            seed=seed,
            verbose=verbose,
            fold_verbose=fold_verbose,
        )
        
        actual_epochs = kfold_result["optimal_epochs"]
        
        if verbose:
            print(f"\n--- Retraining on Full Data ({actual_epochs} epochs) ---")
        
        # Reset seed and create the model for final training
        set_seed(seed)
        model = create_model(model_name, num_classes=num_classes)
    
    elif not has_val:
        if verbose:
            print("NOTE: No validation set (fault-size split). Training for fixed epochs.")

    # Train (val_loader may be empty for fault-size split)
    train_result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader if has_val else None,
        mode=mode,
        epochs=actual_epochs,
        device=device,
        verbose=verbose,
    )

    # Evaluate
    test_metrics = evaluate(model, test_loader, device, mode)

    if verbose:
        if mode == "multilabel":
            print(f"\nTest Macro AUROC: {test_metrics['macro_auroc']:.4f}")
            for name, metrics in test_metrics['per_class_metrics'].items():
                print(f"  {name}: AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")
        else:
            print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
            for name, metrics in test_metrics['per_class_metrics'].items():
                print(f"  {name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

    # Optional noise testing
    noise_results = None
    if test_noise:
        if verbose:
            print("\nRunning noise robustness test...")
        noise_results = evaluate_noise_robustness(
            model, data["X_test"], data["y_test"], mode, device
        )
        if verbose:
            print("Noise robustness summary:")
            for snr, metric in noise_results["summary"].items():
                print(f"  {snr}: {metric:.4f}")

    # Prepare results
    results = {
        "model": model_name,
        "mode": mode,
        "split": split,
        "seed": seed,
        "parameters": params,
        "epochs_trained": train_result["epochs_trained"],
        "best_val_acc": train_result["best_val_acc"],
        "best_epoch": train_result["best_epoch"],
        "test_metrics": test_metrics,
        "noise_robustness": noise_results,
        "kfold_cv": kfold_result,  # None if not used
        "timestamp": datetime.now().isoformat(),
    }

    exp_config = {
        "model": model_name,
        "mode": mode,
        "split": split,
        "seed": seed,
        "epochs": actual_epochs, # Actual epochs used (may differ from input if k-fold CV)
        "epochs_requested": epochs,
        "use_kfold_cv": use_kfold_cv,
        "n_folds": n_folds if use_kfold_cv else None,
        "lr": config.LEARNING_RATE,
        "weight_decay": config.WEIGHT_DECAY,
        "dropout": config.DROPOUT,
        "batch_size": config.BATCH_SIZE,
    }

    # Save
    save_experiment(
        exp_dir=exp_dir,
        model_state=model.state_dict(),
        results=results,
        exp_config=exp_config,
        history=train_result["history"],
    )

    if verbose:
        print(f"\nSaved to: {exp_dir}")

    return results


def run_study(
    study_name: str,
    force: bool = False,
    verbose: bool = True,
    test_noise: bool = False,
) -> list:
    """
    Run all experiments in a study.
    
    Args:
        study_name: Name of the study (must be defined in config.STUDIES)
        force: Overwrite existing results
        verbose: Print progress
        test_noise: Run noise robustness testing
        
    Returns:
        List of all results
    """
    if study_name not in config.STUDIES:
        available = list(config.STUDIES.keys())
        raise ValueError(f"Unknown study: {study_name}. Available: {available}")

    study = config.STUDIES[study_name]

    if verbose:
        print("=" * 70)
        print(f"STUDY: {study_name}")
        print(f"Description: {study.get('description', '')}")
        print("=" * 70)

    models = study["models"]
    configurations = study["configurations"]
    seeds = study["seeds"]
    epochs = study.get("epochs", config.EPOCHS)
    
    # K-fold CV parameters
    use_kfold_cv = study.get("use_kfold_cv", None) # None means it should auto detect (works as intended, so I will not change this to make it "clearer")
    n_folds = study.get("n_folds", 3)

    total = len(models) * len(configurations) * len(seeds)
    completed = 0
    skipped = 0
    results = []

    for model_name in models:
        for cfg in configurations:
            mode = cfg["mode"]
            split = cfg["split"]
            config_name = cfg["name"]

            for seed in seeds:
                completed += 1
                exp_dir = get_experiment_dir(study_name, model_name, config_name, seed)

                if experiment_exists(exp_dir) and not force:
                    skipped += 1
                    if verbose:
                        print(f"[{completed}/{total}] SKIP: {model_name}_{config_name}_seed{seed}")
                    continue

                if verbose:
                    print(f"\n[{completed}/{total}] Running: {model_name}_{config_name}_seed{seed}")

                result = run_single_experiment(
                    study_name=study_name,
                    model_name=model_name,
                    mode=mode,
                    split=split,
                    config_name=config_name,
                    seed=seed,
                    epochs=epochs,
                    force=force,
                    verbose=verbose,
                    test_noise=test_noise,
                    use_kfold_cv=use_kfold_cv,
                    n_folds=n_folds,
                )

                if result is not None:
                    results.append(result)

    if verbose:
        print("\n" + "=" * 70)
        print(f"STUDY COMPLETE: {study_name}")
        print(f"Total: {total}, New: {len(results)}, Skipped: {skipped}")
        print("=" * 70)

    return results


def aggregate_seed_results(results: list) -> dict:
    """
    Aggregate results across seeds for a single model/config.
    
    Reports mean ± std as recommended by papers:
    - Rosa et al.: 30 seeds
    - ECMCTP: 10 seeds
    - CNN-LSTM: 5-fold CV
    
    Args:
        results: List of result dicts from same model/config, different seeds
        
    Returns:
        Dictionary with the aggregated statistics
    """
    if not results:
        return {}
    
    mode = results[0]["mode"]
    
    if mode == "multilabel":
        aurocs = [r["test_metrics"]["macro_auroc"] for r in results]
        return {
            "n_seeds": len(results),
            "mean_auroc": float(np.mean(aurocs)),
            "std_auroc": float(np.std(aurocs)),
            "min_auroc": float(np.min(aurocs)),
            "max_auroc": float(np.max(aurocs)),
            "seeds": [r["seed"] for r in results],
        }
    else:
        accuracies = [r["test_metrics"]["accuracy"] for r in results]
        f1_scores = [r["test_metrics"]["macro_f1"] for r in results]
        
        return {
            "n_seeds": len(results),
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "min_accuracy": float(np.min(accuracies)),
            "max_accuracy": float(np.max(accuracies)),
            "mean_macro_f1": float(np.mean(f1_scores)),
            "std_macro_f1": float(np.std(f1_scores)),
            "seeds": [r["seed"] for r in results],
        }
