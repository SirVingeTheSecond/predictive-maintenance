"""
Experiment Runner for CWRU Bearing Fault Diagnosis Study.

Executes a comprehensive study comparing model performance under
different data splitting strategies to evaluate generalization.

Experiments:
1. Random split baseline (chunk-based to prevent leakage)
2. Fault-size split with single load (tests severity generalization)
3. Fault-size split with all loads (tests severity generalization with diversity)
4. Cross-load experiments (tests operating condition generalization)
"""

import json
import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from config import config
from data import load_data, create_dataloaders
from models import get_model, count_parameters
from training import train_model, get_predictions


def evaluate_with_metrics(model, test_loader, device: str = None) -> dict:
    """
    Comprehensive model evaluation with per-class metrics.

    Returns:
        Dictionary containing accuracy, classification report, and confusion matrix
    """
    if device is None:
        device = config["device"]

    predictions, labels = get_predictions(model, test_loader, device)

    accuracy = np.mean(predictions == labels)

    present_classes = sorted(set(labels))
    target_names = [config["class_names"][i] for i in present_classes]

    report = classification_report(
        labels, predictions,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    conf_matrix = confusion_matrix(labels, predictions)

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": conf_matrix.tolist(),
        "predictions": predictions.tolist(),
        "labels": labels.tolist(),
    }


def run_single_experiment(
    model_name: str,
    split_strategy: str,
    epochs: int = None,
    train_loads: list = None,
    test_loads: list = None,
    data_dir: str = "data/raw"
) -> dict:
    """
    Run a single experiment with specified configuration.
    """
    if epochs is None:
        epochs = config["epochs"]

    print(f"\n{'=' * 60}")
    print(f"Model: {model_name.upper()} | Strategy: {split_strategy}")
    print(f"{'=' * 60}")

    if split_strategy == "cross_load":
        data = load_data(
            strategy=split_strategy,
            data_dir=data_dir,
            train_loads=train_loads,
            test_loads=test_loads
        )
    else:
        data = load_data(strategy=split_strategy, data_dir=data_dir)

    train_loader, val_loader, test_loader = create_dataloaders(data)

    model = get_model(model_name).to(config["device"])
    params = count_parameters(model)

    result = train_model(
        model, train_loader, val_loader,
        epochs=epochs, model_name=model_name
    )

    test_metrics = evaluate_with_metrics(model, test_loader)

    print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Per-class recall:")
    for cls_name in config["class_names"]:
        if cls_name in test_metrics["report"]:
            recall = test_metrics["report"][cls_name]["recall"]
            print(f"  {cls_name}: {recall:.3f}")

    return {
        "model": model_name,
        "split_strategy": split_strategy,
        "train_loads": train_loads,
        "test_loads": test_loads,
        "parameters": params,
        "epochs_trained": result["epochs_trained"],
        "best_val_acc": result["best_val_acc"],
        "test_accuracy": test_metrics["accuracy"],
        "classification_report": test_metrics["report"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "history": result["history"],
        "timestamp": datetime.now().isoformat(),
    }


def run_comprehensive_study(
    models: list = None,
    epochs: int = None,
    data_dir: str = "data/raw"
) -> list:
    """
    Execute the complete experimental study.
    """
    if models is None:
        models = ["cnn1d", "lstm", "cnnlstm"]
    if epochs is None:
        epochs = config["epochs"]

    all_results = []

    print("=" * 70)
    print("CWRU Bearing Fault Diagnosis - Comprehensive Study")
    print("=" * 70)
    print(f"Device: {config['device']}")
    print(f"Window: {config['window_size']} | Stride: {config['stride']} | Overlap: {100*(1 - config['stride']/config['window_size']):.0f}%")
    print(f"Epochs: {epochs}")

    # Experiment 1: Random Split Baseline
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Random Split (Baseline)")
    print("=" * 70)
    for model_name in models:
        result = run_single_experiment(model_name, "random", epochs, data_dir=data_dir)
        result["experiment"] = "Random Split"
        all_results.append(result)

    # Experiment 2: Fault-Size Split (Single Load)
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: Fault-Size Split - Single Load (Test: {config['test_fault_size']})")
    print("=" * 70)
    for model_name in models:
        result = run_single_experiment(model_name, "fault_size", epochs, data_dir=data_dir)
        result["experiment"] = f"Fault-Size Single ({config['test_fault_size']})"
        all_results.append(result)

    # Experiment 3: Fault-Size Split (All Loads)
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 3: Fault-Size Split - All Loads (Test: {config['test_fault_size']})")
    print("=" * 70)
    for model_name in models:
        result = run_single_experiment(model_name, "fault_size_all_loads", epochs, data_dir=data_dir)
        result["experiment"] = f"Fault-Size All Loads ({config['test_fault_size']})"
        all_results.append(result)

    # Experiment 4: Cross-Load (1772 -> 1750)
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Cross-Load (1772 -> 1750)")
    print("=" * 70)
    for model_name in models:
        result = run_single_experiment(
            model_name, "cross_load", epochs,
            train_loads=["1772"], test_loads=["1750"],
            data_dir=data_dir
        )
        result["experiment"] = "Cross-Load (1750)"
        all_results.append(result)

    # Experiment 5: Cross-Load (1772 -> 1730)
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Cross-Load (1772 -> 1730)")
    print("=" * 70)
    for model_name in models:
        result = run_single_experiment(
            model_name, "cross_load", epochs,
            train_loads=["1772"], test_loads=["1730"],
            data_dir=data_dir
        )
        result["experiment"] = "Cross-Load (1730)"
        all_results.append(result)

    # Experiment 6: Cross-Load (1772 -> Both)
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Cross-Load (1772 -> 1750+1730)")
    print("=" * 70)
    for model_name in models:
        result = run_single_experiment(
            model_name, "cross_load", epochs,
            train_loads=["1772"], test_loads=["1750", "1730"],
            data_dir=data_dir
        )
        result["experiment"] = "Cross-Load (Both)"
        all_results.append(result)

    return all_results


def print_summary_table(results: list):
    """Print formatted summary table of all experiment results."""
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Experiment':<30} {'Model':<10} {'Val Acc':<10} {'Test Acc':<10}")
    print("-" * 90)

    experiments = []
    for r in results:
        if r["experiment"] not in experiments:
            experiments.append(r["experiment"])

    for exp in experiments:
        exp_results = [r for r in results if r["experiment"] == exp]
        for r in exp_results:
            print(f"{r['experiment']:<30} {r['model']:<10} "
                  f"{r['best_val_acc']:<10.4f} {r['test_accuracy']:<10.4f}")
        print("-" * 90)

    print("\nAVERAGE TEST ACCURACY BY EXPERIMENT:")
    for exp in experiments:
        exp_results = [r for r in results if r["experiment"] == exp]
        avg_acc = np.mean([r["test_accuracy"] for r in exp_results])
        print(f"  {exp:<30}: {avg_acc:.4f}")


def save_results(results: list, filename: str = "results/comprehensive_results.json"):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    serializable_results = []
    for r in results:
        r_copy = r.copy()
        if "history" in r_copy:
            r_copy["history"] = {
                k: [float(v) for v in vals]
                for k, vals in r_copy["history"].items()
            }
        serializable_results.append(r_copy)

    with open(filename, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])

    results = run_comprehensive_study(
        models=["cnn1d", "lstm", "cnnlstm"],
        epochs=50,
        data_dir="data/raw"
    )

    print_summary_table(results)
    save_results(results)
