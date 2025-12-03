"""
Experiment Runner for CWRU Bearing Fault Diagnosis Study.

Executes a comprehensive study comparing model performance under
different data splitting strategies to evaluate generalization.

Experiments:
1. Random split baseline (potential data leakage)
2. Fault-size split with single load (tests severity generalization)
3. Fault-size split with all loads (tests severity generalization with more diversity)
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

    # Get class names present in the test set
    present_classes = sorted(set(labels))
    target_names = [config["class_names"][i] for i in present_classes]

    report = classification_report(
        labels, predictions,
        labels=present_classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    conf_matrix = confusion_matrix(labels, predictions, labels=present_classes)

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

    Args:
        model_name: Architecture to use (cnn1d, lstm, cnnlstm)
        split_strategy: Data splitting method
        epochs: Maximum training epochs
        train_loads: For cross_load, which loads to train on
        test_loads: For cross_load, which loads to test on
        data_dir: Directory containing data files

    Returns:
        Dictionary containing all experiment results and metadata
    """
    if epochs is None:
        epochs = config["epochs"]

    print(f"\n{'=' * 60}")
    print(f"Model: {model_name} | Split: {split_strategy}")
    print(f"{'=' * 60}")

    # Load data based on strategy
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

    # Initialize and train model
    model = get_model(model_name).to(config["device"])
    params = count_parameters(model)

    result = train_model(
        model, train_loader, val_loader,
        epochs=epochs, model_name=model_name
    )

    # Evaluate on test set
    test_metrics = evaluate_with_metrics(model, test_loader)

    print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"\nPer-class results:")
    for cls_name in config["class_names"]:
        if cls_name in test_metrics["report"]:
            cls_report = test_metrics["report"][cls_name]
            print(f"  {cls_name}: P={cls_report['precision']:.3f} "
                  f"R={cls_report['recall']:.3f} F1={cls_report['f1-score']:.3f}")

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

    Runs all experiments in sequence:
    1. Random split baseline
    2. Fault-size split (single load)
    3. Fault-size split (all loads)
    4. Cross-load: 1772 -> 1750
    5. Cross-load: 1772 -> 1730
    6. Cross-load: 1772 -> both

    Args:
        models: List of model names to evaluate
        epochs: Maximum training epochs per experiment
        data_dir: Directory containing data files

    Returns:
        List of result dictionaries for all experiments
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
    print(f"Classification mode: {config['classification_mode']}")
    print(f"Window size: {config['window_size']}, Stride: {config['stride']}")
    print(f"Epochs: {epochs}")

    # Experiment 1: Random Split Baseline
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Random Split (Baseline)")
    print("=" * 70)
    for model_name in models:
        result = run_single_experiment(
            model_name, "random", epochs, data_dir=data_dir
        )
        result["experiment"] = "Random Split"
        all_results.append(result)

    # Experiment 2: Fault-Size Split (Single Load)
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: Fault-Size Split - Single Load (Test on {config['test_fault_size']})")
    print("=" * 70)
    for model_name in models:
        result = run_single_experiment(
            model_name, "fault_size", epochs, data_dir=data_dir
        )
        result["experiment"] = f"Fault-Size Single ({config['test_fault_size']})"
        all_results.append(result)

    # Experiment 3: Fault-Size Split (All Loads)
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 3: Fault-Size Split - All Loads (Test on {config['test_fault_size']})")
    print("=" * 70)
    for model_name in models:
        result = run_single_experiment(
            model_name, "fault_size_all_loads", epochs, data_dir=data_dir
        )
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
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Experiment':<30} {'Model':<10} {'Val Acc':<10} {'Test Acc':<10} {'Drop':<10}")
    print("-" * 90)

    # Group results by experiment
    experiments = []
    for r in results:
        if r["experiment"] not in experiments:
            experiments.append(r["experiment"])

    # Get baseline accuracy for drop calculation
    baseline_acc = {}
    for r in results:
        if r["experiment"] == "Random Split":
            baseline_acc[r["model"]] = r["test_accuracy"]

    for exp in experiments:
        exp_results = [r for r in results if r["experiment"] == exp]
        for r in exp_results:
            drop = baseline_acc.get(r["model"], 1.0) - r["test_accuracy"]
            print(f"{r['experiment']:<30} {r['model']:<10} "
                  f"{r['best_val_acc']:<10.4f} {r['test_accuracy']:<10.4f} "
                  f"{drop:<10.4f}")
        print("-" * 90)

    # Print averages by experiment
    print("\nAVERAGE TEST ACCURACY BY EXPERIMENT:")
    print("-" * 50)
    for exp in experiments:
        exp_results = [r for r in results if r["experiment"] == exp]
        avg_acc = np.mean([r["test_accuracy"] for r in exp_results])
        print(f"  {exp:<30}: {avg_acc:.4f}")


def print_per_class_analysis(results: list):
    """Print per-class recall analysis across experiments."""
    print("\n" + "=" * 90)
    print("PER-CLASS ANALYSIS (Recall by Fault Type)")
    print("=" * 90)
    print(f"{'Experiment':<25} {'Model':<10} {'Normal':<8} {'Ball':<8} {'IR':<8} {'OR':<8}")
    print("-" * 90)

    experiments = []
    for r in results:
        if r["experiment"] not in experiments:
            experiments.append(r["experiment"])

    for exp in experiments:
        exp_results = [r for r in results if r["experiment"] == exp]
        for r in exp_results:
            report = r["classification_report"]
            recalls = []
            for cls in ["Normal", "Ball", "IR", "OR"]:
                if cls in report:
                    recalls.append(f"{report[cls]['recall']:.3f}")
                else:
                    recalls.append("N/A")

            print(f"{r['experiment']:<25} {r['model']:<10} "
                  f"{recalls[0]:<8} {recalls[1]:<8} {recalls[2]:<8} {recalls[3]:<8}")
        print("-" * 90)


def save_results(results: list, filename: str = "results/comprehensive_results.json"):
    """Save results to JSON file for later analysis."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
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


def print_key_findings(results: list):
    """Print summary of key findings from the study."""
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Calculate averages
    experiments = {}
    for r in results:
        exp = r["experiment"]
        if exp not in experiments:
            experiments[exp] = []
        experiments[exp].append(r["test_accuracy"])

    baseline = np.mean(experiments.get("Random Split", [1.0]))

    print(f"\n1. Random split baseline: {baseline * 100:.1f}%")

    print("\n2. Fault-size generalization:")
    for exp_name in ["Fault-Size Single (014)", "Fault-Size All Loads (014)"]:
        if exp_name in experiments:
            avg = np.mean(experiments[exp_name])
            drop = (baseline - avg) * 100
            print(f"   - {exp_name}: {avg * 100:.1f}% (drop: {drop:.1f}%)")

    print("\n3. Cross-load generalization:")
    for exp_name in ["Cross-Load (1750)", "Cross-Load (1730)", "Cross-Load (Both)"]:
        if exp_name in experiments:
            avg = np.mean(experiments[exp_name])
            drop = (baseline - avg) * 100
            print(f"   - Train 1772 -> Test {exp_name.split('(')[1].replace(')', '')}: "
                  f"{avg * 100:.1f}% (drop: {drop:.1f}%)")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])

    # Run comprehensive study
    results = run_comprehensive_study(
        models=["cnn1d", "lstm", "cnnlstm"],
        epochs=50,
        data_dir="data/raw"
    )

    # Print analysis
    print_summary_table(results)
    print_per_class_analysis(results)
    print_key_findings(results)

    # Save results
    save_results(results)

    print("\n" + "=" * 70)
