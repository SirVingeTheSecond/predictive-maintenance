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

"""
Experiment runner comparing feature modes and architectures.
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
    """Comprehensive model evaluation."""
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
    }


def run_experiment(
        model_name: str,
        split_strategy: str,
        feature_mode: str = None,
        epochs: int = None,
        train_loads: list = None,
        test_loads: list = None,
        data_dir: str = "data/raw"
) -> dict:
    """Run single experiment."""
    if epochs is None:
        epochs = config["epochs"]
    if feature_mode is None:
        feature_mode = config["feature_mode"]

    # Temporarily set feature mode
    original_mode = config["feature_mode"]
    config["feature_mode"] = feature_mode

    print(f"\n{'=' * 60}")
    print(f"Model: {model_name.upper()} | Strategy: {split_strategy} | Features: {feature_mode}")
    print(f"{'=' * 60}")

    if split_strategy == "cross_load":
        data = load_data(strategy=split_strategy, data_dir=data_dir,
                         train_loads=train_loads, test_loads=test_loads)
    else:
        data = load_data(strategy=split_strategy, data_dir=data_dir)

    train_loader, val_loader, test_loader = create_dataloaders(data)

    model = get_model(model_name).to(config["device"])
    params = count_parameters(model)
    print(f"Parameters: {params:,}")

    result = train_model(model, train_loader, val_loader, epochs=epochs, model_name=model_name)
    test_metrics = evaluate_with_metrics(model, test_loader)

    print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    print("Per-class recall:")
    for cls_name in config["class_names"]:
        if cls_name in test_metrics["report"]:
            recall = test_metrics["report"][cls_name]["recall"]
            print(f"  {cls_name}: {recall:.3f}")

    # Restore original feature mode
    config["feature_mode"] = original_mode

    return {
        "model": model_name,
        "split_strategy": split_strategy,
        "feature_mode": feature_mode,
        "parameters": params,
        "epochs_trained": result["epochs_trained"],
        "best_val_acc": result["best_val_acc"],
        "test_accuracy": test_metrics["accuracy"],
        "classification_report": test_metrics["report"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "timestamp": datetime.now().isoformat(),
    }


def run_improvement_study(data_dir: str = "data/raw") -> list:
    """
    Compare improvements on fault-size generalization task.

    Tests:
    1. Time domain (baseline)
    2. FFT features
    3. Time + FFT combined
    4. Deep CNN with FFT
    """
    all_results = []

    print("=" * 70)
    print("IMPROVEMENT STUDY: Fault-Size Generalization")
    print("=" * 70)

    configurations = [
        # (model, feature_mode, description)
        ("cnn1d", "time", "CNN1D + Time Domain (Baseline)"),
        ("cnn1d", "fft", "CNN1D + FFT Features"),
        ("cnn1d", "both", "CNN1D + Time + FFT"),
        ("cnn1d_deep", "fft", "Deep CNN (ResNet-style) + FFT"),
        ("lstm", "fft", "LSTM + FFT"),
        ("cnnlstm", "fft", "CNN-LSTM + FFT"),
    ]

    for model_name, feature_mode, description in configurations:
        print(f"\n{'#' * 70}")
        print(f"# {description}")
        print(f"{'#' * 70}")

        result = run_experiment(
            model_name=model_name,
            split_strategy="fault_size_all_loads",
            feature_mode=feature_mode,
            epochs=100,
            data_dir=data_dir
        )
        result["description"] = description
        all_results.append(result)

    return all_results


def print_improvement_summary(results: list):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("IMPROVEMENT STUDY RESULTS")
    print("=" * 90)
    print(f"{'Configuration':<40} {'Test Acc':<10} {'Normal':<8} {'Ball':<8} {'IR':<8} {'OR':<8}")
    print("-" * 90)

    for r in results:
        report = r["classification_report"]
        normal = report.get("Normal", {}).get("recall", 0)
        ball = report.get("Ball", {}).get("recall", 0)
        ir = report.get("IR", {}).get("recall", 0)
        or_recall = report.get("OR", {}).get("recall", 0)

        print(f"{r['description']:<40} {r['test_accuracy']:<10.4f} "
              f"{normal:<8.3f} {ball:<8.3f} {ir:<8.3f} {or_recall:<8.3f}")


def save_results(results: list, filename: str = "results/improvement_results.json"):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])

    results = run_improvement_study(data_dir="data/raw")
    print_improvement_summary(results)
    save_results(results)
