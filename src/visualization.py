"""
Visualization Utilities for Experiment Results.

Generates publication-quality figures for the research paper including
confusion matrices, training curves, and comparative bar charts.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import config


# Configure matplotlib for publication quality
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    save_path: str = None
):
    """
    Plot a confusion matrix with annotations.

    Args:
        conf_matrix: Square confusion matrix array
        class_names: List of class names for axis labels
        title: Plot title
        normalize: If True, show percentages instead of counts
        save_path: If provided, save figure to this path
    """
    if normalize:
        conf_matrix = conf_matrix.astype(float)
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix = np.divide(
            conf_matrix, row_sums,
            where=row_sums != 0,
            out=np.zeros_like(conf_matrix)
        )

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=1 if normalize else None,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.show()


def plot_training_curves(
    history: dict,
    title: str = "Training History",
    save_path: str = None
):
    """
    Plot training and validation loss and accuracy curves.

    Args:
        history: Dictionary with train_loss, val_loss, train_acc, val_acc lists
        title: Plot title
        save_path: If provided, save figure to this path
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train")
    axes[1].plot(epochs, history["val_acc"], "r-", label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.show()


def plot_experiment_comparison(
    results: list,
    save_path: str = None
):
    """
    Create bar chart comparing test accuracy across all experiments and models.

    Args:
        results: List of experiment result dictionaries
        save_path: If provided, save figure to this path
    """
    # Extract unique experiments and models
    experiments = []
    for r in results:
        if r["experiment"] not in experiments:
            experiments.append(r["experiment"])

    models = ["cnn1d", "lstm", "cnnlstm"]

    # Organize data for plotting
    data = {model: [] for model in models}
    for exp in experiments:
        for model in models:
            acc = None
            for r in results:
                if r["experiment"] == exp and r["model"] == model:
                    acc = r["test_accuracy"]
                    break
            data[model].append(acc if acc is not None else 0)

    # Create grouped bar chart
    x = np.arange(len(experiments))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2ecc71", "#3498db", "#9b59b6"]
    for i, (model, color) in enumerate(zip(models, colors)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[model], width, label=model.upper(), color=color)

        # Add value labels on bars
        for bar, val in zip(bars, data[model]):
            height = bar.get_height()
            ax.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90 if len(experiments) > 4 else 0
            )

    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Performance Across Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.show()


def plot_per_class_recall_heatmap(
    results: list,
    save_path: str = None
):
    """
    Create heatmap showing per-class recall for each experiment and model.

    Reveals which fault types generalize well and which fail under
    different splitting strategies.

    Args:
        results: List of experiment result dictionaries
        save_path: If provided, save figure to this path
    """
    models = ["cnn1d", "lstm", "cnnlstm"]
    classes = ["Normal", "Ball", "IR", "OR"]

    experiments = []
    for r in results:
        if r["experiment"] not in experiments:
            experiments.append(r["experiment"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    for idx, model in enumerate(models):
        # Build recall matrix
        recall_matrix = []
        exp_labels = []

        for exp in experiments:
            for r in results:
                if r["experiment"] == exp and r["model"] == model:
                    row = []
                    for cls in classes:
                        if cls in r["classification_report"]:
                            row.append(r["classification_report"][cls]["recall"])
                        else:
                            row.append(np.nan)
                    recall_matrix.append(row)
                    exp_labels.append(exp)
                    break

        recall_matrix = np.array(recall_matrix)

        # Create heatmap
        sns.heatmap(
            recall_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            xticklabels=classes,
            yticklabels=exp_labels,
            ax=axes[idx],
            vmin=0,
            vmax=1,
            cbar=idx == 2,
        )

        axes[idx].set_title(f"{model.upper()}")
        axes[idx].set_xlabel("Fault Type")
        if idx == 0:
            axes[idx].set_ylabel("Experiment")

    fig.suptitle("Per-Class Recall Across Experiments", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.show()


def plot_accuracy_drop_chart(
    results: list,
    save_path: str = None
):
    """
    Visualize accuracy drop from baseline for each experiment.

    Clearly shows the impact of different splitting strategies
    on model generalization.

    Args:
        results: List of experiment result dictionaries
        save_path: If provided, save figure to this path
    """
    # Get baseline accuracies
    baseline = {}
    for r in results:
        if r["experiment"] == "Random Split":
            baseline[r["model"]] = r["test_accuracy"]

    experiments = []
    for r in results:
        if r["experiment"] not in experiments and r["experiment"] != "Random Split":
            experiments.append(r["experiment"])

    models = ["cnn1d", "lstm", "cnnlstm"]

    # Calculate drops
    drops = {model: [] for model in models}
    for exp in experiments:
        for model in models:
            for r in results:
                if r["experiment"] == exp and r["model"] == model:
                    drop = (baseline.get(model, 1.0) - r["test_accuracy"]) * 100
                    drops[model].append(drop)
                    break

    # Create chart
    x = np.arange(len(experiments))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#e74c3c", "#f39c12", "#8e44ad"]
    for i, (model, color) in enumerate(zip(models, colors)):
        offset = (i - 1) * width
        ax.bar(x + offset, drops[model], width, label=model.upper(), color=color)

    ax.set_ylabel("Accuracy Drop from Baseline (%)")
    ax.set_title("Generalization Gap by Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.show()


def load_results(filepath: str = "results/comprehensive_results.json") -> list:
    """Load experiment results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def generate_all_figures(
    results: list = None,
    results_path: str = "results/comprehensive_results.json",
    output_dir: str = "figures"
):
    """
    Generate all publication figures from experiment results.

    Args:
        results: List of result dictionaries (loads from file if None)
        results_path: Path to results JSON file
        output_dir: Directory to save figures
    """
    if results is None:
        results = load_results(results_path)

    os.makedirs(output_dir, exist_ok=True)

    print("Generating figures...")

    # Experiment comparison bar chart
    plot_experiment_comparison(
        results,
        save_path=os.path.join(output_dir, "experiment_comparison.png")
    )

    # Per-class recall heatmap
    plot_per_class_recall_heatmap(
        results,
        save_path=os.path.join(output_dir, "per_class_recall.png")
    )

    # Accuracy drop chart
    plot_accuracy_drop_chart(
        results,
        save_path=os.path.join(output_dir, "accuracy_drop.png")
    )

    # Confusion matrices for key experiments
    key_experiments = [
        "Random Split",
        "Fault-Size Single (014)",
        "Fault-Size All Loads (014)",
        "Cross-Load (Both)"
    ]

    for exp in key_experiments:
        for r in results:
            if r["experiment"] == exp and r["model"] == "cnn1d":
                conf_matrix = np.array(r["confusion_matrix"])

                # Determine class names present in this experiment
                labels = r.get("labels", [])
                if labels:
                    present_classes = sorted(set(labels))
                    class_names = [config["class_names"][i] for i in present_classes]
                else:
                    class_names = config["class_names"]

                safe_name = exp.replace(" ", "_").replace("(", "").replace(")", "")
                plot_confusion_matrix(
                    conf_matrix,
                    class_names,
                    title=f"Confusion Matrix: {exp} (CNN1D)",
                    save_path=os.path.join(output_dir, f"confusion_{safe_name}.png")
                )
                break

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    try:
        results = load_results()
        generate_all_figures(results)
    except FileNotFoundError:
        print("No results file found. Run experiment.py first.")
