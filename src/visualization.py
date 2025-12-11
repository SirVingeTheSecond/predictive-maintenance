import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config


# Style config
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']


def plot_training_curves(history: dict, save_path: str = None):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train', color=COLORS[0])
    axes[0].plot(epochs, history['val_loss'], label='Validation', color=COLORS[1])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], label='Train', color=COLORS[0])
    axes[1].plot(epochs, history['val_acc'], label='Validation', color=COLORS[1])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    normalize: bool = True,
    save_path: str = None,
    title: str = "Confusion Matrix",
):
    """Plot confusion matrix heatmap."""
    if normalize:
        cm_plot = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = '.2f'
    else:
        cm_plot = cm
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_model_comparison(
    stats: dict,
    config_name: str,
    metric: str = "accuracy",
    save_path: str = None,
):
    """Plot bar chart comparing models."""
    if config_name not in stats:
        print(f"Configuration not found: {config_name}")
        return

    models_data = stats[config_name]
    models = []
    means = []
    stds = []

    for model in ["cnn", "lstm", "cnnlstm"]:
        if model in models_data and metric in models_data[model]:
            models.append(model.upper())
            means.append(models_data[model][metric]["mean"])
            stds.append(models_data[model][metric]["std"])

    if not models:
        print(f"No data for metric: {metric}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=COLORS[:len(models)], edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{config_name}: Model Comparison")

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.01,
            f'{mean:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_per_class_comparison(
    stats: dict,
    config_name: str,
    save_path: str = None,
):
    """Plot grouped bar chart of per-class metrics."""
    if config_name not in stats:
        print(f"Configuration not found: {config_name}")
        return

    models_data = stats[config_name]

    # Determine classes and metric type
    sample = list(models_data.values())[0]
    if "accuracy" in sample:
        classes = ["Normal", "Ball", "IR", "OR"]
        metric_prefix = "recall_"
        ylabel = "Recall"
    else:
        classes = ["Ball", "IR", "OR"]
        metric_prefix = "auroc_"
        ylabel = "AUROC"

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(classes))
    width = 0.25
    offsets = [-width, 0, width]

    for i, model in enumerate(["cnn", "lstm", "cnnlstm"]):
        if model not in models_data:
            continue

        means = []
        stds = []
        for cls in classes:
            m = models_data[model].get(f"{metric_prefix}{cls}", {})
            means.append(m.get("mean", 0))
            stds.append(m.get("std", 0))

        ax.bar(
            x + offsets[i],
            means,
            width,
            yerr=stds,
            label=model.upper(),
            color=COLORS[i],
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{config_name}: Per-Class Performance")
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_study_overview(stats: dict, save_path: str = None):
    """Plot overview of all configurations for a study."""
    configs = list(stats.keys())
    n_configs = len(configs)

    fig, axes = plt.subplots(1, n_configs, figsize=(5 * n_configs, 5), squeeze=False)
    axes = axes.flatten()

    for idx, config_name in enumerate(configs):
        models_data = stats[config_name]
        ax = axes[idx]

        # Determine primary metric
        sample = list(models_data.values())[0]
        if "accuracy" in sample:
            metric = "accuracy"
            ylabel = "Accuracy"
        else:
            metric = "macro_auroc"
            ylabel = "Macro AUROC"

        models = []
        means = []
        stds = []

        for model in ["cnn", "lstm", "cnnlstm"]:
            if model in models_data and metric in models_data[model]:
                models.append(model.upper())
                means.append(models_data[model][metric]["mean"])
                stds.append(models_data[model][metric]["std"])

        if models:
            x = np.arange(len(models))
            ax.bar(x, means, yerr=stds, capsize=5, color=COLORS[:len(models)])
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.set_ylabel(ylabel)
            ax.set_title(config_name.replace("_", "\n"))
            ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_all_figures(study_name: str, stats: dict = None):
    """Generate all figures for a study."""
    from analysis import load_study_results, aggregate_results, compute_statistics

    if stats is None:
        results = load_study_results(study_name)
        aggregated = aggregate_results(results)
        stats = compute_statistics(aggregated)

    figures_dir = os.path.join(config.FIGURES_DIR, study_name)
    os.makedirs(figures_dir, exist_ok=True)

    plot_study_overview(
        stats,
        save_path=os.path.join(figures_dir, "overview.png"),
    )
    print(f"Saved: {figures_dir}/overview.png")

    # Per-config plots
    for config_name in stats.keys():
        # Model comparison
        sample = list(stats[config_name].values())[0]
        metric = "accuracy" if "accuracy" in sample else "macro_auroc"

        plot_model_comparison(
            stats,
            config_name,
            metric=metric,
            save_path=os.path.join(figures_dir, f"{config_name}_comparison.png"),
        )
        print(f"Saved: {figures_dir}/{config_name}_comparison.png")

        # Per-class
        plot_per_class_comparison(
            stats,
            config_name,
            save_path=os.path.join(figures_dir, f"{config_name}_per_class.png"),
        )
        print(f"Saved: {figures_dir}/{config_name}_per_class.png")
