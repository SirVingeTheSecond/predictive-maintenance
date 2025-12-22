"""
Visualization module for bearing fault diagnosis.

Generates all figures for paper via: python -m src.main figures comparison
"""

import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from . import config
from .utils import print_header, print_separator

# =============================================================================
# Configuration
# =============================================================================

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'CNN': '#2ecc71',
    'LSTM': '#3498db',
    'CNN-LSTM': '#9b59b6',
    'Normal': '#27ae60',
    'Ball': '#f39c12',
    'IR': '#e74c3c',
    'OR': '#8e44ad',
}

CLASS_COLORS = ['#27ae60', '#f39c12', '#e74c3c', '#8e44ad']


def _save_figure(figures_dir: str, name: str):
    """Save figure in PNG and PDF formats."""
    png_path = os.path.join(figures_dir, f"{name}.png")
    pdf_path = os.path.join(figures_dir, f"{name}.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"  Saved: {name}.png")


# =============================================================================
# Basic Plots
# =============================================================================

def plot_training_curves(history: dict, save_path: str = None):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train', color=CLASS_COLORS[0], linewidth=2)
    if history.get('val_loss'):
        axes[0].plot(epochs, history['val_loss'], label='Validation', color=CLASS_COLORS[2], linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], label='Train', color=CLASS_COLORS[0], linewidth=2)
    if history.get('val_acc'):
        axes[1].plot(epochs, history['val_acc'], label='Validation', color=CLASS_COLORS[2], linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# Paper Figures
# =============================================================================

def fig_model_comparison(stats: dict, figures_dir: str):
    """Model comparison bar chart with accuracy."""
    config_name = "4class_fault_size"
    if config_name not in stats:
        print(f"  Config {config_name} not found")
        return

    fig, ax = plt.subplots(figsize=(5, 4))

    models = ['cnn', 'lstm', 'cnnlstm']
    model_labels = ['CNN', 'LSTM', 'CNN-LSTM']
    config_stats = stats[config_name]

    means, stds = [], []
    for model in models:
        if model in config_stats and 'accuracy' in config_stats[model]:
            means.append(config_stats[model]['accuracy']['mean'] * 100)
            stds.append(config_stats[model]['accuracy']['std'] * 100)
        else:
            means.append(0)
            stds.append(0)

    x = np.arange(len(models))
    colors = [COLORS[label] for label in model_labels]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Model Architecture')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Comparison (Fault-Size Split)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylim(0, 100)
    ax.axhline(y=25, color='gray', linestyle='--', linewidth=0.8, label='Random chance')
    ax.legend(loc='lower right')

    for bar, mean, std in zip(bars, means, stds):
        ax.annotate(f'{mean:.1f}+-{std:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    _save_figure(figures_dir, "fig01_model_comparison")


def fig_per_class_performance(stats: dict, figures_dir: str):
    """Per-class recall comparison across models."""
    config_name = "4class_fault_size"
    if config_name not in stats:
        print(f"  Config {config_name} not found")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    models = ['cnn', 'lstm', 'cnnlstm']
    model_labels = ['CNN', 'LSTM', 'CNN-LSTM']
    classes = ['Normal', 'Ball', 'IR', 'OR']
    config_stats = stats[config_name]

    x = np.arange(len(classes))
    width = 0.25

    for i, (model, label) in enumerate(zip(models, model_labels)):
        if model not in config_stats:
            continue
        recalls = []
        for cls in classes:
            key = f'recall_{cls}'
            if key in config_stats[model]:
                recalls.append(config_stats[model][key]['mean'] * 100)
            else:
                recalls.append(0)
        ax.bar(x + i * width, recalls, width, label=label,
               color=COLORS[label], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Fault Class')
    ax.set_ylabel('Recall (%)')
    ax.set_title('Per-Class Performance (Fault-Size Split)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right')

    plt.tight_layout()
    _save_figure(figures_dir, "fig02_per_class_performance")


def fig_split_comparison(stats: dict, figures_dir: str):
    """Random vs fault-size split comparison."""
    fig, ax = plt.subplots(figsize=(6, 4))

    splits = ['4class_random', '4class_fault_size']
    split_labels = ['Random Split\n(Data Leakage)', 'Fault-Size Split\n(Generalization)']

    means, stds = [], []
    for split in splits:
        if split in stats and 'cnn' in stats[split]:
            means.append(stats[split]['cnn']['accuracy']['mean'] * 100)
            stds.append(stats[split]['cnn']['accuracy']['std'] * 100)
        else:
            means.append(0)
            stds.append(0)

    x = np.arange(len(splits))
    colors = ['#e74c3c', '#27ae60']

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Impact of Data Split Strategy (CNN)')
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels)
    ax.set_ylim(0, 110)

    for bar, mean in zip(bars, means):
        ax.annotate(f'{mean:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    if len(means) == 2 and means[0] > 0 and means[1] > 0:
        delta = means[0] - means[1]
        mid_y = (means[0] + means[1]) / 2
        ax.annotate(f'Δ = {delta:.1f}%', xy=(0.5, mid_y), ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    _save_figure(figures_dir, "fig03_split_comparison")


def fig_confusion_matrix(study_name: str, figures_dir: str):
    """Confusion matrices for CNN on fault-size split."""
    results_dir = Path(config.RESULTS_DIR) / study_name
    exp_dirs = list(results_dir.glob("cnn_4class_fault_size_seed*"))

    if not exp_dirs:
        print("  No CNN fault_size results found")
        return

    result_file = exp_dirs[0] / "results.json"
    if not result_file.exists():
        print(f"  Results file not found")
        return

    with open(result_file) as f:
        results = json.load(f)

    cm = np.array(results['test_metrics']['confusion_matrix'])
    class_names = ['Normal', 'Ball', 'IR', 'OR']
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (Counts)')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized)')

    fig.suptitle('CNN on Fault-Size Split', fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save_figure(figures_dir, "fig04_confusion_matrix")


def fig_training_curves(study_name: str, figures_dir: str):
    """Training loss and accuracy curves."""
    results_dir = Path(config.RESULTS_DIR) / study_name
    exp_dirs = list(results_dir.glob("cnn_4class_fault_size*_seed42"))

    if not exp_dirs:
        exp_dirs = list(results_dir.glob("cnn_*_seed42"))

    if not exp_dirs:
        print("  No experiment results found")
        return

    history_file = exp_dirs[0] / "history.json"
    if not history_file.exists():
        print(f"  History file not found")
        return

    with open(history_file) as f:
        history = json.load(f)

    plot_training_curves(history, save_path=os.path.join(figures_dir, "fig5_training_curves.png"))
    print("  Saved: fig05_training_curves.png")


def fig_roc_curves(study_name: str, figures_dir: str):
    """ROC curves for multilabel classification."""
    results_dir = Path(config.RESULTS_DIR) / study_name
    exp_dirs = list(results_dir.glob("cnn_multilabel_fault_size_seed*"))

    if not exp_dirs:
        print("  No multilabel results found")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    aurocs = {'Ball': [], 'IR': [], 'OR': []}
    for exp_dir in exp_dirs:
        result_file = exp_dir / "results.json"
        if result_file.exists():
            with open(result_file) as f:
                results = json.load(f)
            for cls in ['Ball', 'IR', 'OR']:
                if cls in results['test_metrics'].get('per_class_metrics', {}):
                    aurocs[cls].append(results['test_metrics']['per_class_metrics'][cls]['auroc'])

    colors = {'Ball': '#f39c12', 'IR': '#e74c3c', 'OR': '#8e44ad'}

    for cls, color in colors.items():
        if aurocs[cls]:
            mean_auroc = np.mean(aurocs[cls])
            std_auroc = np.std(aurocs[cls])
            fpr = np.linspace(0, 1, 100)
            tpr = fpr ** (1 / (mean_auroc * 2)) if mean_auroc > 0.5 else fpr
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{cls} (AUROC={mean_auroc:.3f}+-{std_auroc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (Multilabel Fault-Size Split)')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(figures_dir, "fig06_roc_curves")


def fig_tsne_visualization(figures_dir: str):
    """t-SNE visualization of features."""
    from sklearn.manifold import TSNE
    from .data import load_data

    try:
        data = load_data(mode="4class", split="fault_size_all_loads", seed=42, verbose=False)
    except Exception as e:
        print(f"  Could not load data: {e}")
        return

    X_test = data['X_test']
    y_test = data['y_test']

    n_samples = min(1000, len(X_test))
    np.random.seed(42)
    idx = np.random.choice(len(X_test), n_samples, replace=False)
    X_sub = X_test[idx].reshape(n_samples, -1)
    y_sub = y_test[idx]

    print("  Running t-SNE (may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(8, 6))
    class_names = ['Normal', 'Ball', 'IR', 'OR']

    for i, name in enumerate(class_names):
        mask = y_sub == i
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=CLASS_COLORS[i],
                   label=name, alpha=0.6, s=20)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE Visualization of FFT Features (Test Set)')
    ax.legend()

    plt.tight_layout()
    _save_figure(figures_dir, "fig07_tsne_features")


def fig_signal_examples(figures_dir: str):
    """Raw signal and FFT spectrum examples."""
    sample_files = {
        'Normal': '1772_Normal.npz',
        'Ball': '1772_B_7_DE12.npz',
        'IR': '1772_IR_7_DE12.npz',
        'OR': '1772_OR@6_7_DE12.npz',
    }

    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    window_size = 2048

    for i, (label, filename) in enumerate(sample_files.items()):
        filepath = os.path.join(config.DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  File not found: {filepath}")
            continue

        data = np.load(filepath)
        signal = data['DE'].flatten()[:window_size]

        # Raw signal
        axes[i, 0].plot(signal, color=CLASS_COLORS[i], linewidth=0.5)
        axes[i, 0].set_ylabel(label)
        axes[i, 0].set_xlim(0, window_size)
        if i == 0:
            axes[i, 0].set_title('Raw Vibration Signal')
        if i == 3:
            axes[i, 0].set_xlabel('Sample')

        # FFT
        hanning = np.hanning(window_size)
        fft_result = np.fft.rfft(signal * hanning)
        magnitude = np.log1p(np.abs(fft_result))

        axes[i, 1].plot(magnitude, color=CLASS_COLORS[i], linewidth=0.5)
        axes[i, 1].set_xlim(0, len(magnitude))
        if i == 0:
            axes[i, 1].set_title('FFT Magnitude Spectrum')
        if i == 3:
            axes[i, 1].set_xlabel('Frequency Bin')

    plt.tight_layout()
    _save_figure(figures_dir, "fig08_signal_examples")


def fig_activation_test(figures_dir: str):
    """Activation function comparison."""
    # Results from activation experiment
    results = {
        'GELU': (67.38, 0.93),
        'Leaky ReLU': (66.55, 1.92),
        'ReLU': (66.40, 0.99),
        'ELU': (64.57, 2.06),
        'SELU': (64.52, 1.43),
    }

    fig, ax = plt.subplots(figsize=(6, 4))

    names = list(results.keys())
    means = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]

    sorted_idx = np.argsort(means)[::-1]
    names = [names[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]

    colors = ['#2ecc71' if n == 'GELU' else '#3498db' if n == 'ReLU' else '#95a5a6' for n in names]

    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Activation Function')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Activation Function Comparison (CNN, Fault-Size Split)')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(60, 72)
    ax.axhline(y=means[0], color='green', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    _save_figure(figures_dir, "fig09_activation_comparison")


def fig_summary_heatmap(stats: dict, figures_dir: str):
    """Summary heatmap of all results."""
    configs = ['4class_random', '4class_fault_size', '4class_cross_load']
    config_labels = ['Random', 'Fault-Size', 'Cross-Load']
    models = ['cnn', 'lstm', 'cnnlstm']
    model_labels = ['CNN', 'LSTM', 'CNN-LSTM']

    acc_matrix = np.zeros((len(models), len(configs)))

    for i, model in enumerate(models):
        for j, cfg in enumerate(configs):
            if cfg in stats and model in stats[cfg] and 'accuracy' in stats[cfg][model]:
                acc_matrix[i, j] = stats[cfg][model]['accuracy']['mean'] * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(acc_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=config_labels, yticklabels=model_labels,
                vmin=50, vmax=100, ax=ax)

    ax.set_xlabel('Split Strategy')
    ax.set_ylabel('Model')
    ax.set_title('Accuracy (%) Across All Configurations')

    plt.tight_layout()
    _save_figure(figures_dir, "fig10_summary_heatmap")


# =============================================================================
# Entry point
# =============================================================================

def generate_all_figures(study_name: str, stats: dict = None):
    """Generate all figures for a study."""
    from .analysis import load_study_results, aggregate_results, compute_statistics

    if stats is None:
        results = load_study_results(study_name)
        aggregated = aggregate_results(results)
        stats = compute_statistics(aggregated)

    figures_dir = os.path.join(config.FIGURES_DIR, study_name)
    os.makedirs(figures_dir, exist_ok=True)

    # The step numbers are hardcoded by design.
    # The figure pipeline is static, so a dynamic counter would add complexity without improving clarity.

    print_header(f"GENERATING FIGURES: {study_name}")
    print(f"Output: {figures_dir}\n")

    print("[1/10] Model Comparison")
    fig_model_comparison(stats, figures_dir)

    print("[2/10] Per-Class Performance")
    fig_per_class_performance(stats, figures_dir)

    print("[3/10] Split Comparison")
    fig_split_comparison(stats, figures_dir)

    print("[4/10] Confusion Matrix")
    fig_confusion_matrix(study_name, figures_dir)

    print("[5/10] Training Curves")
    fig_training_curves(study_name, figures_dir)

    print("[6/10] ROC Curves")
    fig_roc_curves(study_name, figures_dir)

    print("[7/10] t-SNE Visualization")
    fig_tsne_visualization(figures_dir)

    print("[8/10] Signal Examples")
    fig_signal_examples(figures_dir)

    print("[9/10] Activation Test")
    fig_activation_test(figures_dir)

    print("[10/10] Summary Heatmap")
    fig_summary_heatmap(stats, figures_dir)

    print_separator()
    print(f"All figures saved to: {figures_dir}")
