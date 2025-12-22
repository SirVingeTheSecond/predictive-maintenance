import os
import json

import numpy as np
import matplotlib.pyplot as plt

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
    """Save figure in PNG and PDF."""
    png_path = os.path.join(figures_dir, f"{name}.png")
    pdf_path = os.path.join(figures_dir, f"{name}.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"  Saved: {name}.png")


# =============================================================================
# Let the Plots begin
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

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str = None,
                          normalize: bool = False, title: str = None):
    """Plot confusion matrix with nice formatting."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title or 'Confusion Matrix')

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# Paper-Specific Figures
# =============================================================================

def fig_model_comparison(stats: dict, figures_dir: str):
    """Generate main model comparison figure for paper."""
    # Pick the right key
    config_name = '4class_fault_size_all_loads'
    if config_name not in stats:
        config_name = '4class_fault_size'

    if config_name not in stats:
        print(f"  WARNING: fault_size config not found in stats, skipping")
        return

    config_stats = stats[config_name]

    models = []
    means = []
    stds = []
    colors = []

    for model in ['cnn', 'lstm', 'cnnlstm']:
        if model in config_stats and 'accuracy' in config_stats[model]:
            models.append({'cnn': 'CNN', 'lstm': 'LSTM', 'cnnlstm': 'CNN-LSTM'}[model])
            means.append(config_stats[model]['accuracy']['mean'] * 100)
            stds.append(config_stats[model]['accuracy']['std'] * 100)
            colors.append(COLORS[{'cnn': 'CNN', 'lstm': 'LSTM', 'cnnlstm': 'CNN-LSTM'}[model]])

    fig, ax = plt.subplots(figsize=(6, 5))

    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Model Architecture')
    ax.set_title('Model Comparison (Fault-Size Split)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 100)
    ax.axhline(y=25, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random chance')
    ax.legend()

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    _save_figure(figures_dir, "fig01_model_comparison")


def fig_per_class_performance(stats: dict, figures_dir: str):
    """Per-class recall comparison."""
    config_name = '4class_fault_size_all_loads'
    if config_name not in stats:
        config_name = '4class_fault_size'

    if config_name not in stats:
        print(f"  WARNING: fault_size config not found, skipping")
        return

    config_stats = stats[config_name]
    classes = ['Normal', 'Ball', 'IR', 'OR']
    model_names = ['CNN', 'LSTM', 'CNN-LSTM']
    model_keys = ['cnn', 'lstm', 'cnnlstm']

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(classes))
    width = 0.25

    for i, (model_key, model_name) in enumerate(zip(model_keys, model_names)):
        if model_key not in config_stats:
            continue

        recalls = []
        for cls in classes:
            key = f"recall_{cls}"  # analysis.py uses recall_{cls} format
            if key in config_stats[model_key]:
                recalls.append(config_stats[model_key][key]['mean'] * 100)
            else:
                recalls.append(0)

        ax.bar(x + i * width, recalls, width, label=model_name,
               color=COLORS[model_name], edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Recall (%)')
    ax.set_xlabel('Fault Class')
    ax.set_title('Per-Class Performance (Fault-Size Split)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 105)
    ax.legend()

    plt.tight_layout()
    _save_figure(figures_dir, "fig02_per_class_performance")


def fig_split_comparison(stats: dict, figures_dir: str):
    """Compare random vs fault-size split performance."""
    fig, ax = plt.subplots(figsize=(6, 5))

    splits = ['4class_random', '4class_fault_size_all_loads']
    alt_splits = ['4class_random', '4class_fault_size']
    labels = ['Random Split\n(Data Leakage)', 'Fault-Size Split\n(Generalization)']
    colors_split = ['#e74c3c', '#2ecc71']

    means = []
    stds = []

    for split, alt in zip(splits, alt_splits):
        cfg = split if split in stats else alt
        if cfg in stats and 'cnn' in stats[cfg]:
            means.append(stats[cfg]['cnn']['accuracy']['mean'] * 100)
            stds.append(stats[cfg]['cnn']['accuracy']['std'] * 100)
        else:
            means.append(0)
            stds.append(0)

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors_split, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Impact of Data Split Strategy (CNN)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Delta annotation
    if len(means) == 2 and means[0] > 0 and means[1] > 0:
        delta = means[0] - means[1]
        ax.annotate(f'Δ = {delta:.1f}%', xy=(0.5, (means[0] + means[1]) / 2),
                    fontsize=10, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    _save_figure(figures_dir, "fig03_split_comparison")


def fig_confusion_matrix(study_name: str, figures_dir: str):
    """Generate confusion matrix from best model."""
    from .experiment import list_experiments, load_experiment

    # Find a CNN fault-size experiment
    experiments = list_experiments(study_name)
    target_exp = None

    for exp_dir in experiments:
        if 'cnn_4class_fault_size' in exp_dir and 'seed42' in exp_dir:
            target_exp = exp_dir
            break

    if target_exp is None:
        # Try alternative
        for exp_dir in experiments:
            if 'cnn_4class_fault' in exp_dir:
                target_exp = exp_dir
                break

    if target_exp is None:
        print(f"  WARNING: No CNN fault-size experiment found, skipping")
        return

    exp_data = load_experiment(target_exp)
    results = exp_data.get('results', {})
    test_metrics = results.get('test_metrics', {})
    cm = test_metrics.get('confusion_matrix')

    if cm is None:
        print(f"  WARNING: No confusion matrix in results, skipping")
        return

    cm = np.array(cm)
    classes = ['Normal', 'Ball', 'IR', 'OR']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Raw counts
    im1 = axes[0].imshow(cm, cmap='Blues')
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_xticks(np.arange(len(classes)))
    axes[0].set_yticks(np.arange(len(classes)))
    axes[0].set_xticklabels(classes)
    axes[0].set_yticklabels(classes)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    for i in range(len(classes)):
        for j in range(len(classes)):
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                         color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.colorbar(im1, ax=axes[0])

    # Normalized
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    im2 = axes[1].imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_xticks(np.arange(len(classes)))
    axes[1].set_yticks(np.arange(len(classes)))
    axes[1].set_xticklabels(classes)
    axes[1].set_yticklabels(classes)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    for i in range(len(classes)):
        for j in range(len(classes)):
            axes[1].text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center',
                         color='white' if cm_norm[i, j] > 0.5 else 'black')

    plt.colorbar(im2, ax=axes[1])

    plt.suptitle('CNN on Fault-Size Split', fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save_figure(figures_dir, "fig04_confusion_matrix")


def fig_training_curves(study_name: str, figures_dir: str):
    """Training curves from a representative experiment."""
    from .experiment import list_experiments, load_experiment

    experiments = list_experiments(study_name)
    target_exp = None

    # Prefer random split (has validation data)
    for exp_dir in experiments:
        if 'cnn_4class_random' in exp_dir and 'seed42' in exp_dir:
            target_exp = exp_dir
            break

    if target_exp is None:
        for exp_dir in experiments:
            if 'cnn_4class' in exp_dir:
                target_exp = exp_dir
                break

    if target_exp is None:
        print(f"  WARNING: No suitable experiment found, skipping")
        return

    exp_data = load_experiment(target_exp)
    history = exp_data.get('history', {})

    if not history or 'train_loss' not in history:
        print(f"  WARNING: No training history found, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train', color=CLASS_COLORS[0], linewidth=2)
    if history.get('val_loss') and any(v > 0 for v in history['val_loss']):
        axes[0].plot(epochs, history['val_loss'], label='Validation', color=CLASS_COLORS[2], linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], label='Train', color=CLASS_COLORS[0], linewidth=2)
    if history.get('val_acc') and any(v > 0 for v in history['val_acc']):
        axes[1].plot(epochs, history['val_acc'], label='Validation', color=CLASS_COLORS[2], linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()

    plt.tight_layout()
    _save_figure(figures_dir, "fig05_training_curves")

# ToDo
def fig_roc_curves(study_name: str, figures_dir: str):
    """ROC curves for multilabel classification."""
    from .experiment import list_experiments, load_experiment

    experiments = list_experiments(study_name)

    # Collect all multilabel fault-size results
    roc_data = {'Ball': [], 'IR': [], 'OR': []}

    for exp_dir in experiments:
        if 'cnn_multilabel_fault_size' in exp_dir:
            exp_data = load_experiment(exp_dir)
            results = exp_data.get('results', {})
            test_metrics = results.get('test_metrics', {})

            for class_name in ['Ball', 'IR', 'OR']:
                per_class = test_metrics.get('per_class_metrics', {}).get(class_name, {})
                if 'fpr' in per_class and 'tpr' in per_class:
                    roc_data[class_name].append({
                        'fpr': per_class['fpr'],
                        'tpr': per_class['tpr'],
                        'auroc': per_class.get('auroc', 0)
                    })

    if not any(roc_data.values()):
        print(f"  WARNING: No ROC data found, skipping")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    colors = {'Ball': '#f39c12', 'IR': '#e74c3c', 'OR': '#8e44ad'}

    for class_name, data_list in roc_data.items():
        if not data_list:
            continue

        # Use first seed's data, average AUROC across seeds
        fpr = data_list[0]['fpr']
        tpr = data_list[0]['tpr']
        aurocs = [d['auroc'] for d in data_list]
        mean_auroc = np.mean(aurocs)
        std_auroc = np.std(aurocs)

        ax.plot(fpr, tpr, color=colors[class_name], linewidth=2,
                label=f'{class_name} (AUROC={mean_auroc:.3f}+-{std_auroc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (Multilabel Fault-Size Split)')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(figures_dir, "fig06_roc_curves")


def fig_tsne_visualization(figures_dir: str):
    """t-SNE visualization of FFT features."""
    from sklearn.manifold import TSNE
    from .data import load_data

    # Load data
    data = load_data(mode='4class', split='fault_size_all_loads', seed=42, verbose=False)

    X_test = data['X_test'].reshape(data['X_test'].shape[0], -1)
    y_test = data['y_test']

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X_test)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    classes = ['Normal', 'Ball', 'IR', 'OR']

    for i, (cls, color) in enumerate(zip(classes, CLASS_COLORS)):
        mask = y_test == i
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                   c=color, label=cls, alpha=0.6, s=30)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    # Depends on config.FEATURE_MODE - not the most trivial solution
    feature_name = config.FEATURE_MODE.upper()
    ax.set_title(f't-SNE Visualization of {feature_name} Features (Test Set)')

    ax.legend()

    plt.tight_layout()
    _save_figure(figures_dir, "fig07_tsne_features")


def fig_signal_examples(figures_dir: str):
    """Example signals from each fault class."""
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
    """Activation function comparison - LOADS FROM FILE, NOT HARDCODED."""
    # Try to load activation results from file
    activation_results_path = os.path.join(config.RESULTS_DIR, 'activation_test_results.json')

    if os.path.exists(activation_results_path):
        with open(activation_results_path) as f:
            results = json.load(f)
        print(f"  Loaded activation results from {activation_results_path}")
    else:
        print(f"  SKIPPING fig09_activation_comparison")
        return

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
    models = ['CNN', 'LSTM', 'CNN-LSTM']
    model_keys = ['cnn', 'lstm', 'cnnlstm']
    splits = ['Random', 'Fault-Size', 'Cross-Load']
    split_keys = ['4class_random', '4class_fault_size_all_loads', '4class_cross_load']
    alt_keys = ['4class_random', '4class_fault_size', '4class_cross_load']

    acc_matrix = np.zeros((len(models), len(splits)))

    for i, model in enumerate(model_keys):
        for j, (split, alt) in enumerate(zip(split_keys, alt_keys)):
            cfg = split if split in stats else alt
            if cfg in stats and model in stats[cfg] and 'accuracy' in stats[cfg][model]:
                acc_matrix[i, j] = stats[cfg][model]['accuracy']['mean'] * 100

    fig, ax = plt.subplots(figsize=(7, 4))

    im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)

    ax.set_xticks(np.arange(len(splits)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(splits)
    ax.set_yticklabels(models)
    ax.set_xlabel('Split Strategy')
    ax.set_ylabel('Model')
    ax.set_title('Accuracy (%) Across All Configurations')

    for i in range(len(models)):
        for j in range(len(splits)):
            val = acc_matrix[i, j]
            color = 'white' if val < 70 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=11)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    _save_figure(figures_dir, "fig10_summary_heatmap")


# =============================================================================
# Main Generation
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
