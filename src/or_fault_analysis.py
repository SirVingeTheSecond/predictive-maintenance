"""
OR Fault Severity Non-Monotonicity Analysis (Publication Quality).

Investigates why OR faults fail catastrophically on fault-size split
while Ball and IR faults generalize successfully.

Usage:
    python or_fault_analysis_v2.py --data_dir path/to/cwru/data --output_dir results/analysis
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color scheme
COLORS = {
    'Ball': '#2ecc71',      # Green
    'IR': '#3498db',        # Blue
    'OR': '#e74c3c',        # Red
    '007': '#3498db',       # Blue
    '014': '#e74c3c',       # Red
    '021': '#27ae60',       # Green
    'monotonic': '#27ae60', # Green
    'non_monotonic': '#e74c3c',  # Red
}

# =============================================================================
# Configuration
# =============================================================================

WINDOW_SIZE = 2048
STRIDE = 512
FAULT_TYPES = ["Ball", "IR", "OR"]
SEVERITIES = ["007", "014", "021"]
LOADS = ["1772", "1750", "1730"]

FILE_PATTERNS = {
    "Normal": "{load}_Normal.npz",
    "Ball_007": "{load}_B_7_DE12.npz",
    "Ball_014": "{load}_B_14_DE12.npz",
    "Ball_021": "{load}_B_21_DE12.npz",
    "IR_007": "{load}_IR_7_DE12.npz",
    "IR_014": "{load}_IR_14_DE12.npz",
    "IR_021": "{load}_IR_21_DE12.npz",
    "OR_007": "{load}_OR@6_7_DE12.npz",
    "OR_014": "{load}_OR@6_14_DE12.npz",
    "OR_021": "{load}_OR@6_21_DE12.npz",
}


# =============================================================================
# Feature Extraction (matching data.py)
# =============================================================================

def extract_windows(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Extract sliding windows from signal."""
    num_windows = (len(signal) - window_size) // stride + 1
    windows = np.zeros((num_windows, window_size), dtype=np.float32)
    for i in range(num_windows):
        start = i * stride
        windows[i] = signal[start:start + window_size]
    return windows


def compute_fft_features(windows: np.ndarray) -> np.ndarray:
    """Compute FFT magnitude spectrum with log scaling."""
    hanning = np.hanning(windows.shape[1])
    windowed = windows * hanning
    fft_result = np.fft.rfft(windowed, axis=1)
    magnitude = np.abs(fft_result)
    log_magnitude = np.log1p(magnitude)
    mean = log_magnitude.mean(axis=1, keepdims=True)
    std = log_magnitude.std(axis=1, keepdims=True) + 1e-8
    normalized = (log_magnitude - mean) / std
    return normalized.astype(np.float32)


# =============================================================================
# Data Loading
# =============================================================================

def load_features_for_class(class_name: str, data_dir: str) -> np.ndarray:
    """Load and compute FFT features for a specific class across all loads."""
    all_features = []
    for load in LOADS:
        filename = FILE_PATTERNS[class_name].format(load=load)
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  Warning: Missing {filepath}")
            continue
        data = np.load(filepath)
        signal = data["DE"].flatten()
        windows = extract_windows(signal, WINDOW_SIZE, STRIDE)
        features = compute_fft_features(windows)
        all_features.append(features)
    if all_features:
        return np.vstack(all_features)
    return np.array([])


def load_all_features(data_dir: str) -> dict:
    """Load FFT features for all fault types and severities."""
    features = {}
    for fault_type in FAULT_TYPES:
        features[fault_type] = {}
        for severity in SEVERITIES:
            class_name = f"{fault_type}_{severity}"
            print(f"Loading {class_name}...")
            features[fault_type][severity] = load_features_for_class(class_name, data_dir)
            print(f"  Samples: {len(features[fault_type][severity])}")
    print("Loading Normal...")
    features["Normal"] = load_features_for_class("Normal", data_dir)
    print(f"  Samples: {len(features['Normal'])}")
    return features


# =============================================================================
# Distance Metrics
# =============================================================================

def compute_centroid_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """Compute Euclidean distance between feature centroids."""
    centroid1 = np.mean(feat1, axis=0)
    centroid2 = np.mean(feat2, axis=0)
    return np.linalg.norm(centroid1 - centroid2)


def compute_monotonicity_ratio(features: dict, fault_type: str) -> dict:
    """Compute monotonicity ratio for a fault type."""
    d_007_014 = compute_centroid_distance(
        features[fault_type]["007"], features[fault_type]["014"])
    d_014_021 = compute_centroid_distance(
        features[fault_type]["014"], features[fault_type]["021"])
    d_007_021 = compute_centroid_distance(
        features[fault_type]["007"], features[fault_type]["021"])
    path_through_014 = d_007_014 + d_014_021
    direct_path = d_007_021
    ratio = path_through_014 / direct_path if direct_path > 0 else float('inf')
    return {
        "d_007_014": d_007_014,
        "d_014_021": d_014_021,
        "d_007_021": d_007_021,
        "path_through_014": path_through_014,
        "direct_path": direct_path,
        "ratio": ratio,
    }


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_all(features: dict) -> dict:
    """Run all analyses and return results."""
    results = {
        'monotonicity': {},
        'cross_class': {},
    }

    # Monotonicity analysis
    for fault_type in FAULT_TYPES:
        results['monotonicity'][fault_type] = compute_monotonicity_ratio(features, fault_type)

    # Cross-class distances from OR_014
    or_014 = features["OR"]["014"]
    distances = {}
    for fault_type in FAULT_TYPES:
        for severity in SEVERITIES:
            key = f"{fault_type}_{severity}"
            distances[key] = compute_centroid_distance(or_014, features[fault_type][severity])
    distances["Normal"] = compute_centroid_distance(or_014, features["Normal"])
    results['cross_class'] = distances

    return results


def print_results(results: dict) -> None:
    """Print analysis results to console."""
    print("\n" + "=" * 70)
    print("MONOTONICITY ANALYSIS")
    print("=" * 70)
    print("\nRatio = (d_007_014 + d_014_021) / d_007_021")
    print("Ratio ≈ 1.0: monotonic | Ratio >> 1.0: non-monotonic")
    print("-" * 70)

    for fault_type in FAULT_TYPES:
        r = results['monotonicity'][fault_type]
        print(f"\n{fault_type}:")
        print(f"  d(007, 014) = {r['d_007_014']:.2f}")
        print(f"  d(014, 021) = {r['d_014_021']:.2f}")
        print(f"  d(007, 021) = {r['d_007_021']:.2f}")
        print(f"  Ratio: {r['ratio']:.3f}")

    print("\n" + "=" * 70)
    print("CROSS-CLASS DISTANCES FROM OR_014")
    print("=" * 70)
    sorted_dist = sorted(results['cross_class'].items(), key=lambda x: x[1])
    for name, dist in sorted_dist:
        marker = ""
        if name in ["OR_007", "OR_021"]:
            marker = " ← TRAIN"
        elif name == "IR_014":
            marker = " ← CLOSEST OTHER TYPE"
        print(f"  {name:<12}: {dist:.2f}{marker}")


# =============================================================================
# Publication-Quality Visualizations
# =============================================================================

def plot_main_figure(features: dict, results: dict, output_dir: str) -> None:
    """Generate main analysis figure with 4 panels."""
    fig = plt.figure(figsize=(14, 10))

    # Create grid: 2 rows, with top row having 3 columns, bottom row having 2 columns
    gs = fig.add_gridspec(2, 6, hspace=0.3, wspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0:2])  # Ball spectrum
    ax2 = fig.add_subplot(gs[0, 2:4])  # IR spectrum
    ax3 = fig.add_subplot(gs[0, 4:6])  # OR spectrum
    ax4 = fig.add_subplot(gs[1, 0:3])  # Monotonicity ratios
    ax5 = fig.add_subplot(gs[1, 3:6])  # Cross-class distances

    # Panel A-C: Mean FFT Spectra
    axes_spectra = [ax1, ax2, ax3]
    for idx, fault_type in enumerate(FAULT_TYPES):
        ax = axes_spectra[idx]
        for severity in SEVERITIES:
            mean_spectrum = np.mean(features[fault_type][severity], axis=0)
            color = COLORS[severity]
            linestyle = '-' if severity != '014' else '--'
            linewidth = 1.5 if severity != '014' else 2.0
            ax.plot(mean_spectrum, label=severity, color=color,
                   linestyle=linestyle, linewidth=linewidth, alpha=0.85)

        ax.set_title(f"{fault_type} Fault", fontweight='bold')
        ax.set_xlabel("Frequency Bin")
        ax.set_ylabel("Log Magnitude (norm.)")
        ax.legend(title="Severity", loc='upper right', framealpha=0.9)
        ax.set_xlim(0, len(mean_spectrum))
        ax.grid(True, alpha=0.3, linestyle=':')

        # Add panel label
        ax.text(-0.12, 1.05, chr(65 + idx), transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top')

    # Panel D: Monotonicity Ratios
    ax = ax4
    ratios = [results['monotonicity'][ft]['ratio'] for ft in FAULT_TYPES]
    colors_bar = [COLORS[ft] for ft in FAULT_TYPES]

    bars = ax.bar(FAULT_TYPES, ratios, color=colors_bar, edgecolor='black',
                  linewidth=1.2, alpha=0.85, width=0.6)

    # Add threshold line
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5,
              label='Perfect monotonicity', zorder=1)

    # Add value labels on bars
    for bar, ratio, ft in zip(bars, ratios, FAULT_TYPES):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
               f'{ratio:.2f}', ha='center', va='bottom',
               fontsize=11, fontweight='bold')

    ax.set_ylabel("Monotonicity Ratio")
    ax.set_title("Severity Non-Monotonicity by Fault Type", fontweight='bold')
    ax.set_ylim(0, 2.5)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    ax.text(-0.12, 1.05, 'D', transform=ax.transAxes,
           fontsize=14, fontweight='bold', va='top')

    # Panel E: Cross-class distances from OR_014
    ax = ax5

    # Get distances and sort
    distances = results['cross_class']
    classes_to_show = ['OR_007', 'IR_014', 'OR_021', 'Ball_014', 'IR_007']
    dists = [distances[c] for c in classes_to_show]

    # Create horizontal bar chart
    y_pos = np.arange(len(classes_to_show))
    colors_dist = []
    for c in classes_to_show:
        if c.startswith('OR'):
            colors_dist.append(COLORS['OR'])
        elif c.startswith('IR'):
            colors_dist.append(COLORS['IR'])
        else:
            colors_dist.append(COLORS['Ball'])

    bars = ax.barh(y_pos, dists, color=colors_dist, edgecolor='black',
                   linewidth=1.2, alpha=0.85, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes_to_show)
    ax.set_xlabel("Centroid Distance from OR_014")
    ax.set_title("Cross-Class Distances (OR_014 Reference)", fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', axis='x')

    # Add value labels
    for i, (bar, dist) in enumerate(zip(bars, dists)):
        width = bar.get_width()
        label = f'{dist:.1f}'
        if classes_to_show[i] == 'OR_007':
            label += ' (train)'
        elif classes_to_show[i] == 'IR_014':
            label += ' (closest other)'
        elif classes_to_show[i] == 'OR_021':
            label += ' (train)'
        ax.text(width + 0.3, bar.get_y() + bar.get_height()/2,
               label, ha='left', va='center', fontsize=9)

    ax.set_xlim(0, max(dists) + 8)
    ax.text(-0.12, 1.05, 'E', transform=ax.transAxes,
           fontsize=14, fontweight='bold', va='top')

    # Add figure title
    fig.suptitle("OR Fault Severity Generalization: Root Cause Analysis",
                fontsize=14, fontweight='bold', y=0.98)

    output_path = os.path.join(output_dir, "or_fault_analysis.png")
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def plot_severity_progression(features: dict, output_dir: str) -> None:
    """Plot PCA severity progression for each fault type."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, fault_type in enumerate(FAULT_TYPES):
        ax = axes[idx]

        # Compute centroids
        centroids = {}
        for severity in SEVERITIES:
            centroids[severity] = np.mean(features[fault_type][severity], axis=0)

        # PCA via SVD
        all_features = np.vstack([features[fault_type][s] for s in SEVERITIES])
        mean_feat = np.mean(all_features, axis=0)
        centered = all_features - mean_feat
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pc1, pc2 = Vt[0], Vt[1]

        # Project centroids
        coords = {}
        for severity in SEVERITIES:
            proj = centroids[severity] - mean_feat
            coords[severity] = (np.dot(proj, pc1), np.dot(proj, pc2))

        # Draw direct path (007 -> 021) - dashed gray
        ax.plot([coords['007'][0], coords['021'][0]],
               [coords['007'][1], coords['021'][1]],
               color='gray', linestyle='--', linewidth=2, alpha=0.7,
               label='Direct path (007→021)', zorder=1)

        # Draw path through 014 - solid with color
        ax.plot([coords['007'][0], coords['014'][0]],
               [coords['007'][1], coords['014'][1]],
               color=COLORS['014'], linestyle='-', linewidth=2, alpha=0.7, zorder=2)
        ax.plot([coords['014'][0], coords['021'][0]],
               [coords['014'][1], coords['021'][1]],
               color=COLORS['014'], linestyle='-', linewidth=2, alpha=0.7,
               label='Path through 014', zorder=2)

        # Plot centroids
        markers = {'007': 'o', '014': 'X', '021': 's'}
        sizes = {'007': 120, '014': 180, '021': 120}

        for severity in SEVERITIES:
            x, y = coords[severity]
            ax.scatter(x, y, c=COLORS[severity], s=sizes[severity],
                      marker=markers[severity], edgecolors='black',
                      linewidths=1.5, label=severity, zorder=3)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{fault_type} Fault", fontweight='bold')
        ax.legend(loc='best', framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':')

        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='datalim')

        # Panel label
        ax.text(-0.12, 1.05, chr(65 + idx), transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top')

    fig.suptitle("Severity Progression in Feature Space (PCA)",
                fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "severity_progression.png")
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def plot_or_detail(features: dict, output_dir: str) -> None:
    """Plot detailed OR fault analysis showing unique bin 123 peak."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: Low frequency detail (bins 0-200)
    ax = axes[0]
    for severity in SEVERITIES:
        mean_spectrum = np.mean(features["OR"][severity], axis=0)[:200]
        color = COLORS[severity]
        linestyle = '-' if severity != '014' else '--'
        linewidth = 1.5 if severity != '014' else 2.5
        ax.plot(mean_spectrum, label=f"OR_{severity}", color=color,
               linestyle=linestyle, linewidth=linewidth, alpha=0.85)

    # Highlight bin 123
    ax.axvline(x=123, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.annotate('Bin 123\n(OR_014 only)', xy=(123, 2.8), xytext=(145, 3.2),
               fontsize=9, ha='left',
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Log Magnitude (normalized)")
    ax.set_title("OR Fault: Low Frequency Detail", fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xlim(0, 200)
    ax.text(-0.1, 1.05, 'A', transform=ax.transAxes,
           fontsize=14, fontweight='bold', va='top')

    # Panel B: Full spectrum comparison
    ax = axes[1]
    for severity in SEVERITIES:
        mean_spectrum = np.mean(features["OR"][severity], axis=0)
        color = COLORS[severity]
        linestyle = '-' if severity != '014' else '--'
        linewidth = 1.2 if severity != '014' else 2.0
        ax.plot(mean_spectrum, label=f"OR_{severity}", color=color,
               linestyle=linestyle, linewidth=linewidth, alpha=0.85)

    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Log Magnitude (normalized)")
    ax.set_title("OR Fault: Full Spectrum", fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.text(-0.1, 1.05, 'B', transform=ax.transAxes,
           fontsize=14, fontweight='bold', va='top')

    fig.suptitle("OR_014 Unique Spectral Characteristics",
                fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "or_spectral_detail.png")
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="OR Fault Severity Analysis")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to CWRU data directory")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory for plots")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("OR FAULT SEVERITY NON-MONOTONICITY ANALYSIS")
    print("=" * 70)
    print(f"\nData directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}\n")

    # Load data
    features = load_all_features(args.data_dir)

    # Run analysis
    results = analyze_all(features)
    print_results(results)

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    plot_main_figure(features, results, args.output_dir)
    plot_severity_progression(features, args.output_dir)
    plot_or_detail(features, args.output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
