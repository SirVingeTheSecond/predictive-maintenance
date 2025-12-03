"""
CWRU Bearing Dataset Loading and Preprocessing.

Implements multiple data splitting strategies to evaluate model
generalization under different conditions:
- Random: baseline with potential data leakage
- Fault-size: tests generalization to unseen severity levels
- Fault-size-all-loads: fault-size split with training diversity from all loads
- Cross-load: tests generalization to unseen operating conditions
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import config, get_file_mapping


class CWRUDataset(Dataset):
    """PyTorch Dataset wrapper for CWRU bearing signals."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = torch.from_numpy(signals).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


def load_raw_signals_for_load(load: str, data_dir: str) -> dict:
    """
    Load raw vibration signals for a specific motor load.

    Args:
        load: Motor load identifier (1772, 1750, or 1730)
        data_dir: Directory containing the .npz data files

    Returns:
        Dictionary mapping class names to signal data and metadata
    """
    signals = {}
    mode = config["classification_mode"]
    file_mapping = get_file_mapping(load)

    for filename, class_name, fault_size, class_idx_4 in file_mapping:
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing data file: {filepath}")

        data = np.load(filepath)
        signal = data["DE"].flatten()

        if mode == "4class":
            label_idx = class_idx_4
        else:
            label_idx = config["class_names_10"].index(class_name)

        signals[f"{load}_{class_name}"] = {
            "signal": signal,
            "fault_size": fault_size if fault_size else "none",
            "label_idx": label_idx,
            "load": load,
            "class_name": class_name,
        }

    return signals


def extract_windows(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Extract normalized sliding windows from a signal.

    Per-window z-score normalization removes amplitude variations
    while preserving the relative pattern structure within each window.

    Args:
        signal: Raw vibration signal
        window_size: Number of samples per window
        stride: Step size between consecutive windows

    Returns:
        Array of shape (num_windows, window_size) with normalized windows
    """
    num_windows = (len(signal) - window_size) // stride + 1
    windows = np.zeros((num_windows, window_size), dtype=np.float32)

    for i in range(num_windows):
        start = i * stride
        window = signal[start:start + window_size]
        mean = window.mean()
        std = window.std() + 1e-8
        windows[i] = (window - mean) / std

    return windows


def split_signal_into_chunks(
    signal: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> dict:
    """
    Split a signal into train, validation, and test segments.

    Uses chunk-based splitting to prevent segment-level data leakage.
    The signal is divided into 10 chunks which are shuffled and then
    assigned to splits, ensuring no temporal proximity between sets.

    Args:
        signal: Raw vibration signal to split
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        seed: Random seed for reproducible shuffling

    Returns:
        Dictionary with train, val, and test signal segments
    """
    rng = np.random.RandomState(seed)

    n_chunks = 10
    chunk_size = len(signal) // n_chunks
    chunks = [signal[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]
    rng.shuffle(chunks)

    # Allocate chunks based on ratios (6:2:2 for 60:20:20 split)
    n_test = int(n_chunks * test_ratio)
    n_val = int(n_chunks * val_ratio)

    test_chunks = chunks[:n_test] if n_test > 0 else []
    val_chunks = chunks[n_test:n_test + n_val]
    train_chunks = chunks[n_test + n_val:]

    return {
        "train": np.concatenate(train_chunks) if train_chunks else np.array([]),
        "val": np.concatenate(val_chunks) if val_chunks else np.array([]),
        "test": np.concatenate(test_chunks) if test_chunks else np.array([]),
    }


def load_data_random_split(data_dir: str) -> dict:
    """
    Load data with random split from a single load.

    This represents the typical approach in literature which can
    lead to data leakage due to temporal correlation between windows.
    """
    signals = load_raw_signals_for_load("1772", data_dir)
    window_size = config["window_size"]
    stride = config["stride"]
    seed = config["seed"]

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for key, data in signals.items():
        split = split_signal_into_chunks(
            data["signal"], val_ratio=0.2, test_ratio=0.2, seed=seed
        )

        for split_name, X_list, y_list in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test)
        ]:
            if len(split[split_name]) > 0:
                windows = extract_windows(split[split_name], window_size, stride)
                X_list.append(windows)
                y_list.extend([data["label_idx"]] * len(windows))

    return _finalize_data(X_train, y_train, X_val, y_val, X_test, y_test)


def load_data_fault_size_split(data_dir: str, use_all_loads: bool = False) -> dict:
    """
    Load data with fault-size based split.

    Holds out one fault severity entirely for testing to evaluate
    whether models can generalize to unseen damage levels.

    Args:
        data_dir: Directory containing data files
        use_all_loads: If True, use all motor loads for training diversity
                      If False, use only load 1772
    """
    test_fault_size = config["test_fault_size"]
    window_size = config["window_size"]
    stride = config["stride"]
    seed = config["seed"]

    loads_to_use = config["available_loads"] if use_all_loads else ["1772"]

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for load in loads_to_use:
        signals = load_raw_signals_for_load(load, data_dir)

        for key, data in signals.items():
            signal = data["signal"]
            label = data["label_idx"]
            fault_size = data["fault_size"]

            if fault_size == test_fault_size:
                # This severity goes entirely to test set
                windows = extract_windows(signal, window_size, stride)
                X_test.append(windows)
                y_test.extend([label] * len(windows))

            elif fault_size == "none":
                # Normal class: split between train, val, AND test
                # This ensures all 4 classes are present in test set
                split = split_signal_into_chunks(
                    signal, val_ratio=0.2, test_ratio=0.2, seed=seed
                )

                train_windows = extract_windows(split["train"], window_size, stride)
                val_windows = extract_windows(split["val"], window_size, stride)
                test_windows = extract_windows(split["test"], window_size, stride)

                X_train.append(train_windows)
                y_train.extend([label] * len(train_windows))
                X_val.append(val_windows)
                y_val.extend([label] * len(val_windows))
                X_test.append(test_windows)
                y_test.extend([label] * len(test_windows))

            else:
                # Other fault severities: split between train and validation only
                split = split_signal_into_chunks(
                    signal, val_ratio=0.2, test_ratio=0.0, seed=seed
                )

                train_windows = extract_windows(split["train"], window_size, stride)
                val_windows = extract_windows(split["val"], window_size, stride)

                X_train.append(train_windows)
                y_train.extend([label] * len(train_windows))
                X_val.append(val_windows)
                y_val.extend([label] * len(val_windows))

    return _finalize_data(X_train, y_train, X_val, y_val, X_test, y_test)


def load_data_cross_load_split(
    train_loads: list,
    test_loads: list,
    data_dir: str
) -> dict:
    """
    Load data with cross-load split.

    Trains on one operating condition and tests on different conditions
    to evaluate robustness to varying motor speeds.

    Args:
        train_loads: List of load identifiers for training
        test_loads: List of load identifiers for testing
        data_dir: Directory containing data files
    """
    window_size = config["window_size"]
    stride = config["stride"]
    seed = config["seed"]

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    # Load training data from specified loads
    for load in train_loads:
        signals = load_raw_signals_for_load(load, data_dir)

        for key, data in signals.items():
            split = split_signal_into_chunks(
                data["signal"], val_ratio=0.2, test_ratio=0.0, seed=seed
            )

            train_windows = extract_windows(split["train"], window_size, stride)
            val_windows = extract_windows(split["val"], window_size, stride)

            X_train.append(train_windows)
            y_train.extend([data["label_idx"]] * len(train_windows))
            X_val.append(val_windows)
            y_val.extend([data["label_idx"]] * len(val_windows))

    # Load test data from different loads
    for load in test_loads:
        signals = load_raw_signals_for_load(load, data_dir)

        for key, data in signals.items():
            windows = extract_windows(data["signal"], window_size, stride)
            X_test.append(windows)
            y_test.extend([data["label_idx"]] * len(windows))

    return _finalize_data(X_train, y_train, X_val, y_val, X_test, y_test)


def _finalize_data(
    X_train: list, y_train: list,
    X_val: list, y_val: list,
    X_test: list, y_test: list
) -> dict:
    """
    Concatenate window lists and add channel dimension for CNN input.
    """
    X_train = np.concatenate(X_train)[:, np.newaxis, :]
    X_val = np.concatenate(X_val)[:, np.newaxis, :]
    X_test = np.concatenate(X_test)[:, np.newaxis, :] if X_test else np.array([])

    return {
        "X_train": X_train,
        "y_train": np.array(y_train, dtype=np.int64),
        "X_val": X_val,
        "y_val": np.array(y_val, dtype=np.int64),
        "X_test": X_test,
        "y_test": np.array(y_test, dtype=np.int64),
    }


def load_data(
    strategy: str = None,
    data_dir: str = "data/raw",
    train_loads: list = None,
    test_loads: list = None,
    use_all_loads: bool = False
) -> dict:
    """
    Main data loading function with configurable split strategy.

    Args:
        strategy: Split strategy (random, fault_size, fault_size_all_loads, cross_load)
        data_dir: Directory containing data files
        train_loads: For cross_load strategy, which loads to train on
        test_loads: For cross_load strategy, which loads to test on
        use_all_loads: For fault_size strategy, whether to use all loads

    Returns:
        Dictionary containing train, validation, and test data arrays
    """
    if strategy is None:
        strategy = config["split_strategy"]

    if strategy == "random":
        data = load_data_random_split(data_dir)
        print(f"Mode: {config['classification_mode']}, Split: random")

    elif strategy == "fault_size":
        data = load_data_fault_size_split(data_dir, use_all_loads=False)
        print(f"Mode: {config['classification_mode']}, Split: fault_size (test={config['test_fault_size']})")

    elif strategy == "fault_size_all_loads":
        data = load_data_fault_size_split(data_dir, use_all_loads=True)
        print(f"Mode: {config['classification_mode']}, Split: fault_size_all_loads (test={config['test_fault_size']})")

    elif strategy == "cross_load":
        if train_loads is None:
            train_loads = config["cross_load_train"]
        if test_loads is None:
            test_loads = config["cross_load_test"]
        data = load_data_cross_load_split(train_loads, test_loads, data_dir)
        print(f"Mode: {config['classification_mode']}, Split: cross_load")
        print(f"  Train loads: {train_loads}")
        print(f"  Test loads: {test_loads}")

    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    print(f"  Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")

    return data


def create_dataloaders(data: dict, batch_size: int = None) -> tuple:
    """
    Create PyTorch DataLoaders from data dictionary.

    Args:
        data: Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
        batch_size: Batch size for training (uses config default if None)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config["batch_size"]

    train_loader = DataLoader(
        CWRUDataset(data["X_train"], data["y_train"]),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        CWRUDataset(data["X_val"], data["y_val"]),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        CWRUDataset(data["X_test"], data["y_test"]),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Verification of data loading for each strategy
    print("=" * 60)
    print("Data Loading Verification")
    print("=" * 60)

    for strategy in ["random", "fault_size", "fault_size_all_loads"]:
        print(f"\n{strategy.upper()}:")
        data = load_data(strategy=strategy)
        print(f"  Train labels: {np.unique(data['y_train'])}")
        print(f"  Test labels: {np.unique(data['y_test'])}")
