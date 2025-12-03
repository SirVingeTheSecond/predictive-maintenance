"""
CWRU Bearing Dataset Loading and Preprocessing.

Implements multiple data splitting strategies to evaluate model
generalization under different conditions:
- Random: baseline with chunk-based splitting to prevent leakage
- Fault-size: tests generalization to unseen severity levels
- Fault-size-all-loads: fault-size split with training diversity from all loads
- Cross-load: tests generalization to unseen operating conditions

Design decisions based on literature review:
- 75% overlap (stride 512 with window 2048) matches common practice
- Chunk-based splitting prevents segment-level data leakage
- Normal class included in test set for fault-size experiments
"""
"""
CWRU Bearing Dataset Loading with FFT Features and Data Augmentation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import config, get_file_mapping


class CWRUDataset(Dataset):
    """PyTorch Dataset with optional data augmentation."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.signals = torch.from_numpy(signals).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment and config["augmentation"]["enabled"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.signals[idx].clone()
        y = self.labels[idx]

        if self.augment:
            x = self._apply_augmentation(x)

        return x, y

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to the signal."""
        aug_config = config["augmentation"]

        # Add Gaussian noise
        if aug_config["noise_std"] > 0:
            noise = torch.randn_like(x) * aug_config["noise_std"]
            x = x + noise

        # Random amplitude scaling
        scale_min, scale_max = aug_config["scale_range"]
        scale = torch.empty(1).uniform_(scale_min, scale_max).item()
        x = x * scale

        # Random time shift (circular)
        shift_max = aug_config["time_shift_max"]
        if shift_max > 0:
            shift = torch.randint(-shift_max, shift_max + 1, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=-1)

        return x


def load_raw_signals_for_load(load: str, data_dir: str) -> dict:
    """Load raw vibration signals for a specific motor load."""
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
    """Extract sliding windows from signal."""
    num_windows = (len(signal) - window_size) // stride + 1
    windows = np.zeros((num_windows, window_size), dtype=np.float32)

    for i in range(num_windows):
        start = i * stride
        windows[i] = signal[start:start + window_size]

    return windows


def compute_fft_features(windows: np.ndarray) -> np.ndarray:
    """
    Compute FFT magnitude spectrum for each window.

    Returns the first half of the spectrum (positive frequencies only)
    with log scaling for better dynamic range.
    """
    # Apply Hanning window to reduce spectral leakage
    hanning = np.hanning(windows.shape[1])
    windowed = windows * hanning

    # Compute FFT
    fft_result = np.fft.rfft(windowed, axis=1)

    # Get magnitude spectrum (log scale for better dynamic range)
    magnitude = np.abs(fft_result)
    log_magnitude = np.log1p(magnitude)  # log(1 + x) to handle zeros

    # Normalize per sample
    mean = log_magnitude.mean(axis=1, keepdims=True)
    std = log_magnitude.std(axis=1, keepdims=True) + 1e-8
    normalized = (log_magnitude - mean) / std

    return normalized.astype(np.float32)


def compute_time_features(windows: np.ndarray) -> np.ndarray:
    """Normalize time-domain windows."""
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True) + 1e-8
    normalized = (windows - mean) / std
    return normalized.astype(np.float32)


def process_windows(windows: np.ndarray, feature_mode: str = None) -> np.ndarray:
    """
    Process windows according to feature mode.

    Args:
        windows: Raw signal windows
        feature_mode: "time", "fft", or "both"

    Returns:
        Processed features with channel dimension
    """
    if feature_mode is None:
        feature_mode = config["feature_mode"]

    if feature_mode == "time":
        features = compute_time_features(windows)
        return features[:, np.newaxis, :]

    elif feature_mode == "fft":
        features = compute_fft_features(windows)
        return features[:, np.newaxis, :]

    elif feature_mode == "both":
        time_features = compute_time_features(windows)
        fft_features = compute_fft_features(windows)
        # Stack as two channels
        return np.stack([time_features, fft_features], axis=1)

    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")


def split_signal_into_chunks(
        signal: np.ndarray,
        val_ratio: float,
        test_ratio: float,
        seed: int
) -> dict:
    """Split signal into train/val/test using chunk-based approach."""
    rng = np.random.RandomState(seed)

    n_chunks = 10
    chunk_size = len(signal) // n_chunks
    chunks = [signal[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]
    rng.shuffle(chunks)

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
    """Load data with random split."""
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
                features = process_windows(windows)
                X_list.append(features)
                y_list.extend([data["label_idx"]] * len(features))

    return _finalize_data(X_train, y_train, X_val, y_val, X_test, y_test)


def load_data_fault_size_split(data_dir: str, use_all_loads: bool = False) -> dict:
    """Load data with fault-size based split."""
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
                windows = extract_windows(signal, window_size, stride)
                features = process_windows(windows)
                X_test.append(features)
                y_test.extend([label] * len(features))

            elif fault_size == "none":
                split = split_signal_into_chunks(
                    signal, val_ratio=0.2, test_ratio=0.2, seed=seed
                )

                for split_name, X_list, y_list in [
                    ("train", X_train, y_train),
                    ("val", X_val, y_val),
                    ("test", X_test, y_test)
                ]:
                    if len(split[split_name]) > 0:
                        windows = extract_windows(split[split_name], window_size, stride)
                        features = process_windows(windows)
                        X_list.append(features)
                        y_list.extend([label] * len(features))

            else:
                split = split_signal_into_chunks(
                    signal, val_ratio=0.2, test_ratio=0.0, seed=seed
                )

                for split_name, X_list, y_list in [
                    ("train", X_train, y_train),
                    ("val", X_val, y_val),
                ]:
                    if len(split[split_name]) > 0:
                        windows = extract_windows(split[split_name], window_size, stride)
                        features = process_windows(windows)
                        X_list.append(features)
                        y_list.extend([label] * len(features))

    return _finalize_data(X_train, y_train, X_val, y_val, X_test, y_test)


def load_data_cross_load_split(
        train_loads: list,
        test_loads: list,
        data_dir: str
) -> dict:
    """Load data with cross-load split."""
    window_size = config["window_size"]
    stride = config["stride"]
    seed = config["seed"]

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for load in train_loads:
        signals = load_raw_signals_for_load(load, data_dir)

        for key, data in signals.items():
            split = split_signal_into_chunks(
                data["signal"], val_ratio=0.2, test_ratio=0.0, seed=seed
            )

            for split_name, X_list, y_list in [
                ("train", X_train, y_train),
                ("val", X_val, y_val),
            ]:
                if len(split[split_name]) > 0:
                    windows = extract_windows(split[split_name], window_size, stride)
                    features = process_windows(windows)
                    X_list.append(features)
                    y_list.extend([data["label_idx"]] * len(features))

    for load in test_loads:
        signals = load_raw_signals_for_load(load, data_dir)

        for key, data in signals.items():
            windows = extract_windows(data["signal"], window_size, stride)
            features = process_windows(windows)
            X_test.append(features)
            y_test.extend([data["label_idx"]] * len(features))

    return _finalize_data(X_train, y_train, X_val, y_val, X_test, y_test)


def _finalize_data(
        X_train: list, y_train: list,
        X_val: list, y_val: list,
        X_test: list, y_test: list
) -> dict:
    """Concatenate arrays and return data dictionary."""
    X_train = np.concatenate(X_train)
    X_val = np.concatenate(X_val)
    X_test = np.concatenate(X_test) if X_test else np.array([])

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
) -> dict:
    """Main data loading function."""
    if strategy is None:
        strategy = config["split_strategy"]

    if strategy == "random":
        data = load_data_random_split(data_dir)

    elif strategy == "fault_size":
        data = load_data_fault_size_split(data_dir, use_all_loads=False)

    elif strategy == "fault_size_all_loads":
        data = load_data_fault_size_split(data_dir, use_all_loads=True)

    elif strategy == "cross_load":
        if train_loads is None:
            train_loads = config["cross_load_train"]
        if test_loads is None:
            test_loads = config["cross_load_test"]
        data = load_data_cross_load_split(train_loads, test_loads, data_dir)

    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    print(f"Strategy: {strategy} | Features: {config['feature_mode']}")
    print(f"Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")
    print(f"Input shape: {data['X_train'].shape}")

    return data


def create_dataloaders(data: dict, batch_size: int = None, augment_train: bool = True) -> tuple:
    """Create DataLoaders with optional augmentation."""
    if batch_size is None:
        batch_size = config["batch_size"]

    train_loader = DataLoader(
        CWRUDataset(data["X_train"], data["y_train"], augment=augment_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        CWRUDataset(data["X_val"], data["y_val"], augment=False),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        CWRUDataset(data["X_test"], data["y_test"], augment=False),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
