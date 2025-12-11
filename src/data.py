import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config


class SignalDataset(Dataset):
    """Dataset for vibration signals."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray, mode: str):
        """
        Args:
            signals: Array of shape (N, 1, seq_len)
            labels: Array of shape (N,) for multiclass or (N, 3) for multilabel
            mode: Classification mode ("4class", "10class", "multilabel")
        """
        self.signals = torch.from_numpy(signals).float()
        self.mode = mode

        if mode == "multilabel":
            self.labels = torch.from_numpy(labels).float()
        else:
            self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


# =============================================================================
# Signal processing
# =============================================================================

def load_raw_signal(filepath: str) -> np.ndarray:
    """Load raw vibration signal from .npz file."""
    data = np.load(filepath)
    return data["DE"].flatten()


def extract_windows(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Extract windows from signal."""
    n_windows = (len(signal) - window_size) // stride + 1
    windows = np.zeros((n_windows, window_size), dtype=np.float32)

    for i in range(n_windows):
        start = i * stride
        windows[i] = signal[start:start + window_size]

    return windows


def compute_fft_features(windows: np.ndarray) -> np.ndarray:
    """Compute normalized log-FFT spectrum."""
    # Apply Hanning window
    hanning = np.hanning(windows.shape[1])
    windowed = windows * hanning

    # FFT
    fft_result = np.fft.rfft(windowed, axis=1)
    magnitude = np.abs(fft_result)

    # Log scaling
    log_magnitude = np.log1p(magnitude)

    # Normalize per sample
    mean = log_magnitude.mean(axis=1, keepdims=True)
    std = log_magnitude.std(axis=1, keepdims=True) + 1e-8
    normalized = (log_magnitude - mean) / std

    return normalized.astype(np.float32)


def compute_time_features(windows: np.ndarray) -> np.ndarray:
    """Normalize time-domain windows."""
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True) + 1e-8
    return ((windows - mean) / std).astype(np.float32)


def process_windows(windows: np.ndarray, feature_mode: str) -> np.ndarray:
    """
    Process windows according to feature mode.
    
    Returns array of shape (N, channels, seq_len)
    """
    if feature_mode == "time":
        features = compute_time_features(windows)
        return features[:, np.newaxis, :]

    elif feature_mode == "fft":
        features = compute_fft_features(windows)
        return features[:, np.newaxis, :]

    elif feature_mode == "both":
        time_feat = compute_time_features(windows)
        fft_feat = compute_fft_features(windows)
        # Pad FFT to match time length
        pad = time_feat.shape[1] - fft_feat.shape[1]
        fft_padded = np.pad(fft_feat, ((0, 0), (0, pad)), mode='constant')
        return np.stack([time_feat, fft_padded], axis=1)

    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")


# =============================================================================
# Data splitting
# =============================================================================

def split_signal_chunks(
    signal: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> dict:
    """Split signal into chunks for train/val/test."""
    rng = np.random.RandomState(seed)

    n_chunks = 10
    chunk_size = len(signal) // n_chunks
    chunks = [signal[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]
    rng.shuffle(chunks)

    n_test = int(n_chunks * test_ratio)
    n_val = int(n_chunks * val_ratio)

    return {
        "train": np.concatenate(chunks[n_test + n_val:]) if chunks[n_test + n_val:] else np.array([]),
        "val": np.concatenate(chunks[n_test:n_test + n_val]) if n_val > 0 else np.array([]),
        "test": np.concatenate(chunks[:n_test]) if n_test > 0 else np.array([]),
    }


def get_label(mode: str, idx_4: int, idx_10: int, multilabel: list):
    """Get the correct label for classification mode."""
    if mode == "4class":
        return idx_4
    elif mode == "10class":
        return idx_10
    elif mode == "multilabel":
        return np.array(multilabel, dtype=np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# Data loading
# =============================================================================

def load_data_random_split(
    mode: str,
    data_dir: str,
    seed: int,
    feature_mode: str = None,
) -> dict:
    """Load data with random split (single load, all severities mixed)."""
    if feature_mode is None:
        feature_mode = config.FEATURE_MODE

    signals = {}
    file_mapping = config.get_file_mapping("1772")

    # Load all signals
    for filename, class_name, fault_size, idx_4, idx_10, multilabel in file_mapping:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        signals[class_name] = {
            "signal": load_raw_signal(filepath),
            "label": get_label(mode, idx_4, idx_10, multilabel),
        }

    # Process each class
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for class_name, data in signals.items():
        split = split_signal_chunks(data["signal"], val_ratio=0.2, test_ratio=0.2, seed=seed)

        for split_name, X_list, y_list in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            if len(split[split_name]) > 0:
                windows = extract_windows(split[split_name], config.WINDOW_SIZE, config.STRIDE)
                features = process_windows(windows, feature_mode)
                X_list.append(features)
                y_list.extend([data["label"]] * len(features))

    return _finalize_data(X_train, y_train, X_val, y_val, X_test, y_test, mode)


def load_data_fault_size_split(
    mode: str,
    data_dir: str,
    seed: int,
    use_all_loads: bool = True,
    feature_mode: str = None,
    create_val_split: bool = False,
) -> dict:
    """
    Load data with fault-size split (train on 007/021, test on 014).
    
    NOTE (from the papers):
    - Test set (014) is held out for generalization evaluation
    - Training pool (007, 021) should NOT be split into train/val
    - Validation within training pool does not predict 014 performance
    - Using K-fold CV within training pool for hyperparameter selection
    
    Args:
        mode: Classification mode
        data_dir: Path to data
        seed: Random seed
        use_all_loads: Use all motor loads
        feature_mode: Feature extraction mode
        create_val_split: If True, create val from training pool (for k-fold CV)
                         If False, all training data goes to train
    """
    if feature_mode is None:
        feature_mode = config.FEATURE_MODE

    loads = config.MOTOR_LOADS if use_all_loads else ["1772"]
    test_fault_size = config.TEST_FAULT_SIZE

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for load in loads:
        file_mapping = config.get_file_mapping(load)

        for filename, class_name, fault_size, idx_4, idx_10, multilabel in file_mapping:
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found: {filepath}")

            signal = load_raw_signal(filepath)
            label = get_label(mode, idx_4, idx_10, multilabel)

            if fault_size == test_fault_size:
                # Test severity -> test set (completely held out)
                windows = extract_windows(signal, config.WINDOW_SIZE, config.STRIDE)
                features = process_windows(windows, feature_mode)
                X_test.append(features)
                y_test.extend([label] * len(features))

            elif fault_size is None:
                # Normal class -> split: some to test, rest to training pool
                split = split_signal_chunks(signal, val_ratio=0.0, test_ratio=0.2, seed=seed)
                
                # Test portion
                if len(split["test"]) > 0:
                    windows = extract_windows(split["test"], config.WINDOW_SIZE, config.STRIDE)
                    features = process_windows(windows, feature_mode)
                    X_test.append(features)
                    y_test.extend([label] * len(features))
                
                # Training portion (no val split from training pool)
                if len(split["train"]) > 0:
                    windows = extract_windows(split["train"], config.WINDOW_SIZE, config.STRIDE)
                    features = process_windows(windows, feature_mode)
                    X_train.append(features)
                    y_train.extend([label] * len(features))

            else:
                # Training severities (007, 021) -> ALL to training pool
                # NO validation split - use k-fold CV instead
                windows = extract_windows(signal, config.WINDOW_SIZE, config.STRIDE)
                features = process_windows(windows, feature_mode)
                X_train.append(features)
                y_train.extend([label] * len(features))

    # For fault-size split, validation set is empty by design
    # K-fold CV should be used within training pool for selecting hyperparameters
    return _finalize_data(X_train, y_train, X_val, y_val, X_test, y_test, mode)


def _finalize_data(
    X_train: list, y_train: list,
    X_val: list, y_val: list,
    X_test: list, y_test: list,
    mode: str,
) -> dict:
    """Concatenate and format data arrays."""
    X_train = np.concatenate(X_train) if X_train else np.array([])
    X_val = np.concatenate(X_val) if X_val else np.array([])
    X_test = np.concatenate(X_test) if X_test else np.array([])

    if mode == "multilabel":
        y_train = np.array(y_train, dtype=np.float32) if y_train else np.array([])
        y_val = np.array(y_val, dtype=np.float32) if y_val else np.array([])
        y_test = np.array(y_test, dtype=np.float32) if y_test else np.array([])
    else:
        y_train = np.array(y_train, dtype=np.int64) if y_train else np.array([])
        y_val = np.array(y_val, dtype=np.int64) if y_val else np.array([])
        y_test = np.array(y_test, dtype=np.int64) if y_test else np.array([])

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


# =============================================================================
# Interface stuff
# =============================================================================

def load_data_cross_load_split(
    mode: str,
    data_dir: str,
    seed: int,
    train_loads: list = None,
    test_load: str = None,
    feature_mode: str = None,
) -> dict:
    """Load data with cross-load split (train on some loads, test on held-out load)."""
    if feature_mode is None:
        feature_mode = config.FEATURE_MODE
    if train_loads is None:
        train_loads = ["1772", "1750"]
    if test_load is None:
        test_load = "1730"

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    # Training loads
    for load in train_loads:
        file_mapping = config.get_file_mapping(load)

        for filename, class_name, fault_size, idx_4, idx_10, multilabel in file_mapping:
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found: {filepath}")

            signal = load_raw_signal(filepath)
            label = get_label(mode, idx_4, idx_10, multilabel)

            # Split into train/val
            split = split_signal_chunks(signal, val_ratio=0.2, test_ratio=0.0, seed=seed)

            for split_name, X_list, y_list in [
                ("train", X_train, y_train),
                ("val", X_val, y_val),
            ]:
                if len(split[split_name]) > 0:
                    windows = extract_windows(split[split_name], config.WINDOW_SIZE, config.STRIDE)
                    features = process_windows(windows, feature_mode)
                    X_list.append(features)
                    y_list.extend([label] * len(features))

    # Test load (entirely held out)
    file_mapping = config.get_file_mapping(test_load)

    for filename, class_name, fault_size, idx_4, idx_10, multilabel in file_mapping:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        signal = load_raw_signal(filepath)
        label = get_label(mode, idx_4, idx_10, multilabel)

        windows = extract_windows(signal, config.WINDOW_SIZE, config.STRIDE)
        features = process_windows(windows, feature_mode)
        X_test.append(features)
        y_test.extend([label] * len(features))

    return _finalize_data(X_train, y_train, X_val, y_val, X_test, y_test, mode)


def load_data(
    mode: str,
    split: str,
    seed: int = None,
    data_dir: str = None,
    verbose: bool = True,
) -> dict:
    """
    Load dataset with specified configuration.
    
    Args:
        mode: Classification mode ("4class", "10class", "multilabel")
        split: Split strategy ("random", "fault_size", "fault_size_all_loads", "cross_load")
        seed: Random seed
        data_dir: Path to data directory
        verbose: Print data statistics
        
    Returns:
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
    """
    if seed is None:
        seed = config.DEFAULT_SEED
    if data_dir is None:
        data_dir = config.DATA_DIR

    if split == "random":
        data = load_data_random_split(mode, data_dir, seed)
    elif split == "fault_size":
        data = load_data_fault_size_split(mode, data_dir, seed, use_all_loads=False)
    elif split == "fault_size_all_loads":
        data = load_data_fault_size_split(mode, data_dir, seed, use_all_loads=True)
    elif split == "cross_load":
        data = load_data_cross_load_split(mode, data_dir, seed)
    else:
        raise ValueError(f"Unknown split strategy: {split}")

    if verbose:
        print(f"Data: mode={mode}, split={split}, feature={config.FEATURE_MODE}")
        print(f"  Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")
        print(f"  Shape: {data['X_train'].shape}")

    return data


def create_dataloaders(
    data: dict,
    mode: str,
    batch_size: int = None,
) -> tuple:
    """
    Create DataLoaders from data dictionary.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    train_loader = DataLoader(
        SignalDataset(data["X_train"], data["y_train"], mode),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        SignalDataset(data["X_val"], data["y_val"], mode),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    test_loader = DataLoader(
        SignalDataset(data["X_test"], data["y_test"], mode),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
