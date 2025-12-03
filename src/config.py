"""
Configuration for CWRU Bearing Fault Diagnosis Experiments.

Centralizes hyperparameters and dataset configurations to ensure
reproducibility across different experimental setups.
"""

import torch

config = {
    # Hardware
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,

    # Signal processing
    # 2048 samples at 12kHz captures approximately 170ms per window
    "window_size": 2048,
    # 512 stride gives 75% overlap, matching literature practices
    # Papers typically use 50-75% overlap, NOT 97%
    "stride": 512,

    # Classification setup
    # 4class groups severities by fault type for generalization testing
    "classification_mode": "4class",
    "class_names": ["Normal", "Ball", "IR", "OR"],
    "class_names_10": [
        "Normal",
        "Ball_007", "Ball_014", "Ball_021",
        "IR_007", "IR_014", "IR_021",
        "OR_007", "OR_014", "OR_021"
    ],
    "num_classes": 4,

    # Dataset structure
    "available_loads": ["1772", "1750", "1730"],

    # Split strategy options:
    # - random: standard split with chunk-based leakage prevention
    # - fault_size: hold out one severity for testing (single load)
    # - fault_size_all_loads: hold out one severity using all loads
    # - cross_load: train on one load and test on different loads
    "split_strategy": "random",

    # Fault-size split settings
    "test_fault_size": "014",

    # Cross-load split settings
    "cross_load_train": ["1772"],
    "cross_load_test": ["1750", "1730"],

    # Training hyperparameters
    "batch_size": 64,
    "epochs": 50,

    # LSTM requires higher learning rate due to gradient flow challenges
    "learning_rates": {
        "cnn1d": 1e-3,
        "lstm": 1e-2,
        "cnnlstm": 1e-3,
    },

    # Early stopping prevents overfitting while allowing convergence
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.001,
}


def get_file_mapping(load: str) -> list:
    """
    Return dataset file mapping for a specific motor load.

    Args:
        load: Motor load identifier (1772, 1750, or 1730)

    Returns:
        List of tuples containing (filename, class_name, fault_size, class_idx)
    """
    return [
        (f"{load}_Normal.npz", "Normal", None, 0),
        (f"{load}_B_7_DE12.npz", "Ball_007", "007", 1),
        (f"{load}_B_14_DE12.npz", "Ball_014", "014", 1),
        (f"{load}_B_21_DE12.npz", "Ball_021", "021", 1),
        (f"{load}_IR_7_DE12.npz", "IR_007", "007", 2),
        (f"{load}_IR_14_DE12.npz", "IR_014", "014", 2),
        (f"{load}_IR_21_DE12.npz", "IR_021", "021", 2),
        (f"{load}_OR@6_7_DE12.npz", "OR_007", "007", 3),
        (f"{load}_OR@6_14_DE12.npz", "OR_014", "014", 3),
        (f"{load}_OR@6_21_DE12.npz", "OR_021", "021", 3),
    ]
