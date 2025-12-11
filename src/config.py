
# Device is determined at runtime to avoid requiring torch at import
DEVICE = None  # Set by get_device() in utils.py

# Seeds for multi-run experiments
# METHODOLOGY NOTE: Other papers recommend 5-30 seeds for reliability
# - Benchmark paper: 30 seeds
# - ECMCTP: 10 experiments averaged
# - CNN-LSTM: 5-fold CV
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
DEFAULT_SEED = 42

DATA_DIR = "data/raw"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

# =============================================================================
# Signal processing
# =============================================================================

WINDOW_SIZE = 2048
STRIDE = 512  # 75% overlap
SAMPLING_RATE = 12000  # Hz

# Feature extraction: "time", "fft", "both"
FEATURE_MODE = "fft"

# =============================================================================
# Dataset structure
# =============================================================================

MOTOR_LOADS = ["1772", "1750", "1730"]
FAULT_SIZES = ["007", "014", "021"]
TEST_FAULT_SIZE = "014"  # Held out for generalization test

# File mapping: (filename_template, class_name, fault_size, 4class_idx, 10class_idx, multilabel)
def get_file_mapping(load: str) -> list:
    """Return file mapping for a motor load."""
    return [
        (f"{load}_Normal.npz", "Normal", None, 0, 0, [0, 0, 0]),
        (f"{load}_B_7_DE12.npz", "Ball_007", "007", 1, 1, [1, 0, 0]),
        (f"{load}_B_14_DE12.npz", "Ball_014", "014", 1, 2, [1, 0, 0]),
        (f"{load}_B_21_DE12.npz", "Ball_021", "021", 1, 3, [1, 0, 0]),
        (f"{load}_IR_7_DE12.npz", "IR_007", "007", 2, 4, [0, 1, 0]),
        (f"{load}_IR_14_DE12.npz", "IR_014", "014", 2, 5, [0, 1, 0]),
        (f"{load}_IR_21_DE12.npz", "IR_021", "021", 2, 6, [0, 1, 0]),
        (f"{load}_OR@6_7_DE12.npz", "OR_007", "007", 3, 7, [0, 0, 1]),
        (f"{load}_OR@6_14_DE12.npz", "OR_014", "014", 3, 8, [0, 0, 1]),
        (f"{load}_OR@6_21_DE12.npz", "OR_021", "021", 3, 9, [0, 0, 1]),
    ]

# =============================================================================
# Classification modes
# =============================================================================

CLASS_NAMES = {
    "4class": ["Normal", "Ball", "IR", "OR"],
    "10class": [
        "Normal",
        "Ball_007", "Ball_014", "Ball_021",
        "IR_007", "IR_014", "IR_021",
        "OR_007", "OR_014", "OR_021",
    ],
    "multilabel": ["Ball", "IR", "OR"],
}

NUM_CLASSES = {
    "4class": 4,
    "10class": 10,
    "multilabel": 3,
}

# =============================================================================
# Training hyperparameters
# =============================================================================

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.001

# Studies can be seen declarative specifications of experiments to run.
# Each study defines what combinations to test.
#
# METHODOLOGY NOTE (from other papers):
# - For fault_size splits: validation set is empty, no early stopping
# - Report mean +- std across seeds (the idea is to test the reliability)

STUDIES = {
    # Main comparison study for the paper
    "comparison": {
        "description": "Full comparison of models across configurations (5 seeds)",
        "models": ["cnn", "lstm", "cnnlstm"],
        "configurations": [
            {"mode": "4class", "split": "random", "name": "4class_random"},
            {"mode": "4class", "split": "fault_size_all_loads", "name": "4class_fault_size"},
            {"mode": "4class", "split": "cross_load", "name": "4class_cross_load"},
            {"mode": "10class", "split": "random", "name": "10class_random"},
            {"mode": "multilabel", "split": "random", "name": "multilabel_random"},
            {"mode": "multilabel", "split": "fault_size_all_loads", "name": "multilabel_fault_size"},
        ],
        "seeds": [42, 123, 456, 789, 1024],
        "epochs": 100,
    },

    # A study with 10 seeds
    "comparison_full": {
        "description": "Full comparison with 10 seeds for statistical reliability",
        "models": ["cnn", "lstm", "cnnlstm"],
        "configurations": [
            {"mode": "4class", "split": "random", "name": "4class_random"},
            {"mode": "4class", "split": "fault_size_all_loads", "name": "4class_fault_size"},
            {"mode": "4class", "split": "cross_load", "name": "4class_cross_load"},
            {"mode": "10class", "split": "random", "name": "10class_random"},
            {"mode": "multilabel", "split": "random", "name": "multilabel_random"},
            {"mode": "multilabel", "split": "fault_size_all_loads", "name": "multilabel_fault_size"},
        ],
        "seeds": [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144],
        "epochs": 100,
    },

    # Quick test to verify setup
    "quick_test": {
        "description": "Quick validation run",
        "models": ["cnn"],
        "configurations": [
            {"mode": "4class", "split": "fault_size_all_loads", "name": "4class_fault_size"},
        ],
        "seeds": [42],
        "epochs": 10,
    },

    # Generalization study (fault-size and cross-load splits)
    "generalization": {
        "description": "Focus on generalization performance (5 seeds)",
        "models": ["cnn", "lstm", "cnnlstm"],
        "configurations": [
            {"mode": "4class", "split": "fault_size_all_loads", "name": "4class_fault_size"},
            {"mode": "4class", "split": "cross_load", "name": "4class_cross_load"},
            {"mode": "multilabel", "split": "fault_size_all_loads", "name": "multilabel_fault_size"},
        ],
        "seeds": [42, 123, 456, 789, 1024],
        "epochs": 100,
    },

    # Fault-size study with K-fold CV
    "fault_size_kfold": {
        "description": "Fault-size generalization with K-fold CV for epoch selection (arXiv 2407.14625)",
        "models": ["cnn", "lstm", "cnnlstm"],
        "configurations": [
            {"mode": "4class", "split": "fault_size_all_loads", "name": "4class_fault_size"},
            {"mode": "multilabel", "split": "fault_size_all_loads", "name": "multilabel_fault_size"},
        ],
        "seeds": [42, 123, 456, 789, 1024],
        "epochs": 100,  # Max epochs for k-fold CV
        "use_kfold_cv": True,
        "n_folds": 3,
    },
}

# =============================================================================
# Hyperparameter search
# =============================================================================

SWEEPS = {
    "hyperparameter_search": {
        "description": "Grid search for optimal hyperparameters",
        "base_config": {
            "mode": "4class",
            "split": "fault_size_all_loads",
        },
        "param_grid": {
            "model": ["cnn", "lstm", "cnnlstm"],
            "lr": [1e-4, 5e-4, 1e-3],
            "dropout": [0.2, 0.3, 0.5],
            "weight_decay": [0, 1e-4, 1e-3],
        },
        "screening_epochs": 20,
        "screening_threshold": 0.45,
        "full_epochs": 100,
    },
}
