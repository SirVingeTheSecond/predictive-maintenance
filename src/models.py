"""
Neural Network Architectures for Time-Series Classification.

Implements three architectures commonly used for vibration-based
fault diagnosis: CNN, LSTM, and a hybrid CNN-LSTM model.
"""

import torch
import torch.nn as nn

from config import config


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for vibration signal classification.

    Architecture uses progressively increasing filter counts to capture
    hierarchical features from local patterns to global structure.
    """

    def __init__(self, num_classes: int = None):
        super().__init__()

        if num_classes is None:
            num_classes = config["num_classes"]

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for capturing temporal dependencies in signals.

    Initial convolution reduces sequence length to make LSTM training
    feasible. Bidirectional processing captures context from both directions.
    """

    def __init__(self, num_classes: int = None, hidden_size: int = 128):
        super().__init__()

        if num_classes is None:
            num_classes = config["num_classes"]

        self.conv_downsample = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv_downsample(x)
        x = x.permute(0, 2, 1)

        lstm_out, (hidden, cell) = self.lstm(x)
        hidden_concat = torch.cat((hidden[-2], hidden[-1]), dim=1)

        x = self.classifier(hidden_concat)
        return x


class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM combining local feature extraction with sequence modeling.

    CNN layers extract discriminative local features while LSTM captures
    temporal relationships between feature activations.
    """

    def __init__(self, num_classes: int = None, hidden_size: int = 64):
        super().__init__()

        if num_classes is None:
            num_classes = config["num_classes"]

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        lstm_out, (hidden, cell) = self.lstm(x)
        hidden_concat = torch.cat((hidden[-2], hidden[-1]), dim=1)

        x = self.classifier(hidden_concat)
        return x


def get_model(model_name: str, num_classes: int = None) -> nn.Module:
    """
    Factory function to create model instances by name.

    Args:
        model_name: One of "cnn1d", "lstm", or "cnnlstm"
        num_classes: Number of output classes (uses config default if None)

    Returns:
        Initialized model instance
    """
    models = {
        "cnn1d": CNN1D,
        "lstm": LSTMClassifier,
        "cnnlstm": CNNLSTM,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name](num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
