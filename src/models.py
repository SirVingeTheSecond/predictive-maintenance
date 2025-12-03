"""
Neural Network Architectures for Time-Series Classification.

Implements three architectures commonly used for vibration-based
fault diagnosis: CNN, LSTM, and a hybrid CNN-LSTM model.
"""

"""
Neural Network Architectures for Bearing Fault Diagnosis.
"""

import torch
import torch.nn as nn

from config import config


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class CNN1D(nn.Module):
    """Basic 1D CNN (original architecture)."""

    def __init__(self, num_classes: int = None, in_channels: int = 1):
        super().__init__()

        if num_classes is None:
            num_classes = config["num_classes"]

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
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


class CNN1DDeep(nn.Module):
    """Deeper CNN with residual connections (ResNet-style)."""

    def __init__(self, num_classes: int = None, in_channels: int = 1):
        super().__init__()

        if num_classes is None:
            num_classes = config["num_classes"]

        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _make_layer(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier."""

    def __init__(self, num_classes: int = None, hidden_size: int = 128, in_channels: int = 1):
        super().__init__()

        if num_classes is None:
            num_classes = config["num_classes"]

        self.conv_downsample = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=4, padding=3),
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

        _, (hidden, _) = self.lstm(x)
        hidden_concat = torch.cat((hidden[-2], hidden[-1]), dim=1)

        x = self.classifier(hidden_concat)
        return x


class CNNLSTM(nn.Module):
    """Hybrid CNN-LSTM model."""

    def __init__(self, num_classes: int = None, hidden_size: int = 64, in_channels: int = 1):
        super().__init__()

        if num_classes is None:
            num_classes = config["num_classes"]

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
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

        _, (hidden, _) = self.lstm(x)
        hidden_concat = torch.cat((hidden[-2], hidden[-1]), dim=1)

        x = self.classifier(hidden_concat)
        return x


def get_model(model_name: str, num_classes: int = None, in_channels: int = None) -> nn.Module:
    """Factory function to create models."""

    if in_channels is None:
        # Determine from feature mode
        if config["feature_mode"] == "both":
            in_channels = 2
        else:
            in_channels = 1

    models = {
        "cnn1d": lambda: CNN1D(num_classes=num_classes, in_channels=in_channels),
        "cnn1d_deep": lambda: CNN1DDeep(num_classes=num_classes, in_channels=in_channels),
        "lstm": lambda: LSTMClassifier(num_classes=num_classes, in_channels=in_channels),
        "cnnlstm": lambda: CNNLSTM(num_classes=num_classes, in_channels=in_channels),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    return models[model_name]()


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
