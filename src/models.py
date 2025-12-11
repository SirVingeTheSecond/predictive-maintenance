import torch
import torch.nn as nn

import config


# =============================================================================
# Weight init
# =============================================================================

def _init_weights(module):
    """Init weights using Kaiming initialization."""
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# =============================================================================
# Models
# =============================================================================

class CNN(nn.Module):
    """
    1D Convolutional Neural Network.
    
    Architecture:
        Conv1d(1 -> 32, k=7) -> BN -> ReLU -> MaxPool -> Dropout
        Conv1d(32 -> 64, k=5) -> BN -> ReLU -> MaxPool -> Dropout
        Conv1d(64 -> 128, k=3) -> BN -> ReLU -> AdaptiveAvgPool
        Linear(128 -> 64) -> ReLU -> Dropout -> Linear(64 -> num_classes)

    Parameters: ~36K (for num_classes=4)
    """

    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = None):
        super().__init__()

        if dropout is None:
            dropout = config.DROPOUT

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.apply(_init_weights)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LSTM(nn.Module):
    """
    Bidirectional LSTM with CNN preprocessing.
    
    Architecture:
        Conv1d downsampling (helps reduce sequence length)
        2-layer Bidirectional LSTM
        Classifier on concatenated final hidden states
    
    Parameters: ~181K (for num_classes=4)
    """

    def __init__(self, num_classes: int, in_channels: int = 1,
                 hidden_size: int = 128, dropout: float = None):
        super().__init__()

        if dropout is None:
            dropout = config.DROPOUT

        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.apply(_init_weights)

    def forward(self, x):
        # Downsample: (batch, channels, seq) -> (batch, 64, seq/16)
        x = self.downsample(x)

        # Reshape for LSTM: (batch, seq, channels)
        x = x.permute(0, 2, 1)

        # LSTM
        _, (hidden, _) = self.lstm(x)

        # Concatenate forward and backward final hidden states
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)

        # Classify
        x = self.classifier(hidden_cat)
        return x


class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM architecture.
    
    Architecture:
        Deeper CNN feature extraction
        2-layer Bidirectional LSTM
        Classifier on concatenated final hidden states
    
    Parameters: ~238K (for num_classes=4)
    """

    def __init__(self, num_classes: int, in_channels: int = 1,
                 hidden_size: int = 128, dropout: float = None):
        super().__init__()

        if dropout is None:
            dropout = config.DROPOUT

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.apply(_init_weights)

    def forward(self, x):
        # CNN features: (batch, 1, seq) -> (batch, 128, seq/8)
        x = self.cnn(x)

        # Reshape for LSTM: (batch, seq, channels)
        x = x.permute(0, 2, 1)

        # LSTM
        _, (hidden, _) = self.lstm(x)

        # Concatenate bidirectional hidden states
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)

        # Classify
        x = self.classifier(hidden_cat)
        return x


# =============================================================================
# Model registry
# =============================================================================

MODELS = {
    "cnn": CNN,
    "lstm": LSTM,
    "cnnlstm": CNNLSTM,
}


def create_model(name: str, num_classes: int, dropout: float = None) -> nn.Module:
    """
    Create a model instance.
    
    Args:
        name: Model name ("cnn", "lstm", "cnnlstm")
        num_classes: Number of output classes
        dropout: Dropout rate (optional)
        
    Returns:
        The instance of the Model
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")

    kwargs = {"num_classes": num_classes}
    if dropout is not None:
        kwargs["dropout"] = dropout

    return MODELS[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Model Parameter Counts:")

    x = torch.randn(2, 1, 1025)

    for name in MODELS:
        model = create_model(name, num_classes=4)
        params = count_parameters(model)
        y = model(x)
        print(f"{name.upper():10} {params:>10,} params  |  {x.shape} -> {y.shape}")
