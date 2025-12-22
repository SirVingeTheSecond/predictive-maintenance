import torch
import torch.nn as nn

from . import config


# =============================================================================
# Activation functions
# =============================================================================

def get_activation(name: str = "gelu", inplace: bool = True) -> nn.Module:
    """
    Get activation function by name.

    Args:
        name: Activation name ("relu", "leaky_relu", "gelu", "selu", "elu")
        inplace: Use inplace if supported

    Returns:
        Activation module
    """
    activations = {
        "relu": lambda: nn.ReLU(inplace=inplace),
        "leaky_relu": lambda: nn.LeakyReLU(0.1, inplace=inplace),
        "gelu": lambda: nn.GELU(), # No inplace support GELU
        "selu": lambda: nn.SELU(inplace=inplace),
        "elu": lambda: nn.ELU(inplace=inplace),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    return activations[name]()


# =============================================================================
# Weight init
# =============================================================================

def _get_init_nonlinearity(activation: str) -> str:
    """Map activation name to torch init nonlinearity."""
    mapping = {
        "relu": "relu",
        "leaky_relu": "leaky_relu",
        "gelu": "relu",  # GELU is similar to ReLU for init purposes
        "selu": "linear",  # SELU uses lecun_normal (linear approximation)
        "elu": "relu",
    }
    return mapping.get(activation, "relu")


def _init_weights(module, nonlinearity: str = "relu"):
    """Init weights using Kaiming initialization."""
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# =============================================================================
# Models
# =============================================================================

class CNN(nn.Module):
    """
    1D Convolutional Neural Network.

    Architecture:
        Conv1d(1 -> 32, k=7) -> BN -> Act -> MaxPool -> Dropout
        Conv1d(32 -> 64, k=5) -> BN -> Act -> MaxPool -> Dropout
        Conv1d(64 -> 128, k=3) -> BN -> Act -> AdaptiveAvgPool
        Linear(128 -> 64) -> Act -> Dropout -> Linear(64 -> num_classes)

    Parameters: ~36K (for num_classes=4)
    """

    def __init__(self, num_classes: int, in_channels: int = 1,
                 dropout: float = None, activation: str = "relu"):
        super().__init__()

        if dropout is None:
            dropout = config.DROPOUT

        # Lighter dropout in conv blocks, full dropout in classifier
        conv_dropout = dropout * 0.67

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            get_activation(activation),
            nn.MaxPool1d(2),
            nn.Dropout(conv_dropout),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            get_activation(activation),
            nn.MaxPool1d(2),
            nn.Dropout(conv_dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            get_activation(activation),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        nonlinearity = _get_init_nonlinearity(activation)
        self.apply(lambda m: _init_weights(m, nonlinearity))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNN_VGG(nn.Module):
    """
    VGG-style 1D CNN.

    Architecture:
        2x Conv1d(1 -> 64, k=3) -> BN -> Act -> MaxPool -> Dropout
        2x Conv1d(64 -> 128, k=3) -> BN -> Act -> MaxPool -> Dropout
        2x Conv1d(128 -> 256, k=3) -> BN -> Act -> MaxPool -> Dropout
        2x Conv1d(256 -> 256, k=3) -> BN -> Act -> AdaptiveAvgPool
        Linear(256 -> 128) -> Act -> Dropout -> Linear(128 -> num_classes)

    Parameters: ~470K (for num_classes=4)
    """

    def __init__(self, num_classes: int, in_channels: int = 1,
                 dropout: float = None, activation: str = "relu"):
        super().__init__()

        if dropout is None:
            dropout = config.DROPOUT

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            get_activation(activation),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            get_activation(activation),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            get_activation(activation),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            get_activation(activation),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            get_activation(activation),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            get_activation(activation),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            get_activation(activation),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            get_activation(activation),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        nonlinearity = _get_init_nonlinearity(activation)
        self.apply(lambda m: _init_weights(m, nonlinearity))

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
                 hidden_size: int = 128, dropout: float = None,
                 activation: str = "relu"):
        super().__init__()

        if dropout is None:
            dropout = config.DROPOUT

        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(64),
            get_activation(activation),
            nn.Conv1d(64, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64),
            get_activation(activation),
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
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        nonlinearity = _get_init_nonlinearity(activation)
        self.apply(lambda m: _init_weights(m, nonlinearity))

    def forward(self, x):
        x = self.downsample(x)
        x = x.permute(0, 2, 1)
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        x = self.classifier(hidden_cat)
        return x


class LSTM_Large(nn.Module):
    """
    Bidirectional LSTM with increased capacity.

    Architecture:
        Conv1d downsampling
        3-layer Bidirectional LSTM (hidden=256)
        Classifier on concatenated final hidden states

    Parameters: ~1.1M (for num_classes=4)
    """

    def __init__(self, num_classes: int, in_channels: int = 1,
                 hidden_size: int = 256, dropout: float = None,
                 activation: str = "relu"):
        super().__init__()

        if dropout is None:
            dropout = config.DROPOUT

        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(64),
            get_activation(activation),
            nn.Conv1d(64, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64),
            get_activation(activation),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        nonlinearity = _get_init_nonlinearity(activation)
        self.apply(lambda m: _init_weights(m, nonlinearity))

    def forward(self, x):
        x = self.downsample(x)
        x = x.permute(0, 2, 1)
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
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
                 hidden_size: int = 128, dropout: float = None,
                 activation: str = "relu"):
        super().__init__()

        if dropout is None:
            dropout = config.DROPOUT

        # Lighter dropout in conv blocks (a bit lighter here as LSTM provides additional regularization)
        conv_dropout = dropout * 0.33

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            get_activation(activation),
            nn.MaxPool1d(2),
            nn.Dropout(conv_dropout),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            get_activation(activation),
            nn.MaxPool1d(2),
            nn.Dropout(conv_dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            get_activation(activation),
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
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        nonlinearity = _get_init_nonlinearity(activation)
        self.apply(lambda m: _init_weights(m, nonlinearity))

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        x = self.classifier(hidden_cat)
        return x


class CNNLSTM_VGG(nn.Module):
    """
    VGG-style CNN combined with deeper LSTM.

    Architecture:
        VGG-style CNN (3x3 kernels, 64->128->256 filters)
        3-layer Bidirectional LSTM (hidden=256)
        Classifier on concatenated final hidden states

    Parameters: ~1.8M (for num_classes=4)
    """

    def __init__(self, num_classes: int, in_channels: int = 1,
                 hidden_size: int = 256, dropout: float = None,
                 activation: str = "relu"):
        super().__init__()

        if dropout is None:
            dropout = config.DROPOUT

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            get_activation(activation),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            get_activation(activation),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            get_activation(activation),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            get_activation(activation),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            get_activation(activation),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            get_activation(activation),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        nonlinearity = _get_init_nonlinearity(activation)
        self.apply(lambda m: _init_weights(m, nonlinearity))

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        x = self.classifier(hidden_cat)
        return x


# =============================================================================
# Model registry
# =============================================================================

MODELS = {
    "cnn": CNN,
    "cnn_vgg": CNN_VGG,
    "lstm": LSTM,
    "lstm_large": LSTM_Large,
    "cnnlstm": CNNLSTM,
    "cnnlstm_vgg": CNNLSTM_VGG,
}


def create_model(name: str, num_classes: int, dropout: float = None,
                 activation: str = "relu") -> nn.Module:
    """
    Create a model instance.

    Args:
        name: Model name ("cnn", "cnn_vgg", "lstm", "lstm_large", "cnnlstm", "cnnlstm_vgg")
        num_classes: Number of output classes
        dropout: Dropout rate (optional)
        activation: Activation function ("relu", "leaky_relu", "gelu", "selu", "elu")

    Returns:
        The instance of the Model
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")

    kwargs = {"num_classes": num_classes, "activation": activation}
    if dropout is not None:
        kwargs["dropout"] = dropout

    return MODELS[name](**kwargs)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    from .utils import count_parameters

    print("Model Parameter Counts:")
    print("=" * 70)

    x = torch.randn(2, 1, 1025)

    for name in MODELS:
        model = create_model(name, num_classes=4)
        params = count_parameters(model)
        y = model(x)
        print(f"{name.upper():15} {params:>10,} params  |  {x.shape} -> {y.shape}")

    print("\nActivation Function Test (CNN):")
    print("-" * 70)
    for act in ["relu", "leaky_relu", "gelu", "selu", "elu"]:
        model = create_model("cnn", num_classes=4, activation=act)
        y = model(x)
        print(f"{act:12} -> output shape: {y.shape}")
