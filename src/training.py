"""
Training Utilities for Neural Network Models.

Provides training loop, early stopping, and evaluation functions
with support for model-specific learning rates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import config


class EarlyStopping:
    """
    Monitors validation loss to prevent overfitting.

    Restores best model weights when training is stopped early
    to ensure the returned model is the best performing one.
    """

    def __init__(self, patience: int = None, min_delta: float = None):
        if patience is None:
            patience = config["early_stopping_patience"]
        if min_delta is None:
            min_delta = config["early_stopping_min_delta"]

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore_best_weights(self, model: nn.Module):
        """Load the best model weights observed during training."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> tuple:
    """
    Execute one training epoch.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: str
) -> tuple:
    """
    Evaluate model on a dataset.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = None,
    model_name: str = None,
    verbose: bool = True
) -> dict:
    """
    Train a model with early stopping and model-specific learning rate.

    Args:
        model: Neural network to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Maximum number of training epochs
        model_name: Model identifier for learning rate lookup
        verbose: Whether to print progress

    Returns:
        Dictionary containing training history and best validation accuracy
    """
    if epochs is None:
        epochs = config["epochs"]

    device = config["device"]
    model = model.to(device)

    lr = config["learning_rates"].get(model_name, 1e-3)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if verbose:
            print(f"Epoch {epoch + 1:3d}/{epochs} | "
                  f"Train: {train_acc:.4f} | Val: {val_acc:.4f}")

        if early_stopping(val_loss, model):
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    early_stopping.restore_best_weights(model)

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "epochs_trained": len(history["train_loss"]),
    }


def get_predictions(model: nn.Module, data_loader, device: str = None) -> tuple:
    """
    Get model predictions and true labels for a dataset.

    Returns:
        Tuple of (predictions array, labels array)
    """
    if device is None:
        device = config["device"]

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)
