import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

import config


class EarlyStopping:

    def __init__(self, patience: int = None, min_delta: float = None):
        if patience is None:
            patience = config.EARLY_STOPPING_PATIENCE
        if min_delta is None:
            min_delta = config.EARLY_STOPPING_MIN_DELTA

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save state from
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        """Restore best model state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def get_criterion(mode: str) -> nn.Module:
    """Get loss function for classification mode."""
    if mode == "multilabel":
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()


def train_epoch(model, loader, criterion, optimizer, device, mode):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)

        # Compute accuracy
        if mode == "multilabel":
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).all(dim=1).sum().item()
        else:
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    return total_loss / total, correct / total


def evaluate_epoch(model, loader, criterion, device, mode):
    """Run one evaluation epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

            if mode == "multilabel":
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).all(dim=1).sum().item()
            else:
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

    return total_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    mode: str,
    epochs: int = None,
    lr: float = None,
    weight_decay: float = None,
    device: str = None,
    verbose: bool = True,
) -> dict:
    """
    Full training loop.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader (can be None for fault-size split)
        mode: Classification mode
        epochs: Number of epochs
        lr: Learning rate
        weight_decay: Weight decay
        device: Device to train on
        verbose: Print progress
        
    Returns:
        Dictionary with history and best metrics
    """
    if epochs is None:
        epochs = config.EPOCHS
    if lr is None:
        lr = config.LEARNING_RATE
    if weight_decay is None:
        weight_decay = config.WEIGHT_DECAY
    if device is None:
        from utils import get_device
        device = get_device()

    model = model.to(device)

    criterion = get_criterion(mode)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # Only use early stopping if we have a validation set
    use_early_stopping = val_loader is not None and len(val_loader.dataset) > 0
    early_stopping = EarlyStopping() if use_early_stopping else None

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None # Well...

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, mode)
        
        # Validation (if available)
        if use_early_stopping:
            val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device, mode)
        else:
            val_loss, val_acc = 0.0, 0.0

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if use_early_stopping and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
        elif not use_early_stopping and train_acc > best_val_acc:
            # Without validation, track best training (less meaningful)
            best_val_acc = train_acc
            best_epoch = epoch

        if verbose:
            if use_early_stopping:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"LR: {current_lr:.2e}")
            else:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"LR: {current_lr:.2e}")

        if early_stopping is not None and early_stopping(val_loss, model):
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    # Restore best model if we used early stopping
    if early_stopping is not None:
        early_stopping.restore_best(model)

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "epochs_trained": len(history["train_loss"]),
    }


def train_kfold_cv(
    model_class,
    model_kwargs: dict,
    train_dataset,
    mode: str,
    n_folds: int = 3,
    epochs: int = None,
    lr: float = None,
    weight_decay: float = None,
    device: str = None,
    seed: int = None,
    verbose: bool = True,
    fold_verbose: bool = False,
) -> dict:
    """
    K-fold cross-validation for hyperparameter selection.
    
    NOTE (Benchmark paper arXiv 2407.14625 -> Rosa et al.):
    This performs CV WITHIN the training data to select hyperparameters.
    The held-out test set (014 severity) is NEVER touched during CV.
    
    This can be used to determine:
    1. Optimal number of epochs (from early stopping across folds)
    2. Expected validation accuracy range
    
    Args:
        model_class: Model class/factory to instantiate
        model_kwargs: Arguments for model constructor
        train_dataset: Full training dataset
        mode: Classification mode
        n_folds: Number of CV folds
        epochs: Training epochs per fold
        lr: Learning rate
        weight_decay: Weight decay
        device: Device
        seed: Random seed for fold splitting
        verbose: Print fold summaries
        fold_verbose: Print per-epoch training details (can be noisy)
        
    Returns:
        Dictionary with per-fold results and aggregated statistics including
        optimal_epochs for subsequent training
    """
    if epochs is None:
        epochs = config.EPOCHS
    if lr is None:
        lr = config.LEARNING_RATE
    if weight_decay is None:
        weight_decay = config.WEIGHT_DECAY
    if device is None:
        from utils import get_device
        device = get_device()
    if seed is None:
        seed = config.DEFAULT_SEED
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(train_dataset)))):
        if verbose:
            print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
        
        # Create fold data loaders
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Create new model for this fold
        model = model_class(**model_kwargs).to(device)
        
        # Train with early stopping
        result = train(
            model, train_loader, val_loader, mode,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            device=device, verbose=fold_verbose
        )
        
        fold_results.append({
            "fold": fold_idx + 1,
            "best_val_acc": result["best_val_acc"],
            "best_epoch": result["best_epoch"],
            "epochs_trained": result["epochs_trained"],
            "final_val_loss": result["history"]["val_loss"][-1] if result["history"]["val_loss"] else None,
        })
        
        if verbose:
            print(f"Fold {fold_idx + 1}: Val Acc = {result['best_val_acc']:.4f}, "
                  f"Best Epoch = {result['best_epoch']}, "
                  f"Stopped at = {result['epochs_trained']}")
    
    # Aggregate results
    val_accs = [r["best_val_acc"] for r in fold_results]
    best_epochs = [r["best_epoch"] for r in fold_results]
    
    # Optimal epochs: use median to account for outliers -> should in theory be more robust?
    # Just adding a small buffer (10%) for safety when retraining on full data
    median_best_epoch = int(np.median(best_epochs))
    optimal_epochs = max(median_best_epoch, 10) # At least 10 epochs
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"K-Fold CV Summary ({n_folds} folds)")
        print(f"{'='*60}")
        print(f"Val Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
        print(f"Best epochs per fold: {best_epochs}")
        print(f"Optimal epochs for retraining: {optimal_epochs}")
    
    return {
        "fold_results": fold_results,
        "mean_val_acc": float(np.mean(val_accs)),
        "std_val_acc": float(np.std(val_accs)),
        "mean_best_epoch": float(np.mean(best_epochs)),
        "median_best_epoch": median_best_epoch,
        "optimal_epochs": optimal_epochs,
        "n_folds": n_folds,
    }
