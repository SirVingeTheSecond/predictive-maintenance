import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)

import config


def get_predictions(model, loader, device, mode):
    """
    Get model predictions.
    
    Args:
        model: Trained model
        loader: Data loader
        device: Device
        mode: Classification mode
        
    Returns:
        For multiclass: (predictions, probabilities, labels)
        For multilabel: (probabilities, labels)
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            if mode == "multilabel":
                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.cpu().numpy())
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())

    if mode == "multilabel":
        return np.array(all_probs), np.array(all_labels)
    else:
        return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def evaluate_multiclass(model, loader, device, mode) -> dict:
    """
    Evaluate multiclass classification with metrics.
    
    Returns:
        Dictionary with accuracy, precision, recall, F1, AUROC, confusion matrix
    """
    predictions, probabilities, labels = get_predictions(model, loader, device, mode)

    accuracy = float(accuracy_score(labels, predictions))

    # Per-class and macro metrics
    class_names = config.CLASS_NAMES[mode]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    # Per-class metrics
    per_class_metrics = {}
    for i, name in enumerate(class_names):
        if i < len(precision):
            per_class_metrics[name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]) if i < len(support) else 0,
            }

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    # AUROC (this is for probabilities)
    try:
        if probabilities.shape[1] == len(class_names):
            macro_auroc = float(roc_auc_score(
                labels, probabilities, multi_class='ovr', average='macro'
            ))
        else:
            macro_auroc = float('nan')
    except:
        macro_auroc = float('nan')

    return {
        "accuracy": accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "macro_auroc": macro_auroc,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_multilabel(model, loader, device) -> dict:
    """
    Evaluate multilabel classification with metrics.
    
    Returns:
        Dictionary with AUROC scores and per-class metrics
    """
    probs, labels = get_predictions(model, loader, device, mode="multilabel")

    class_names = config.CLASS_NAMES["multilabel"]
    
    # Per-class AUROC
    aurocs = {}
    for i, name in enumerate(class_names):
        if len(np.unique(labels[:, i])) > 1:
            aurocs[name] = float(roc_auc_score(labels[:, i], probs[:, i]))
        else:
            aurocs[name] = float('nan')

    # Macro average (excluding NaN)
    valid = [v for v in aurocs.values() if not np.isnan(v)]
    aurocs["macro"] = float(np.mean(valid)) if valid else float('nan')

    # Binary predictions for precision/recall/F1
    predictions = (probs > 0.5).astype(int)

    per_class_metrics = {}
    for i, name in enumerate(class_names):
        p, r, f, _ = precision_recall_fscore_support(
            labels[:, i], predictions[:, i], average='binary', zero_division=0
        )
        per_class_metrics[name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "auroc": aurocs[name],
        }

    return {
        "macro_auroc": aurocs["macro"],
        "per_class_auroc": aurocs,
        "per_class_metrics": per_class_metrics,
    }


def evaluate(model, loader, device, mode) -> dict:
    """
    Evaluate model based on classification mode.
    
    Args:
        model: Trained model
        loader: Test data loader
        device: Device
        mode: Classification mode
        
    Returns:
        Dictionary with metrics
    """
    if mode == "multilabel":
        return evaluate_multilabel(model, loader, device)
    else:
        return evaluate_multiclass(model, loader, device, mode)


# =============================================================================
# Noise testing (VERY EXPERIMENTAL)
# =============================================================================

def add_gaussian_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Gaussian white noise to signal at specified SNR level.

    Based on found papers (from TF-MDA and ECMTP, I should remember to cite these later).
    
    Args:
        signal: Input signal array
        snr_db: Signal-to-noise ratio in decibels
        
    Returns:
        Noisy signal
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


def add_noise_to_batch(X: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise to a batch of samples."""
    X_noisy = np.zeros_like(X)
    for i in range(len(X)):
        for c in range(X.shape[1]):
            X_noisy[i, c] = add_gaussian_noise(X[i, c], snr_db)
    return X_noisy


def evaluate_noise_robustness(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mode: str,
    device: str,
    snr_levels: list = None,
) -> dict:
    """
    Evaluate model robustness under various noise levels.
    
    Based on found papers (from TF-MDA and ECMTP, I should remember to cite these later).

    Tests SNR from -6dB to +6dB.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        mode: Classification mode
        device: Device
        snr_levels: List of SNR levels in dB
        
    Returns:
        Dictionary with metrics at each noise level
    """
    from data import SignalDataset
    from torch.utils.data import DataLoader
    
    if snr_levels is None:
        snr_levels = [-6, -4, -2, 0, 2, 4, 6]
    
    results = {}
    
    # Clean baseline
    clean_loader = DataLoader(
        SignalDataset(X_test, y_test, mode),
        batch_size=config.BATCH_SIZE, shuffle=False
    )
    clean_metrics = evaluate(model, clean_loader, device, mode)
    results["clean"] = clean_metrics
    
    # Test at each noise level
    for snr in snr_levels:
        X_noisy = add_noise_to_batch(X_test.copy(), snr)
        noisy_loader = DataLoader(
            SignalDataset(X_noisy, y_test, mode),
            batch_size=config.BATCH_SIZE, shuffle=False
        )
        noisy_metrics = evaluate(model, noisy_loader, device, mode)
        results[f"snr_{snr}dB"] = noisy_metrics
    
    # Summary
    if mode == "multilabel":
        summary = {
            "clean": clean_metrics["macro_auroc"],
            **{f"{snr}dB": results[f"snr_{snr}dB"]["macro_auroc"] for snr in snr_levels}
        }
    else:
        summary = {
            "clean": clean_metrics["accuracy"],
            **{f"{snr}dB": results[f"snr_{snr}dB"]["accuracy"] for snr in snr_levels}
        }
    
    results["summary"] = summary
    return results
