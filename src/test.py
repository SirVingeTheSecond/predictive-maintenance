import os
import json

import numpy as np
import torch

import config
from data import load_data, create_dataloaders
from models import create_model, count_parameters
from evaluation import evaluate, get_predictions
from utils import get_device


def load_checkpoint(checkpoint_path: str) -> tuple:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model.pth file
        
    Returns:
        (model, exp_config) tuple
    """
    # Find config.json in same directory
    exp_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(exp_dir, "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        exp_config = json.load(f)
    
    # Create model
    model_name = exp_config["model"]
    mode = exp_config["mode"]
    dropout = exp_config.get("dropout", config.DROPOUT)
    num_classes = config.NUM_CLASSES[mode]
    
    model = create_model(model_name, num_classes=num_classes, dropout=dropout)
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    return model, exp_config


def print_confusion_matrix(cm: np.ndarray, class_names: list):
    """Print confusion matrix."""
    n = len(class_names)
    
    # Header
    header = "          " + "".join(f"{name:>10}" for name in class_names)
    print(header)
    print("-" * len(header))
    
    # Rows
    for i, name in enumerate(class_names):
        row = f"{name:<10}" + "".join(f"{cm[i, j]:>10}" for j in range(n))
        print(row)


def print_conditional_probabilities(cm: np.ndarray, class_names: list):
    """Print P(predicted | true) for each class."""
    print("\nConditional Probabilities P(predicted | true):")
    print("-" * 50)
    
    for i, true_name in enumerate(class_names):
        row_sum = cm[i].sum()
        if row_sum == 0:
            continue
            
        probs = cm[i] / row_sum
        print(f"\nGiven true={true_name}:")
        for j, pred_name in enumerate(class_names):
            if probs[j] > 0.01:  # Only show significant probabilities
                print(f"  P({pred_name}) = {probs[j]:.3f}")


def test_checkpoint(
    checkpoint_path: str,
    split: str = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate a saved checkpoint.
    
    Args:
        checkpoint_path: Path to model.pth
        split: Override split strategy (optional)
        verbose: Print detailed output
        
    Returns:
        Dictionary with test metrics
    """
    # Load model and config
    model, exp_config = load_checkpoint(checkpoint_path)
    
    model_name = exp_config["model"]
    mode = exp_config["mode"]
    original_split = exp_config["split"]
    seed = exp_config.get("seed", config.DEFAULT_SEED)
    
    # Use original split unless overridden
    if split is None:
        split = original_split
    
    device = get_device()
    model = model.to(device)
    model.eval()
    
    if verbose:
        print("CHECKPOINT EVALUATION")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Model:      {model_name}")
        print(f"Mode:       {mode}")
        print(f"Split:      {split}" + (" (override)" if split != original_split else ""))
        print(f"Parameters: {count_parameters(model):,}")
    
    # Load data
    data = load_data(mode=mode, split=split, seed=seed, verbose=verbose)
    _, _, test_loader = create_dataloaders(data, mode)
    
    # Evaluate
    metrics = evaluate(model, test_loader, device, mode)
    
    if verbose:
        print("RESULTS")
        
        if mode == "multilabel":
            print(f"\nMacro AUROC: {metrics['auroc']['macro']:.4f}")
            print("\nPer-class AUROC:")
            for name in config.CLASS_NAMES["multilabel"]:
                auroc = metrics['auroc'].get(name, float('nan'))
                recall = metrics['per_class_recall'].get(name, 0)
                print(f"  {name:<10} AUROC: {auroc:.4f}  Recall: {recall:.4f}")
        else:
            print(f"\nAccuracy: {metrics['accuracy']:.4f}")
            print("\nPer-class Recall:")
            for name, recall in metrics['per_class_recall'].items():
                print(f"  {name:<10} {recall:.4f}")
            
            # Confusion matrix
            cm = np.array(metrics['confusion_matrix'])
            class_names = config.CLASS_NAMES[mode]
            
            print("\nConfusion Matrix:")
            print_confusion_matrix(cm, class_names)
            print_conditional_probabilities(cm, class_names)
    
    return metrics


def compare_splits(checkpoint_path: str):
    """
    Evaluate checkpoint on all split strategies.
    
    Useful for understanding generalization.
    """
    model, exp_config = load_checkpoint(checkpoint_path)
    mode = exp_config["mode"]

    print("CROSS-SPLIT EVALUATION")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mode: {mode}")
    
    splits = ["random", "fault_size_all_loads", "cross_load"]
    results = {}
    
    for split in splits:
        print(f"\n--- {split} ---")
        try:
            metrics = test_checkpoint(checkpoint_path, split=split, verbose=False)
            
            if "accuracy" in metrics:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                results[split] = metrics['accuracy']
            else:
                print(f"  Macro AUROC: {metrics['auroc']['macro']:.4f}")
                results[split] = metrics['auroc']['macro']
                
        except Exception as e:
            print(f"  Error: {e}")
            results[split] = None
    
    return results
