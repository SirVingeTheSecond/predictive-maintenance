import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from torch.utils.data import DataLoader

from src import config
from src.models import create_model
from src.data import load_data, SignalDataset
from src.training import train
from src.evaluation import evaluate
from src.utils import get_device, set_seed, count_parameters, print_header, print_separator
from src.augmentation import AugmentedSignalDataset, get_augmentations


# Results directory
RESULTS_DIR = Path(config.RESULTS_DIR) / "augmentation"

# Augmentations to test
AUGMENTATIONS = ["none", "noise_only", "warp_only", "scale_only", "light", "moderate", "heavy"]


def run_single_experiment(
    model_name: str = "cnn",
    augmentation_name: str = "none",
    mode: str = "4class",
    split: str = "fault_size_all_loads",
    seed: int = None,
    epochs: int = None,
    verbose: bool = True,
) -> dict:
    """Run single experiment with specified augmentation."""
    if seed is None:
        seed = config.DEFAULT_SEED
    if epochs is None:
        epochs = config.EPOCHS

    set_seed(seed)
    device = get_device()

    if verbose:
        print_header(f"Model: {model_name} | Aug: {augmentation_name} | Seed: {seed}", width=60)

    # Load data
    data = load_data(mode=mode, split=split, seed=seed, verbose=False)

    # Create datasets
    base_train_dataset = SignalDataset(data["X_train"], data["y_train"], mode)

    if augmentation_name == "none":
        train_dataset = base_train_dataset
    else:
        augmentation = get_augmentations(augmentation_name)
        train_dataset = AugmentedSignalDataset(
            base_train_dataset,
            augmentations=augmentation,
            augment_prob=0.8
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        SignalDataset(data["X_val"], data["y_val"], mode),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    ) if len(data["X_val"]) > 0 else None

    test_loader = DataLoader(
        SignalDataset(data["X_test"], data["y_test"], mode),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    # Create and train model
    num_classes = config.NUM_CLASSES[mode]
    model = create_model(model_name, num_classes, dropout=config.DROPOUT)
    model = model.to(device)

    if verbose:
        print(f"Parameters: {count_parameters(model):,}")
        print(f"Training samples: {len(train_dataset)}")

    train_result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode=mode,
        epochs=epochs,
        device=device,
        verbose=verbose,
    )

    # Evaluate
    test_metrics = evaluate(model, test_loader, device, mode)

    result = {
        "model": model_name,
        "augmentation": augmentation_name,
        "mode": mode,
        "split": split,
        "seed": seed,
        "epochs": epochs,
        "test_metrics": test_metrics,
        "train_history": {
            "final_train_acc": train_result["history"]["train_acc"][-1],
            "final_train_loss": train_result["history"]["train_loss"][-1],
        },
        "timestamp": datetime.now().isoformat(),
    }

    if verbose:
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {test_metrics.get('macro_f1', 'N/A')}")

    return result


def run_sweep(
    models: list = None,
    augmentations: list = None,
    seeds: list = None,
    epochs: int = None,
    verbose: bool = True,
):
    """Run full augmentation sweep."""
    if models is None:
        models = ["cnn"]
    if augmentations is None:
        augmentations = AUGMENTATIONS
    if seeds is None:
        seeds = config.EXPERIMENT_SEEDS
    if epochs is None:
        epochs = config.EPOCHS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    total = len(models) * len(augmentations) * len(seeds)
    current = 0

    print_header("AUGMENTATION SWEEP")
    print(f"Total experiments: {total}")
    print(f"Models: {models}")
    print(f"Augmentations: {augmentations}")
    print(f"Seeds: {seeds}")
    print()

    for model_name in models:
        for augmentation_name in augmentations:
            for seed in seeds:
                current += 1
                print(f"[{current}/{total}] {model_name} + {augmentation_name} (seed={seed})")

                try:
                    result = run_single_experiment(
                        model_name=model_name,
                        augmentation_name=augmentation_name,
                        seed=seed,
                        epochs=epochs,
                        verbose=verbose,
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    all_results.append({
                        "model": model_name,
                        "augmentation": augmentation_name,
                        "seed": seed,
                        "error": str(e),
                    })

    # Save results
    output_file = RESULTS_DIR / "augmentation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print_separator()
    print(f"Results saved to: {output_file}")

    # Print summary
    print_summary(all_results)

    return all_results


def print_summary(results: list):
    """Print summary of augmentation results."""
    # Aggregate by augmentation
    augmentation_stats = defaultdict(list)
    for r in results:
        if "error" not in r:
            augmentation_stats[r["augmentation"]].append(r["test_metrics"]["accuracy"])

    print_separator()
    print("SUMMARY")
    print_separator()
    print(f"{'Augmentation':<12} {'Accuracy (%)':<15} {'Std':<10} {'N':<5}")
    print_separator(char="-", width=45)

    summary = []
    for aug_name, accs in sorted(augmentation_stats.items()):
        mean = np.mean(accs) * 100
        std = np.std(accs) * 100
        summary.append((aug_name, mean, std, len(accs)))

    for aug_name, mean, std, n in sorted(summary, key=lambda x: -x[1]):
        print(f"{aug_name:<12} {mean:<15.2f} {std:<10.2f} {n:<5}")

    if summary:
        best = max(summary, key=lambda x: x[1])
        print()
        print(f"Best: {best[0]} ({best[1]:.2f}% ± {best[2]:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run augmentation experiments")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "single"],
        default="quick",
        help="Experiment mode",
    )
    parser.add_argument(
        "--augmentation",
        default="moderate",
        help="Augmentation for single mode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_SEED,
        help="Seed for single mode"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs (default: config.EPOCHS)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    if args.mode == "quick":
        # Quick test: fewer augmentations, 1 seed, 50 epochs
        run_sweep(
            augmentations=["none", "noise_only", "moderate"],
            seeds=[config.DEFAULT_SEED],
            epochs=50,
            verbose=args.verbose,
        )
    elif args.mode == "full":
        # Full sweep: all augmentations, all seeds
        run_sweep(
            seeds=config.EXPERIMENT_SEEDS,
            epochs=args.epochs,
            verbose=args.verbose,
        )
    else:
        # Single experiment
        run_single_experiment(
            augmentation_name=args.augmentation,
            seed=args.seed,
            epochs=args.epochs,
            verbose=True,
        )


if __name__ == "__main__":
    main()
