import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.data import load_data, create_dataloaders
from src.models import create_model
from src.training import train
from src.evaluation import evaluate
from src.utils import set_seed, get_device, print_header, print_separator


def main():
    device = get_device()
    results = []

    # Use config values
    activations = config.ACTIVATIONS
    seeds = config.EXPERIMENT_SEEDS

    print_header("ACTIVATION FUNCTION TESTS")
    print(f"Model: CNN")
    print(f"Split: fault_size_all_loads")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Seeds: {seeds}")
    print(f"Activations: {activations}")
    print()

    # Load data once (same for all experiments)
    data = load_data(
        mode="4class",
        split="fault_size_all_loads",
        seed=config.DEFAULT_SEED,
        verbose=False
    )

    for activation in activations:
        print_separator()
        print(f"Testing: {activation}")
        print_separator()

        accuracies = []

        for seed in seeds:
            set_seed(seed)

            # Create model with specified activation
            model = create_model(
                "cnn",
                num_classes=config.NUM_CLASSES["4class"],
                dropout=config.DROPOUT,
                activation=activation,
            ).to(device)

            # Create data loaders
            train_loader, val_loader, test_loader = create_dataloaders(data, "4class")

            # Train
            train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader if len(val_loader.dataset) > 0 else None,
                mode="4class",
                epochs=config.EPOCHS,
                device=device,
                verbose=False,
            )

            # Evaluate
            metrics = evaluate(model, test_loader, device, "4class")
            accuracies.append(metrics["accuracy"])
            print(f"  {activation:12} seed={seed}: {metrics['accuracy']:.4f}")

        # Compute statistics
        mean_acc = sum(accuracies) / len(accuracies)
        std_acc = (sum((a - mean_acc) ** 2 for a in accuracies) / len(accuracies)) ** 0.5
        results.append((activation, mean_acc * 100, std_acc * 100))
        print(f"  {activation:12} MEAN: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
        print()

    # Print summary
    print_separator()
    print("SUMMARY")
    print_separator()
    print(f"{'Activation':<12} {'Accuracy (%)':<15} {'Std':<10}")
    print_separator(char="-", width=40)

    for activation, mean, std in sorted(results, key=lambda x: -x[1]):
        print(f"{activation:<12} {mean:<15.2f} {std:<10.2f}")

    best = max(results, key=lambda x: x[1])
    print()
    print(f"Best: {best[0]} ({best[1]:.2f}% ± {best[2]:.2f}%)")


if __name__ == "__main__":
    main()
