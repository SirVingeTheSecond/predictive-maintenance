import argparse
import os
import sys

import config


def cmd_run(args):
    """Run a study."""
    from runner import run_study

    if args.study not in config.STUDIES:
        print(f"Error: Unknown study '{args.study}'")
        print(f"Available: {list(config.STUDIES.keys())}")
        sys.exit(1)

    run_study(args.study, force=args.force, verbose=not args.quiet)


def cmd_analyze(args):
    """Analyze results from a given study."""
    from analysis import analyze

    analyze(args.study)


def cmd_export(args):
    """Export results to file."""
    from analysis import export_csv, generate_typst_table, export_results

    if args.format == "csv":
        export_csv(args.study)
    elif args.format == "typst":
        typst_content = generate_typst_table(args.study)
        output_path = os.path.join(config.RESULTS_DIR, args.study, "results.typ")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(typst_content)
        print(f"Exported Typst to: {output_path}")
    elif args.format == "json":
        export_results(args.study)
    else:
        print(f"Unknown format: {args.format}")


def cmd_figures(args):
    """Generate figures for study."""
    from analysis import load_study_results, aggregate_results, compute_statistics
    from visualization import generate_all_figures

    results = load_study_results(args.study)
    if not results:
        print(f"Error: No results found for study '{args.study}'")
        sys.exit(1)

    aggregated = aggregate_results(results)
    stats = compute_statistics(aggregated)
    generate_all_figures(args.study, stats)


def cmd_sweep(args):
    """Run hyperparameter search."""
    from sweep import run_sweep

    if args.sweep not in config.SWEEPS:
        print(f"Error: Unknown sweep '{args.sweep}'")
        print(f"Available: {list(config.SWEEPS.keys())}")
        sys.exit(1)

    run_sweep(args.sweep, seed=args.seed, verbose=not args.quiet)


def cmd_sweep_results(args):
    """Show sweep results."""
    from sweep import print_sweep_leaderboard

    print_sweep_leaderboard(args.sweep, top_n=args.top)


def cmd_list(args):
    """List available studies and sweeps."""
    print("\nAvailable Studies:")

    for name, study in config.STUDIES.items():
        desc = study.get("description", "")
        n_models = len(study.get("models", []))
        n_configs = len(study.get("configurations", []))
        n_seeds = len(study.get("seeds", []))
        total = n_models * n_configs * n_seeds

        print(f"\n  {name}")
        print(f"    {desc}")
        print(f"    {n_models} models x {n_configs} configs x {n_seeds} seeds = {total} experiments")

    print("\n\nAvailable Sweeps:")

    for name, sweep in config.SWEEPS.items():
        desc = sweep.get("description", "")
        param_grid = sweep.get("param_grid", {})
        n_combos = 1
        for values in param_grid.values():
            n_combos *= len(values)

        print(f"\n  {name}")
        print(f"    {desc}")
        print(f"    {n_combos} combinations")


def cmd_status(args):
    """Show study completion status."""
    from experiment import list_experiments, get_experiment_dir, experiment_exists

    study = config.STUDIES.get(args.study)
    if study is None:
        print(f"Error: Unknown study '{args.study}'")
        sys.exit(1)

    models = study["models"]
    configurations = study["configurations"]
    seeds = study["seeds"]

    total = len(models) * len(configurations) * len(seeds)
    completed = len(list_experiments(args.study))

    print(f"\nStudy: {args.study}")
    print(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

    for cfg in configurations:
        config_name = cfg["name"]
        print(f"\n  {config_name}:")

        for model in models:
            done = 0
            for seed in seeds:
                exp_dir = get_experiment_dir(args.study, model, config_name, seed)
                if experiment_exists(exp_dir):
                    done += 1

            status = "DONE" if done == len(seeds) else f"{done}/{len(seeds)}"
            print(f"    {model:<10} {status}")


def cmd_train(args):
    """Run single experiment."""
    from runner import run_single_experiment

    run_single_experiment(
        study_name="manual",
        model_name=args.model,
        mode=args.mode,
        split=args.split,
        config_name=f"{args.mode}_{args.split}",
        seed=args.seed,
        epochs=args.epochs,
        force=args.force,
        verbose=True,
    )


def cmd_test(args):
    """Evaluate a saved checkpoint."""
    from test import test_checkpoint, compare_splits

    if args.compare_splits:
        compare_splits(args.checkpoint)
    else:
        test_checkpoint(args.checkpoint, split=args.split, verbose=True)

# This might seem pretty overwhelming, oh well...
def main():
    parser = argparse.ArgumentParser(
        description="CWRU Bearing Fault Diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run
    p_run = subparsers.add_parser("run", help="Run a predefined study")
    p_run.add_argument("study", help="Study name")
    p_run.add_argument("--force", action="store_true", help="Overwrite existing results")
    p_run.add_argument("--quiet", action="store_true", help="Minimal output")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze study results")
    p_analyze.add_argument("study", help="Study name")

    # figures
    p_figures = subparsers.add_parser("figures", help="Generate figures")
    p_figures.add_argument("study", help="Study name")

    # list
    subparsers.add_parser("list", help="List available studies and sweeps")

    # status
    p_status = subparsers.add_parser("status", help="Show study status")
    p_status.add_argument("study", help="Study name")

    # train
    p_train = subparsers.add_parser("train", help="Run single experiment")
    p_train.add_argument("--model", default="cnn", choices=["cnn", "lstm", "cnnlstm"])
    p_train.add_argument("--mode", default="4class", choices=["4class", "10class", "multilabel"])
    p_train.add_argument("--split", default="fault_size_all_loads",
                         choices=["random", "fault_size", "fault_size_all_loads", "cross_load"])
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--force", action="store_true", help="Overwrite existing results")

    # test
    p_test = subparsers.add_parser("test", help="Evaluate saved checkpoint")
    p_test.add_argument("checkpoint", help="Path to model.pth")
    p_test.add_argument("--split", default=None,
                        choices=["random", "fault_size", "fault_size_all_loads", "cross_load"],
                        help="Override split strategy (uses original if not specified)")
    p_test.add_argument("--compare-splits", action="store_true",
                        help="Evaluate checkpoint on all split strategies")

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Run hyperparameter sweep")
    p_sweep.add_argument("sweep", help="Sweep name")
    p_sweep.add_argument("--seed", type=int, default=42)
    p_sweep.add_argument("--quiet", action="store_true", help="Minimal output")

    # sweep-results
    p_sweep_results = subparsers.add_parser("sweep-results", help="Show sweep leaderboard")
    p_sweep_results.add_argument("sweep", help="Sweep name")
    p_sweep_results.add_argument("--top", type=int, default=10, help="Number of results to show")

    # export
    p_export = subparsers.add_parser("export", help="Export results to file")
    p_export.add_argument("study", help="Study name")
    p_export.add_argument("--format", default="csv", choices=["csv", "typst", "json"],
                          help="Output format")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "run": cmd_run,
        "analyze": cmd_analyze,
        "figures": cmd_figures,
        "list": cmd_list,
        "status": cmd_status,
        "train": cmd_train,
        "test": cmd_test,
        "sweep": cmd_sweep,
        "sweep-results": cmd_sweep_results,
        "export": cmd_export,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
