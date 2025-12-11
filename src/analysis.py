import os
import json
from collections import defaultdict

import numpy as np

import config
from experiment import list_experiments, load_experiment, parse_experiment_name


def load_study_results(study_name: str) -> list:
    """
    Load all experiment results from a study.
    
    Returns:
        List of result dictionaries
    """
    experiments = list_experiments(study_name)
    results = []

    for exp_dir in experiments:
        exp_data = load_experiment(exp_dir)
        if "results" in exp_data:
            parsed = parse_experiment_name(exp_dir)
            exp_data["results"]["_parsed"] = parsed
            results.append(exp_data["results"])

    return results


def aggregate_results(results: list) -> dict:
    """
    Aggregate results by model and configuration.
    
    Returns:
        Nested dictionary: {config_name: {model: {metric: [values]}}}
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for r in results:
        parsed = r.get("_parsed", {})
        model = parsed.get("model") or r.get("model")
        config_name = parsed.get("config") or f"{r.get('mode')}_{r.get('split')}"

        # Extract metrics
        test_metrics = r.get("test_metrics", {})

        if "accuracy" in test_metrics:
            aggregated[config_name][model]["accuracy"].append(test_metrics["accuracy"])

            # Per-class recall
            for cls, recall in test_metrics.get("per_class_recall", {}).items():
                aggregated[config_name][model][f"recall_{cls}"].append(recall)

        if "auroc" in test_metrics:
            aurocs = test_metrics["auroc"]
            aggregated[config_name][model]["macro_auroc"].append(aurocs.get("macro", np.nan))

            for cls in config.CLASS_NAMES["multilabel"]:
                if cls in aurocs:
                    aggregated[config_name][model][f"auroc_{cls}"].append(aurocs[cls])

    return aggregated


def compute_statistics(aggregated: dict) -> dict:
    """
    Compute mean +- std for aggregated results.
    
    Returns:
        {config_name: {model: {metric: {"mean": x, "std": y, "n": n}}}}
    """
    stats = {}

    for config_name, models in aggregated.items():
        stats[config_name] = {}

        for model, metrics in models.items():
            stats[config_name][model] = {}

            for metric, values in metrics.items():
                values = [v for v in values if not np.isnan(v)]
                if values:
                    stats[config_name][model][metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "n": len(values),
                    }

    return stats


def print_summary_table(study_name: str, stats: dict = None):
    """Print formatted summary table for a study."""
    if stats is None:
        results = load_study_results(study_name)
        aggregated = aggregate_results(results)
        stats = compute_statistics(aggregated)

    print(f"STUDY: {study_name}")

    for config_name, models in sorted(stats.items()):
        print(f"\n{config_name}:")

        # Determine which metrics to show
        sample_model = list(models.values())[0] if models else {}
        has_auroc = "macro_auroc" in sample_model
        has_accuracy = "accuracy" in sample_model

        if has_accuracy:
            # Multiclass table
            print(f"  {'Model':<12} {'Accuracy':<18} {'Normal':<12} {'Ball':<12} {'IR':<12} {'OR':<12}")

            for model in ["cnn", "lstm", "cnnlstm"]:
                if model not in models:
                    continue

                m = models[model]
                acc = m.get("accuracy", {})
                acc_str = f"{acc.get('mean', 0):.3f} ± {acc.get('std', 0):.3f}" if acc else "-"

                recalls = []
                for cls in ["Normal", "Ball", "IR", "OR"]:
                    r = m.get(f"recall_{cls}", {})
                    if r:
                        recalls.append(f"{r.get('mean', 0):.3f}")
                    else:
                        recalls.append("-")

                print(f"  {model.upper():<12} {acc_str:<18} {recalls[0]:<12} {recalls[1]:<12} {recalls[2]:<12} {recalls[3]:<12}")

        elif has_auroc:
            # Multilabel table
            print(f"  {'Model':<12} {'Macro AUROC':<18} {'Ball':<15} {'IR':<15} {'OR':<15}")

            for model in ["cnn", "lstm", "cnnlstm"]:
                if model not in models:
                    continue

                m = models[model]
                macro = m.get("macro_auroc", {})
                macro_str = f"{macro.get('mean', 0):.3f} ± {macro.get('std', 0):.3f}" if macro else "-"

                aurocs = []
                for cls in ["Ball", "IR", "OR"]:
                    a = m.get(f"auroc_{cls}", {})
                    if a:
                        aurocs.append(f"{a.get('mean', 0):.3f} ± {a.get('std', 0):.3f}")
                    else:
                        aurocs.append("-")

                print(f"  {model.upper():<12} {macro_str:<18} {aurocs[0]:<15} {aurocs[1]:<15} {aurocs[2]:<15}")

    print("\n" + "=" * 90)


def export_results(study_name: str, output_path: str = None):
    """Export study results to JSON."""
    if output_path is None:
        output_path = os.path.join(config.RESULTS_DIR, study_name, "summary.json")

    results = load_study_results(study_name)
    aggregated = aggregate_results(results)
    stats = compute_statistics(aggregated)

    summary = {
        "study_name": study_name,
        "n_experiments": len(results),
        "statistics": stats,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Exported to: {output_path}")

    return summary


def export_csv(study_name: str, output_path: str = None) -> str:
    """
    Export study results to CSV.
    
    Args:
        study_name: Name of the study
        output_path: Output file path (optional)
        
    Returns:
        CSV string
    """
    results = load_study_results(study_name)
    aggregated = aggregate_results(results)
    stats = compute_statistics(aggregated)

    if output_path is None:
        output_path = os.path.join(config.RESULTS_DIR, study_name, "results.csv")

    lines = []

    for config_name, models in sorted(stats.items()):
        sample = list(models.values())[0]
        has_accuracy = "accuracy" in sample

        if has_accuracy:
            lines.append("config,model,accuracy_mean,accuracy_std,recall_Normal,recall_Ball,recall_IR,recall_OR")
            for model in ["cnn", "lstm", "cnnlstm"]:
                if model not in models:
                    continue
                m = models[model]
                acc = m.get("accuracy", {})
                recalls = [m.get(f"recall_{c}", {}).get("mean", "") for c in ["Normal", "Ball", "IR", "OR"]]
                lines.append(f"{config_name},{model},{acc.get('mean', '')},{acc.get('std', '')},{','.join(map(str, recalls))}")
        else:
            lines.append("config,model,macro_auroc_mean,macro_auroc_std,auroc_Ball,auroc_IR,auroc_OR")
            for model in ["cnn", "lstm", "cnnlstm"]:
                if model not in models:
                    continue
                m = models[model]
                macro = m.get("macro_auroc", {})
                aurocs = [m.get(f"auroc_{c}", {}).get("mean", "") for c in ["Ball", "IR", "OR"]]
                lines.append(f"{config_name},{model},{macro.get('mean', '')},{macro.get('std', '')},{','.join(map(str, aurocs))}")

    csv_content = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(csv_content)

    print(f"Exported CSV to: {output_path}")
    return csv_content


def generate_typst_table(study_name: str, config_filter: str = None) -> str:
    """
    Generate Typst table for publication.
    
    Args:
        study_name: Name of the study
        config_filter: Only include configurations matching this pattern
        
    Returns:
        Typst table string
    """
    results = load_study_results(study_name)
    aggregated = aggregate_results(results)
    stats = compute_statistics(aggregated)

    lines = []
    lines.append("#figure(")
    lines.append("  table(")
    lines.append("    columns: 5,")
    lines.append("    align: (left, center, center, center, center),")
    lines.append("    [Model], [Accuracy], [Ball], [IR], [OR],")

    for config_name, models in sorted(stats.items()):
        if config_filter and config_filter not in config_name:
            continue

        # Section header
        lines.append(f"    table.cell(colspan: 5)[_{config_name.replace('_', ' ')}_],")

        for model in ["cnn", "lstm", "cnnlstm"]:
            if model not in models:
                continue

            m = models[model]

            if "accuracy" in m:
                acc = m["accuracy"]
                acc_str = f"${acc['mean']:.3f} plus.minus {acc['std']:.3f}$"

                recalls = []
                for cls in ["Ball", "IR", "OR"]:
                    r = m.get(f"recall_{cls}", {})
                    if r:
                        recalls.append(f"${r['mean']:.3f}$")
                    else:
                        recalls.append("--")

                lines.append(f"    [{model.upper()}], [{acc_str}], [{recalls[0]}], [{recalls[1]}], [{recalls[2]}],")

            elif "macro_auroc" in m:
                macro = m["macro_auroc"]
                macro_str = f"${macro['mean']:.3f} plus.minus {macro['std']:.3f}$"

                aurocs = []
                for cls in ["Ball", "IR", "OR"]:
                    a = m.get(f"auroc_{cls}", {})
                    if a:
                        aurocs.append(f"${a['mean']:.3f}$")
                    else:
                        aurocs.append("--")

                lines.append(f"    [{model.upper()}], [{macro_str}], [{aurocs[0]}], [{aurocs[1]}], [{aurocs[2]}],")

    lines.append("  ),")
    lines.append("  caption: [Results],")
    lines.append(")")

    return "\n".join(lines)


def analyze(study_name: str):
    """
    Full analysis of a study: print summary and export.
    """
    print_summary_table(study_name)
    export_results(study_name)
