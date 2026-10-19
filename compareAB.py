"""Compare every saved modelA and modelB optimizer run."""

import argparse
import csv
import json
from pathlib import Path


METRICS = [
    ("Total parameters", "total_parameters", ",.0f"),
    ("Trainable parameters", "trainable_parameters", ",.0f"),
    ("Estimated model size on disk (KB)", "model_file_size_kb", ",.1f"),
    ("Float32 weight size (KB)", "float32_weight_size_kb", ",.1f"),
    ("MACs per inference", "macs_per_inference", ",.0f"),
    ("Training time per epoch (s)", "average_epoch_time_seconds", ".2f"),
    ("Median epoch time (s)", "median_epoch_time_seconds", ".2f"),
    ("Test loss", "test_loss", ".4f"),
    ("Test accuracy", "test_accuracy", ".4f"),
    ("Test precision (macro)", "test_precision_macro", ".4f"),
    ("Test recall (macro)", "test_recall_macro", ".4f"),
    ("Test F1 (macro)", "test_f1_macro", ".4f"),
    ("Test F1 (weighted)", "test_f1_weighted", ".4f"),
    ("Best validation loss", "best_validation_loss", ".4f"),
    ("Best validation accuracy", "best_validation_accuracy", ".4f"),
]


def load_results(results_dir):
    """Load each saved run from results/modelA/<optimizer> and modelB/<optimizer>."""
    loaded = {}
    for model_name in ("modelA", "modelB"):
        model_dir = results_dir / model_name
        if not model_dir.is_dir():
            continue
        for metrics_path in sorted(model_dir.glob("*/metrics.json")):
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                print(f"Skipping {metrics_path}: {error}")
                continue
            optimizer = metrics_path.parent.name
            run_name = f"{model_name}_{optimizer}"
            loaded[run_name] = {
                "model": model_name,
                "optimizer_key": optimizer,
                "optimizer": metrics.get("optimizer", optimizer),
                "metrics_path": str(metrics_path),
                **metrics,
            }
    return loaded


def format_value(value, format_spec):
    if value is None:
        return "N/A"
    return format(value, format_spec)


def build_markdown(results):
    lines = [
        "# Saved Model Comparison",
        "",
        "Each column is one saved model and optimizer run. Missing values are shown as `N/A`.",
        "",
        "| Metric | " + " | ".join(results) + " |",
        "|---|" + "---:|" * len(results),
    ]

    for label, key, format_spec in METRICS:
        values = [
            format_value(results[run_name].get(key), format_spec)
            for run_name in results
        ]
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    if not results:
        return "# Saved Model Comparison\n\nNo optimizer results were found.\n"
    return "\n".join(lines).rstrip() + "\n"


def build_csv_rows(results):
    rows = []
    for run_name, run in results.items():
        row = {
            "run": run_name,
            "model": run["model"],
            "optimizer_key": run["optimizer_key"],
            "optimizer": run.get("optimizer", run["optimizer_key"]),
            "metrics_path": run["metrics_path"],
        }
        row.update({key: run.get(key) for _, key, _ in METRICS})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare every saved modelA and modelB optimizer run."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="directory containing modelA and modelB result folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for generated comparison files; defaults to results directory",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(results_dir)

    markdown_path = output_dir / "comparison_table_AB.md"
    csv_path = output_dir / "comparison_table_AB.csv"
    json_path = output_dir / "comparison_table_AB.json"

    markdown_path.write_text(build_markdown(results), encoding="utf-8")
    rows = build_csv_rows(results)
    fieldnames = ["run", "model", "optimizer_key", "optimizer", "metrics_path"] + [
        key for _, key, _ in METRICS
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Comparison report saved to: {markdown_path}")
    print(f"CSV data saved to: {csv_path}")
    print(f"JSON data saved to: {json_path}")


if __name__ == "__main__":
    main()
