"""Compare saved modelB, MobileNetV2, and EfficientNetB0 results."""

import argparse
import csv
import json
from pathlib import Path


MODEL_DIRS = {
    "modelB": ("modelB",),
    "MobileNetV2": (("mobilenetv2",), ("mobile", "mobilenetv2")),
    "EfficientNetB0": (("efficientnetb0",), ("mobile", "efficientnetb0")),
}
METRICS = [
    ("Total parameters", "total_parameters", ",.0f"),
    ("Trainable parameters", "trainable_parameters", ",.0f"),
    ("Model size (MB)", "model_size_mb", ".2f"),
    ("Model size (KB)", "model_file_size_kb", ",.1f"),
    ("Float32 weight size (MB)", "float32_weight_size_mb", ".2f"),
    ("MACs per inference", "macs_per_inference", ",.0f"),
    ("Average epoch time (s)", "average_epoch_time_seconds", ".2f"),
    ("Median epoch time (s)", "median_epoch_time_seconds", ".2f"),
    ("Test loss", "test_loss", ".4f"),
    ("Test accuracy", "test_accuracy", ".4f"),
    ("Test precision (macro)", "test_precision_macro", ".4f"),
    ("Test recall (macro)", "test_recall_macro", ".4f"),
    ("Test F1 (macro)", "test_f1_macro", ".4f"),
    ("Test F1 (weighted)", "test_f1_weighted", ".4f"),
]


def load_metrics(results_dir):
    runs = {}
    for label, locations in MODEL_DIRS.items():
        if label == "modelB":
            model_dir = results_dir / locations[0]
            paths = sorted(model_dir.glob("*/metrics.json"))
        else:
            paths = [
                results_dir.joinpath(*location) / "metrics.json"
                for location in locations
            ]
        for metrics_path in paths:
            if not metrics_path.is_file():
                continue
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                print(f"Skipping {metrics_path}: {error}")
                continue
            run_name = (
                f"modelB_{metrics_path.parent.name}"
                if label == "modelB"
                else label
            )
            runs[run_name] = {
                "model": label,
                "metrics_path": str(metrics_path),
                **metrics,
            }
    return runs


def metric_value(metrics, key):
    value = metrics.get(key)
    if value is None and key == "trainable_parameters":
        value = metrics.get("trainable_parameters_phase2")
    if value is None and key == "model_size_mb":
        size_kb = metrics.get("model_file_size_kb")
        if size_kb is not None:
            return size_kb / 1024
    if value is None and key == "model_file_size_kb":
        size_mb = metrics.get("model_size_mb")
        if size_mb is not None:
            return size_mb * 1024
    if value is None and key == "float32_weight_size_mb":
        size_kb = metrics.get("float32_weight_size_kb")
        if size_kb is not None:
            return size_kb / 1024
    return value


def format_value(value, format_spec):
    return "N/A" if value is None else format(value, format_spec)


def build_markdown(runs):
    if not runs:
        return "# modelB and SOTA Comparison\n\nNo saved results found.\n"
    lines = [
        "# modelB and Lightweight SOTA Comparison",
        "",
        "Each column is a saved run. Missing models or metrics are shown as `N/A`.",
        "",
        "| Metric | " + " | ".join(runs) + " |",
        "|---|" + "---:|" * len(runs),
    ]
    for label, key, format_spec in METRICS:
        values = [
            format_value(metric_value(runs[run], key), format_spec)
            for run in runs
        ]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def build_csv_rows(runs):
    rows = []
    for run_name, metrics in runs.items():
        row = {
            "run": run_name,
            "model": metrics["model"],
            "metrics_path": metrics["metrics_path"],
        }
        for _, key, _ in METRICS:
            row[key] = metric_value(metrics, key)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare saved resource-constrained modelB with MobileNetV2 and EfficientNetB0."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_metrics(results_dir)
    markdown_path = output_dir / "comparison_modelB_sota.md"
    csv_path = output_dir / "comparison_modelB_sota.csv"
    json_path = output_dir / "comparison_modelB_sota.json"
    markdown_path.write_text(build_markdown(runs), encoding="utf-8")
    fieldnames = ["run", "model", "metrics_path"] + [key for _, key, _ in METRICS]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(build_csv_rows(runs))
    json_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"Comparison report saved to: {markdown_path}")
    print(f"CSV data saved to: {csv_path}")
    print(f"JSON data saved to: {json_path}")


if __name__ == "__main__":
    main()
