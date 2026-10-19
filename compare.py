import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results"
A = json.loads((ROOT / "modelA" / "metrics.json").read_text())
B = json.loads((ROOT / "modelB" / "metrics.json").read_text())

rows = [
    ("Trainable parameters", f"{A['trainable_parameters']:,}", f"{B['trainable_parameters']:,}"),
    ("Total parameters", f"{A['total_parameters']:,}", f"{B['total_parameters']:,}"),
    ("MACs per inference", f"{A['macs_per_inference']:,}", f"{B['macs_per_inference']:,}"),
    ("Float32 weight size (KB)", f"{A['float32_weight_size_kb']:.1f}", f"{B['float32_weight_size_kb']:.1f}"),
    ("Model file on disk (KB)", f"{A['model_file_size_kb']:.1f}", f"{B['model_file_size_kb']:.1f}"),
    ("Training time / epoch (s)", f"{A['average_epoch_time_seconds']:.2f}", f"{B['average_epoch_time_seconds']:.2f}"),
    ("Test accuracy", f"{A['test_accuracy']:.4f}", f"{B['test_accuracy']:.4f}"),
    ("Test precision (macro)", f"{A['test_precision_macro']:.4f}", f"{B['test_precision_macro']:.4f}"),
    ("Test recall (macro)", f"{A['test_recall_macro']:.4f}", f"{B['test_recall_macro']:.4f}"),
    ("Test F1 (macro)", f"{A['test_f1_macro']:.4f}", f"{B['test_f1_macro']:.4f}"),
]

lines = ["| Metric | Model A (Standard) | Model B (Depthwise Separable) |",
         "|---|---|---|"]
lines += [f"| {m} | {a} | {b} |" for m, a, b in rows]

ratio = A["trainable_parameters"] / B["trainable_parameters"]
mac_ratio = A["macs_per_inference"] / B["macs_per_inference"]
lines += ["", f"Parameter reduction: {ratio:.1f}x | MAC reduction: {mac_ratio:.1f}x"]

table = "\n".join(lines)
print(table)
(ROOT / "comparison_table.md").write_text(table, encoding="utf-8")
print(f"\nSaved to {ROOT / 'comparison_table.md'}")