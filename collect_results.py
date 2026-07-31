#!/usr/bin/env python
"""
Collect benchmark results and produce a summary table.

Reads results from results/paper_benchmark/ and produces:
    1. A console summary table
    2. A Markdown table file (benchmark_results.md)
    3. A CSV file (benchmark_results.csv)
"""

import os
import json
import pandas as pd


BENCHMARK_DIR = "results/paper_benchmark"

CONFIG_NAMES = [
    "MNIST_2dim", "MNIST_3dim",
    "FMNIST_2dim", "FMNIST_3dim",
    "cifar10_2dim", "cifar10_3dim",
]

EXPECTED = {
    "MNIST_2dim":    {"dataset": "MNIST",    "hidden": [196, 64, 10],
                      "params": 26570,  "target_acc": 97.71},
    "MNIST_3dim":    {"dataset": "MNIST",    "hidden": [196, 64, 10],
                      "params": 39754,  "target_acc": 97.35},
    "FMNIST_2dim":   {"dataset": "FMNIST",   "hidden": [784, 256, 10],
                      "params": 407306, "target_acc": 88.82},
    "FMNIST_3dim":   {"dataset": "FMNIST",   "hidden": [784, 256, 10],
                      "params": 610570, "target_acc": 88.44},
    "cifar10_2dim":  {"dataset": "cifar10",  "hidden": [3072, 1024, 256, 10],
                      "params": 6824714, "target_acc": 47.74},
    "cifar10_3dim":  {"dataset": "cifar10",  "hidden": [3072, 1024, 256, 10],
                      "params": 10235146, "target_acc": 49.69},
}


def find_best_accuracy(config_name: str) -> dict:
    """Locate the best-config JSON for a benchmark run and extract key metrics."""
    config_dir = os.path.join(BENCHMARK_DIR, config_name, "univariate")
    if not os.path.isdir(config_dir):
        return None

    # Walk subdirectories to find the dataset folder
    for ds in os.listdir(config_dir):
        ds_dir = os.path.join(config_dir, ds)
        if not os.path.isdir(ds_dir):
            continue
        config_file = os.path.join(ds_dir, "configs")
        if not os.path.isdir(config_file):
            continue
        # Find any *_best_config.json
        for f in sorted(os.listdir(config_file)):
            if f.endswith("_best_config.json"):
                path = os.path.join(config_file, f)
                with open(path) as fh:
                    data = json.load(fh)
                return {
                    "config_name": config_name,
                    "dataset": data.get("dataset", ds),
                    "test_acc": data.get("test_acc", 0.0) * 100,
                    "noise_std": data.get("noise_std", 0.0),
                    "learning_rate": data.get("learning_rate"),
                    "hidden_dims": data.get("HIDDEN_DIMS"),
                    "batch_size": data.get("batch_size"),
                    "norm_layer": data.get("norm_layer"),
                    "td_basis": data.get("td_basis_types"),
                    "config_path": path,
                }
    return None


def main():
    rows = []
    for name in CONFIG_NAMES:
        info = find_best_accuracy(name)
        exp = EXPECTED.get(name, {})
        if info:
            info["expected_acc"] = exp.get("target_acc", None)
            info["expected_params"] = exp.get("params", None)
            rows.append(info)
            status = "✓" if info["test_acc"] > 0 else "?"
            print(f"  {status} {name:20s}  test_acc={info['test_acc']:.2f}%  "
                  f"(target: {exp.get('target_acc', '?')}%)")
        else:
            print(f"  ✗ {name:20s}  NO RESULTS YET")
            rows.append({"config_name": name, "dataset": exp.get("dataset", "?"),
                         "test_acc": None})

    if not rows:
        print("No results found.")
        return

    df = pd.DataFrame(rows)

    # --- CSV ---
    csv_path = os.path.join(BENCHMARK_DIR, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")

    # --- Markdown ---
    md_path = os.path.join(BENCHMARK_DIR, "benchmark_results.md")
    with open(md_path, "w") as f:
        f.write("# KANalogue Benchmark Results\n\n")
        f.write("| Config | Dataset | Test Acc (%) | Expected (%) | Params |\n")
        f.write("|--------|---------|-------------|-------------|--------|\n")
        for _, r in df.iterrows():
            acc_str = f"{r['test_acc']:.2f}" if r['test_acc'] is not None else "—"
            f.write(f"| {r['config_name']} | {r['dataset']} | "
                    f"{acc_str} | {r.get('expected_acc', '—')} | "
                    f"{r.get('expected_params', '—')} |\n")
    print(f"Markdown saved to {md_path}")


if __name__ == "__main__":
    main()
