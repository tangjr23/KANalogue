#!/usr/bin/env python
"""
KANalogue multi-seed benchmark — 6 configs × 10 random seeds on GPU.

Runs each paper configuration 10 times with different random seeds,
collects all accuracy results, and writes a clean summary JSON.

Output: results/benchmark_summary.json
"""

import os
import sys
import json
import time
import random
import argparse
import subprocess

# ---------------------------------------------------------------------------
# Paper configurations (same as run_benchmark.py)
# ---------------------------------------------------------------------------
BENCHMARK_CONFIGS = [
    {
        "name": "MNIST_2dim",
        "dataset": "MNIST",
        "hidden": [64],
        "td_basis_types": ["Z15", "ZAZ_21"],
        "architecture": "[196, 64, 10]",
        "parameters_number": 26570,
        "batch_size": 256,
        "learning_rate": 0.001,
        "acti": "None",
        "norm_layer": "layer",
        "fit_mode": "univariate",
        "basis_type": "pos-larger",
        "neg_input": False,
    },
    {
        "name": "MNIST_3dim",
        "dataset": "MNIST",
        "hidden": [64],
        "td_basis_types": ["Z21", "Z15", "ZAZ_21"],
        "architecture": "[196, 64, 10]",
        "parameters_number": 39754,
        "batch_size": 256,
        "learning_rate": 0.001,
        "acti": "None",
        "norm_layer": "layer",
        "fit_mode": "univariate",
        "basis_type": "pos-larger",
        "neg_input": False,
    },
    {
        "name": "FMNIST_2dim",
        "dataset": "FMNIST",
        "hidden": [256],
        "td_basis_types": ["Z15", "ZAZ_21"],
        "architecture": "[784, 256, 10]",
        "parameters_number": 407306,
        "batch_size": 256,
        "learning_rate": 0.001,
        "acti": "None",
        "norm_layer": "layer",
        "fit_mode": "univariate",
        "basis_type": "pos-larger",
        "neg_input": False,
    },
    {
        "name": "FMNIST_3dim",
        "dataset": "FMNIST",
        "hidden": [256],
        "td_basis_types": ["Z21", "Z15", "ZAZ_21"],
        "architecture": "[784, 256, 10]",
        "parameters_number": 610570,
        "batch_size": 256,
        "learning_rate": 0.001,
        "acti": "None",
        "norm_layer": "layer",
        "fit_mode": "univariate",
        "basis_type": "pos-larger",
        "neg_input": False,
    },
    {
        "name": "cifar10_2dim",
        "dataset": "cifar10",
        "hidden": [1024, 256],
        "td_basis_types": ["Z21", "ZAZ_21"],
        "architecture": "[3072, 1024, 256, 10]",
        "parameters_number": 6824714,
        "batch_size": 512,
        "learning_rate": 0.001,
        "acti": "None",
        "norm_layer": "layer",
        "fit_mode": "univariate",
        "basis_type": "pos-larger",
        "neg_input": False,
    },
    {
        "name": "cifar10_3dim",
        "dataset": "cifar10",
        "hidden": [1024, 256],
        "td_basis_types": ["A21", "Z21", "ZAZ_21"],
        "architecture": "[3072, 1024, 256, 10]",
        "parameters_number": 10235146,
        "batch_size": 512,
        "learning_rate": 0.001,
        "acti": "None",
        "norm_layer": "layer",
        "fit_mode": "univariate",
        "basis_type": "pos-larger",
        "neg_input": False,
    },
]

# Shared settings
N_SEEDS = 10
MAX_EPOCHS = 500
PATIENCE = 15
BASE_EXP_NAME = "multiseed_benchmark"


def build_cmd(cfg: dict, device: str, seed: int) -> list:
    """Build the command-line argument list for ``kanalogue train``."""
    return [
        sys.executable, "-m", "kanalogue.cli.train",
        "--dataset", cfg["dataset"],
        "--device", device,
        "--hidden"] + [str(h) for h in cfg["hidden"]] + [
        "--batch_size", str(cfg["batch_size"]),
        "--learning_rate", str(cfg["learning_rate"]),
        "--acti", cfg["acti"],
        "--norm_layer", cfg["norm_layer"],
        "--fit_mode", cfg["fit_mode"],
        "--basis_type", cfg["basis_type"],
        "--max_epochs", str(MAX_EPOCHS),
        "--patience", str(PATIENCE),
        "--exp_name", f"{BASE_EXP_NAME}/{cfg['name']}/seed_{seed}",
        "--td_basis_types"] + cfg["td_basis_types"] + [
        "--seed", str(seed),
        "--search_mode", "custom",
        "--model_type", "tdkan",
    ]


def read_best_accuracy(config_name: str, seed: int) -> float:
    """Parse the best_config.json and return test accuracy (0.0--1.0)."""
    config_dir = (
        f"results/new_structure_exps/{BASE_EXP_NAME}/{config_name}/"
        f"seed_{seed}"
    )
    for root, dirs, files in os.walk(config_dir):
        for f in files:
            if f.endswith("_best_config.json"):
                path = os.path.join(root, f)
                with open(path) as fh:
                    data = json.load(fh)
                return data.get("test_acc", 0.0)
    return 0.0


def run_one_seed(cfg: dict, device: str, seed: int) -> dict:
    """Run one seed of one config and return the result dict."""
    cmd = build_cmd(cfg, device, seed)

    print(f"\n  [{cfg['name']}] seed={seed}  ", end="", flush=True)

    start = time.time()
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        elapsed = (time.time() - start) / 60.0
        acc = read_best_accuracy(cfg["name"], seed)
        print(f"acc={acc*100:.2f}%  ({elapsed:.1f} min)")
        return {
            "config": cfg["name"],
            "seed": seed,
            "accuracy": round(acc * 100, 2),
            "architecture": cfg["architecture"],
            "parameters_number": cfg["parameters_number"],
        }
    except subprocess.CalledProcessError as e:
        elapsed = (time.time() - start) / 60.0
        print(f"FAILED ({elapsed:.1f} min)  exit={e.returncode}")
        # Try to read accuracy anyway (might have partial results)
        acc = read_best_accuracy(cfg["name"], seed)
        return {
            "config": cfg["name"],
            "seed": seed,
            "accuracy": round(acc * 100, 2) if acc > 0 else None,
            "architecture": cfg["architecture"],
            "parameters_number": cfg["parameters_number"],
        }


def main():
    parser = argparse.ArgumentParser(description="KANalogue multi-seed benchmark")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Which configs to run (e.g. MNIST_2dim MNIST_3dim)")
    args = parser.parse_args()

    configs = BENCHMARK_CONFIGS
    if args.configs:
        configs = [c for c in configs if c["name"] in args.configs]

    total_runs = len(configs) * N_SEEDS
    print(f"Multi-seed benchmark: {len(configs)} configs × {N_SEEDS} seeds = {total_runs} runs")
    print(f"Device: {args.device}")

    # Generate seeds deterministically but varied (not 42)
    random.seed(0)
    seed_pool = random.sample(range(1, 10000), N_SEEDS)

    results = []
    run_count = 0
    for cfg in configs:
        cfg_results = []
        for seed in seed_pool:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}]", end="")
            if args.dry_run:
                cmd = build_cmd(cfg, args.device, seed)
                print(f"  [DRY-RUN] {' '.join(cmd[:8])}...")
                cfg_results.append({
                    "config": cfg["name"],
                    "seed": seed,
                    "accuracy": None,
                    "architecture": cfg["architecture"],
                    "parameters_number": cfg["parameters_number"],
                })
            else:
                r = run_one_seed(cfg, args.device, seed)
                cfg_results.append(r)
        results.extend(cfg_results)

    # Build the summary
    summary = {
        "description": "KANalogue multi-seed benchmark (10 random seeds per config)",
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "device": args.device,
        "total_runs": total_runs,
        "results": results,
    }

    # Compute per-config statistics
    for cfg in configs:
        cfg_accs = [r["accuracy"] for r in results
                    if r["config"] == cfg["name"] and r["accuracy"] is not None]
        if cfg_accs:
            cfg_accs_sorted = sorted(cfg_accs)
            n = len(cfg_accs_sorted)
            print(f"\n  {cfg['name']}: mean={sum(cfg_accs)/n:.2f}%  "
                  f"min={min(cfg_accs):.2f}%  max={max(cfg_accs):.2f}%  "
                  f"median={cfg_accs_sorted[n//2]:.2f}%  n={n}/{N_SEEDS}")

    # Save
    summary_path = "results/benchmark_summary.json"
    os.makedirs("results", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
