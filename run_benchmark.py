#!/usr/bin/env python
"""
KANalogue benchmark — reproduces the comparison-table results from the paper.

Runs 6 configurations (3 datasets × 2 basis dimensions) with seed 42.

Configurations (matching Table 1 / tab:comparison_architectures):

    MNIST           [196, 64, 10]          2-dim: 26,570  params
    MNIST           [196, 64, 10]          3-dim: 39,754  params
    FashionMNIST    [784, 256, 10]         2-dim: 407,306 params
    FashionMNIST    [784, 256, 10]         3-dim: 610,570 params
    CIFAR-10        [3072, 1024, 256, 10]  2-dim: 6,824,714  params
    CIFAR-10        [3072, 1024, 256, 10]  3-dim: 10,235,146 params

Usage:
    python run_benchmark.py [--device cuda:0] [--dry-run]
"""

import os
import sys
import json
import time
import argparse
import subprocess

# ---------------------------------------------------------------------------
# Paper configurations
# ---------------------------------------------------------------------------

BENCHMARK_CONFIGS = [
    # === MNIST ===
    {
        "name": "MNIST_2dim",
        "dataset": "MNIST",
        "hidden": [64],                              # input(196) → 64 → 10
        "td_basis_types": ["Z15", "ZAZ_21"],        # 2-dim basis
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
        "hidden": [64],                              # input(196) → 64 → 10
        "td_basis_types": ["Z21", "Z15", "ZAZ_21"],  # 3-dim basis
        "batch_size": 256,
        "learning_rate": 0.001,
        "acti": "None",
        "norm_layer": "layer",
        "fit_mode": "univariate",
        "basis_type": "pos-larger",
        "neg_input": False,
    },
    # === FashionMNIST ===
    {
        "name": "FMNIST_2dim",
        "dataset": "FMNIST",
        "hidden": [256],                             # input(784) → 256 → 10
        "td_basis_types": ["Z15", "ZAZ_21"],        # 2-dim basis
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
        "hidden": [256],                             # input(784) → 256 → 10
        "td_basis_types": ["Z21", "Z15", "ZAZ_21"],  # 3-dim basis
        "batch_size": 256,
        "learning_rate": 0.001,
        "acti": "None",
        "norm_layer": "layer",
        "fit_mode": "univariate",
        "basis_type": "pos-larger",
        "neg_input": False,
    },
    # === CIFAR-10 ===
    {
        "name": "cifar10_2dim",
        "dataset": "cifar10",
        "hidden": [1024, 256],                       # input(3072) → 1024 → 256 → 10
        "td_basis_types": ["Z21", "ZAZ_21"],        # 2-dim basis
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
        "hidden": [1024, 256],                       # input(3072) → 1024 → 256 → 10
        "td_basis_types": ["A21", "Z21", "ZAZ_21"],  # 3-dim basis
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
SEED = 42
MAX_EPOCHS = 500
PATIENCE = 15
EXP_NAME = "paper_benchmark"


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
        "--exp_name", f"{EXP_NAME}/{cfg['name']}",
        "--td_basis_types"] + cfg["td_basis_types"] + [
        "--seed", str(seed),
        "--search_mode", "custom",
        "--model_type", "tdkan",
    ]


def run_one(cfg: dict, device: str, seed: int, dry_run: bool = False):
    """Run a single training configuration."""
    cmd = build_cmd(cfg, device, seed)

    print("\n" + "=" * 72)
    print(f"  {cfg['name']}: {cfg['dataset']}  hidden={cfg['hidden']}  "
          f"basis={cfg['td_basis_types']}  lr={cfg['learning_rate']}  "
          f"batch={cfg['batch_size']}  seed={seed}")
    print("=" * 72)

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return {"config": cfg["name"], "status": "dry-run"}

    start = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.time() - start
        print(f"\n  {cfg['name']}: COMPLETED in {elapsed/60:.1f} min")
        return {"config": cfg["name"], "status": "ok", "elapsed_min": elapsed/60}
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"\n  {cfg['name']}: FAILED after {elapsed/60:.1f} min (exit {e.returncode})")
        return {"config": cfg["name"], "status": "failed", "elapsed_min": elapsed/60}


def main():
    parser = argparse.ArgumentParser(description="KANalogue paper benchmark")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Which configs to run (e.g. MNIST_2dim MNIST_3dim)")
    args = parser.parse_args()

    configs = BENCHMARK_CONFIGS
    if args.configs:
        configs = [c for c in configs if c["name"] in args.configs]

    print(f"Running {len(configs)} / {len(BENCHMARK_CONFIGS)} configs")
    print(f"Device: {args.device}, Seed: {SEED}, Max epochs: {MAX_EPOCHS}, "
          f"Patience: {PATIENCE}")
    print(f"Experiment: {EXP_NAME}")

    results = []
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Starting {cfg['name']}...")
        r = run_one(cfg, args.device, SEED, dry_run=args.dry_run)
        results.append(r)

    # Print summary
    print("\n" + "=" * 72)
    print("  BENCHMARK SUMMARY")
    print("=" * 72)
    for r in results:
        status_icon = "✓" if r["status"] == "ok" else ("✗" if r["status"] == "failed" else "○")
        elapsed = f"  ({r.get('elapsed_min', 0):.1f} min)" if "elapsed_min" in r else ""
        print(f"  {status_icon} {r['config']:20s}  {r['status']}{elapsed}")

    # Save summary JSON
    summary_path = f"results/{EXP_NAME}/benchmark_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "seed": SEED,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "device": args.device,
            "results": results,
        }, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
