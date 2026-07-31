#!/usr/bin/env python
"""
Comparison model benchmark — MLP-RTD, MLP-CMTD, KAN-BSpline, KAN-Gottlieb.

Runs all 12 configs (4 models × 3 datasets) with seed=42 then 10 random seeds.
Records results to results/comparison_summary.json.
"""

import os, sys, json, time, random, re, argparse, subprocess

# Paper configurations
BENCHMARK = [
    # ===== MNIST [196, 64, 10] =====
    {"name": "MLP-RTD_MNIST",      "model": "mlp-rtd",
     "dataset": "MNIST",    "hidden": [64],     "batch_size": 512,
     "lr": 0.01,    "params": 13258,     "arch": "[196, 64, 10]", "target": 97.17},
    {"name": "MLP-CMTD_MNIST",     "model": "mlp-cmtd",
     "dataset": "MNIST",    "hidden": [64],     "batch_size": 512,
     "lr": 0.0003,  "params": 13258,     "arch": "[196, 64, 10]", "target": 96.73},
    {"name": "KAN-BSpline_MNIST",  "model": "kan-bspline",
     "dataset": "MNIST",    "hidden": [64],     "batch_size": 256,
     "lr": 0.0004,  "params": 92288,     "arch": "[196, 64, 10]", "target": 97.68,
     "grid_size": 3, "spline_order": 3},
    {"name": "KAN-Gottlieb_MNIST", "model": "kan-gottlieb",
     "dataset": "MNIST",    "hidden": [64],     "batch_size": 1024,
     "lr": 0.00005, "params": 52866,     "arch": "[196, 64, 10]", "target": 93.15,
     "degree": 3},

    # ===== FashionMNIST [784, 256, 10] =====
    {"name": "MLP-RTD_FMNIST",      "model": "mlp-rtd",
     "dataset": "FMNIST",   "hidden": [256],    "batch_size": 256,
     "lr": 0.003,   "params": 203530,    "arch": "[784, 256, 10]", "target": 88.26},
    {"name": "MLP-CMTD_FMNIST",     "model": "mlp-cmtd",
     "dataset": "FMNIST",   "hidden": [256],    "batch_size": 256,
     "lr": 0.0001,  "params": 203530,    "arch": "[784, 256, 10]", "target": 86.68},
    {"name": "KAN-BSpline_FMNIST",  "model": "kan-bspline",
     "dataset": "FMNIST",   "hidden": [256],    "batch_size": 256,
     "lr": 0.0001,  "params": 1219584,   "arch": "[784, 256, 10]", "target": 89.19,
     "grid_size": 2, "spline_order": 3},
    {"name": "KAN-Gottlieb_FMNIST", "model": "kan-gottlieb",
     "dataset": "FMNIST",   "hidden": [256],    "batch_size": 1024,
     "lr": 0.0005,  "params": 813570,    "arch": "[784, 256, 10]", "target": 87.62,
     "degree": 3},

    # ===== CIFAR-10 [3072, 1024, 256, 10] =====
    {"name": "MLP-RTD_cifar10",      "model": "mlp-rtd",
     "dataset": "cifar10",  "hidden": [1024, 256], "batch_size": 1024,
     "lr": 0.0005,  "params": 3411722,   "arch": "[3072, 1024, 256, 10]", "target": 47.35},
    {"name": "MLP-CMTD_cifar10",     "model": "mlp-cmtd",
     "dataset": "cifar10",  "hidden": [1024, 256], "batch_size": 512,
     "lr": 0.0001,  "params": 3411722,   "arch": "[3072, 1024, 256, 10]", "target": 38.99},
    {"name": "KAN-BSpline_cifar10",  "model": "kan-bspline",
     "dataset": "cifar10",  "hidden": [1024, 256], "batch_size": 1024,
     "lr": 0.0005,  "params": 20462592,  "arch": "[3072, 1024, 256, 10]", "target": 58.99,
     "grid_size": 2, "spline_order": 3},
    {"name": "KAN-Gottlieb_cifar10", "model": "kan-gottlieb",
     "dataset": "cifar10",  "hidden": [1024, 256], "batch_size": 1024,
     "lr": 0.0003,  "params": 13644291,  "arch": "[3072, 1024, 256, 10]", "target": 49.30,
     "degree": 3},
]

MAX_EPOCHS = 500
PATIENCE = 15

TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "kanalogue", "cli", "train_comparison.py")


def build_cmd(cfg, device, seed):
    cmd = [
        sys.executable, "-m", "kanalogue.cli.train_comparison",
        "--dataset", cfg["dataset"],
        "--device", device,
        "--hidden"] + [str(h) for h in cfg["hidden"]] + [
        "--batch_size", str(cfg["batch_size"]),
        "--learning_rate", str(cfg["lr"]),
        "--model_type", cfg["model"],
        "--max_epochs", str(MAX_EPOCHS),
        "--patience", str(PATIENCE),
        "--exp_name", f"comparison_models/{cfg['name']}/seed_{seed}",
        "--seed", str(seed),
    ]
    # Add model-specific kwargs
    for k in ["grid_size", "spline_order", "degree"]:
        if k in cfg:
            cmd += [f"--{k}", str(cfg[k])]
    return cmd


def read_accuracy(name, seed):
    base = f"results/new_structure_exps/comparison_models/{name}/seed_{seed}"
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.endswith("_best_config.json"):
                with open(os.path.join(root, f)) as fh:
                    return json.load(fh).get("test_acc", 0.0) * 100
    return None


def main():
    p = argparse.ArgumentParser(description="Comparison model benchmark")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--n-seeds", type=int, default=10,
                   help="Number of random seeds after seed=42")
    p.add_argument("--seed42-only", action="store_true",
                   help="Only run seed=42, skip random seeds")
    args = p.parse_args()

    # Seed list: 42 first, then N random seeds
    random.seed(0)
    seeds = [42] + random.sample(range(1, 10000), args.n_seeds)
    if args.seed42_only:
        seeds = [42]

    total = len(BENCHMARK) * len(seeds)
    print(f"Comparison benchmark: {len(BENCHMARK)} configs × {len(seeds)} seeds = {total} runs")

    results = []
    run_count = 0
    for cfg in BENCHMARK:
        for seed in seeds:
            run_count += 1
            print(f"\n[{run_count}/{total}] {cfg['name']} seed={seed}  ", end="", flush=True)
            if args.dry_run:
                cmd = build_cmd(cfg, args.device, seed)
                print(f"[DRY-RUN] {' '.join(cmd[:8])}...")
                results.append({"config": cfg["name"], "model": cfg["model"],
                                "dataset": cfg["dataset"], "seed": seed,
                                "accuracy": None, "architecture": cfg["arch"],
                                "parameters_number": cfg["params"]})
                continue

            start = time.time()
            try:
                subprocess.run(build_cmd(cfg, args.device, seed),
                             check=True, capture_output=True)
                acc = read_accuracy(cfg["name"], seed)
                elapsed = (time.time() - start) / 60
                print(f"acc={acc:.2f}%  ({elapsed:.1f} min)" if acc else f"NO RESULT ({elapsed:.1f} min)")
            except subprocess.CalledProcessError as e:
                elapsed = (time.time() - start) / 60
                acc = read_accuracy(cfg["name"], seed)
                print(f"FAILED ({elapsed:.1f} min)" +
                      (f"  partial_acc={acc:.2f}%" if acc else ""))

            results.append({
                "config": cfg["name"], "model": cfg["model"],
                "dataset": cfg["dataset"], "seed": seed,
                "accuracy": round(acc, 2) if acc else None,
                "architecture": cfg["arch"],
                "parameters_number": cfg["params"],
            })

            # Incremental save
            summary = {
                "description": "Comparison model benchmark (MLP-RTD, MLP-CMTD, KAN-BSpline, KAN-Gottlieb)",
                "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "device": args.device,
                "total_runs": total, "completed_runs": len([r for r in results if r["accuracy"] is not None]),
                "results": results,
            }
            with open("results/comparison_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    # Final stats
    print("\n===== COMPARISON BENCHMARK COMPLETE =====")
    for cfg in BENCHMARK:
        accs = [r["accuracy"] for r in results if r["config"] == cfg["name"] and r["accuracy"] is not None]
        if accs:
            print(f"  {cfg['name']:25s}  n={len(accs)}  mean={sum(accs)/len(accs):.2f}%  "
                  f"min={min(accs):.2f}%  max={max(accs):.2f}%  target={cfg['target']}%")


if __name__ == "__main__":
    main()
