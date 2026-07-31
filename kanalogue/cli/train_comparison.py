#!/usr/bin/env python
"""Training CLI for comparison models (MLP-RTD, MLP-CMTD, KAN-BSpline, KAN-Gottlieb)."""

import os, re, sys, time, math, json, random, argparse
from collections import defaultdict

import numpy as np
import torch, torch.nn as nn, torch.optim as optim

from kanalogue.comparison_models import build_comparison_model, count_parameters
from kanalogue.loader import get_dataloaders
from kanalogue.train import write_to_csv


def validate_comparison(model, test_loader, criterion, device):
    """Validate without noise_std (comparison models don't support it)."""
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return total_loss / len(test_loader), correct / len(test_loader.dataset)


def parse_args():
    p = argparse.ArgumentParser(description="Train comparison models")
    p.add_argument("--dataset", default="MNIST", choices=["MNIST","FMNIST","cifar10"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--hidden", type=int, nargs="+", default=[64])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--model_type", default="mlp-rtd",
                   choices=["mlp-rtd","mlp-cmtd","kan-bspline","kan-gottlieb"])
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--exp_name", default="comparison_test")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--grid_size", type=int, default=2)
    p.add_argument("--spline_order", type=int, default=3)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--neg_input", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # CUDA device
    m = re.match(r"cuda:(\d+)", args.device)
    if m:
        os.environ["CUDA_VISIBLE_DEVICES"] = m.group(1)
    device = args.device if args.device == "cpu" else "cuda:0"

    dataset = args.dataset
    hidden_dims = args.hidden
    parent_folder = f"results/new_structure_exps/{args.exp_name}"

    # Determine input dim
    if dataset == "MNIST":
        resize, input_dim = (14, 14), 196
    elif dataset == "FMNIST":
        resize, input_dim = (28, 28), 784
    else:
        resize, input_dim = (3, 32, 32), 3072

    # Data
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        dataset_name=dataset, batch_size=args.batch_size, resize=resize,
        pixel_range_neg=args.neg_input,
    )

    # Model
    kwargs = {}
    if args.model_type == "kan-bspline":
        kwargs = {"grid_size": args.grid_size, "spline_order": args.spline_order}
    elif args.model_type == "kan-gottlieb":
        kwargs = {"degree": args.degree}

    model = build_comparison_model(args.model_type, input_dim, hidden_dims, 10, **kwargs).to(device)
    n_params = count_parameters(model)
    print(f"\n[{args.model_type}] {dataset}  hidden={hidden_dims}  "
          f"lr={args.learning_rate}  batch={args.batch_size}  seed={args.seed}")
    print(f"  Params: {n_params}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state_dict = None
    trigger_times = 0
    title = f"{args.model_type}_{dataset}"

    for epoch in range(args.max_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Flatten for MLP/KAN
            inputs = inputs.view(inputs.size(0), -1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        val_loss, val_acc = validate_comparison(model, val_loader, criterion, device)

        if epoch % 5 == 0 or epoch < 5:
            print(f"  Epoch {epoch:03d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restore best weights and evaluate
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    _, test_acc = validate_comparison(model, test_loader, criterion, device)

    # Save results
    results_dir = f"{parent_folder}/{dataset}"
    os.makedirs(f"{results_dir}/configs", exist_ok=True)
    os.makedirs(f"{results_dir}/weights", exist_ok=True)

    best_config = {
        "model_type": args.model_type, "dataset": dataset,
        "HIDDEN_DIMS": hidden_dims, "learning_rate": args.learning_rate,
        "batch_size": args.batch_size, "test_acc": test_acc,
        "parameters_number": n_params, "seed": args.seed,
        "max_epochs": args.max_epochs, "patience": args.patience,
    }
    with open(f"{results_dir}/configs/{title}_best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    torch.save(model.state_dict(), f"{results_dir}/weights/best.pth")

    print(f"\n  Best val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f} "
          f"({test_acc*100:.2f}%)")
    print(f"  Results saved to {results_dir}")


if __name__ == "__main__":
    main()
