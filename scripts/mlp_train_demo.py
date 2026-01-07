from utils.imports import *
import pandas as pd
from utils.arguments import MLP_KAN_train_parse_arguments
# ===== 你已有的工具函数（假设已实现） =====
# from utils.data import get_dataloaders
# from utils.model_factory import build_model
# from utils.train_utils import validate, train_with_early_stopping_epoch
# from utils.csv_utils import write_to_csv

# ---------------------------
# 简单 Early Stopping 训练
# ---------------------------
def train_with_early_stopping_epoch(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    max_epochs=30,
    patience=10,
    min_delta=1e-4,
):
    best_val_loss = float("inf")
    trigger_times = 0

    for epoch in range(max_epochs):
        # ---- train ----
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # ---- validate ----
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

        # ---- early stopping ----
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = model.state_dict()
            best_epoch = epoch
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    return best_epoch + 1, val_loss, val_acc, train_loss, train_acc


# ---------------------------
# 验证 / 测试
# ---------------------------
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


# ---------------------------
# CSV 行准备（MLP-only）
# ---------------------------
def prepare_csv_row(
    config,
    total_param,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    test_acc,
    epochs_used,
    duration_sec,
):
    return {
        "model_name": config["model_name"],
        "learning_rate": config["learning_rate"],
        "hidden_dims": "-".join(map(str, config["HIDDEN_DIMS"])),
        "batch_size": config["batch_size"],
        # "TD_basis": None,
        # "noise_std": None,
        "total_parameter": total_param,
        "train_loss": f"{train_loss:.4f}",
        "train_acc": f"{train_acc:.4f}",
        "val_loss": f"{val_loss:.4f}",
        "val_acc": f"{val_acc:.4f}",
        "test_acc": f"{test_acc:.4f}",
        "epochs_used": epochs_used,
        "train_time": f"{duration_sec:.2f}",
    }


# ---------------------------
# Grid Search / Benchmark
# ---------------------------
def grid_search_train(device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"

    results = []
    csv_rows = []

    # ====== MLP-only 配置 ======
    param_grid = {
        "model_name": ["BSplineKAN"],
        "learning_rate": [1e-4],
        "HIDDEN_DIMS": [[64]],
        "batch_size": [32],
    }

    def generate_configs(grid):
        keys = grid.keys()
        values = grid.values()
        for v in itertools.product(*values):
            yield dict(zip(keys, v))

    # ====== 数据 ======
    for config in generate_configs(param_grid):
        train_loader, val_loader, test_loader, input_dim = get_dataloaders(
            dataset_name="MNIST",
            batch_size=config["batch_size"],
            resize=(14, 14),
        )


        model = build_MLP_KAN_model(
            model_name=config["model_name"],
            input_dim=input_dim,
            hidden_dims=config["HIDDEN_DIMS"],
        ).to(device)
        print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")

        total_param = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        optimizer = optim.Adam(
            model.parameters(), lr=config["learning_rate"]
        )
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()

        epochs_used, val_loss, val_acc, train_loss, train_acc = (
            train_with_early_stopping_epoch(
                model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                device,
            )
        )

        duration_sec = time.time() - start_time
        _, test_acc = validate(model, test_loader, criterion, device)

        print(
            f"[DONE] test_acc={test_acc:.4f}, "
            f"val_acc={val_acc:.4f}, "
            f"time={duration_sec:.2f}s"
        )

        csv_rows.append(
            prepare_csv_row(
                config,
                total_param,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                test_acc,
                epochs_used,
                duration_sec,
            )
        )

        results.append((config, test_acc))

    df = pd.DataFrame(csv_rows)
    os.makedirs("results/mlp", exist_ok=True)
    df.to_csv("results/mlp/results.csv", index=False)

    return results


# ---------------------------
# main
# ---------------------------
if __name__ == "__main__":
    import itertools
    # import os
    # print(">>> before import torch, CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    # import torch
    # print(">>> after import torch, CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

    # print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    # print("torch device count =", torch.cuda.device_count())
    # print("current device =", torch.cuda.current_device())
    # print("device name =", torch.cuda.get_device_name(0))

    grid_search_train(device="cuda")
