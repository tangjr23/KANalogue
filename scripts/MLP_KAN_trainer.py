from utils.imports import *
import pandas as pd
from utils.arguments import MLP_KAN_train_parse_arguments

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
def grid_search_train(args):
# def search_train_mlp(args):
    """
    Unified MLP training with grid / random search + early stopping
    """
    device = args.device if torch.cuda.is_available() else "cpu"
    dataset = args.dataset
    model_name = args.model_name

    max_epochs = args.max_epochs
    patience = args.patience

    parent_folder = "results/mlp"
    if args.exp_name:
        parent_folder = os.path.join(parent_folder, args.exp_name)
    os.makedirs(parent_folder, exist_ok=True)

    # --------------------------------------------------
    # 1. Build search space
    # --------------------------------------------------
    if dataset == "MNIST":
        param_grid = {
            "learning_rate": [args.learning_rate],
            "HIDDEN_DIMS": [[64], [256], [128, 16]],
            "batch_size": [32],
        }
        resize = (14, 14)

    elif dataset == "FMNIST":
        param_grid = {
            "learning_rate": [args.learning_rate],
            "HIDDEN_DIMS": [[512, 128], [256, 64], [128, 64, 16]],
            "batch_size": [32],
        }
        resize = (28, 28)

    elif dataset == "cifar10":
        param_grid = {
            "learning_rate": [args.learning_rate],
            "HIDDEN_DIMS": [[1024], [1024, 256], [1024, 256, 64]],
            "batch_size": [args.batch_size],
        }
        resize = (3, 32, 32)

    # generate configs
    def generate_configs(grid):
        keys = grid.keys()
        values = grid.values()
        for v in itertools.product(*values):
            yield dict(zip(keys, v))

    if args.search_mode == "random":
        lr_range = (1e-4, 5e-3) if dataset != "cifar10" else (5e-5, 2e-3)
        batch_choices = [32, 64, 128] if dataset != "cifar10" else [256, 512]

        def random_config():
            return {
                "learning_rate": 10 ** random.uniform(
                    math.log10(lr_range[0]), math.log10(lr_range[1])
                ),
                "HIDDEN_DIMS": args.hidden,
                "batch_size": random.choice(batch_choices),
            }

        configs = [random_config() for _ in range(20)]
    else:
        configs = list(generate_configs(param_grid))

    # --------------------------------------------------
    # 2. Search loop
    # --------------------------------------------------
    best_accuracy = 0.0
    best_config = None
    results = []
    csv_rows = []

    for i, config in enumerate(configs):
        print(f"\n===== Trial {i+1}/{len(configs)} =====")

        # ---- data ----
        train_loader, val_loader, test_loader, input_dim = get_dataloaders(
            dataset_name=dataset,
            batch_size=config["batch_size"],
            resize=resize,
        )

        # ---- model ----
        model = build_MLP_KAN_model(
            model_name=model_name,
            input_dim=input_dim,
            hidden_dims=config["HIDDEN_DIMS"],
        ).to(device)

        total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(
            f"Model={model_name} | "
            f"LR={config['learning_rate']:.5f}, "
            f"HIDDEN={config['HIDDEN_DIMS']}, "
            f"Batch={config['batch_size']} | "
            f"Params={total_param}"
        )

        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        criterion = nn.CrossEntropyLoss()

        # ---- train ----
        start_time = time.time()

        epochs_used, val_loss, val_acc, train_loss, train_acc = (
            train_with_early_stopping_epoch(
                model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                device,
                max_epochs=max_epochs,
                patience=patience,
            )
        )

        duration_sec = time.time() - start_time

        # ---- test ----
        _, test_acc = validate(model, test_loader, criterion, device)

        print(
            f"[DONE] test_acc={test_acc:.4f}, "
            f"val_acc={val_acc:.4f}, "
            f"time={duration_sec:.2f}s"
        )

        # ---- log ----
        row = prepare_csv_row(
            config={**config, "model_name": model_name},
            total_param=total_param,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            test_acc=test_acc,
            epochs_used=epochs_used,
            duration_sec=duration_sec,
        )
        csv_rows.append(row)
        results.append((config, test_acc))

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_config = config

    # --------------------------------------------------
    # 3. Save results
    # --------------------------------------------------
    df = pd.DataFrame(csv_rows)
    save_dir = os.path.join(parent_folder, dataset)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, f"{model_name}.csv"), index=False)

    print("\n====== Best Result ======")
    print(f"Best Config: {best_config}")
    print(f"Best Accuracy: {best_accuracy:.4f}")

    return results, best_config

def main():
    parser = MLP_KAN_train_parse_arguments()
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        args.device = "cuda"
    else:
        args.device = "cpu"

    print("\n========== Training Config ==========")
    print(f"Dataset     : {args.dataset}")
    print(f"Model       : {args.model_name}")
    print(f"Search Mode : {args.search_mode}")
    print(f"Hidden Dims : {args.hidden}")
    print(f"Device      : {args.device}")
    print("=====================================\n")

    grid_search_train(args)


if __name__ == "__main__":
    main()

