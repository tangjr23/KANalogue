import sys

def print_epoch_status(epoch, train_loss, val_loss, val_acc, tag_path, file_name, best_loss, first=False):
    """动态刷新终端中的三行训练状态"""
    lines = [
        f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}",
        f"Appended activation stats for epoch {epoch} to",
        f"{tag_path}", 
        f"{file_name}", 
        f"New best model saved (val_loss={best_loss:.4f})"
    ]

    # 若非首轮输出，则向上移动5行并清空（覆盖旧内容）
    if not first:
        sys.stdout.write("\033[F\033[K" * 5)
        sys.stdout.flush()

    # 输出新5行
    for line in lines:
        print(line)
    sys.stdout.flush()
