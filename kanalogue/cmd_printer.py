import sys

def print_epoch_status(epoch, train_loss, val_loss, val_acc, tag_path, file_name, best_loss, first=False):
    """Dynamically refresh three-line training status in terminal."""
    lines = [
        f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}",
        f"Appended activation stats for epoch {epoch} to",
        f"{tag_path}",
        f"{file_name}",
        f"New best model saved (val_loss={best_loss:.4f})"
    ]

    # If not first round, move up 5 lines and clear (overwrite old content)
    if not first:
        sys.stdout.write("\033[F\033[K" * 5)
        sys.stdout.flush()

    # Output new 5 lines
    for line in lines:
        print(line)
    sys.stdout.flush()
