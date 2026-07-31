import torch
import torch.nn as nn

import pandas as pd
import os

from kanalogue.model import TDiode_KANLayer

def record_activation_stats(name, module, inp, out, activation_stats, decimals=4):
    """
    Record per-layer output and input statistics.
    module: current layer object
    inp: forward input (tuple)
    out: forward output Tensor
    """
    # Output y
    y = out
    if not torch.is_tensor(y):
        return
    y_detached = y.detach()

    # Input x (take first element)
    x = inp[0] if isinstance(inp, tuple) else inp
    if not torch.is_tensor(x):
        x = None
    else:
        x = x.detach()

    # Save to activation_stats
    stats = {
        'y_min': round(float(y_detached.min().cpu()), decimals),
        'y_max': round(float(y_detached.max().cpu()), decimals),
        'y_mean': round(float(y_detached.mean().cpu()), decimals),
    }

    if x is not None:
        stats.update({
            'x_min': round(float(x.min().cpu()), decimals),
            'x_max': round(float(x.max().cpu()), decimals),
            'x_mean': round(float(x.mean().cpu()), decimals),
        })

    activation_stats[name] = stats

def attach_activation_hooks(model,
                            layer_types=(nn.Linear, nn.Conv2d, nn.LayerNorm, TDiode_KANLayer),
                            activation_stats=None):
    """Register forward hooks for specified layer types."""
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, layer_types):
            hook = layer.register_forward_hook(lambda m, inp, out, n=name:
                                               record_activation_stats(n, m, inp, out,
                                                                       activation_stats=activation_stats))
            hooks.append(hook)
    return hooks

def clear_activation_stats(activation_stats):
    activation_stats.clear()

def save_activation_stats(epoch, title, tag="MNIST", save_path="results", activation_stats=None):
    """
    Write all epoch activation statistics into a single CSV file.
    Each epoch is separated by an 'Epoch_X' marker.
    """
    if len(activation_stats) == 0:
        print(f"[Warning] No activation stats to save for epoch {epoch}.")
        return

    # Build DataFrame
    df_epoch = pd.DataFrame.from_dict(activation_stats, orient='index')
    df_epoch.insert(0, "Epoch", f"Epoch_{epoch}")  # Add Epoch identifier column

    # File path
    os.makedirs(f"{save_path}/{tag}/train_process", exist_ok=True)
    csv_file = f"{save_path}/{tag}/train_process/{title}_all_epochs.csv"

    # Check if file exists
    file_exists = os.path.isfile(csv_file)

    # Append mode (one epoch at a time)
    with open(csv_file, "a", newline='') as f:
        if not file_exists:
            # First write, include header
            df_epoch.to_csv(f, index_label="Layer", header=True)
        else:
            # Append, don't repeat header
            f.write(f"\n# === Epoch {epoch} ===\n")
            df_epoch.to_csv(f, index_label="Layer", header=False)
