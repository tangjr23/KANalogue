import torch
import torch.nn as nn

import pandas as pd
import os

from utils.model import TDiode_KANLayer

def record_activation_stats(name, module, inp, out, activation_stats, decimals=4):
    """
    记录每层输出和输入统计信息。
    module: 当前层对象
    inp: forward 的输入 (tuple)
    out: forward 输出 Tensor
    """
    # 输出 y
    y = out
    if not torch.is_tensor(y):
        return
    y_detached = y.detach()

    # 输入 x (取第一个元素)
    x = inp[0] if isinstance(inp, tuple) else inp
    if not torch.is_tensor(x):
        x = None
    else:
        x = x.detach()

    # 保存到 activation_stats
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
    """为指定类型的子层注册 forward hook"""
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
    将所有 epoch 的激活统计信息写入同一个 CSV 文件中。
    每个 epoch 之间通过 'Epoch_X' 标记隔开。
    """
    if len(activation_stats) == 0:
        print(f"[Warning] No activation stats to save for epoch {epoch}.")
        return

    # 构造 DataFrame
    df_epoch = pd.DataFrame.from_dict(activation_stats, orient='index')
    df_epoch.insert(0, "Epoch", f"Epoch_{epoch}")  # 添加 Epoch 标识列

    # 文件路径
    os.makedirs(f"{save_path}/{tag}/train_process", exist_ok=True)
    csv_file = f"{save_path}/{tag}/train_process/{title}_all_epochs.csv"

    # 检查文件是否存在
    file_exists = os.path.isfile(csv_file)

    # 以追加模式写入（每次一个 epoch）
    with open(csv_file, "a", newline='') as f:
        if not file_exists:
            # 首次写入，包含表头
            df_epoch.to_csv(f, index_label="Layer", header=True)
        else:
            # 追加写入，不重复表头
            f.write(f"\n# === Epoch {epoch} ===\n")
            df_epoch.to_csv(f, index_label="Layer", header=False)
