import torch
import torch.optim as optim
import os
import copy
import csv
import pandas as pd
import itertools

def train(model, train_loader, criterion, optimizer, device):
    model.train()  # 设置为训练模式，启用 dropout、BN 等
    total_loss = 0  # 累加损失
    correct = 0     # 累加正确预测数量
    global_step = 0  # 全局步数计数器
    
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 数据和标签转移到 GPU/CPU

        # 针对使用 LBFGS 优化器的情况（少见）
        if isinstance(optimizer, optim.LBFGS):
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                return loss
            loss = optimizer.step(closure)  # 特殊调用方式
        else:
            optimizer.zero_grad()            # 梯度清零
            output = model(data)             # 前向传播
            loss = criterion(output, target) # 计算损失
            loss.backward()                  # 反向传播
            optimizer.step()                 # 更新参数
            loss = loss.item()               # 获取标量 loss 值（避免 tensor 累加）

        total_loss += loss  # 累加 batch 损失

        global_step += 1    # 增加全局步数计数器

        # 获取预测类别
        pred = output.argmax(dim=1, keepdim=True)
        # 统计正确预测数量
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    # 返回平均损失和准确率
    return total_loss / len(train_loader), correct / len(train_loader.dataset)


def validate(model, test_loader, criterion, device, 
             noise_std=0.0, noise_distri='binary'):
    model.eval()  # 设置为评估模式，禁用 dropout、BN
    total_loss = 0
    correct = 0

    with torch.no_grad():  # 禁用梯度计算（节省内存与加速）
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 移动数据到设备
            output = model(data, noise_std=noise_std, 
                           noise_mode=noise_distri)            # 前向传播
            loss = criterion(output, target)                   # 计算损失
            total_loss += loss.item()                          # 累加损失

            pred = output.argmax(dim=1, keepdim=True)          # 获取预测
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计正确数量

    # 返回平均损失和准确率
    return total_loss / len(test_loader), correct / len(test_loader.dataset)

# 定义带 Early Stopping 的训练过程
def train_with_early_stopping_epoch(model, train_loader, val_loader, optimizer, criterion, device,
                                    max_epochs, patience=5, min_delta=1e-3, noise_std=0.0):
    """    
    Args:
        model (nn.Module): 要训练的模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        optimizer (torch.optim.Optimizer): 优化器（如 Adam）
        criterion (nn.Module): 损失函数（如 CrossEntropyLoss）
        device (str): 设备（'cuda' 或 'cpu'）
        max_epochs (int): 最大训练轮数
        patience (int): Early Stopping 的容忍轮数
        min_delta (float): 验证损失改进的最小阈值，小于该值认为未改进

    Returns:
        Tuple[int, float, float]: 实际训练轮数、最佳验证损失、最佳验证准确率
    """
    best_loss = float('inf')        # 当前最佳验证损失
    best_accuracy = 0.0             # 当前最佳验证准确率
    counter = 0                     # 没有提升的 epoch 计数器
    best_weights = None             # 最佳模型参数（用于恢复）

    model.to(device)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(max_epochs):
        # === 训练 ===
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)

        # === 验证 ===
        val_loss, val_accuracy = validate(model, val_loader, criterion, device, noise_std=noise_std)

        # 保存日志
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)

        # 如果验证损失有显著下降，保存当前模型
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_accuracy = val_accuracy
            counter = 0  # 重置容忍计数器
            best_weights = copy.deepcopy(model.state_dict())
        else:
            counter += 1  # 无明显改善，增加容忍计数器

        # 如果容忍计数器超过设定值，则提前停止训练
        if counter >= patience:
            print(f"Early stopped at epoch {epoch+1} | Best Val Loss: {best_loss:.4f}")
            break

    # 恢复到性能最好的模型参数
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # 返回训练轮数、最优验证损失和验证准确率
    return epoch + 1, best_loss, best_accuracy, train_loss, train_acc

def train_with_noise(model, train_loader, val_loader, optimizer, criterion, device,
                        max_epochs, patience=10, min_delta=1e-4, noise_std=0.0):

    best_loss = float('inf')        # 当前最佳验证损失
    best_accuracy = 0.0             # 当前最佳验证准确率
    counter = 0                     # 没有提升的 epoch 计数器
    best_weights = None             # 最佳模型参数（用于恢复）

    model.to(device)

    for epoch in range(max_epochs):
        # === 训练 ===
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)

        # === 验证 ===
        val_loss, val_accuracy = validate(model, val_loader, criterion, device, noise_std=noise_std)

        # 如果验证损失有显著下降，保存当前模型
        if best_loss - val_loss > min_delta:
            best_loss = val_loss
            best_accuracy = val_accuracy
            counter = 0  # 重置容忍计数器
            best_weights = copy.deepcopy(model.state_dict())
        else:
            counter += 1  # 无明显改善，增加容忍计数器

        # 如果容忍计数器超过设定值，则提前停止训练
        if counter >= patience:
            print(f"Early stopped at epoch {epoch+1} | Best Val Loss: {best_loss:.4f}")
            break

    # 恢复到性能最好的模型参数
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # 返回训练轮数、最优验证损失和验证准确率
    return epoch + 1, train_loss, train_acc, val_loss, val_accuracy

def prepare_csv_data(config, 
                     noise_std, 
                     total_parameter, 
                     train_loss, train_acc, val_loss, val_acc, 
                     test_acc, epochs_used, duration_sec):
    """
    准备一条用于保存到 CSV 的训练结果数据记录。

    Args:
        config (dict): 当前训练配置，包括学习率、隐藏层结构等
        total_parameter (int): 模型的总可训练参数数量
        acc (float): 测试集准确率
        val_loss (float): 验证集损失
        val_acc (float): 验证集准确率
        epochs_used (int): 实际训练的轮数（考虑 Early Stopping）
        duration_sec (float): 总训练时间（单位：秒）

    Returns:
        dict: 一条 CSV 数据行，字段名对应写入 CSV 文件的表头
    """
    return {
        # 'model_name': config['models_types'],
        'learning_rate': config['learning_rate'],
        'hidden_dims': '-'.join(map(str, config['HIDDEN_DIMS'])),   # 将隐藏层维度列表转换为字符串
        'TD_basis': '-'.join(config['td_basis_types']),             # 将基底类型列表连接成字符串        
        'noise_std': noise_std,   # 添加噪声列
        'batch_size': config['batch_size'],
        'total_parameter': total_parameter,
        'train_loss': f"{train_loss:.4f}",
        'train_acc': f"{train_acc:.4f}",           # 保留4位小数
        'val_loss': f"{val_loss:.4f}",
        'val_acc': f"{val_acc:.4f}",
        'test_acc': f"{test_acc:.4f}",
        'epochs_used': epochs_used,
        'train_time': f"{duration_sec:.2f}" # 保留2位小数
    }


def write_to_csv(filename, data, fieldnames, save_dir, idx=None):
    """
    将训练结果写入 CSV 文件。

    Args:
        filename (str): 保存的 CSV 文件名（如 'result.csv'）
        data (list of dict): 多条训练记录，每条是一个字典
        fieldnames (list of str): CSV 表头字段名（与数据字典中的 key 对应）
        save_dir (str): 保存文件的目录路径

    Returns:
        None
    """
    # 创建目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    
    # 拼接完整的文件路径
    filepath = os.path.join(save_dir, filename)

    # 写入 CSV 文件
    mode = 'w' if idx == 0 or idx is None else 'a'
    with open(filepath, mode=mode, newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w' or f.tell() == 0:
            writer.writeheader()    # 写入表头
        writer.writerows(data)      # 写入多行数据

def save_epoch_params(model, epoch, tag="MNIST"):
    """保存每个 epoch 的模型参数"""
    os.makedirs(f"checkpoints/{tag}", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{tag}/epoch_{epoch}.pt")

# 从参数字典生成所有组合的配置，用于网格搜索。
def generate_configs(param_grid):
    keys = list(param_grid.keys())  # 获取所有超参数名
    for values in itertools.product(*param_grid.values()):  # 笛卡尔积生成所有组合
        yield dict(zip(keys, values))  # 将参数名和值组合成 dict 并返回

def clear_files(path):
    import glob
    for file in glob.glob(path):
        if os.path.isfile(file) and os.path.exists(file):
            os.remove(file)
            print(f"File {file} removed.", end='\r')
