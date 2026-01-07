import os
import pandas as pd

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_MNIST_dataloaders(batch_size=64, resize=(14, 14), 
                          normalize=False, pixel_range_neg=True, 
                          val_ratio=0.1):
    """
    获取经过预处理的 MNIST 数据加载器

    Args:
        batch_size (int): 每个 batch 的样本数
        resize (tuple): 是否调整图像大小（默认为 14x14）
        normalize (bool): 是否标准化到均值0.1307/方差0.3081
        pixel_range_neg (bool): 是否将像素值放缩到 [-1, 1]
        val_ratio (float): 验证集比例
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int]: 训练集、验证集、测试集的 DataLoader 和输入维度
    """
    transform = get_base_transform(resize, normalize, pixel_range_neg)

    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_dataset, val_dataset = get_val_dataset(full_train_dataset, val_ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    input_dim = resize[0] * resize[1]
    return train_loader, val_loader, test_loader, input_dim

def get_FashionMNIST_dataloaders(batch_size=64, resize=(28, 28), 
                                 normalize=False, pixel_range_neg=True, 
                                 val_ratio=0.1):
    """
    获取经过预处理的 FashionMNIST 数据加载器

    Args:
        batch_size (int): 每个 batch 的样本数
        resize (tuple): 是否调整图像大小
        normalize (bool): 是否标准化
        pixel_range_neg (bool): 是否将像素值放缩到 [-1, 1]
        val_ratio (float): 验证集比例
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int]: 训练集、验证集、测试集的 DataLoader 和输入维度
    """
    transform = get_base_transform(resize, normalize, pixel_range_neg)

    full_train_dataset = datasets.FashionMNIST(root='./data', train=True, 
                                               download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, 
                                         download=True, transform=transform)
    
    train_dataset, val_dataset = get_val_dataset(full_train_dataset, val_ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    input_dim = resize[0] * resize[1]
    return train_loader, val_loader, test_loader, input_dim

def get_cifar10_dataloaders(batch_size=64, resize=None, 
                            normalize=None, pixel_range_neg=None, 
                            val_ratio=0.1):
    """
    获取经过预处理的 CIFAR-10 数据加载器

    Args:
        batch_size (int): 每个 batch 的样本数
        val_ratio (float): 验证集比例
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int]: 训练集、验证集、测试集的 DataLoader 和输入维度
    """
    train_transform = get_cifar10_transform()
    
    # 测试集使用简单的变换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    full_train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                          download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                    download=True, transform=test_transform)
    
    train_dataset, val_dataset = get_val_dataset(full_train_dataset, val_ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    input_dim = 3 * 32 * 32  # CIFAR-10 固定尺寸
    return train_loader, val_loader, test_loader, input_dim

def get_val_dataset(full_train_dataset, val_ratio=0.1):
    """
    从完整训练集中分割出验证集
    
    Args:
        full_train_dataset: 完整的训练数据集
        val_ratio (float): 验证集比例
    
    Returns:
        Tuple[Dataset, Dataset]: 训练集和验证集
    """
    train_size = int((1 - val_ratio) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size]
    )
    return train_dataset, val_dataset

def get_base_transform(resize=(14, 14), normalize=False, pixel_range_neg=True):
    """
    获取基础的数据变换组合
    
    Args:
        resize (tuple): 调整图像大小
        normalize (bool): 是否标准化
        pixel_range_neg (bool): 是否将像素值放缩到 [-1, 1]
    
    Returns:
        transforms.Compose: 数据变换组合
    """
    transform_list = []
    if resize:
        transform_list.append(transforms.Resize(resize))
    transform_list.append(transforms.ToTensor())
    if pixel_range_neg:
        transform_list.append(transforms.Lambda(lambda x: x * 2 - 1))
    if normalize:
        # 不同数据集使用不同的标准化参数
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    
    return transforms.Compose(transform_list)

def get_cifar10_transform():
    """
    获取CIFAR-10专用的数据增强变换
    
    Returns:
        transforms.Compose: CIFAR-10数据变换组合
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

# 统一的数据加载器获取函数
def get_dataloaders(dataset_name, **kwargs):
    """
    统一接口获取不同数据集的数据加载器
    
    Args:
        dataset_name (str): 数据集名称 ('MNIST', 'FashionMNIST', 'CIFAR10')
        **kwargs: 传递给具体数据集函数的参数
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int]: 训练集、验证集、测试集的 DataLoader 和输入维度
    """
    dataset_functions = {
        'MNIST': get_MNIST_dataloaders,
        'FMNIST': get_FashionMNIST_dataloaders,
        'cifar10': get_cifar10_dataloaders
    }
    
    if dataset_name not in dataset_functions:
        raise ValueError(f"不支持的数据集: {dataset_name}。可选: {list(dataset_functions.keys())}")
    
    return dataset_functions[dataset_name](**kwargs)

def load_line_params(csv_path, device=torch.device('cuda:0')):
    nodes_df = pd.read_csv(f'{csv_path}.csv')
    line_params = {
        "v0": torch.tensor(nodes_df["v0"].values, dtype=torch.float64, device=device),
        "v1": torch.tensor(nodes_df["v1"].values, dtype=torch.float64, device=device),
        "i0": torch.tensor(nodes_df["i0"].values, dtype=torch.float64, device=device),
        "i1": torch.tensor(nodes_df["i1"].values, dtype=torch.float64, device=device),
    }
    
    return line_params

def load_natural_spline_params(csv_path, device=torch.device('cuda:0')):
    nodes_df = pd.read_csv(f'{csv_path}_nodes.csv')
    coeff_df = pd.read_csv(f'{csv_path}_coefficients.csv')    
    spline_params = {
        "x": torch.tensor(nodes_df["x"].values, dtype=torch.float64, device=device),
        "y": torch.tensor(nodes_df["y"].values, dtype=torch.float64, device=device),
        "a": torch.tensor(coeff_df["a"].values, dtype=torch.float64, device=device),
        "b": torch.tensor(coeff_df["b"].values, dtype=torch.float64, device=device),
        "c": torch.tensor(coeff_df["c"].values, dtype=torch.float64, device=device),
        "d": torch.tensor(coeff_df["d"].values, dtype=torch.float64, device=device),
    }
    
    return spline_params

def load_univariate_spline_params(csv_path, device=torch.device('cuda:0')):
    nodes_df = pd.read_csv(f'{csv_path}_nodes.csv')
    knots_df = pd.read_csv(f'{csv_path}_knots.csv')
    coeff_df = pd.read_csv(f'{csv_path}_coefficients.csv')    

    spline_params = {
        "x": torch.tensor(nodes_df["x"].values, dtype=torch.float64, device=device),
        "xs": torch.tensor(knots_df["xs"].values, dtype=torch.float64, device=device),
        "y": torch.tensor(nodes_df["y"].values, dtype=torch.float64, device=device),
        "a": torch.tensor(coeff_df["a"].values, dtype=torch.float64, device=device),
        "b": torch.tensor(coeff_df["b"].values, dtype=torch.float64, device=device),
        "c": torch.tensor(coeff_df["c"].values, dtype=torch.float64, device=device),
        "d": torch.tensor(coeff_df["d"].values, dtype=torch.float64, device=device),
    }
    
    return spline_params

def load_poly_params(csv_path, device=torch.device('cuda:0')):
    nodes_df = pd.read_csv(f'{csv_path}_nodes.csv')
    coeff_df = pd.read_csv(f'{csv_path}_coefficients.csv')
    clamp_df = pd.read_csv(f"{csv_path}_clamp_params.csv")
    poly_params = {
        "x": torch.tensor(nodes_df["x"].values, dtype=torch.float64, device=device),
        "y": torch.tensor(nodes_df["y"].values, dtype=torch.float64, device=device),
        "coeffs": torch.tensor(coeff_df["coefficient"].values, dtype=torch.float64, device=device),
        "degree": torch.tensor(coeff_df["degree"].values, dtype=torch.float64, device=device),
        "V_max": torch.tensor(clamp_df["V_max"].values, dtype=torch.float64, device=device),
        "V_min": torch.tensor(clamp_df["V_min"].values, dtype=torch.float64, device=device),
    }
    
    return poly_params

def load_piecewise(basis_mode='pos-norm', fit_mode='univariate', fit_degree=10, 
                   device=torch.device('cuda:0')):
    basis_mode_mapping = {
        'pos-norm': 'PosNorm',      ## [0, 1]
        'neg-norm': 'NegNorm',      ## [-1, 1]
        'odd-sym': 'OddSym',        ## artificially odd-symmetic func
        'neg-ori': 'NegOri',        ## [-1.2, 1.2]
        'pos-ori': 'PosOri',        ## [0, 1.2]
        'pos-larger': 'PosLarger',  ## domain: [0, 1.2], range: [0, 1]
        'neg-larger': 'NegLarger',  ## domain: [-1.2, 1.2], range: [-1, 1]
    }
    basis_title = basis_mode_mapping.get(basis_mode, 'Unknown')

    parent_folder = f'IVcurve/{fit_mode}'
    fit_loaders = {
        'spline': load_natural_spline_params, 
        'poly': load_poly_params, 
        'fft': load_natural_spline_params, 
        'univariate': load_univariate_spline_params, 
        'line': load_line_params, 
    }
    loader_func = fit_loaders[fit_mode]

    if fit_mode == 'poly':
        basis_mode = f'deg{fit_degree}/{basis_mode}'

    td_params = {
        "A21": loader_func(f"{parent_folder}/{basis_mode}/a21", device),
        "Z21": loader_func(f"{parent_folder}/{basis_mode}/z21", device),
        "Z15": loader_func(f"{parent_folder}/{basis_mode}/z15", device),
        "ZAZ_21": loader_func(f"{parent_folder}/{basis_mode}/zaz21", device),
    }

    return td_params, basis_title

def load_best_config(parent_folder, dataset, title):
    path = f'{parent_folder}/{dataset}/configs/{title}.json'
    if os.path.exists(path):
        import json
        with open(path, 'r')  as f:
            print(f'Loaded best config from\n{path}.')
            return json.load(f)
    else:
        print(f'Warning: {path} not found!')
        return None
