import os
import pandas as pd

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Resolve the data directory relative to the project root.
# When installed with `pip install -e .`, __file__ is inside the kanalogue package.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PACKAGE_DIR, 'data')


def get_MNIST_dataloaders(batch_size=64, resize=(14, 14),
                          normalize=False, pixel_range_neg=True,
                          val_ratio=0.1):
    """
    Get preprocessed MNIST data loaders.

    Args:
        batch_size (int): samples per batch
        resize (tuple): resize images (default 14x14)
        normalize (bool): standardise to mean 0.1307 / std 0.3081
        pixel_range_neg (bool): scale pixel values to [-1, 1]
        val_ratio (float): validation set ratio

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int]: train, val, test loaders and input dim
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
    Get preprocessed FashionMNIST data loaders.

    Args:
        batch_size (int): samples per batch
        resize (tuple): resize images
        normalize (bool): standardise
        pixel_range_neg (bool): scale pixel values to [-1, 1]
        val_ratio (float): validation set ratio

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int]: train, val, test loaders and input dim
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
    Get preprocessed CIFAR-10 data loaders.

    Args:
        batch_size (int): samples per batch
        val_ratio (float): validation set ratio

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int]: train, val, test loaders and input dim
    """
    train_transform = get_cifar10_transform()

    # Test set uses simple transforms
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

    input_dim = 3 * 32 * 32  # CIFAR-10 fixed size
    return train_loader, val_loader, test_loader, input_dim


def get_val_dataset(full_train_dataset, val_ratio=0.1):
    """
    Split validation set from full training set.

    Args:
        full_train_dataset: complete training dataset
        val_ratio (float): validation set ratio

    Returns:
        Tuple[Dataset, Dataset]: training and validation sets
    """
    train_size = int((1 - val_ratio) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size]
    )
    return train_dataset, val_dataset


def get_base_transform(resize=(14, 14), normalize=False, pixel_range_neg=True):
    """
    Get basic data transform composition.

    Args:
        resize (tuple): resize images
        normalize (bool): standardise
        pixel_range_neg (bool): scale pixel values to [-1, 1]

    Returns:
        transforms.Compose: data transform composition
    """
    transform_list = []
    if resize:
        transform_list.append(transforms.Resize(resize))
    transform_list.append(transforms.ToTensor())
    if pixel_range_neg:
        transform_list.append(transforms.Lambda(lambda x: x * 2 - 1))
    if normalize:
        # Different datasets use different normalisation parameters
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))

    return transforms.Compose(transform_list)


def get_cifar10_transform():
    """
    Get CIFAR-10-specific data augmentation transforms.

    Returns:
        transforms.Compose: CIFAR-10 data transform composition
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


# Unified data loader access function
def get_dataloaders(dataset_name, **kwargs):
    """
    Unified interface for obtaining data loaders for different datasets.

    Args:
        dataset_name (str): dataset name ('MNIST', 'FashionMNIST', 'CIFAR10')
        **kwargs: arguments passed to the specific dataset function

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int]: train, val, test loaders and input dim
    """
    dataset_functions = {
        'MNIST': get_MNIST_dataloaders,
        'FMNIST': get_FashionMNIST_dataloaders,
        'cifar10': get_cifar10_dataloaders
    }

    if dataset_name not in dataset_functions:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Options: {list(dataset_functions.keys())}")

    return dataset_functions[dataset_name](**kwargs)


# ---------------------------------------------------------------------------
# IV-curve parameter loaders
# ---------------------------------------------------------------------------

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
    """
    Load all four precomputed spline-parameter sets from data/processed/.

    Args:
        basis_mode: label for the basis normalisation mode
                    (used for title only; data is loaded from flat directory).
        fit_mode:   'univariate', 'spline', 'poly', 'fft', or 'line'.
        fit_degree: polynomial degree (only used when fit_mode == 'poly').
        device:     torch device to place tensors on.

    Returns:
        td_params:  dict mapping basis name -> parameter dict
        basis_title: string label for the basis mode
    """
    basis_mode_mapping = {
        'pos-norm': 'PosNorm',      # [0, 1]
        'neg-norm': 'NegNorm',      # [-1, 1]
        'odd-sym': 'OddSym',        # artificially odd-symmetric function
        'neg-ori': 'NegOri',        # [-1.2, 1.2]
        'pos-ori': 'PosOri',        # [0, 1.2]
        'pos-larger': 'PosLarger',  # domain: [0, 1.2], range: [0, 1]
        'neg-larger': 'NegLarger',  # domain: [-1.2, 1.2], range: [-1, 1]
    }
    basis_title = basis_mode_mapping.get(basis_mode, 'Unknown')

    # Data is stored flat under data/processed/
    parent_folder = os.path.join(_DATA_DIR, 'processed')

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

    def _resolve_csv_prefix(base_dir, basis_mode_dir, key):
        """Try subdirectory layout first, then fall back to flat."""
        subdir_prefix = os.path.join(base_dir, basis_mode_dir, key)
        flat_prefix = os.path.join(base_dir, key)
        if os.path.exists(f"{subdir_prefix}_nodes.csv") or \
           os.path.exists(f"{subdir_prefix}_coefficients.csv") or \
           os.path.exists(f"{subdir_prefix}.csv"):
            return subdir_prefix
        return flat_prefix

    td_params = {
        "A21": loader_func(_resolve_csv_prefix(parent_folder, basis_mode, "a21"), device),
        "Z21": loader_func(_resolve_csv_prefix(parent_folder, basis_mode, "z21"), device),
        "Z15": loader_func(_resolve_csv_prefix(parent_folder, basis_mode, "z15"), device),
        "ZAZ_21": loader_func(_resolve_csv_prefix(parent_folder, basis_mode, "zaz21"), device),
    }

    return td_params, basis_title


def load_best_config(parent_folder, dataset, title):
    path = f'{parent_folder}/{dataset}/configs/{title}.json'
    if os.path.exists(path):
        import json
        with open(path, 'r') as f:
            print(f'Loaded best config from\n{path}.')
            return json.load(f)
    else:
        print(f'Warning: {path} not found!')
        return None
