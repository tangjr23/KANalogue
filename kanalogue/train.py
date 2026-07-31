import torch
import torch.optim as optim
import os
import copy
import csv
import pandas as pd
import itertools

def train(model, train_loader, criterion, optimizer, device):
    model.train()  # Set to training mode, enable dropout, BN, etc.
    total_loss = 0  # Accumulate loss
    correct = 0     # Accumulate correct predictions
    global_step = 0  # Global step counter

    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move data and labels to GPU/CPU

        # Special case for LBFGS optimizer
        if isinstance(optimizer, optim.LBFGS):
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                return loss
            loss = optimizer.step(closure)  # Special call pattern
        else:
            optimizer.zero_grad()            # Clear gradients
            output = model(data)             # Forward pass
            loss = criterion(output, target) # Compute loss
            loss.backward()                  # Backward pass
            optimizer.step()                 # Update parameters
            loss = loss.item()               # Get scalar loss value (avoid tensor accumulation)

        total_loss += loss  # Accumulate batch loss

        global_step += 1    # Increment global step counter

        # Get predicted class
        pred = output.argmax(dim=1, keepdim=True)
        # Count correct predictions
        correct += pred.eq(target.view_as(pred)).sum().item()

    # Return average loss and accuracy
    return total_loss / len(train_loader), correct / len(train_loader.dataset)


def validate(model, test_loader, criterion, device,
             noise_std=0.0, noise_distri='binary'):
    model.eval()  # Set to evaluation mode, disable dropout, BN
    total_loss = 0
    correct = 0

    with torch.no_grad():  # Disable gradient computation (save memory & speed up)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Move data to device
            output = model(data, noise_std=noise_std,
                           noise_mode=noise_distri)            # Forward pass
            loss = criterion(output, target)                   # Compute loss
            total_loss += loss.item()                          # Accumulate loss

            pred = output.argmax(dim=1, keepdim=True)          # Get prediction
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct

    # Return average loss and accuracy
    return total_loss / len(test_loader), correct / len(test_loader.dataset)


def train_with_early_stopping_epoch(model, train_loader, val_loader, optimizer, criterion, device,
                                    max_epochs, patience=5, min_delta=1e-3, noise_std=0.0):
    """
    Args:
        model (nn.Module): model to train
        train_loader (DataLoader): training data loader
        val_loader (DataLoader): validation data loader
        optimizer (torch.optim.Optimizer): optimizer (e.g. Adam)
        criterion (nn.Module): loss function (e.g. CrossEntropyLoss)
        device (str): device ('cuda' or 'cpu')
        max_epochs (int): maximum training epochs
        patience (int): early stopping tolerance epochs
        min_delta (float): minimum validation loss improvement threshold

    Returns:
        Tuple[int, float, float]: actual epochs trained, best val loss, best val accuracy
    """
    best_loss = float('inf')        # Current best validation loss
    best_accuracy = 0.0             # Current best validation accuracy
    counter = 0                     # No-improvement epoch counter
    best_weights = None             # Best model parameters (for restoration)

    model.to(device)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(max_epochs):
        # === Training ===
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)

        # === Validation ===
        val_loss, val_accuracy = validate(model, val_loader, criterion, device, noise_std=noise_std)

        # Save logs
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)

        # If validation loss significantly decreased, save current model
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_accuracy = val_accuracy
            counter = 0  # Reset tolerance counter
            best_weights = copy.deepcopy(model.state_dict())
        else:
            counter += 1  # No significant improvement, increment tolerance counter

        # If tolerance counter exceeds set value, stop training early
        if counter >= patience:
            print(f"Early stopped at epoch {epoch+1} | Best Val Loss: {best_loss:.4f}")
            break

    # Restore best-performing model parameters
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Return epochs trained, best validation loss and validation accuracy
    return epoch + 1, best_loss, best_accuracy, train_loss, train_acc


def train_with_noise(model, train_loader, val_loader, optimizer, criterion, device,
                     max_epochs, patience=10, min_delta=1e-4, noise_std=0.0):

    best_loss = float('inf')        # Current best validation loss
    best_accuracy = 0.0             # Current best validation accuracy
    counter = 0                     # No-improvement epoch counter
    best_weights = None             # Best model parameters (for restoration)

    model.to(device)

    for epoch in range(max_epochs):
        # === Training ===
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)

        # === Validation ===
        val_loss, val_accuracy = validate(model, val_loader, criterion, device, noise_std=noise_std)

        # If validation loss significantly decreased, save current model
        if best_loss - val_loss > min_delta:
            best_loss = val_loss
            best_accuracy = val_accuracy
            counter = 0  # Reset tolerance counter
            best_weights = copy.deepcopy(model.state_dict())
        else:
            counter += 1  # No significant improvement, increment tolerance counter

        # If tolerance counter exceeds set value, stop training early
        if counter >= patience:
            print(f"Early stopped at epoch {epoch+1} | Best Val Loss: {best_loss:.4f}")
            break

    # Restore best-performing model parameters
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Return epochs trained, training and validation metrics
    return epoch + 1, train_loss, train_acc, val_loss, val_accuracy


def prepare_csv_data(config,
                     noise_std,
                     total_parameter,
                     train_loss, train_acc, val_loss, val_acc,
                     test_acc, epochs_used, duration_sec):
    """
    Prepare one training result data record for CSV export.

    Args:
        config (dict): current training configuration, including learning rate, hidden layer structure, etc.
        total_parameter (int): total trainable parameters of the model
        acc (float): test set accuracy
        val_loss (float): validation set loss
        val_acc (float): validation set accuracy
        epochs_used (int): actual training epochs (considering Early Stopping)
        duration_sec (float): total training time (seconds)

    Returns:
        dict: one CSV data row, field names correspond to CSV header
    """
    return {
        'learning_rate': config['learning_rate'],
        'hidden_dims': '-'.join(map(str, config['HIDDEN_DIMS'])),   # Convert hidden dims list to string
        'TD_basis': '-'.join(config['td_basis_types']),             # Join basis type list into string
        'noise_std': noise_std,   # Add noise column
        'batch_size': config['batch_size'],
        'total_parameter': total_parameter,
        'train_loss': f"{train_loss:.4f}",
        'train_acc': f"{train_acc:.4f}",           # 4 decimal places
        'val_loss': f"{val_loss:.4f}",
        'val_acc': f"{val_acc:.4f}",
        'test_acc': f"{test_acc:.4f}",
        'epochs_used': epochs_used,
        'train_time': f"{duration_sec:.2f}" # 2 decimal places
    }


def write_to_csv(filename, data, fieldnames, save_dir, idx=None):
    """
    Write training results to CSV file.

    Args:
        filename (str): CSV file name to save (e.g. 'result.csv')
        data (list of dict): multiple training records, each a dict
        fieldnames (list of str): CSV header field names (matching keys in data dict)
        save_dir (str): directory path for saving the file

    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Build full file path
    filepath = os.path.join(save_dir, filename)

    # Write CSV file
    mode = 'w' if idx == 0 or idx is None else 'a'
    with open(filepath, mode=mode, newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w' or f.tell() == 0:
            writer.writeheader()    # Write header
        writer.writerows(data)      # Write multiple data rows


def save_epoch_params(model, epoch, tag="MNIST"):
    """Save model parameters for each epoch."""
    os.makedirs(f"checkpoints/{tag}", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{tag}/epoch_{epoch}.pt")


def generate_configs(param_grid):
    """Generate all combinations of configs from a parameter dict for grid search."""
    keys = list(param_grid.keys())  # Get all hyperparameter names
    for values in itertools.product(*param_grid.values()):  # Cartesian product generates all combinations
        yield dict(zip(keys, values))  # Combine parameter names and values into dict and return


def clear_files(path):
    import glob
    for file in glob.glob(path):
        if os.path.isfile(file) and os.path.exists(file):
            os.remove(file)
            print(f"File {file} removed.", end='\r')
