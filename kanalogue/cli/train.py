#!/usr/bin/env python
"""
KANanlogue training CLI.

Usage:
    kanalogue train --dataset MNIST --device cuda:0 --acti None \\
                    --max_epochs 10000 --patience 10 --batch_size 256 \\
                    --exp_name rough_search --norm_layer batch

    kanalogue train --help   for all options.
"""

import os
import re
import sys
import time
import math
import json
import random
import itertools
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Parse args early so we can set CUDA_VISIBLE_DEVICES before torch init
# ---------------------------------------------------------------------------
from kanalogue.arguments import train_parse_arguments

# --- torch (import after CUDA device is configured) ---
import torch
import torch.nn as nn
import torch.optim as optim

# --- kanalogue modules ---
from kanalogue.model import build_model
from kanalogue.loader import load_piecewise, load_best_config, get_dataloaders
from kanalogue.train import (
    validate, prepare_csv_data, write_to_csv,
    save_epoch_params, clear_files, generate_configs,
)
from kanalogue.visualize import train_process_visualize, plot_batch_of_exp_result
from kanalogue.cmd_printer import print_epoch_status
from kanalogue.hooker import (
    attach_activation_hooks, clear_activation_stats, save_activation_stats,
)

activation_stats = defaultdict(dict)


def search_train(args):
    best_config = None
    best_accuracy = 0.0
    results = []

    parent_folder = 'results/new_structure_exps'
    dataset = args.dataset
    device = args.device if args.device == 'cpu' else 'cuda:0'
    acti = args.acti
    max_epochs = args.max_epochs
    patience = args.patience
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    norm_layer = args.norm_layer
    add_noise = args.add_noise

    if args.exp_name:
        exp_name = args.exp_name
        parent_folder = f'{parent_folder}/{exp_name}'

    td_params, td_title = load_piecewise(basis_mode=args.basis_type, fit_mode=args.fit_mode, device=device)
    title = f'{td_title}Basis_{acti}_{norm_layer}Norm'
    clear_files(f'{parent_folder}/{dataset}/train_process/{title}_all_epochs.csv')

    # grid search
    if add_noise:
        if dataset == 'MNIST':
            td_basis_combinations = [['Z21'], ['Z15', 'ZAZ_21'],
                                     ['Z21', 'Z15', 'ZAZ_21'],
                                     ['A21', 'Z21', 'Z15', 'ZAZ_21']]
        elif dataset == 'FMNIST':
            td_basis_combinations = [['Z15'], ['Z15', 'Z15'],
                                     ['Z21', 'Z15', 'ZAZ_21'],
                                     ['A21', 'Z21', 'Z15', 'ZAZ_21']]
        elif dataset == 'cifar10':
            td_basis_combinations = [['ZAZ_21'], ['Z21', 'ZAZ_21'],
                                     ['Z21', 'Z15', 'ZAZ_21'],
                                     ['A21', 'Z21', 'Z15', 'ZAZ_21']]

    if dataset == 'cifar10':
        param_grid = {
            'learning_rate': [learning_rate],
            'HIDDEN_DIMS': [[1024], [1024, 256], [1024, 256, 64]],
            'batch_size': [batch_size],
            'td_basis_types': [['A21', 'Z21', 'ZAZ_21'],]
        }
    elif dataset == 'MNIST':
        param_grid = {
            'learning_rate': [0.0013, ],
            'HIDDEN_DIMS': [[64], [256], [128, 16]],
            'batch_size': [32,],
            'td_basis_types': [['Z21', 'Z15', 'ZAZ_21'], ['Z15', 'ZAZ_21']],
        }
    elif dataset == 'FMNIST':
        param_grid = {
            'learning_rate': [learning_rate],
            'HIDDEN_DIMS': [[512, 128],
                            [256, 64], [128, 64, 16]],
            'batch_size': [32],
            'td_basis_types': [['Z21', 'Z15', 'ZAZ_21'], ['Z15', 'ZAZ_21']],
        }

    hidden = args.hidden
    # random search
    if dataset == 'MNIST':
        lr_range = (1e-4, 5e-3)
        hidden_choices = [hidden]
        batch_choices = [32, 64, 128]
    elif dataset == 'cifar10':
        lr_range = (5e-5, 2e-3)
        hidden_choices = [hidden]
        batch_choices = [256, 512, 1024]
    elif dataset == 'FMNIST':
        lr_range = (1e-4, 5e-3)
        hidden_choices = [hidden]
        batch_choices = [32, 64, 128]

    def random_config():
        config = {
            "learning_rate": 10 ** random.uniform(math.log10(lr_range[0]), math.log10(lr_range[1])),
            "HIDDEN_DIMS": random.choice(hidden_choices),
            "batch_size": random.choice(batch_choices),
            "td_basis_types": random.choice(td_basis_combinations),
        }
        return config

    if args.search_mode == 'random':
        n_trials = 20
        configs = [random_config() for _ in range(n_trials)]
        print(f"Random Search: {n_trials} trials")
    elif args.search_mode == 'grid':
        configs = list(generate_configs(param_grid))
        print(f"Grid Search: {len(configs)} configs")
    else:
        best_config = load_best_config(parent_folder=parent_folder,
                                       dataset=dataset, title='ref')
        configs = []
        if args.search_mode in ('best', 'custom', 'even_early'):
            if best_config is None:
                # No previous config — build one from CLI args
                _basis = (args.td_basis_types if args.td_basis_types
                          else (td_basis_combinations[0] if add_noise
                                else param_grid['td_basis_types'][0]))
                best_config = {
                    "learning_rate": args.learning_rate,
                    "HIDDEN_DIMS": args.hidden,
                    "batch_size": args.batch_size,
                    "td_basis_types": _basis,
                }
                print(f'No previous best config found; created from CLI args: {best_config}')
            if args.search_mode in ('custom', 'even_early'):
                best_config["HIDDEN_DIMS"] = args.hidden
                if args.td_basis_types is not None:
                    best_config["td_basis_types"] = list(args.td_basis_types)
                if args.search_mode == 'even_early':
                    best_config["learning_rate"] = args.learning_rate
                    best_config["batch_size"] = args.batch_size
            configs.append(best_config)
        elif args.search_mode == 'ablation' or args.search_mode == 'same-param':
            main_basis_types = ['A21', 'Z21', 'Z15', 'ZAZ_21']
            td_basis_combinations = []

            for r in range(1, len(main_basis_types) + 1):
                for combo in itertools.combinations(main_basis_types, r):
                    combo_list = list(combo)
                    td_basis_combinations.append(combo_list)

            if best_config and args.search_mode == 'same-param':
                hidden_dim = best_config["HIDDEN_DIMS"][0]
            for basis in td_basis_combinations:
                if best_config:
                    from copy import deepcopy
                    cfg = deepcopy(best_config)
                    cfg['td_basis_types'] = basis
                    if args.search_mode == 'same-param':
                        cfg['HIDDEN_DIMS'][0] = int(hidden_dim * 2 / len(basis))
                configs.append(cfg)

            print('Doing Ablation Study Towards Basis.')

    final_best_val_acc = 0
    for i, config in enumerate(configs):
        csv_data = []

        if dataset == 'MNIST':
            resize = (14, 14)
        elif dataset == 'FMNIST':
            resize = (28, 28)
        else:
            resize = (3, 32, 32)
        train_loader, val_loader, test_loader, input_dim = get_dataloaders(
            dataset_name=dataset,
            batch_size=config['batch_size'],
            resize=resize,
            pixel_range_neg=args.neg_input,
        )

        model = build_model(
            input_dim=input_dim,
            hidden_dims=config['HIDDEN_DIMS'],
            td_basis_types=config['td_basis_types'],
            td_params=td_params,
            acti=acti,
            norm_layer=norm_layer,
            fit_mode=args.fit_mode,
            model_type=args.model_type,
        ).to(device)

        total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"\n[Training config] "
            f"LR={config['learning_rate']:.5f}, "
            f"HIDDEN={config['HIDDEN_DIMS']}, "
            f"Batch={config['batch_size']}, "
            f"TD={config['td_basis_types']} | "
            f"Total Params: {total_param}"
        )

        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        hooks = attach_activation_hooks(model, activation_stats=activation_stats)

        start_time = time.time()
        best_val_acc = 0
        best_state_dict = None

        trigger_times = 0
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        for epoch in range(max_epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                correct += (outputs.argmax(1) == targets).sum().item()
                total += targets.size(0)

            train_loss = total_loss / total
            train_acc = correct / total

            # === validation ===
            val_loss, val_acc = validate(model, val_loader, criterion,
                                         device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            epochs = range(1, len(train_losses) + 1)
            train_process_visualize(epochs=epochs,
                                    train_losses=train_losses, val_losses=val_losses,
                                    train_accuracies=train_accuracies, val_accuracies=val_accuracies,
                                    dataset=dataset, title=title, parent_folder=parent_folder)

            save_activation_stats(epoch, title=title, tag=dataset, save_path=parent_folder,
                                  activation_stats=activation_stats)
            clear_activation_stats(activation_stats)
            save_epoch_params(model, epoch, tag=dataset)

            print_epoch_status(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=val_acc,
                tag_path=f"{parent_folder}/{dataset}/train_process/...",
                file_name=f"{title}_all_epochs.csv",
                best_loss=val_loss,
                first=(epoch == 0)
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = model.state_dict()

                if best_val_acc > final_best_val_acc:
                    final_best_val_acc = best_val_acc
                    weight_path = f"{parent_folder}/{dataset}/weights"
                    os.makedirs(weight_path, exist_ok=True)
                    torch.save(best_state_dict,
                               f"{weight_path}/{title}_best_model.pt")
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("===== Early stopping triggered. =====")
                    break

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": loss,
            }, f"{parent_folder}/{dataset}/weights/checkpoint.pth")
        duration_sec = time.time() - start_time

        # Restore best weights before final evaluation
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            print(f"Restored best model (val_acc={best_val_acc:.4f}) for final evaluation.")

        noise_stds = [0.0, 0.01, 0.05, 0.09, 0.13, 0.17, 0.21, 0.26,
                      0.31, 0.36, 0.41, 0.46] if add_noise else [0.0]
        for noise_std in noise_stds:
            _, test_acc = validate(model, test_loader, criterion,
                                   device, noise_std=noise_std)

            csv_data.append(prepare_csv_data(
                config=config,
                noise_std=noise_std,
                total_parameter=total_param,
                train_loss=train_loss, train_acc=train_acc,
                val_loss=val_loss, val_acc=val_acc,
                test_acc=test_acc, epochs_used=epoch, duration_sec=duration_sec
            ))

            results.append((config, noise_std, test_acc))
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_config = config

            best_config_dir = f'{parent_folder}/{dataset}/configs'
            os.makedirs(best_config_dir, exist_ok=True)
            best_config_path = f'{best_config_dir}/{title}_best_config.json'

            best_config_more = best_config.copy()
            best_config_more['dataset'] = dataset
            best_config_more['patience'] = patience
            best_config_more['norm_function'] = acti
            best_config_more['norm_layer'] = norm_layer

            best_config_more['noise_std'] = noise_std
            best_config_more['test_acc'] = best_accuracy
            with open(best_config_path, 'w') as f:
                json.dump(best_config_more, f, indent=4)

        for h in hooks:
            h.remove()

        os.makedirs(f"{parent_folder}/{dataset}", exist_ok=True)
        csv_fieldnames = [
            'learning_rate', 'hidden_dims', 'batch_size', 'TD_basis',
            'total_parameter', 'noise_std',
            'train_loss', 'train_acc', 'val_loss', 'val_acc',
            'test_acc', 'epochs_used', 'train_time'
        ]
        write_to_csv(f'{title}.csv', csv_data, csv_fieldnames,
                     save_dir=f'{parent_folder}/{dataset}', idx=i)

        plot_batch_of_exp_result(csv_data=csv_data,
                                 parent_folder=parent_folder,
                                 dataset=dataset, title=title)

    torch.save(model.state_dict(), f"{parent_folder}/{dataset}/weights/best.pth")

    print("\n====== Best Result ======")
    print(f"Config: {best_config}")
    print(f"Accuracy: {best_accuracy:.4f}")

    return results, best_config


def main():
    args = train_parse_arguments()

    # --- Reproducibility ---
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        # Make cuDNN deterministic (may slow down training slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f'Random seed set to {args.seed} (deterministic mode)')

    # Set CUDA device early (before any CUDA operations)
    m = re.match(r"cuda:(\d+)", args.device)
    if m is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = m.group(1)

    input_range = '[-1, 1]' if args.neg_input else '[0, 1]'

    print(f"\nDataset: {args.dataset}")
    print(f'Input Range: {input_range}')
    print(f'Activate Function: {args.acti}')
    print(f'Basis Type: {args.basis_type}')
    print(f'Norm Layer: {args.norm_layer}')

    search_train(args=args)


if __name__ == "__main__":
    main()
