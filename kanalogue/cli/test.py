#!/usr/bin/env python
"""
KANanlogue noise-robustness testing CLI.

Loads a trained model and evaluates it across multiple noise standard-deviation
levels, repeating the evaluation ``--noise_num`` times to characterise the
distribution of accuracy under parameter perturbation.

Usage:
    kanalogue test --dataset MNIST --device cuda:0 --exp_name scale-small \\
                   --noise_num 10000 --distribution gauss \\
                   --fit_mode univariate --basis_type pos-larger

    kanalogue test --help   for all options.
"""

import os
import re
import sys
import torch
import torch.nn as nn

from kanalogue.arguments import train_parse_arguments
from kanalogue.model import build_model
from kanalogue.train import validate
from kanalogue.loader import get_dataloaders, load_best_config, load_piecewise
from kanalogue.train import write_to_csv


def prepare_csv_data(config, noise_std, total_parameter, test_acc):
    return {
        'hidden_dims': '-'.join(map(str, config['HIDDEN_DIMS'])),
        'TD_basis': '-'.join(config['td_basis_types']),
        'noise_std': noise_std,
        'total_parameter': total_parameter,
        'test_acc': f"{test_acc:.4f}",
    }


def load_and_add_noise(args):
    parent_folder = 'results/new_structure_exps'
    dataset = args.dataset
    device = args.device if args.device == 'cpu' else 'cuda:0'
    acti = args.acti
    neg_input = args.neg_input
    basis_type = args.basis_type
    fit_mode = args.fit_mode
    norm_layer = args.norm_layer

    if args.exp_name:
        exp_name = args.exp_name
        parent_folder = f'{parent_folder}/{exp_name}'
    parent_folder = f'{parent_folder}/{fit_mode}'

    # Load spline-parameter data
    td_params, _ = load_piecewise(basis_mode=basis_type,
                                  fit_mode=fit_mode, fit_degree=15,
                                  device=device)

    # Load checkpoint
    ckpt_dir = f'{parent_folder}/{dataset}/weights'
    ckpt_path = os.path.join(ckpt_dir, "best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    print(f'Weights loaded from {ckpt_dir}')

    # Load best config
    config = load_best_config(parent_folder=parent_folder,
                              dataset=dataset, title='best')

    if dataset == 'MNIST':
        resize = (14, 14)
    elif dataset == 'FMNIST':
        resize = (28, 28)
    else:
        resize = (3, 32, 32)

    criterion = nn.CrossEntropyLoss()

    _, _, test_loader, input_dim = get_dataloaders(
        dataset_name=dataset,
        batch_size=config['batch_size'],
        resize=resize,
        pixel_range_neg=neg_input,
    )

    model = build_model(
        input_dim=input_dim,
        hidden_dims=config['HIDDEN_DIMS'],
        td_basis_types=config['td_basis_types'],
        td_params=td_params,
        acti=acti, fit_mode=fit_mode,
        norm_layer=norm_layer,
    ).to(device)
    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal Params: {total_param}\n\n')

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded best model, acc={checkpoint.get('best_acc')}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    for i in range(args.noise_num):
        csv_data = []

        sys.stdout.write("\033[A")
        print(f'\033[2KLoop {i + 1}/{args.noise_num}')

        noise_stds = [0.00, 0.01, 0.05, 0.09, 0.13, 0.17,
                      0.21, 0.26, 0.31, 0.36, 0.41, 0.46]
        for noise_std in noise_stds:
            _, test_acc = validate(model, test_loader, criterion, device,
                                   noise_std=noise_std,
                                   noise_distri=args.distribution)

            print(f'\033[2KNoise Std: {noise_std:.2f}\t Test Acc: {test_acc}', end='\r')

            csv_data.append(prepare_csv_data(
                config=config,
                noise_std=noise_std,
                total_parameter=total_param,
                test_acc=test_acc,
            ))
        csv_fieldnames = [
            'hidden_dims', 'TD_basis',
            'total_parameter', 'noise_std',
            'test_acc'
        ]
        write_to_csv(f'{args.distribution}_noise.csv', csv_data, csv_fieldnames,
                     save_dir=f'{parent_folder}/{dataset}', idx=i)
    print(f'Results saved into {parent_folder}/{dataset}/{args.distribution}_noise.csv')


def main():
    args = train_parse_arguments()

    # Set CUDA device early
    m = re.match(r"cuda:(\d+)", args.device)
    if m is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = m.group(1)

    print(f"\nAdd Noise into Model")
    print(f'Dataset: {args.dataset}')
    print(f'Noise Distribution: {args.distribution}')
    print(f'Fit Mode: {args.fit_mode}')
    print(f'Basis Type: {args.basis_type}\n')

    load_and_add_noise(args=args)


if __name__ == "__main__":
    main()
