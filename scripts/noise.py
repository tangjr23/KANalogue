# Usage
# python noise.py --dataset MNIST
#                 --device cuda:0
#                 --exp_name test
#                 --noise_num 10000
#                 --distribution gauss
# Use `python noise.py -h` or `python noise.py --help` to see more.

import os
import re
import sys
import torch
import torch.nn as nn

from utils.arguments import train_parse_arguments
args = train_parse_arguments()
m = None
m = re.match(r"cuda:(\d+)", args.device)
if m is not None:
    physical_id = m.group(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_id

from utils.model import build_model
from utils.train import validate
from utils.loader import get_dataloaders, load_best_config, load_piecewise
from utils.train import write_to_csv

def prepare_csv_data(config, 
                     noise_std, 
                     total_parameter, 
                     test_acc):
    return {
        'hidden_dims': '-'.join(map(str, config['HIDDEN_DIMS'])), 
        'TD_basis': '-'.join(config['td_basis_types']),        
        'noise_std': noise_std,
        'total_parameter': total_parameter,
        'test_acc': f"{test_acc:.4f}",
    }

def load_and_add_noise(args, 
                       ckpt_dir="/path/to/your/experiment"):
    parent_folder = 'results'
    dataset = args.dataset
    device = args.device if args.device == 'cpu' else 'cuda:0'

    if args.exp_name:
        exp_name = args.exp_name
        parent_folder = f'{parent_folder}/{exp_name}'
    parent_folder = f'{parent_folder}/univariate'
    
    td_params, _ = load_piecewise(basis_mode='pos-larger', 
                                         fit_mode='univariate', fit_degree=15, 
                                         device=device)

    ckpt_dir = f'{parent_folder}/{dataset}/weights' 
    ckpt_path = os.path.join(ckpt_dir, "best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    print(f'Weights loaded form {ckpt_dir}')

    config = load_best_config(parent_folder=parent_folder, 
                              dataset=dataset, title=f'best')

    if dataset == 'MNIST':
        # resize = (28, 28) if args.search_mode == 'grid' else (14, 14)
        resize = (14, 14)
    elif dataset == 'FMNIST':
        resize=(28, 28)
    else:
        resize=(3, 32, 32) 

    criterion = nn.CrossEntropyLoss()

    _, _, test_loader, input_dim = get_dataloaders(
        dataset_name=dataset, 
        batch_size=config['batch_size'], 
        resize=resize, 
        pixel_range_neg=False, 
    )

    model = build_model(
        input_dim=input_dim,
        hidden_dims=config['HIDDEN_DIMS'],
        td_basis_types=config['td_basis_types'],
        td_params=td_params, 
        acti=None, fit_mode='univariate', 
        norm_layer='None', 
    ).to(device)
    model = model.to(device)
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

    print(f"\nAdd Noise into Model")
    print(f'Dataset: {args.dataset}')
    print(f'Noise Distribution: {args.distribution}\n')

    load_and_add_noise(args=args)

if __name__ == "__main__":
    main()


