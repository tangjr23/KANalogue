# KANanlog: Fully analogue in-memory neural computing via quantum tunneling effect. 

## Introduction
This repository contains code for training KAN networks using the I-V characteristics of Tunnel Diodes as basis functions, tested on MNIST, FashionMNIST, and cifar10 datasets.

## Overview
### Useful Files
+ `trainer.py`: Main training script for KAN networks.
+ `scripts/`: Contains Jupyter notebooks for running experiments.
+ `utils/`: Utility functions for data handling, model definitions, and training routines.

### Main Project Tree
```plaintext
.
├── README.md
├── data
├── IVcurve
├── results
│   ├── ...
│   ├── exp1
│   │   ├── MNIST
|   │   │   ├── ...
|   │   │   ├── configs
|   │   │   └── results.csv
│   │   ├── FMNIST
|   │   │   ├── ...
|   │   │   ├── configs
|   │   │   └── results.csv
│   │   └── cifar10
|   │       ├── ...
|   │       ├── configs
|   │       └── results.csv
│   └──exp2
├── scripts
│   ├── fit.py
│   ├── trainer.py
│   └── ...
├── utils
│   ├── arguments.py
│   ├── basis.py
│   ├── imports.py
│   ├── loader.py
│   ├── model.py
│   ├── train.py
│   └── ...
└── ...
```

## Train
### Setup
To set up the environment, you can use either conda or pip.   
+ conda methord
    ```bash
    conda env create -f environment.yaml
    ```

    > If you are training on NVIDIA Geforce RTX 5090 GPU, use [environment5090.yaml](./environment5090.yaml) instead.

+ pip method
    ```bash
    pip install -r requirements.txt
    ```
### Run
To train a KAN network, run the following command:
```bash
python train_exp.py --exp_name exp1 \
                    --dataset MNIST \
                    --acti None \
                    --norm_layer batch
```
You could use `python trainer.py --help` or `python trainer.py -h` to see all available options.
