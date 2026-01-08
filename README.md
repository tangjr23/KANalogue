# KANanlog: Fully analogue in-memory neural computing via quantum tunneling effect. 

## Introduction
This repository contains the source code for the paper **"Fully analogue in-memory neural computing via quantum tunneling effect"**. 
It provides the official implementation of KANalogue, which utilizes the
I–V characteristics of tunnel diodes as basis functions. Experiments are
conducted on MNIST, FashionMNIST, and CIFAR-10 datasets.
<!-- This repository contains code for training KAN networks using the I-V characteristics of Tunnel Diodes as basis functions, tested on MNIST, FashionMNIST, and cifar10 datasets. -->

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

## Installation
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

## Usage
### Running the KANalogue
To train a KAN network, run the following command:
```bash
python trainer.py --exp_name exp1 \
                    --dataset MNIST \
                    --acti None \
                    --norm_layer batch
```
You could use `python trainer.py --help` or `python trainer.py -h` to see all available options. Below are its available arguments:
```bash
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE
                        Learning rate of training progress.
  --batch_size BATCH_SIZE
                        Integer. 256 by default.
  --dataset {MNIST,FMNIST,cifar10}
                        Choose a dataset: MNIST, FMNIST, cifar10
  --device DEVICE       Train Device: cuda or cpu
  --acti {None,PosHC,NegHC,sigmoid,tanh}
                        Choose the activate function: PosHC, NegHC, sigmoid, tanh, or None
  --max_epochs MAX_EPOCHS
                        Integer. 10_000 by default.
  --patience PATIENCE   Integer. 10 by default.
  --neg_input           Range of input. False for [0, 1] and True for [-1, 1]
  --basis_type {pos-norm,pos-ori,neg-norm,neg-ori,odd-sym,pos-larger,neg-larger}
                        Type of basis functions.
  --basis_combine {2,3}
                        Combination type of basis functions. 2 for two basis, [A21, Z21]. 3 for three basis, [A21, Z21, Z15]
  --exp_name EXP_NAME   Name this exp. Then all the results will be in results/exp-name.
  --search_mode {grid,random,ablation,best,same-param,custom,even_early}
                        Searching mode. Random search by default.
  --norm_layer {layer,batch,None}
                        Type of the norm layer. "layer" for LayerNorm, "batch" for "BatchNorm", and "None" for no norm layer.
  --add_noise           Add noise or not. Write in cmmd for True.
  --hidden HIDDEN [HIDDEN ...]
                        Hidden dims. [64] by default. Entering "1024 256" for [1024, 256].
  --noise_num NOISE_NUM
                        Hidden dims. [64] by default
  --distribution {binary,uniform,gauss}
                        Noise distribution. "binary" by default.
```

## Reproducing Figures and Tables
### Expressive role of basis function diversity
We compare different choice and combination of basis function in KANalogue. To train fix architecture model, run the following command:
```bash

```
To train matched parameter budget model, run the following command:
```bash

```

### Architectural comparison and cross bar node effciency
We present a comparison of an MLP using a tunnel diode activation function and the original KAN. To train the models, run the following command:
```bash
python -m scripts.MLP_KAN_trainer.py --exp_name exp1 \
                    --dataset MNIST \
                    --model_name MLP_CMTDAF
```
Other baseline models can be trained by specifying different values
for the `--model_name` argument, including `MLP_RTDAF`, `BSplineKAN`,
and `GottliebKAN`.
You could use `python -m scripts.MLP_KAN_trainer.py --help` or `python -m scripts.MLP_KAN_trainer.py -h` to see all available options.

### Robustness to analogue pertubations
We evaluate the robustness of KANalogue under different levels of relative
coefficient perturbations. To test pertubation model, run the following command:
```bash

```