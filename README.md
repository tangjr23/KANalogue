# KANanlogue: Fully analogue in-memory neural computing via quantum tunnelling effect.

## Introduction

This repository contains code for training KAN networks using the I–V characteristics
of tunnel diodes as basis functions, tested on MNIST, FashionMNIST, and CIFAR-10 datasets.

The core idea is to use physical quantum-tunnelling device characteristics as the
nonlinear computation primitive in a KAN-like architecture — replacing learned or
fixed mathematical activation functions with spline fits to real tunnel-diode IV curves.

## Project Structure

```plaintext
.
├── README.md
├── FOLDERINFO.md                   # Detailed architecture & data-flow documentation
├── setup.py                        # Package installer + CLI entry-points
├── requirements.txt                # Pip dependencies
├── environment.yaml                # Conda env (Python 3.9, torch 2.0.0)
├── environment5090.yaml            # Conda env for RTX 5090
│
├── kanalogue/                      # Main Python package
│   ├── __init__.py
│   ├── arguments.py                # Argument parsers
│   ├── basis.py                    # Spline & polynomial evaluation
│   ├── cmd_printer.py              # Terminal output formatting
│   ├── hooker.py                   # Activation hook recording
│   ├── loader.py                   # Data loaders + IV-curve param loading
│   ├── model.py                    # TDiode_KANLayer + model builder
│   ├── train.py                    # Train / validate / early-stopping / CSV I/O
│   ├── visualize.py                # Training-curve and heatmap plotting
│   └── cli/                        # CLI entry-points
│       ├── __init__.py
│       ├── main.py                 # Command dispatcher
│       ├── train.py                # `kanalogue train`
│       ├── test.py                 # `kanalogue test`
│       ├── noise.py                # `kanalogue noise`
│       └── fit.py                  # `kanalogue fit`
│
├── data/
│   ├── raw/                        # Raw tunnel-diode IV-curve CSVs
│   └── processed/                  # Fitted spline parameter CSVs
│
└── results/                        # Experiment outputs (created at runtime)
```

## Setup

### Conda

```bash
conda env create -f environment.yaml
```

> If you are training on an NVIDIA GeForce RTX 5090 GPU, use
> [environment5090.yaml](./environment5090.yaml) instead.

### Pip

```bash
pip install -r requirements.txt
```

### Install KANanlogue as a package (recommended)

```bash
pip install -e .
```

This makes the `kanalogue` command available system-wide.

## Usage

After installing with `pip install -e .`, the following CLI commands are available:

### Train

Train a KAN network with tunnel-diode basis functions:

```bash
kanalogue train --dataset MNIST \
                --device cuda:0 \
                --acti None \
                --max_epochs 10000 \
                --patience 10 \
                --batch_size 256 \
                --exp_name my_experiment \
                --norm_layer batch
```

Run `kanalogue train --help` to see all available options, including grid search,
random search, ablation study, and custom configuration modes.

### Test (Noise Robustness)

Evaluate noise robustness on a trained model:

```bash
kanalogue test --dataset MNIST \
               --device cuda:0 \
               --exp_name my_experiment \
               --noise_num 10000 \
               --distribution gauss
```

### Noise Analysis

Generate publication-quality noise-sensitivity plots:

```bash
kanalogue noise --noise_mode binary \
                --datasets MNIST FMNIST \
                --broken
```

### Fit IV Curves

Fit splines / polynomials to raw tunnel-diode IV-curve CSVs:

```bash
# Fit univariate splines to all raw CSVs
kanalogue fit --mode univariate \
              --input data/raw \
              --output data/processed \
              --smoothing 0.1

# Batch process multiple folders
kanalogue fit --batch \
              --mode spline \
              --input data/raw \
              --output data/processed
```

## Key Design Decisions

- **Float64 throughout** — all computation uses `torch.float64` for numerical precision.
- **Basis functions are fixed, not learned** — spline parameters are precomputed from
  physical measurements; only linear-combination coefficients are trained.
- **Noise as a first-class evaluation primitive** — the `validate()` and
  `TDiode_KANLayer.forward()` methods accept `noise_std` and `noise_mode` for
  systematic robustness testing.
- **KAN-like multiplicative gating** — each layer computes
  `sum_over_basis_types(basis(x) * coefficient[i,o,d])`.

## Reference

For more details on the architecture, data flow, and basis types, see
[FOLDERINFO.md](./FOLDERINFO.md).
