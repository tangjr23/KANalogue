#!/usr/bin/env python
"""
KANanlogue main CLI entry point.

Usage:
    kanalogue train  --dataset MNIST --device cuda:0 --exp_name my_exp
    kanalogue test   --dataset MNIST --device cuda:0 --exp_name my_exp
    kanalogue noise  --noise_mode binary --datasets MNIST FMNIST
    kanalogue fit    --mode univariate --input data/raw --output data/processed
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: kanalogue <command> [options]")
        print()
        print("Available commands:")
        print("  train    Train KAN networks with tunnel-diode basis functions")
        print("  test     Evaluate noise robustness on trained models")
        print("  noise    Plot noise-sensitivity analysis results")
        print("  fit      Fit splines/polynomials to raw IV-curve CSVs")
        print()
        print("Run 'kanalogue <command> --help' for per-command options.")
        sys.exit(1)

    command = sys.argv[1]
    # Remove 'kanalogue' from argv so the subcommand parser sees its own args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "train":
        from kanalogue.cli.train import main as train_main
        train_main()
    elif command == "test":
        from kanalogue.cli.test import main as test_main
        test_main()
    elif command == "noise":
        from kanalogue.cli.noise import main as noise_main
        noise_main()
    elif command == "fit":
        from kanalogue.cli.fit import main as fit_main
        fit_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, test, noise, fit")
        sys.exit(1)


if __name__ == "__main__":
    main()
