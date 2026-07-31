#!/usr/bin/env python
"""Setup script for KANanlogue.

Install in development mode:
    pip install -e .

Then use the CLI:
    kanalogue train --dataset MNIST --device cuda:0 --exp_name my_exp
    kanalogue test  --dataset MNIST --device cuda:0 --exp_name my_exp
    kanalogue noise --noise_mode binary --datasets MNIST FMNIST
    kanalogue fit   --mode univariate --input data/raw --output data/processed
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kanalogue",
    version="0.1.0",
    description="Fully analogue in-memory neural computing via quantum tunnelling effect",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.1",
        "numpy>=1.23.5",
        "scipy>=1.10.1",
        "pandas>=2.0.3",
        "matplotlib>=3.7.2",
        "seaborn",
        "d2l>=1.0.3",
    ],
    entry_points={
        "console_scripts": [
            "kanalogue=kanalogue.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
