#!/usr/bin/env python
"""
KANanlogue noise-analysis visualisation CLI.

Usage:
    kanalogue noise --noise_mode binary --datasets MNIST FMNIST --broken
    kanalogue noise --help   for all options.
"""

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

from scipy.stats import linregress
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, FuncFormatter


# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

dataset_colors = {
    "MNIST": "tab:blue",
    "FMNIST": "tab:green",
}

size_styles = {
    "large": "-",
    "mid": "--",
    "small": ":",
}

# Human-readable parameter-count labels shown in the legend
param_labels = {
    ('MNIST', 'small'):   '20k',
    ('MNIST', 'mid'):     '40k',
    ('MNIST', 'large'):   '80k',
    ('FMNIST', 'small'):  '300k',
    ('FMNIST', 'mid'):    '600k',
    ('FMNIST', 'large'):  '1.2M',
}

ds_dict = {
    'MNIST': 'MNIST',
    'FMNIST': 'FashionMNIST',
    'cifar10': 'CIFAR-10',
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_result(parent="../results/new_structure_exps",
                ds='MNIST', noise_mode='binary'):
    scales = ['small', 'mid', 'large', 'smid', 'middlarge']

    all_df = pd.DataFrame()
    df_analysis = pd.DataFrame()
    for scale in scales:
        parent_folder = f"{parent}/scale-{scale}/univariate/{ds}"
        file_name = f"{noise_mode}_noise.csv"
        if not os.path.exists(f"{parent_folder}/{file_name}"):
            file_name = "noise.csv"
        path = f"{parent_folder}/{file_name}"

        print(f"\033[KLoading from {path}", end='\r')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['scale'] = scale

        all_df = pd.concat([all_df, df], ignore_index=True)
        df_ana = (
            df
            .groupby("noise_std")["test_acc"]
            .agg(["min", "max", "var"])
            .reset_index()
        )
        df_ana["scale"] = scale
        df_analysis = pd.concat([df_analysis, df_ana], ignore_index=True)

    all_df["dataset"] = ds
    df_analysis['dataset'] = ds

    return all_df, df_analysis


def load_all_result(parent="../results/new_structure_exps",
                    noise_mode='binary'):
    datasets = ['MNIST', 'FMNIST']
    all_df = pd.DataFrame()
    all_df_analysis = pd.DataFrame()
    for ds in datasets:
        df, df_analysis = load_result(parent=parent,
                                      ds=ds, noise_mode=noise_mode)
        all_df = pd.concat([all_df, df], ignore_index=True)
        all_df_analysis = pd.concat([all_df_analysis, df_analysis],
                                    ignore_index=True)
    return all_df, all_df_analysis


def export_min_max(df, ds):
    mini = df[df['dataset'] == ds]['min'].min()
    maxi = df[df['dataset'] == ds]['max'].max()
    return mini, maxi


# ---------------------------------------------------------------------------
# Figure 1: Accuracy vs. noise sigma (broken-axis)
# ---------------------------------------------------------------------------

def plot_acc_sigma(df, df_ana,
                   datasets=['MNIST', 'FMNIST', 'cifar10'],
                   noise_mode='binary',
                   broken=True):
    mini, maxi = {}, {}
    for dataset in datasets:
        mini[f'{dataset}'], maxi[f'{dataset}'] = export_min_max(df=df_ana, ds=dataset)

    if_cifar = 'cifar10' in datasets

    df_all = df
    stat_df = (
        df_all
        .groupby(["dataset", "scale", "noise_std"])
        .agg(
            mean=("test_acc", "mean"),
            min=("test_acc", "min"),
            max=("test_acc", "max"),
        )
        .reset_index())

    # Create broken-axis figure
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    if broken:
        width_top, width_bot, width_even_bot = 10, 8, 2
        if if_cifar:
            heights = [width_top, width_bot, width_even_bot]
            grid_num = 3
        else:
            heights = [width_top, width_even_bot]
            grid_num = 2
        gs = gridspec.GridSpec(grid_num, 1, figure=fig,
                               height_ratios=heights,
                               hspace=0.0005)
    else:
        gs = gridspec.GridSpec(1, 1, figure=fig)

    if broken:
        ax_top = fig.add_subplot(gs[0])
        axes_by_dataset = {'MNIST': ax_top, 'FMNIST': ax_top}

        if if_cifar:
            dataset_colors['cifar10'] = "tab:orange"
            ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
            ax_even_bot = fig.add_subplot(gs[2], sharex=ax_bot)
            ax_bot.set_ylim(0.35 + 0.001, 0.55 - 0.001)
            ax_bot.spines["top"].set_visible(False)
            ax_bot.spines["bottom"].set_visible(False)
            ax_bot.tick_params(labelbottom=False)
            ax_bot.tick_params(axis='x', direction='in', length=0)
            ax_bot.grid(axis="y", alpha=0.3)
            axes_by_dataset['cifar10'] = ax_bot
        else:
            ax_even_bot = fig.add_subplot(gs[1], sharex=ax_top)

        ax_top.set_ylim(0.8 + 0.001, 1)
        ax_even_bot.set_ylim(0.00, 0.05 - 0.001)

        ax_top.spines["bottom"].set_visible(False)
        ax_even_bot.spines["top"].set_visible(False)
        ax_top.tick_params(labelbottom=False)
        ax_even_bot.tick_params(labelleft=False)
        ax_top.tick_params(axis='x', direction='in', length=0)
        ax_even_bot.tick_params(axis='y', direction='in', length=0)
        ax_top.grid(axis="y", alpha=0.3)
        ax_even_bot.grid(axis="y", alpha=0.3)
    else:
        ax_all = fig.add_subplot(gs[0])
        axes_by_dataset = {'MNIST': ax_all, 'FMNIST': ax_all}
        if if_cifar:
            axes_by_dataset["cifar10"] = ax_all

    # Draw break marks
    def draw_break_marks(ax_upper, ax_lower, width_upper, width_lower):
        d = 0.01
        kwargs = dict(color='black', clip_on=False, linewidth=1)
        d1 = d * (width_upper / width_lower)
        ax_upper.plot((-d, +d), (-d, +d), transform=ax_upper.transAxes, **kwargs)
        ax_upper.plot((1 - d, 1 + d), (-d, +d), transform=ax_upper.transAxes, **kwargs)
        ax_lower.plot((-d, +d), (1 - d1, 1 + d1), transform=ax_lower.transAxes, **kwargs)
        ax_lower.plot((1 - d, 1 + d), (1 - d1, 1 + d1), transform=ax_lower.transAxes, **kwargs)

    if broken:
        if if_cifar:
            draw_break_marks(ax_top, ax_bot, width_top, width_bot)
            draw_break_marks(ax_bot, ax_even_bot, width_bot, width_even_bot)
        else:
            draw_break_marks(ax_top, ax_even_bot, width_top, width_even_bot)

    axxes = (ax_top, ax_bot) if (broken and if_cifar) else ((ax_top,) if broken else (ax_all,))

    for dataset, color in dataset_colors.items():
        df_ds = stat_df[stat_df["dataset"] == dataset]
        ax = axes_by_dataset.get(dataset)

        for size, ls in size_styles.items():
            d = df_ds[df_ds["scale"] == size].sort_values("noise_std")
            if d.empty:
                continue

            x = d["noise_std"].values
            y_mean = d["mean"].values
            y_min = d["min"].values
            y_max = d["max"].values

            # Build label: include parameter count when available
            label_key = (dataset, size)
            if label_key in param_labels:
                lbl = f"{ds_dict[dataset]}  # param {param_labels[label_key]}"
            else:
                lbl = f"{ds_dict[dataset]} - {size}"

            for axx in axxes:
                axx.fill_between(x, y_min, y_max, color=color, alpha=0.18)
                axx.plot(x, y_min, linestyle=ls, linewidth=2, marker="o",
                         color=color, label=lbl)

    if broken:
        ax_even_bot.set_xlabel("Relative Coefficient Perturbation ($\\sigma$)")
    else:
        ax_all.set_xlabel("Relative Coefficient Perturbation ($\\sigma$)")

    fig.text(-0.02, 0.53, "Accuracy", va='center', rotation='vertical')

    for axx in axxes:
        axx.grid(axis="y", alpha=0.3)

    # Single combined legend
    handles, labels = axxes[0].get_legend_handles_labels()
    bbox_to_anchor_set = (0.08, 0.24) if if_cifar else (0.08, 0.24)
    fig.legend(
        handles=handles, labels=labels,
        bbox_to_anchor=bbox_to_anchor_set,
        loc="upper left", borderaxespad=0., fontsize=7,
    )

    output_dir = '../results/new_structure_exps/scale/noise_fig'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{noise_mode}_noise_sensitivity.pdf",
                format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_dir}/{noise_mode}_noise_sensitivity.svg",
                format='svg', bbox_inches='tight')
    print(f'\033[KFig 1 saved into {output_dir}/')


# ---------------------------------------------------------------------------
# Figure 2: Noise sensitivity vs. parameter count
# ---------------------------------------------------------------------------

def plot_sensiticity(df,
                     datasets=['MNIST', 'FMNIST', 'cifar10'],
                     noise_mode='binary'):
    df["run_id"] = (
        df
        .groupby(["dataset", "total_parameter", "noise_std"])
        .cumcount())

    records = []
    for (dataset, total_param), group in df.groupby(
        ["dataset", "total_parameter"]):
        if group["noise_std"].nunique() < 2:
            continue
        for run_id, sub in group.groupby("run_id"):
            if sub["noise_std"].nunique() < 2:
                continue
            slope, _, _, _, _ = linregress(
                sub["noise_std"],
                sub["test_acc"]
            )
            records.append({
                "dataset": dataset,
                "total_parameter": total_param,
                "run_id": run_id,
                "sensitivity": -slope
            })

    sens_df = pd.DataFrame(records)
    stat = (
        sens_df
        .groupby(["dataset", "total_parameter"])["sensitivity"]
        .agg(["min", "max", "mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    markers = {"MNIST": "o", "FMNIST": "s", "cifar10": "^"}
    colors = {"MNIST": "tab:blue", "FMNIST": "tab:green", "cifar10": "tab:orange"}

    stat["std"] = stat["std"].fillna(0)
    for ds in datasets:
        sub = stat[stat["dataset"] == ds].sort_values("total_parameter")
        ds_mirror = {"MNIST": "MNIST", "FMNIST": "FashionMNIST", "cifar10": "CIFAR-10"}
        ax.errorbar(
            sub["total_parameter"], sub["max"],
            yerr=sub["std"],
            marker=markers.get(ds, "o"),
            color=colors.get(ds, "tab:black"),
            capsize=3, linewidth=1.5,
            label=ds_mirror[ds]
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}k")
    )
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Noise Sensitivity $|dAcc/d\\sigma|$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = '../results/new_structure_exps/scale/noise_fig'
    plt.savefig(f"{output_dir}/{noise_mode}_parameter_sensitivity_analysis.pdf",
                format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_dir}/{noise_mode}_parameter_sensitivity_analysis.svg",
                format='svg', bbox_inches='tight')
    print(f'Fig 2 saved into {output_dir}/')


# ---------------------------------------------------------------------------
# Figure 3: Relative accuracy vs. parameter count (coloured by sigma)
# ---------------------------------------------------------------------------

def plot_acc_scale(df,
                   datasets=['MNIST', 'FMNIST'],
                   noise_mode='binary'):
    stat_df = (
        df
        .groupby(["dataset", "noise_std", "total_parameter"])
        .agg(
            mean=("test_acc", "mean"),
            min=("test_acc", "min"),
            max=("test_acc", "max"),
        )
        .reset_index()
    )

    noise_stds = sorted(stat_df["noise_std"].unique())

    def make_color_series(base_color, n):
        """Create n colours fading from base_color toward white."""
        base = np.array(mcolors.to_rgb(base_color))
        white = np.array([1.0, 1.0, 1.0])
        ts = np.linspace(0.0, 0.6, n)
        return [tuple((1 - t) * base + t * white) for t in ts]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    for dataset, color in dataset_colors.items():
        if dataset not in datasets:
            continue
        noise_colors = make_color_series(color, len(noise_stds))
        noise2color = dict(zip(noise_stds, noise_colors))

        df_ds = stat_df[stat_df["dataset"] == dataset].sort_values("total_parameter")
        for i, noise in enumerate(noise_stds):
            d = df_ds[df_ds["noise_std"] == noise].sort_values("total_parameter")
            if d.empty:
                continue
            if i == 0:
                base_acc = d["min"].values

            ax.plot(
                d["total_parameter"], d["min"].values / base_acc,
                color=noise2color[noise], linewidth=2, marker="o",
                label=f"{ds_dict[dataset]}  $\\sigma$={noise}",
            )

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Relative Accuracy")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}k")
    )

    # Build per-dataset colour legends
    mnist_colors = make_color_series(dataset_colors["MNIST"], len(noise_stds))
    mnist_handles = [Line2D([0], [0], color=mnist_colors[i], lw=2) for i in range(len(noise_stds))]
    mnist_labels = [f"$\\sigma$={n:g}" for n in noise_stds]

    fmnist_colors = make_color_series(dataset_colors["FMNIST"], len(noise_stds))
    fmnist_handles = [Line2D([0], [0], color=fmnist_colors[i], lw=2) for i in range(len(noise_stds))]
    fmnist_labels = [f"$\\sigma$={n:g}" for n in noise_stds]

    leg1 = fig.legend(
        handles=mnist_handles, labels=mnist_labels, title="MNIST",
        loc="lower right", bbox_to_anchor=(0.85, 0.11),
        fontsize=7, title_fontsize=8,
    )
    leg2 = fig.legend(
        handles=fmnist_handles, labels=fmnist_labels, title="FashionMNIST",
        loc="lower right", bbox_to_anchor=(0.97, 0.11),
        fontsize=7, title_fontsize=8,
    )
    leg1._legend_box.align = "left"
    leg2._legend_box.align = "left"

    plt.tight_layout()
    output_dir = '../results/new_structure_exps/scale/noise_fig'
    plt.savefig(f"{output_dir}/{noise_mode}_acc_param.pdf",
                format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_dir}/{noise_mode}_acc_param.svg",
                format='svg', bbox_inches='tight')
    print(f'Fig 3 saved into {output_dir}/')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def arguments():
    parser = argparse.ArgumentParser(description='Options for noise-model analysis.')

    parser.add_argument('--noise_mode', type=str, default='binary',
                        choices=['binary', 'uniform', 'gauss'],
                        help='Noise mode.')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['MNIST', 'FMNIST'],
                        choices=['MNIST', 'FMNIST', 'cifar10'],
                        help='Choices of datasets.')
    parser.add_argument('--broken', action='store_true', default=False,
                        help='Draw broken axis or not. Write --broken for True.')
    parser.add_argument('--parent', type=str, default="../results/new_structure_exps",
                        help='Parent folder of the result files(.csv). "../results/new_structure_exps" by default.')

    return parser.parse_args()


def main():
    args = arguments()
    df, df_ana = load_all_result(parent=args.parent,
                                 noise_mode=args.noise_mode)

    plot_acc_sigma(df, df_ana,
                   datasets=args.datasets,
                   noise_mode=args.noise_mode,
                   broken=args.broken)
    # Uncomment to also produce these figures:
    # plot_sensiticity(df, datasets=args.datasets, noise_mode=args.noise_mode)
    # plot_acc_scale(df, datasets=args.datasets, noise_mode=args.noise_mode)


if __name__ == '__main__':
    main()
