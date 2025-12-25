from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline, PPoly, BSpline
from numpy.fft import fft, ifft, fftfreq

import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './utils'))
from utils.basis import natural_spline_predict
from utils.basis import poly_predict

# ------------------------------------------------------------------------------
# Utilities: I/O and data cleaning
# ------------------------------------------------------------------------------

def load_iv_from_csv(csv_path, v0_col='v0', v1_col='v1', i0_col='i0', i1_col='i1'):
    """
    Read CSV and expand points same as original pattern:
      V_ori = concat(v0, last(v1))
      I_ori = concat(i0, last(i1))

    Then drop duplicate V (keep first), sort by V, and return numpy arrays (V, I).
    """
    df = pd.read_csv(csv_path)
    # Expand like original
    V_ori = np.concatenate([df[v0_col].values, [df[v1_col].values[-1]]])
    I_ori = np.concatenate([df[i0_col].values, [df[i1_col].values[-1]]])

    df2 = pd.DataFrame({'V': V_ori, 'I': I_ori})
    df2 = df2.drop_duplicates(subset=['V'], keep='first')
    df2 = df2.sort_values('V')
    V = df2['V'].values
    I = df2['I'].values
    return V, I, df2  # return df2 for plotting if desired

# ------------------------------------------------------------------------------
# Fitting functions
# ------------------------------------------------------------------------------

def fit_spline_from_arrays(x, y, s=0.1, dtype=torch.float64, 
                           require_grad=True, fit_mode='natural'):
    """
    Fit natural cubic spline using SciPy.CubicSpline.
    Returns a dict of torch tensors:
      {
        "x": nodes (N,)          -- dtype
        "y": values (N,)         -- dtype
        "a","b","c","d": (M,)    -- segment coefficients for each interval, dtype
      }
    Note: SciPy's CubicSpline.c has shape (4, n_intervals) with coefficients for (x-x_i)**3 ... **0
    """
    if fit_mode == 'natural':
        cs = CubicSpline(x, y, bc_type='natural')
        # cs.c shape: (4, n_intervals)  -> order: c0*(x-x_i)**3 + c1*(x-x_i)**2 + c2*(x-x_i) + c3
        c_all = cs.c  # numpy
        a_np, b_np, c_np, d_np = c_all[0, :], c_all[1, :], c_all[2, :], c_all[3, :]
        params = {
            "x": torch.tensor(x, dtype=dtype, requires_grad=False),
            "y": torch.tensor(y, dtype=dtype, requires_grad=False),
            "a": torch.tensor(a_np, dtype=dtype, requires_grad=require_grad),
            "b": torch.tensor(b_np, dtype=dtype, requires_grad=require_grad),
            "c": torch.tensor(c_np, dtype=dtype, requires_grad=require_grad),
            "d": torch.tensor(d_np, dtype=dtype, requires_grad=require_grad),
        }
    elif fit_mode == 'univariate':
        us = UnivariateSpline(x, y, s=s)
        c = us.get_coeffs()
        t, _, k = us._eval_args 
        bs = BSpline(t, c, k)
        pp = PPoly.from_spline(bs)
        c_all = pp.c
        
        a_np, b_np, c_np, d_np = c_all[0, :], c_all[1, :], c_all[2, :], c_all[3, :]
        params = {
            "x": torch.tensor(x, dtype=dtype, requires_grad=False),
            "y": torch.tensor(y, dtype=dtype, requires_grad=False),
            "xs": torch.tensor(pp.x, dtype=dtype, requires_grad=False),
            "a": torch.tensor(a_np, dtype=dtype, requires_grad=require_grad),
            "b": torch.tensor(b_np, dtype=dtype, requires_grad=require_grad),
            "c": torch.tensor(c_np, dtype=dtype, requires_grad=require_grad),
            "d": torch.tensor(d_np, dtype=dtype, requires_grad=require_grad),
        }

    return params

def fit_poly_from_csv(x, y, degree=15, dtype=torch.float64, require_grad=True):
    """
    Fit polynomial (numpy.polyfit) and return dict with torch coeffs and meta.
    coeffs follow numpy.polyfit order: highest degree first.
    """
    coeffs_np = np.polyfit(x, y, deg=degree)  # numpy float64
    coeffs = torch.tensor(coeffs_np, dtype=dtype, requires_grad=require_grad)

    params = {
        "coeffs": coeffs,                 # torch tensor, shape (degree+1,)
        "x": torch.tensor(x, dtype=dtype, requires_grad=False),
        "y": torch.tensor(y, dtype=dtype, requires_grad=False),
        "V_min": torch.tensor(x.min(), dtype=dtype),
        "V_max": torch.tensor(x.max(), dtype=dtype),
        "degree": int(degree),
    }
    return params

# --------------------------------------------------------------------------
# FFT-based smoothing of spline parameters
# --------------------------------------------------------------------------

def fft_smooth_spline(params, 
                      n_samples=4096, cutoff_frac=0.05,
                      filter_type='gaussian', gaussian_sigma_frac=0.02,
                      apply_mirror=True, dtype=torch.float64):
    """
    Apply FFT smoothing to an existing natural spline's sampled curve.
    Returns new smoothed spline parameters (torch.float64, requires_grad=True for coeffs).
    """

    # ---- 1. Setup sampling domain ----
    x_nodes = params["x"].detach().cpu().numpy()
    y_nodes = params["y"].detach().cpu().numpy()
    x_min, x_max = float(x_nodes[0]), float(x_nodes[-1])
    x_uniform = np.linspace(x_min, x_max, n_samples)

    # Sample spline into uniform grid
    with torch.no_grad():
        x_torch = torch.tensor(x_uniform, dtype=dtype)
        y_torch = natural_spline_predict(x_torch, params)
        y_uniform = y_torch.cpu().numpy()

    # ---- 2. Optional mirror padding ----
    if apply_mirror:
        y_ext = np.concatenate([y_uniform[::-1], y_uniform, y_uniform[::-1]])
        x_range = x_max - x_min
        dx = x_range / (n_samples - 1)
        data_for_fft = y_ext
    else:
        data_for_fft = y_uniform
        dx = (x_max - x_min) / (n_samples - 1)

    n_fft = len(data_for_fft)
    freqs = fftfreq(n_fft, d=dx)
    nyquist = 0.5 / dx

    # ---- 3. Construct frequency filter ----
    cutoff_freq = cutoff_frac * nyquist
    if filter_type == 'ideal':
        H = (np.abs(freqs) <= cutoff_freq).astype(np.float64)
    elif filter_type == 'gaussian':
        sigma = gaussian_sigma_frac * nyquist
        H = np.exp(-0.5 * (freqs / sigma) ** 2)
    else:
        raise ValueError("filter_type must be 'ideal' or 'gaussian'")

    # ---- 4. FFT filtering ----
    Y = fft(data_for_fft)
    Y_filtered = Y * H
    y_filtered_ext = np.real(ifft(Y_filtered))

    # ---- 5. Crop back if mirrored ----
    if apply_mirror:
        # start = n_fft // 3
        # end = start + n_samples
        # y_smooth = y_filtered_ext[start:end]
        start = len(y_uniform)
        end = start * 2
        y_smooth = y_filtered_ext[start:end]
    else:
        y_smooth = y_filtered_ext

    # ---- 6. Refit cubic spline to smoothed data ----
    cs = CubicSpline(x_uniform, y_smooth, bc_type='natural')
    a, b, c, d = cs.c  # (4, n_intervals)

    new_params = {
        "x": torch.tensor(x_uniform, dtype=dtype, requires_grad=False),
        "y": torch.tensor(y_smooth, dtype=dtype, requires_grad=False),
        "x_ori": torch.tensor(x_nodes, dtype=dtype, requires_grad=False),
        "y_ori": torch.tensor(y_nodes, dtype=dtype, requires_grad=False),
        "a": torch.tensor(a, dtype=dtype, requires_grad=True),
        "b": torch.tensor(b, dtype=dtype, requires_grad=True),
        "c": torch.tensor(c, dtype=dtype, requires_grad=True),
        "d": torch.tensor(d, dtype=dtype, requires_grad=True),
    }

    return new_params

# ------------------------------------------------------------------------------
# Saving functions
# ------------------------------------------------------------------------------

def save_spline_params(params: dict, outfile_prefix: str):
    """Save nodes and coefficients for cubic spline"""
    Path(outfile_prefix).parent.mkdir(parents=True, exist_ok=True)
    nodes_df = pd.DataFrame({
        'x': params["x"].cpu().numpy(),
        'y': params["y"].cpu().numpy()
    })
    nodes_df.to_csv(f"{outfile_prefix}_nodes.csv", index=False)

    coeffs_df = pd.DataFrame({
        'a': params["a"].detach().cpu().numpy(),
        'b': params["b"].detach().cpu().numpy(),
        'c': params["c"].detach().cpu().numpy(),
        'd': params["d"].detach().cpu().numpy(),
    })
    coeffs_df.to_csv(f"{outfile_prefix}_coefficients.csv", index=False)

def save_univariate_params(params: dict, outfile_prefix: str):
    Path(outfile_prefix).parent.mkdir(parents=True, exist_ok=True)
    nodes_df = pd.DataFrame({
        'x': params["x"].cpu().numpy(),
        'y': params["y"].cpu().numpy()
    })
    nodes_df.to_csv(f"{outfile_prefix}_nodes.csv", index=False)

    coeffs_df = pd.DataFrame({
            'a': params["a"].detach().cpu().numpy(),
            'b': params["b"].detach().cpu().numpy(),
            'c': params["c"].detach().cpu().numpy(),
            'd': params["d"].detach().cpu().numpy(),
        })
    coeffs_df.to_csv(f"{outfile_prefix}_coefficients.csv", index=False)

    knots_df = pd.DataFrame({
        'xs': params["xs"].detach().cpu().numpy(),
    })
    knots_df.to_csv(f"{outfile_prefix}_knots.csv", index=False)

def save_poly_params(params: dict, outfile_prefix: str):
    Path(outfile_prefix).parent.mkdir(parents=True, exist_ok=True)
    coeffs = params["coeffs"].cpu().detach().numpy()
    degs = list(range(len(coeffs)-1, -1, -1))
    df = pd.DataFrame({
        'coefficient': coeffs,
        'degree': degs
    })
    df.to_csv(f"{outfile_prefix}_coefficients.csv", index=False)
    pd.DataFrame({'V_min': [params['V_min'].item()], 'V_max': [params['V_max'].item()]}).to_csv(
        f"{outfile_prefix}_clamp_params.csv", index=False)
    # save nodes for reference
    pd.DataFrame({'x': params['x'].cpu().numpy(), 'y': params['y'].cpu().numpy()}).to_csv(
        f"{outfile_prefix}_nodes.csv", index=False)

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------

def plot_basis(x, y, outpath: str):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, '-', label='basis')
    plt.scatter(x, y)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current')
    plt.title(f'Basis')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()

def plot_fit(params: dict, predict_fn, outpath: str, num_points=2000, show_nodes=True, is_fft=False):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    x_nodes = params['x'].cpu().numpy()
    y_nodes = params['y'].cpu().numpy()
    x_min, x_max = float(x_nodes.min()), float(x_nodes.max())
    x_plot = torch.linspace(x_min, x_max, num_points, dtype=params['x'].dtype)
    with torch.no_grad():
        y_plot = predict_fn(x_plot, params).cpu().numpy()

    plt.figure(figsize=(8,6))
    plt.plot(x_plot.numpy(), y_plot, label='fit')
    if show_nodes:
        if is_fft:
            x_nodes = params['x_ori'].cpu().numpy()
            y_nodes = params['y_ori'].cpu().numpy()
        plt.scatter(x_nodes, y_nodes, color='k', label='data')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------
# Batch processing
# ------------------------------------------------------------------------------

def process_file(filepath: Path, out_dir: Path, mode='spline', degree=15, **kwargs):
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile_prefix = str(out_dir / filepath.stem)
    logging.info("Processing %s -> %s (mode=%s)", filepath, outfile_prefix, mode)

    V, I, _ = load_iv_from_csv(str(filepath))
    if mode == 'line':
        plot_basis(V, I, f"{outfile_prefix}.pdf")
    elif mode == 'spline':
        params = fit_spline_from_arrays(V, I, dtype=torch.float64, fit_mode='natural')
        save_spline_params(params, outfile_prefix)
        plot_fit(params, natural_spline_predict, f"{outfile_prefix}.pdf")
    elif mode == 'poly':
        params = fit_poly_from_csv(V, I, degree=degree, dtype=torch.float64, require_grad=True)
        save_poly_params(params, outfile_prefix)
        plot_fit(params, poly_predict, f"{outfile_prefix}.pdf")
    elif mode == 'fft':
        params = fit_spline_from_arrays(V, I, dtype=torch.float64, fit_mode='natural')
        smoothed_params = fft_smooth_spline(params,
                                            n_samples=kwargs.get('n_samples', 4096),
                                            cutoff_frac=kwargs.get('cutoff_frac', 0.05),
                                            filter_type=kwargs.get('filter_type', 'gaussian'),
                                            gaussian_sigma_frac=kwargs.get('gaussian_sigma_frac', 0.0038),
                                            apply_mirror=kwargs.get('apply_mirror', True))
        save_spline_params(smoothed_params, outfile_prefix)
        plot_fit(smoothed_params, natural_spline_predict, f"{outfile_prefix}.pdf", is_fft=True)
    elif mode == 'univariate':
        params = fit_spline_from_arrays(V, I, s=kwargs.get('s', 0.1), fit_mode='univariate')
        save_univariate_params(params, outfile_prefix)
        plot_fit(params, natural_spline_predict, f'{outfile_prefix}.pdf')
    else:
        raise ValueError(f"Unknown mode: {mode}")

def batch_process(parent: str, folders: list, out_root: str, mode='spline', degree=15, **kwargs):
    success = 0
    for folder in folders:
        in_dir = Path(parent) / folder
        out_dir = Path(out_root) / folder
        if not in_dir.exists():
            logging.warning("Input folder not found: %s", in_dir)
            continue
        csvs = list(in_dir.glob("*.csv"))
        for csvf in csvs:
            try:
                process_file(csvf, out_dir, mode=mode, degree=degree, **kwargs)
                success += 1
            except Exception as e:
                logging.exception("Failed processing %s: %s", csvf, e)
    logging.info("Processed %d files", success)

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def make_argparser():
    p = argparse.ArgumentParser(description="IV curve fitting utilities")
    p.add_argument("--mode", choices=['line', 'spline', 'poly', 'univariate', 'fft'], default='spline')
    p.add_argument("--input", type=str, default='IVcurve/line',
                   help="Input CSV file or parent directory (if --batch)")
    p.add_argument("--output", type=str, default="IVcurve/fitted_curves",
                   help="Output directory or prefix")
    p.add_argument("--batch", action='store_true', help="Process multiple folders under input")
    p.add_argument("--folders", type=str, nargs='*',
                   default=['neg-larger', 'neg-norm', 'neg-ori', 'odd-sym', 'pos-larger', 'pos-norm', 'pos-ori'])
    p.add_argument("--degree", type=int, default=15)
    p.add_argument("--smoothing", type=float, default=0.1)
    return p

def main_cli():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    p = make_argparser()
    args = p.parse_args()

    if args.batch:
        batch_process(parent=args.input, folders=args.folders, out_root=args.output,
                      mode=args.mode, degree=args.degree, s=args.smoothing)
    else:
        inp = Path(args.input)
        out_dir = Path(args.output)
        if inp.is_file():
            process_file(inp, out_dir, mode=args.mode, degree=args.degree, s=args.smoothing)
        else:
            # process all CSVs in directory
            csvs = list(inp.glob("*.csv"))
            for csvf in csvs:
                process_file(csvf, out_dir, mode=args.mode, degree=args.degree, s=args.smoothing)

if __name__ == '__main__':
    main_cli()

