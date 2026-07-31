"""
Comparison model architectures for the KANalogue paper benchmark.

Implements:
  - MLP-RTD:     Standard MLP with ReLU activation
  - MLP-CMTD:    Standard MLP (same arch, used with different hyperparams)
  - KAN-BSpline: KAN using B-spline basis functions
  - KAN-Gottlieb: KAN using Gottlieb polynomial basis functions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# MLP baselines (RTD / CMTD — same architecture, different lr/batch)
# ===========================================================================

def build_mlp(input_dim: int, hidden_dims: list, output_dim: int = 10,
              activation: str = "relu") -> nn.Module:
    """Build a standard MLP: Linear -> Act -> ... -> Linear."""
    dims = [input_dim] + hidden_dims + [output_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


# ===========================================================================
# B-spline KAN (simplified — based on efficient-kan / pykan approach)
# ===========================================================================

class BSplineKANLayer(nn.Module):
    """KAN layer using B-spline basis functions.

    Each edge i->j has (grid_size + spline_order) B-spline basis coefficients
    plus a residual SiLU weight, following the standard KAN parameterisation.
    """

    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 8,
                 spline_order: int = 3, grid_range: tuple = (-1, 1)):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_basis = grid_size + spline_order

        # Build uniform knot vector with extended padding
        lo, hi = grid_range
        step = (hi - lo) / grid_size
        # Extended knots: k extra knots at each end
        knots = torch.cat([
            torch.full((spline_order,), lo - step * spline_order),
            torch.linspace(lo, hi, grid_size + 1),
            torch.full((spline_order,), hi + step * spline_order),
        ])
        self.register_buffer("knots", knots)  # (grid_size + 2*spline_order + 1,)

        self.coeffs = nn.Parameter(
            torch.empty(in_dim, out_dim, self.n_basis)
        )
        self.residual_weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.coeffs, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.residual_weight, a=math.sqrt(5))

    @staticmethod
    def _cox_de_boor(x: torch.Tensor, knots: torch.Tensor, k: int) -> torch.Tensor:
        """Evaluate all k-th order B-spline basis functions at x.

        x: (N,)  clamped to [knots[k], knots[-k-1]]
        knots: (M,)
        k: spline order (3 = cubic)

        Returns (N, M - k - 1) = (N, n_basis)
        """
        N = x.shape[0]
        M = knots.shape[0]
        n_basis = M - k - 1  # = grid_size + k

        x = x.clamp(knots[k], knots[-k - 1])

        # Find interval: knots[j] <= x < knots[j+1]
        # Use the internal knots [k:-k] for searching
        inner = knots[k:-k]  # grid_size + 1 points
        idx = torch.bucketize(x, inner) - 1  # in [0, grid_size - 1]
        idx = idx.clamp(0, len(inner) - 2)   # map to knot index offset
        j = idx + k  # absolute knot index

        # Build N_{i,0} for all i (shape: N, n_basis + k)
        N0 = torch.zeros(N, n_basis + k, device=x.device, dtype=x.dtype)
        N0.scatter_(1, (j - k).unsqueeze(1), 1.0)
        N0.scatter_(1, (j).unsqueeze(1), 1.0)
        # Actually, N_{i,0}(x) = 1 if knots[i] <= x < knots[i+1]
        # For each x with interval j, N_{j,0} = 1
        # Reset and set only the correct interval
        N0.zero_()
        N0.scatter_(1, j.unsqueeze(1), 1.0)

        Nprev = N0  # (N, n_basis + k)
        for p in range(1, k + 1):
            n_cur = n_basis + k - p
            Ncur = torch.zeros(N, n_cur, device=x.device, dtype=x.dtype)
            for i in range(n_cur):
                alpha = (x - knots[i]) / (knots[i + p] - knots[i] + 1e-12)
                beta = (knots[i + p + 1] - x) / (knots[i + p + 1] - knots[i + 1] + 1e-12)
                Ncur[:, i] = alpha * Nprev[:, i] + beta * Nprev[:, i + 1]
            Nprev = Ncur

        # Nprev: (N, n_basis)
        return Nprev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_dim) -> (batch, out_dim)"""
        B, D = x.shape
        # Evaluate B-spline basis for each input dimension
        xf = x.reshape(-1)
        basis = self._cox_de_boor(xf, self.knots, self.spline_order)  # (B*D, n_basis)
        basis = basis.reshape(B, D, self.n_basis)  # (B, D, n_basis)
        # Contract
        spline_out = torch.einsum("bik,iok->bo", basis, self.coeffs)
        residual = torch.einsum("bi,io->bo", F.silu(x), self.residual_weight)
        return spline_out + residual


class BSplineKAN(nn.Module):
    """Multi-layer B-spline KAN."""

    def __init__(self, layer_dims: list, grid_size: int = 8,
                 spline_order: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                BSplineKANLayer(layer_dims[i], layer_dims[i + 1],
                                grid_size=grid_size, spline_order=spline_order)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x


# ===========================================================================
# Gottlieb KAN (using Gottlieb orthogonal polynomials)
# ===========================================================================

class GottliebKANLayer(nn.Module):
    """Single KAN layer using Gottlieb polynomial basis functions.

    Gottlieb polynomials G_n(x) are orthogonal on [-1, 1] with
    recurrence:  (n+2)G_{n+2} = (2n+3)x G_{n+1} - (n+1)G_n
    with G_0 = 1, G_1 = 2x.
    """

    def __init__(self, in_dim: int, out_dim: int, degree: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree  # number of polynomial basis functions

        self.coeffs = nn.Parameter(
            torch.empty(in_dim, out_dim, degree)
        )
        self.residual_weight = nn.Parameter(
            torch.empty(in_dim, out_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.coeffs, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.residual_weight, a=math.sqrt(5))

    def _gottlieb_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Gottlieb polynomials G_0..G_{d-1} at x.

        Returns (..., degree).
        """
        x = x.clamp(-1.0, 1.0)
        polys = [torch.ones_like(x), 2 * x]  # G_0, G_1
        for n in range(1, self.degree - 1):
            # (n+2) G_{n+2} = (2n+3) x G_{n+1} - (n+1) G_n
            g_next = ((2 * n + 3) * x * polys[-1] - (n + 1) * polys[-2]) / (n + 2)
            polys.append(g_next)
        return torch.stack(polys[:self.degree], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_dim) -> (batch, out_dim)"""
        basis = self._gottlieb_basis(x.unsqueeze(-1).expand(-1, -1, 1))
        if basis.dim() == 4:
            basis = basis.squeeze(-2)
        poly_out = torch.einsum("bik,iok->bo", basis, self.coeffs)
        residual = torch.einsum("bi,io->bo", F.silu(x), self.residual_weight)
        return poly_out + residual


class GottliebKAN(nn.Module):
    """Multi-layer Gottlieb KAN."""

    def __init__(self, layer_dims: list, degree: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                GottliebKANLayer(layer_dims[i], layer_dims[i + 1],
                                 degree=degree)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x


# ===========================================================================
# Unified builder
# ===========================================================================

def build_comparison_model(model_type: str, input_dim: int,
                           hidden_dims: list, output_dim: int = 10,
                           **kwargs) -> nn.Module:
    """Build a comparison model by type.

    Parameters
    ----------
    model_type : str
        'mlp-rtd', 'mlp-cmtd', 'kan-bspline', 'kan-gottlieb'
    """
    dims = [input_dim] + hidden_dims + [output_dim]

    if model_type in ('mlp-rtd', 'mlp-cmtd'):
        return build_mlp(input_dim, hidden_dims, output_dim, activation='relu')
    elif model_type == 'kan-bspline':
        return BSplineKAN(dims, grid_size=kwargs.get('grid_size', 8),
                          spline_order=kwargs.get('spline_order', 3))
    elif model_type == 'kan-gottlieb':
        return GottliebKAN(dims, degree=kwargs.get('degree', 4))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
