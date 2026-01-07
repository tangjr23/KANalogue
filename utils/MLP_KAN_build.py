import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import pandas as pd
import torch


# from functools import lru_cache
# from function.data_utils import piecewise_from_excel, piecewise_from_params_vectorized

# ======================================================
# 不同激活函数的 MLP 网络结构
# ======================================================
class RTDAF(nn.Module):
    # 定义类属性（常量）
    q = 1.60218e-19   # Electron charge (C)
    k_B = 1.38065e-23 # Boltzmann constant (J/K)
    T = 300           # Temperature (K)
    a = 0.0039        # Amperes
    b = 0.5           # Volts
    c = 0.0874        # Volts
    d = 0.0073        # Volts
    n1 = 0.0352
    n2 = 0.0031
    h = 0.0367        # Amperes

    def __init__(self, clamp=10, vertical_shift=0.3564077949854996, scale=0.44437022056753417):
        """
        RTDAF Model

        Args:
            clamp (float): Clamp value for voltage.
            vertical_shift (float): Vertical shift in output scaling.
            scale (float): Output scale factor.
        """
        super(RTDAF, self).__init__()
        self.clamp = clamp
        self.vertical_shift = vertical_shift
        self.scale = scale

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RTDAF model.

        Args:
            V (torch.Tensor): Input voltage tensor.

        Returns:
            torch.Tensor: Output current tensor.
        """
        # 使用类属性
        V = torch.clamp(V, -self.clamp, self.clamp)

        term1 = self.a * torch.log(
            (1 + torch.exp(self.q / (self.k_B * self.T) * (self.b - self.c + self.n1 * V))) /
            (1 + torch.exp(self.q / (self.k_B * self.T) * (self.b - self.c - self.n1 * V)))
        )

        term2 = (np.pi / 2) + torch.atan((self.c - self.n1 * V) / self.d)
        term3 = self.h * (torch.exp(self.q / (self.k_B * self.T) * self.n2 * V) - 1)

        return (term1 * term2 + term3 + self.vertical_shift) / self.scale
    
# @lru_cache()
# def load_params_from_excel():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.dirname(current_dir)
#     excel_path = os.path.join(project_root, "IVcurve from my calculation", "IVcurve from my calculation.xlsx")    
#     return piecewise_from_excel(excel_path, "NbHf_a30a21a30_v", "NbHf_a30a21a30_i")
# print(f"[DEBUG] Excel path: {load_params_from_excel()}")

def load_univariate_spline_params(csv_path):#, device=torch.device('cuda:0')
    nodes_df = pd.read_csv(f'{csv_path}_nodes.csv')
    knots_df = pd.read_csv(f'{csv_path}_knots.csv')
    coeff_df = pd.read_csv(f'{csv_path}_coefficients.csv')    

    spline_params = {
        "x": torch.tensor(nodes_df["x"].values, dtype=torch.float64),
        "xs": torch.tensor(knots_df["xs"].values, dtype=torch.float64),
        "y": torch.tensor(nodes_df["y"].values, dtype=torch.float64),
        "a": torch.tensor(coeff_df["a"].values, dtype=torch.float64),
        "b": torch.tensor(coeff_df["b"].values, dtype=torch.float64),
        "c": torch.tensor(coeff_df["c"].values, dtype=torch.float64),
        "d": torch.tensor(coeff_df["d"].values, dtype=torch.float64),
    }
    
    return spline_params

def natural_spline_predict(x: torch.Tensor, params: dict):
    """
    Vectorized evaluation of natural cubic spline using precomputed segment coefficients.
    params must include:
        x: nodes (N,)
        a,b,c,d: tensors of length N-1 (coeffs for each interval)
    x: torch tensor of query points (any shape).
    Returns y of the same shape.
    NOTE: index selection (bucketize) is non-differentiable w.r.t. node positions,
          but gradients flow to a,b,c,d entries (if requires_grad=True).
    """
    
    x_nodes = params.get("xs", params.get("x")).to(x.device).to(x.dtype) 
    a = params["a"].to(x.device).to(x.dtype)
    a = params["a"].to(x.device).to(x.dtype)
    b = params["b"].to(x.device).to(x.dtype)
    c = params["c"].to(x.device).to(x.dtype)
    d = params["d"].to(x.device).to(x.dtype)

    # clamp into domain
    x_flat = x.clone().reshape(-1)
    x_clamped = torch.clamp(x_flat, x_nodes[0], x_nodes[-1])

    # find interval indices: using torch.bucketize (torch.searchsorted alias)
    # idx will be in [0, N-1]; we want interval i such that x in [x_i, x_{i+1})
    idx = torch.bucketize(x_clamped, x_nodes) - 1
    idx = torch.clamp(idx, 0, x_nodes.numel() - 2)

    dx = x_clamped - x_nodes[idx]
    y_flat = a[idx] * dx**3 + b[idx] * dx**2 + c[idx] * dx + d[idx]
    return y_flat.reshape(x.shape)

class CMTDAF(nn.Module):
    def __init__(self):
        super().__init__()
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ivcurve_dir = os.path.join(base, "IVcurve", "univariate", "a21")

        params = load_univariate_spline_params(ivcurve_dir)
        for k, v in params.items():
            self.register_buffer(k, v)

    def forward(self, x):
        params = {
            "x": self.x,
            "xs": self.xs,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
        }
        return natural_spline_predict(x, params)


class MOSFETac(nn.Module):
    """
    Activation function based on 2N7002 MOSFET Id–Vgs characteristics:
        - Zero output below Vth
        - Quadratic growth between Vth and Vmax
        - Constant (clipped) output above Vmax
    """
    def __init__(self, k=6.17, V_th=1, V_max=10.0):
        super(MOSFETac, self).__init__()
        self.k = k
        self.V_th = V_th
        self.V_max = V_max

    def forward(self, x):
        V_th = torch.tensor(self.V_th, dtype=x.dtype, device=x.device)
        V_max = torch.tensor(self.V_max, dtype=x.dtype, device=x.device)

        # Clip input between V_th and V_max
        clipped = torch.clamp(x, min=V_th, max=V_max)

        # Shift and square: (x - V_th)^2
        shifted = clipped - V_th
        output = self.k * shifted.pow(2)

        return output
    
class MLP_model(nn.Module):
    def __init__(self, layer_sizes, activation_fn):
        """
        参数:
            layer_sizes: list[int]，例如 [3*32*32, 2048, 512, 10]
                         表示输入维度、隐藏层维度、输出维度。
            activation_fn: 激活函数，例如 nn.ReLU() 或 nn.Sigmoid()
        """
        super(MLP_model, self).__init__()

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # 最后一层通常不加激活函数
            if i < len(layer_sizes) - 2:
                layers.append(activation_fn)

        self.network = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = x.view(x.size(0), -1)  # 展平输入
        return self.network(x)
    
# ======================================================
# 原始B样条 KAN 网络结构
# ======================================================
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # --- 构造均匀节点网格 ---
        # 官方默认区间 [-2, 2]，步长 h=4/grid_size
        h = 4.0 / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h - 2
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        # --- 两个分支权重：基础线性 + 样条补偿 ---
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.spline_weight.copy_(
                (torch.rand(self.out_features, self.in_features, self.grid_size + self.spline_order) - 0.5) * 0.1
            )

    def b_splines(self, x):
        # x: (batch, in_features)
        x = x.unsqueeze(-1)  # (batch, in_features, 1)
        # degree=0 初值：区间指示函数
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).float()
        # 递推计算高阶
        for k in range(1, self.spline_order + 1):
            left = (x - self.grid[:, :-(k + 1)]) / (
                self.grid[:, k:-1] - self.grid[:, :-(k + 1)]
            ) * bases[:, :, :-1]
            right = (self.grid[:, k + 1:] - x) / (
                self.grid[:, k + 1:] - self.grid[:, 1:-k]
            ) * bases[:, :, 1:]
            bases = left + right
        return bases.contiguous()  # (batch, in_features, grid_size + spline_order)

    def forward(self, x):
        # base 分支：线性层 + SiLU 激活
        base_out = F.linear(F.silu(x), self.base_weight)
        # 样条分支：局部非线性补偿
        spline_out = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        # 两部分相加
        return base_out + spline_out

class BSplineKAN(nn.Module):
    def __init__(self, layer_sizes, grid_size=3, spline_order=3, use_norm=False):
        """
        layer_sizes: list[int]，例如 [784, 512, 128, 10]
        grid_size: 每个输入通道的样条分段数
        spline_order: 样条阶数
        use_norm: 是否在层间使用 LayerNorm
        """
        super().__init__()
        self.use_norm = use_norm
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(
                KANLayer(layer_sizes[i], layer_sizes[i + 1], grid_size, spline_order)
            )
            if use_norm and i < len(layer_sizes) - 2:
                layers.append(nn.LayerNorm(layer_sizes[i + 1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

# ======================================================
# 使用Gottlieb作为基函数 KAN 网络结构
# ======================================================

class GottliebKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(GottliebKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.alpha = nn.Parameter(torch.randn(1))

        self.gottlieb_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.gottlieb_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def _gottlieb(self, n, x, alpha):
        if n == 0:
            return torch.ones_like(x)
        elif n == 1:
            return 2 * alpha * x
        else:
            return 2 * (alpha + n - 1) * x * self._gottlieb(n-1, x, alpha) - (alpha + 2*n - 2) * self._gottlieb(n-2, x, alpha)

    def forward(self, x):
        # Normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        # Compute the Gottlieb basis functions
        gottlieb_basis = []
        for n in range(self.degree + 1):
            gottlieb_basis.append(self._gottlieb(n, x, self.alpha))
        gottlieb_basis = torch.stack(gottlieb_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Gottlieb interpolation
        y = torch.einsum("bid,iod->bo", gottlieb_basis, self.gottlieb_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y
    
class GottliebKAN(nn.Module):
    def __init__(self, layer_sizes, degree=3, use_bn=True, flatten_size=None):
        """
        layer_sizes: list[int]，例如 [784, 512, 128, 10]
        degree: GottliebKANLayer 的阶数
        use_bn: 是否使用 LayerNorm
        flatten_size: 可选，输入展平大小（自动检测为 None 时会自动 flatten）
        """
        super(GottliebKAN, self).__init__()
        self.flatten_size = flatten_size or layer_sizes[0]
        self.use_bn = use_bn
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            layers.append(GottliebKANLayer(in_dim, out_dim, degree))
            
            # 除最后一层外添加 LayerNorm
            if use_bn and i < len(layer_sizes) - 2:
                layers.append(nn.LayerNorm(out_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        if x.dim() > 2:  # 自动展平图像输入
            x = x.view(x.size(0), -1)
        elif self.flatten_size is not None:
            x = x.view(-1, self.flatten_size)
        return self.network(x)



def build_MLP_KAN_model(model_name, input_dim, hidden_dims, output_dim=10):
    if model_name == 'MLP_RTDAF':
        return MLP_model(layer_sizes=[input_dim] + hidden_dims + [output_dim], activation_fn=RTDAF())
    elif model_name == 'MLP_CMTDAF':
        return MLP_model(layer_sizes=[input_dim] + hidden_dims + [output_dim], activation_fn=CMTDAF())
    elif model_name == 'MLP_MOSFETac':
        return MLP_model(layer_sizes=[input_dim] + hidden_dims + [output_dim], activation_fn=MOSFETac())
    elif model_name == 'BSplineKAN':
        return BSplineKAN(layer_sizes=[input_dim] + hidden_dims + [output_dim], grid_size=3, spline_order=3, use_norm=False)
    elif model_name == 'GottliebKAN':
        return GottliebKAN(layer_sizes=[input_dim] + hidden_dims + [output_dim], degree=3)

    else:
        raise ValueError(f"未知模型类型: {model_name}")