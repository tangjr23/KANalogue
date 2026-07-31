import torch
import torch.nn as nn
import torch.nn.functional as F

from kanalogue.basis import natural_spline_predict
from kanalogue.basis import poly_predict
from kanalogue.basis import piecewise_predict


class TDiode_KANLayer(nn.Module):
    """Single KAN layer using tunnel-diode basis functions.

    Each input-output edge is gated by a linear combination of fixed basis
    functions (spline / polynomial / piecewise-linear fits to tunnel-diode IV
    curves).  Only the ``coeffs`` tensor is learned.
    """

    def __init__(self, input_dim, output_dim, degree,
                 td_basis_types, td_params,
                 acti=True, fit_mode='univariate'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.td_basis_types = td_basis_types
        self.td_params = td_params

        self.coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree, dtype=torch.float64))
        nn.init.normal_(self.coeffs, mean=0.0, std=1 / (input_dim * degree))
        self.bias = nn.Parameter(torch.zeros(output_dim, dtype=torch.float64))

        self.activate = acti
        self.act = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        self.fit_mode = fit_mode

    def forward(self, x, noise_std: float = 0.0, noise_mode='binary'):
        x = x.to(torch.float64)
        if self.activate == 'NegHC':
            x = self.act(x)
        elif self.activate == 'PosHC':
            x = torch.clamp(x, min=0.0, max=1.4)
        elif self.activate == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.activate == 'tanh':
            x = torch.tanh(x)

        # Evaluate basis functions according to fit_mode
        TD_basis = []
        for basis_type in self.td_basis_types:
            params = self.td_params[basis_type]
            if self.fit_mode in ('spline', 'fft', 'univariate'):
                TD_basis.append(natural_spline_predict(x=x, params=params))
            elif self.fit_mode == 'poly':
                TD_basis.append(poly_predict(x=x, params=params))
            elif self.fit_mode == 'line':
                TD_basis.append(piecewise_predict(x=x, params=params))
        TD_basis = torch.stack(TD_basis, dim=-1)

        # Noise injection (for robustness evaluation)
        if noise_std > 0:
            if noise_mode == 'binary':
                noise = torch.randint(
                    low=0, high=2,
                    size=tuple(self.coeffs.shape), device=self.coeffs.device
                ) * 2 - 1
                noisy_coeffs = self.coeffs * (1 + noise_std * noise)
            elif noise_mode == 'uniform':
                import math
                a = math.sqrt(3) * noise_std
                noise = 2 * torch.rand_like(self.coeffs) - 1
                noisy_coeffs = self.coeffs * (1 + noise * a)
            elif noise_mode == 'gauss':
                noise = torch.randn_like(self.coeffs)
                noisy_coeffs = self.coeffs * (1 + noise_std * noise)
        else:
            noisy_coeffs = self.coeffs

        y = torch.einsum("bid,iod->bo", TD_basis, noisy_coeffs) + self.bias
        return y.view(-1, self.output_dim)


# ---------------------------------------------------------------------------
# Float64 normalisation layers
# ---------------------------------------------------------------------------

class LayerNorm64(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)
        if self.weight is not None:
            self.weight = nn.Parameter(self.weight.to(torch.float64))
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.to(torch.float64))

    def forward(self, x):
        return F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )


class BatchNorm64(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__(num_features, eps, momentum, affine)
        if self.weight is not None:
            self.weight = nn.Parameter(self.weight.to(torch.float64))
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.to(torch.float64))
        self.running_mean = self.running_mean.to(torch.float64)
        self.running_var = self.running_var.to(torch.float64)

    def forward(self, x):
        if x.dim() == 3:
            original_shape = x.shape
            x = x.reshape(-1, original_shape[-1])
            x = super().forward(x)
            x = x.reshape(original_shape)
            return x
        else:
            return super().forward(x)


# ---------------------------------------------------------------------------
# Pure KAN model builder (fully-connected with TDiode_KANLayers)
# ---------------------------------------------------------------------------

def build_tdkan_model(input_dim, hidden_dims,
                      td_basis_types, td_params,
                      output_dim=10, acti=True, fit_mode='univariate',
                      device=torch.device('cuda:0'),
                      norm_layer='layer', noise_std=0.0):
    """Build a sequential fully-connected KAN model using TDiode_KANLayers."""
    tunnel_order = len(td_basis_types)
    dims = [input_dim] + hidden_dims + [output_dim]
    layers = []

    for i in range(len(dims) - 1):
        layers.append(TDiode_KANLayer(
            dims[i], dims[i + 1], tunnel_order,
            td_basis_types, td_params,
            acti=acti, fit_mode=fit_mode,
        ))
        if i < len(dims) - 2:
            if norm_layer == 'layer':
                layer_norm = LayerNorm64(dims[i + 1])
            elif norm_layer == 'batch':
                layer_norm = BatchNorm64(dims[i + 1])
            elif norm_layer == 'None':
                layer_norm = None

            if layer_norm is not None:
                layer_norm.to(device=device)
                layers.append(layer_norm)

    model = nn.Sequential(*layers)

    class FlattenWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x, noise_std: float = 0.0, noise_mode='binary'):
            x = x.view(x.size(0), -1).to(torch.float64)
            for layer in self.net:
                if isinstance(layer, TDiode_KANLayer):
                    x = layer(x, noise_std=noise_std, noise_mode=noise_mode)
                else:
                    x = layer(x)
            return x

    return FlattenWrapper(model)


# ---------------------------------------------------------------------------
# CNN-KAN hybrid model
# ---------------------------------------------------------------------------

class ConvKANBlock(nn.Module):
    """Conv2d -> KAN (channel-wise) -> BatchNorm"""

    def __init__(self, in_channels, out_channels,
                 td_basis_types, td_params,
                 fit_mode='univariate'):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.kan = TDiode_KANLayer(
            input_dim=out_channels, output_dim=out_channels,
            degree=len(td_basis_types),
            td_basis_types=td_basis_types, td_params=td_params,
            fit_mode=fit_mode
        )

        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x, noise_std=0.0, noise_mode='binary'):
        x = self.conv(x)            # (B, C, H, W)
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)   # (B, H, W, C)
        x = x.reshape(-1, C)        # (BHW, C)

        x = self.kan(x, noise_std, noise_mode)
        x = x.float()
        x = x.reshape(B, H, W, C)
        x = x.permute(0, 3, 1, 2)   # (B, C, H, W)

        return self.norm(x)


class CNN_KAN(nn.Module):
    """Fixed two-stage CNN-KAN with adaptive pooling + KAN head."""

    def __init__(self, in_channels, num_classes,
                 td_basis_types, td_params, fit_mode='univariate'):
        super().__init__()

        self.stage1 = ConvKANBlock(
            in_channels, 32,
            td_basis_types, td_params, fit_mode=fit_mode
        )
        self.stage2 = ConvKANBlock(
            32, 64,
            td_basis_types, td_params, fit_mode=fit_mode
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = TDiode_KANLayer(
            input_dim=64,
            output_dim=num_classes,
            degree=len(td_basis_types),
            td_basis_types=td_basis_types,
            td_params=td_params,
            acti=False,
            fit_mode=fit_mode
        )

    def forward(self, x, noise_std=0.0, noise_mode='binary'):
        x = self.stage1(x, noise_std, noise_mode)
        x = self.stage2(x, noise_std, noise_mode)
        x = self.pool(x).flatten(1)      # (B, 64)
        x = self.head(x, noise_std, noise_mode)
        return x


class CNNKANModel(nn.Module):
    """Wrapper that delegates to build_cnn_kan."""

    def __init__(self, in_channels, hidden_dims,
                 td_basis_types, td_params,
                 output_dim, fit_mode='univariate'):
        super().__init__()
        self.model = build_cnn_kan(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            td_basis_types=td_basis_types,
            td_params=td_params,
            output_dim=output_dim,
            fit_mode=fit_mode
        )

    def forward(self, x, noise_std=0.0, noise_mode='binary'):
        return self.model(x, noise_std=noise_std, noise_mode=noise_mode)


def build_cnn_kan(in_channels, hidden_dims,
                  td_basis_types, td_params,
                  output_dim=10, fit_mode='univariate'):
    """Build a generic CNN-KAN from channel dimensions."""
    channels = [in_channels] + hidden_dims
    layers = []

    for i in range(len(channels) - 1):
        layers.append(ConvKANBlock(
            channels[i], channels[i + 1],
            td_basis_types, td_params,
            fit_mode=fit_mode
        ))
    backbone = nn.Sequential(*layers)

    class CNNWrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = TDiode_KANLayer(
                channels[-1], output_dim,
                len(td_basis_types),
                td_basis_types, td_params,
                acti=False, fit_mode=fit_mode
            )

        def forward(self, x, noise_std=0.0, noise_mode='binary'):
            for layer in self.backbone:
                x = layer(x, noise_std, noise_mode)
            x = self.pool(x).flatten(1)
            return self.head(x, noise_std=noise_std, noise_mode=noise_mode)

    return CNNWrapper(backbone)


# ---------------------------------------------------------------------------
# Unified model builder (dispatches on model_type)
# ---------------------------------------------------------------------------

def build_model(input_dim, hidden_dims,
                td_basis_types, td_params,
                output_dim=10, acti=True, fit_mode='univariate',
                device=torch.device('cuda:0'),
                norm_layer='layer',
                model_type='tdkan', in_channels=3):
    """Build a model, dispatching on ``model_type``.

    Parameters
    ----------
    model_type : str
        ``'tdkan'`` — fully-connected KAN (TDiode_KANLayer stack).
        ``'cnn'``   — CNN-KAN hybrid (ConvKANBlock stack + KAN head).
    """
    if model_type == 'cnn':
        print('===== USING CNN-KAN FOR TRAINING =====')
        model = CNNKANModel(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            td_basis_types=td_basis_types,
            td_params=td_params,
            output_dim=output_dim,
            fit_mode=fit_mode
        )
        return model.to(device)

    elif model_type == 'tdkan':
        return build_tdkan_model(
            input_dim, hidden_dims,
            td_basis_types, td_params,
            output_dim, acti, fit_mode,
            device=device, norm_layer=norm_layer
        )
