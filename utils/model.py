import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.basis import natural_spline_predict

class TDiode_KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, 
                 td_basis_types, td_params, 
                 acti=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.td_basis_types = td_basis_types
        self.td_params = td_params
        self.bias = nn.Parameter(torch.zeros(output_dim))

        self.coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree, dtype=torch.float64))
        nn.init.normal_(self.coeffs, mean=0.0, std=1 / (input_dim * degree))
        self.bias = nn.Parameter(torch.zeros(output_dim, dtype=torch.float64))

        self.activate = acti
        self.act = nn.Hardtanh(min_val=-1.0, max_val=1.0)

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
        
        TD_basis = []
        for basis_type in self.td_basis_types:
            params = self.td_params[basis_type]
            TD_basis.append(natural_spline_predict(x=x, params=params))
        TD_basis = torch.stack(TD_basis, dim=-1)
 
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

def build_model(input_dim, hidden_dims, 
                td_basis_types, td_params, 
                output_dim=10, acti=True, 
                device=torch.device('cuda:0'), 
                norm_layer='layer', noise_std=0.0):
    tunnel_order = len(td_basis_types)
    dims = [input_dim] + hidden_dims + [output_dim]
    layers = []

    for i in range(len(dims) - 1):
        layers.append(TDiode_KANLayer(
            dims[i], dims[i + 1], tunnel_order, 
            td_basis_types, td_params, 
            acti=acti, 
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


