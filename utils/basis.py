import torch

def piecewise_predict(x, params):
    """
    Args:
        x (Tensor): 输入电压值的张量，形状为 (N,)
        params_dict (dict): 分段线性拟合的参数字典，由 piecewise_from_excel 返回，包含每段的起止电压和电流

    Returns:
        Tensor: 对应于输入电压 x 的拟合电流值张量，形状为 (N,)

    功能说明:
        - 使用 params_dict 中的分段参数，对输入张量 x 中的每个值执行分段线性插值
        - 自动判断每个 x 属于哪一段，并计算其对应的线性插值结果
        - 对落在分段范围之外（extrapolation）的值，使用边界电流进行外推处理
    """

    # 构建 segment 边界和斜率的 tensor  
    v0s = params['v0'].to(x.device).to(x.dtype)
    v1s = params['v1'].to(x.device).to(x.dtype)
    i0s = params['i0'].to(x.device).to(x.dtype)
    i1s = params['i1'].to(x.device).to(x.dtype)

    slopes = (i1s - i0s) / (v1s - v0s + 1e-12)  # (S,)

    # 为每个 x 找到它属于哪个 segment（广播比较）
    x_exp = x.unsqueeze(-1)  # (N, 1)
    in_segment = (x_exp >= v0s) & (x_exp <= v1s)  # (N, S)

    # 找出每个 x 所属的 segment 的索引（第一个满足条件的）
    seg_idx = torch.argmax(in_segment.to(torch.int), dim=-1)  # (N,)

    # 利用索引提取对应参数
    v0 = v0s[seg_idx]  # (N,)
    i0 = i0s[seg_idx]  # (N,)
    slope = slopes[seg_idx]  # (N,)

    # 插值计算
    result = i0 + slope * (x - v0)  # (N,)

    # 处理 extrapolation
    below = x < v0s[0]
    above = x > v1s[-1]
    result[below] = i0s[0]
    result[above] = i1s[-1]

    return result

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

def torch_polyval(coeffs: torch.Tensor, x: torch.Tensor):
    x = x.to(coeffs.dtype)
    coeffs = coeffs.to(x.device)
    y = torch.zeros_like(x, dtype=coeffs.dtype)
    for c in coeffs:
        y = y * x + c
    return y

def poly_predict(x: torch.Tensor, params: dict, clamp_left=True, clamp_right=True):
    """
    Evaluate polynomial fit stored in params at input x.
    x may be any shape; returned same shape.
    """
    x = x.to(params["coeffs"].dtype)
    y = torch_polyval(params["coeffs"], x)

    if clamp_left:
        V_min = params["V_min"].to(x.dtype)
        y_left = torch_polyval(params["coeffs"], V_min)
        y = torch.where(x < V_min, y_left, y)
    if clamp_right:
        V_max = params["V_max"].to(x.dtype)
        y_right = torch_polyval(params["coeffs"], V_max)
        y = torch.where(x > V_max, y_right, y)
    return y


