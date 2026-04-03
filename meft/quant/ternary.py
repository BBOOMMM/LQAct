import torch
import torch.nn.functional as F
import math

def _roundclip(x: torch.Tensor, lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
    return torch.clamp(torch.round(x), lo, hi)

def quantize_ternary_group_lastdim(
    W: torch.Tensor,
    group_size: int = 32,
    eps: float = 1e-8,
):
    """
    任意维度张量 W(..., C) 的三值量化，按最后一维 C 分组：

      gamma = mean(|W|)
      alpha_g[g] = mean(|W_group_g|)   (跨所有行/元素，对每组统计)
      codes = RoundClip(W/(gamma+eps), -1, 1) ∈ {-1,0,1}

    返回：
      codes: int8，shape与W相同，取值{-1,0,1}
      alpha_g: float32，shape (G,)
      meta: dict（含orig_shape/C/group_size/Cp等）
    """
    # breakpoint()
    assert W.is_floating_point(), "W 必须是浮点张量"
    assert group_size >= 1

    orig_shape = tuple(int(s) for s in W.shape)
    C = int(orig_shape[-1])
    dtype = W.dtype

    Wf = W.to(torch.float32)
    gamma = Wf.abs().mean().clamp_min(eps)  # 标量Tensor

    # 2D: (N, C)
    W2d = Wf.reshape(-1, C)

    # 分组 + padding 到 Cp
    G = (C + group_size - 1) // group_size
    Cp = G * group_size
    pad_c = Cp - C
    if pad_c:
        W2d = F.pad(W2d, (0, pad_c), value=0.0)

    # (N, G, group_size)
    Wg = W2d.view(-1, G, group_size)

    # 每组一个 alpha（对所有N、组内元素求均值）
    alpha_g = Wg.abs().mean(dim=(0, 2)).clamp_min(eps)  # (G,)

    # 三值 codes（不乘 alpha，alpha 留给反量化/重建）
    q = _roundclip(Wg / (gamma + eps), -1.0, 1.0)  # float in {-1,0,1}
    
    
    
    codes2d = q.view(-1, Cp)[:, :C].contiguous().view(*orig_shape).to(torch.int8)

    meta = {
        "orig_shape": orig_shape,
        "dtype": dtype,
        "C": C,
        "group_size": int(group_size),
        "G": int(G),
        "Cp": int(Cp),
        "pad_c": int(pad_c),
        "gamma": gamma,  # 仅用于对齐公式/调试；反量化不需要
        "eps": float(eps),
    }
    return codes2d, alpha_g, meta


def dequantize_ternary_group_lastdim(
    codes: torch.Tensor,
    alpha_g: torch.Tensor,
    meta: dict,
):
    """
    反量化（重建）：
      W_hat_group = alpha_g[group] * codes_group
    """
    orig_shape = tuple(int(s) for s in meta["orig_shape"])
    out_dtype = meta.get("dtype", torch.float32)
    C = int(meta["C"])
    group_size = int(meta["group_size"])
    G = int(meta["G"])
    Cp = int(meta["Cp"])
    pad_c = int(meta["pad_c"])

    codes_f = codes.to(torch.float32).reshape(-1, C)  # (N, C)

    if pad_c:
        codes_f = F.pad(codes_f, (0, pad_c), value=0.0)  # (N, Cp)

    # (N, G, group_size)
    cg = codes_f.view(-1, G, group_size)
    ag = alpha_g.to(device=codes.device, dtype=torch.float32).view(1, G, 1)

    Wrec = (cg * ag).view(-1, Cp)[:, :C].contiguous().view(*orig_shape)
    return Wrec.to(dtype=out_dtype)







# import torch
# import torch.nn.functional as F

# def _roundclip(x: torch.Tensor, lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
#     return torch.clamp(torch.round(x), lo, hi)

# def quantize_ternary_group_lastdim(
#     W: torch.Tensor,
#     group_size: int = 32,
#     eps: float = 1e-8,
# ):
#     """
#     任意维度张量 W(..., C) 的三值量化，按最后一维 C 分组（group-wise ternary）：

#       alpha_g[g] = mean(|W_group_g|)   (跨所有行/元素，对每组统计)
#       codes = RoundClip(W_group / (alpha_g[g] + eps), -1, 1) ∈ {-1,0,1}

#     反量化：
#       W_hat_group = alpha_g[g] * codes_group

#     返回：
#       codes: int8，shape与W相同，取值{-1,0,1}
#       alpha_g: float32，shape (G,)
#       meta: dict（含orig_shape/C/group_size/Cp等）
#     """
#     assert W.is_floating_point(), "W 必须是浮点张量"
#     assert group_size >= 1

#     orig_shape = tuple(int(s) for s in W.shape)
#     C = int(orig_shape[-1])
#     dtype = W.dtype

#     Wf = W.to(torch.float32)

#     # 2D: (N, C)
#     W2d = Wf.reshape(-1, C)

#     # 分组 + padding 到 Cp
#     G = (C + group_size - 1) // group_size
#     Cp = G * group_size
#     pad_c = Cp - C
#     if pad_c:
#         W2d = F.pad(W2d, (0, pad_c), value=0.0)

#     # (N, G, group_size)
#     Wg = W2d.view(-1, G, group_size)

#     # 每组一个 alpha（对所有N、组内元素求均值）
#     alpha_g = Wg.abs().mean(dim=(0, 2)).clamp_min(eps)  # (G,)

#     # 用 alpha_g 做组内归一化，保证量化/反量化一致
#     ag = alpha_g.view(1, G, 1)  # broadcast
#     q = _roundclip(Wg / ag, -1.0, 1.0)  # float in {-1,0,1}

#     codes2d = q.view(-1, Cp)[:, :C].contiguous().view(*orig_shape).to(torch.int8)

#     meta = {
#         "orig_shape": orig_shape,
#         "dtype": dtype,
#         "C": C,
#         "group_size": int(group_size),
#         "G": int(G),
#         "Cp": int(Cp),
#         "pad_c": int(pad_c),
#         "eps": float(eps),
#     }
#     return codes2d, alpha_g, meta


# def dequantize_ternary_group_lastdim(
#     codes: torch.Tensor,
#     alpha_g: torch.Tensor,
#     meta: dict,
# ):
#     """
#     反量化（重建）：
#       W_hat_group = alpha_g[group] * codes_group
#     """
#     orig_shape = tuple(int(s) for s in meta["orig_shape"])
#     out_dtype = meta.get("dtype", torch.float32)
#     C = int(meta["C"])
#     group_size = int(meta["group_size"])
#     G = int(meta["G"])
#     Cp = int(meta["Cp"])
#     pad_c = int(meta["pad_c"])

#     codes_f = codes.to(torch.float32).reshape(-1, C)  # (N, C)

#     if pad_c:
#         codes_f = F.pad(codes_f, (0, pad_c), value=0.0)  # (N, Cp)

#     # (N, G, group_size)
#     cg = codes_f.view(-1, G, group_size)
#     ag = alpha_g.to(device=codes.device, dtype=torch.float32).view(1, G, 1)

#     Wrec = (cg * ag).view(-1, Cp)[:, :C].contiguous().view(*orig_shape)
#     return Wrec.to(dtype=out_dtype)





def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def fwht_lastdim(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Fast Walsh–Hadamard Transform along last dim.
    要求 last dim = power of 2
    """
    n = x.shape[-1]
    assert (n & (n - 1) == 0), f"FWHT需要长度为2的幂，当前 n={n}"
    y = x
    h = 1
    while h < n:
        y = y.view(*y.shape[:-1], -1, 2 * h)
        a = y[..., :h]
        b = y[..., h : 2 * h]
        y = torch.cat([a + b, a - b], dim=-1)
        y = y.view(*x.shape[:-1], n)
        h *= 2
    if normalize:
        y = y / math.sqrt(n)
    return y

def srht_rotate_channels(x: torch.Tensor, generator: torch.Generator | None = None):
    """
    SRHT: x -> (x * d) @ H  (沿最后一维做Hadamard)
    返回: x_rot, meta(包含d与padding信息)
    """
    assert x.is_floating_point()
    orig_shape = tuple(x.shape)
    C = x.shape[-1]
    Cp2 = _next_pow2(int(C))

    x2d = x.reshape(-1, C)  # (N, C)
    if Cp2 != C:
        x2d = F.pad(x2d, (0, Cp2 - C), value=0.0)  # pad channels

    # random Rademacher signs d in {+1,-1}
    if generator is None:
        d = torch.empty((Cp2,), device=x.device).bernoulli_(0.5).mul_(2).sub_(1)
    else:
        d = torch.empty((Cp2,), device=x.device).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)

    x_signed = x2d * d.view(1, Cp2)
    x_rot = fwht_lastdim(x_signed, normalize=True)  # (N, Cp2)

    meta = {"orig_shape": orig_shape, "C": int(C), "Cp2": int(Cp2), "d": d}
    return x_rot.view(*orig_shape[:-1], Cp2), meta

def srht_unrotate_channels(x_rot: torch.Tensor, meta: dict):
    """
    inverse SRHT: x_rot -> H @ x_rot -> * d
    注意：fwht在normalize=True时自逆（同一个fwht调用两次就是逆）
    """
    C = int(meta["C"])
    Cp2 = int(meta["Cp2"])
    d = meta["d"]

    y2d = x_rot.reshape(-1, Cp2)
    y = fwht_lastdim(y2d, normalize=True)
    y = y * d.view(1, Cp2)
    y = y[:, :C]  # unpad channels
    return y.view(*meta["orig_shape"])

def quantize_ternary_with_srht(
    W: torch.Tensor,
    group_size: int = 32,
    eps: float = 1e-8,
    generator: torch.Generator | None = None,
):
    """
    W --SRHT--> W_rot --ternary group quant--> codes, alpha_g, meta_all
    """
    W_rot, meta_h = srht_rotate_channels(W.to(torch.float32), generator=generator)
    codes, alpha_g, meta_q = quantize_ternary_group_lastdim(W_rot, group_size=group_size, eps=eps)
    meta_all = {"hadamard": meta_h, "quant": meta_q}
    return codes, alpha_g, meta_all

def dequantize_ternary_with_srht(
    codes: torch.Tensor,
    alpha_g: torch.Tensor,
    meta_all: dict,
):
    """
    codes --dequant--> W_rot_hat --invSRHT--> W_hat
    """
    W_rot_hat = dequantize_ternary_group_lastdim(codes, alpha_g, meta_all["quant"])
    W_hat = srht_unrotate_channels(W_rot_hat, meta_all["hadamard"])
    return W_hat