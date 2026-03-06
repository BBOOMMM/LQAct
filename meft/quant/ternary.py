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
#     任意维度张量 W(..., C) 的三值量化，按最后一维 C 分组：

#       gamma = mean(|W|)
#       alpha_g[g] = mean(|W_group_g|)   (跨所有行/元素，对每组统计)
#       codes = RoundClip(W/(gamma+eps), -1, 1) ∈ {-1,0,1}

#     返回：
#       codes: int8，shape与W相同，取值{-1,0,1}
#       alpha_g: float32，shape (G,)
#       meta: dict（含orig_shape/C/group_size/Cp等）
#     """
#     # breakpoint()
#     assert W.is_floating_point(), "W 必须是浮点张量"
#     assert group_size >= 1

#     orig_shape = tuple(int(s) for s in W.shape)
#     C = int(orig_shape[-1])
#     dtype = W.dtype

#     Wf = W.to(torch.float32)
#     gamma = Wf.abs().mean().clamp_min(eps)  # 标量Tensor

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

#     # 三值 codes（不乘 alpha，alpha 留给反量化/重建）
#     q = _roundclip(Wg / (gamma + eps), -1.0, 1.0)  # float in {-1,0,1}
    
    
    
#     codes2d = q.view(-1, Cp)[:, :C].contiguous().view(*orig_shape).to(torch.int8)

#     meta = {
#         "orig_shape": orig_shape,
#         "dtype": dtype,
#         "C": C,
#         "group_size": int(group_size),
#         "G": int(G),
#         "Cp": int(Cp),
#         "pad_c": int(pad_c),
#         "gamma": gamma,  # 仅用于对齐公式/调试；反量化不需要
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







import torch
import torch.nn.functional as F

def _roundclip(x: torch.Tensor, lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
    return torch.clamp(torch.round(x), lo, hi)

def quantize_ternary_group_lastdim(
    W: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-8,
):
    """
    任意维度张量 W(..., C) 的三值量化，按最后一维 C 分组（group-wise ternary）：

      alpha_g[g] = mean(|W_group_g|)   (跨所有行/元素，对每组统计)
      codes = RoundClip(W_group / (alpha_g[g] + eps), -1, 1) ∈ {-1,0,1}

    反量化：
      W_hat_group = alpha_g[g] * codes_group

    返回：
      codes: int8，shape与W相同，取值{-1,0,1}
      alpha_g: float32，shape (G,)
      meta: dict（含orig_shape/C/group_size/Cp等）
    """
    assert W.is_floating_point(), "W 必须是浮点张量"
    assert group_size >= 1

    orig_shape = tuple(int(s) for s in W.shape)
    C = int(orig_shape[-1])
    dtype = W.dtype

    Wf = W.to(torch.float32)

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

    # 用 alpha_g 做组内归一化，保证量化/反量化一致
    ag = alpha_g.view(1, G, 1)  # broadcast
    q = _roundclip(Wg / ag, -1.0, 1.0)  # float in {-1,0,1}

    codes2d = q.view(-1, Cp)[:, :C].contiguous().view(*orig_shape).to(torch.int8)

    meta = {
        "orig_shape": orig_shape,
        "dtype": dtype,
        "C": C,
        "group_size": int(group_size),
        "G": int(G),
        "Cp": int(Cp),
        "pad_c": int(pad_c),
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


