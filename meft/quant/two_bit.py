# import math
# from typing import Any, Tuple

# import torch
# from torch import Tensor

# __all__ = [
#     "quantize_2bit_group",
#     "dequantize_2bit_group",
# ]


# def _pad_last_dim(x2d: Tensor, padded_d: int) -> Tensor:
#     """Pad last dim with zeros to padded_d."""
#     n, d = x2d.shape
#     if d == padded_d:
#         return x2d
#     pad = x2d.new_zeros((n, padded_d - d))
#     return torch.cat([x2d, pad], dim=1)


# def _pack_2bit(codes: Tensor) -> Tensor:
#     """
#     codes: uint8 tensor in [0, 3], shape [N, D_pad]
#     return: uint8 packed, shape [N, ceil(D_pad/4)]
#     """
#     if codes.dtype != torch.uint8:
#         codes = codes.to(torch.uint8)

#     n, d = codes.shape
#     pad_to = int(math.ceil(d / 4) * 4)
#     if pad_to != d:
#         codes = torch.cat([codes, codes.new_zeros((n, pad_to - d))], dim=1)

#     codes4 = codes.view(n, -1, 4)  # [N, nbytes, 4]
#     packed = (
#         (codes4[..., 0] & 0x3)
#         | ((codes4[..., 1] & 0x3) << 2)
#         | ((codes4[..., 2] & 0x3) << 4)
#         | ((codes4[..., 3] & 0x3) << 6)
#     ).to(torch.uint8)
#     return packed


# def _unpack_2bit(packed: Tensor, d_pad: int) -> Tensor:
#     """
#     packed: uint8, shape [N, nbytes]
#     return: uint8 codes in [0,3], shape [N, d_pad]
#     """
#     if packed.dtype != torch.uint8:
#         packed = packed.to(torch.uint8)

#     b = packed
#     c0 = (b & 0x3)
#     c1 = ((b >> 2) & 0x3)
#     c2 = ((b >> 4) & 0x3)
#     c3 = ((b >> 6) & 0x3)
#     codes = torch.stack([c0, c1, c2, c3], dim=-1).reshape(b.shape[0], -1)
#     return codes[:, :d_pad]


# def quantize_2bit_group(
#     x: Tensor,
#     group_size: int = 128,
#     eps: float = 0.0,
# ) -> Tuple[Tensor, Tensor, Any]:
#     """
#     2-bit group-wise quantization on the last dimension.

#     Levels per element (per-group scale alpha):
#         {-2, -1, 0, +1} * alpha

#     Per-group alpha chosen to cover asymmetric range [-2, +1]:
#         alpha = max(max(x), max(-x)/2)

#     Returns:
#         packed: uint8, shape [N, ceil(D_pad/4)] where N=prod(x.shape[:-1])
#         alpha:  x.dtype, shape [N, n_groups]
#         meta:   (orig_shape, group_size, d_pad)
#     """
#     if group_size <= 0:
#         raise ValueError(f"group_size must be positive, got {group_size}")

#     orig_shape = tuple(x.shape)
#     if x.numel() == 0:
#         packed = torch.empty((0, 0), device=x.device, dtype=torch.uint8)
#         alpha = torch.empty((0, 0), device=x.device, dtype=x.dtype)
#         meta = (orig_shape, group_size, 0)
#         return packed, alpha, meta

#     d = orig_shape[-1]
#     n = int(x.numel() // d)
#     x2d = x.reshape(n, d)

#     d_pad = int(math.ceil(d / group_size) * group_size)
#     x2d_pad = _pad_last_dim(x2d, d_pad)  # [N, D_pad]

#     n_groups = d_pad // group_size
#     xg = x2d_pad.view(n, n_groups, group_size)

#     # per-group alpha for range [-2, 1]
#     pos_max = torch.amax(xg, dim=-1)                # [N, n_groups]
#     neg_max = torch.amax(torch.clamp(-xg, min=0), dim=-1)  # max magnitude on negative side
#     alpha = torch.maximum(pos_max, neg_max / 2.0)   # [N, n_groups]

#     if eps > 0:
#         alpha = torch.where(alpha > eps, alpha, alpha.new_zeros(()).expand_as(alpha))

#     alpha_e = alpha.unsqueeze(-1)  # [N, n_groups, 1]
#     inv_alpha = torch.where(alpha_e > 0, 1.0 / alpha_e, torch.zeros_like(alpha_e))

#     # quantize to integers in {-2,-1,0,1}
#     q = torch.round(xg * inv_alpha).to(torch.int16)
#     q = torch.clamp(q, min=-2, max=1)

#     # map q -> code in {0,1,2,3}: (-2->0, -1->1, 0->2, 1->3)
#     codes = (q + 2).to(torch.uint8)
#     codes2d = codes.view(n, d_pad)

#     packed = _pack_2bit(codes2d)
#     meta = (orig_shape, group_size, d_pad)
#     return packed, alpha.to(dtype=x.dtype), meta


# def dequantize_2bit_group(
#     packed: Tensor,
#     alpha: Tensor,
#     meta: Any,
# ) -> Tensor:
#     """
#     Inverse of quantize_2bit_group.

#     packed: uint8, shape [N, ceil(D_pad/4)]
#     alpha:  float tensor, shape [N, n_groups]
#     meta:   (orig_shape, group_size, d_pad)
#     """
#     orig_shape, group_size, d_pad = meta
#     orig_shape = tuple(orig_shape)
#     d = orig_shape[-1]
#     if d_pad == 0:
#         return torch.empty(orig_shape, device=packed.device, dtype=alpha.dtype)

#     n = int(math.prod(orig_shape[:-1])) if len(orig_shape) > 1 else 1
#     codes2d = _unpack_2bit(packed, d_pad).view(n, d_pad)  # [N, D_pad]

#     # code -> q in {-2,-1,0,1}
#     q = (codes2d.to(torch.int16) - 2).to(alpha.dtype)  # [N, D_pad]

#     n_groups = d_pad // group_size
#     if alpha.shape[1] != n_groups:
#         raise ValueError(f"alpha second dim mismatch: got {alpha.shape[1]}, expected {n_groups}")

#     alpha_e = alpha.repeat_interleave(group_size, dim=1)  # [N, D_pad]
#     x2d_pad = q * alpha_e
#     x2d = x2d_pad[:, :d]
#     return x2d.reshape(orig_shape)









import math
from typing import Any, Tuple

import torch
from torch import Tensor

__all__ = [
    "quantize_2bit_group",
    "dequantize_2bit_group",
]


def _pad_last_dim(x2d: Tensor, padded_d: int) -> Tensor:
    """Pad last dim with zeros to padded_d."""
    n, d = x2d.shape
    if d == padded_d:
        return x2d
    pad = x2d.new_zeros((n, padded_d - d))
    return torch.cat([x2d, pad], dim=1)


def _pack_2bit(codes: Tensor) -> Tensor:
    """
    codes: uint8 tensor in [0, 3], shape [N, D_pad]
    return: uint8 packed, shape [N, ceil(D_pad/4)]
    """
    if codes.dtype != torch.uint8:
        codes = codes.to(torch.uint8)

    n, d = codes.shape
    pad_to = int(math.ceil(d / 4) * 4)
    if pad_to != d:
        codes = torch.cat([codes, codes.new_zeros((n, pad_to - d))], dim=1)

    codes4 = codes.view(n, -1, 4)  # [N, nbytes, 4]
    packed = (
        (codes4[..., 0] & 0x3)
        | ((codes4[..., 1] & 0x3) << 2)
        | ((codes4[..., 2] & 0x3) << 4)
        | ((codes4[..., 3] & 0x3) << 6)
    ).to(torch.uint8)
    return packed


def _unpack_2bit(packed: Tensor, d_pad: int) -> Tensor:
    """
    packed: uint8, shape [N, nbytes]
    return: uint8 codes in [0,3], shape [N, d_pad]
    """
    if packed.dtype != torch.uint8:
        packed = packed.to(torch.uint8)

    b = packed
    c0 = (b & 0x3)
    c1 = ((b >> 2) & 0x3)
    c2 = ((b >> 4) & 0x3)
    c3 = ((b >> 6) & 0x3)
    codes = torch.stack([c0, c1, c2, c3], dim=-1).reshape(b.shape[0], -1)
    return codes[:, :d_pad]


def quantize_2bit_group(
    x: Tensor,
    group_size: int = 128,
    eps: float = 0.0,
) -> Tuple[Tensor, Tensor, Any]:
    """
    2-bit group-wise quantization on the last dimension.

    Levels per element (per-group scale alpha):
        {-2, -1, 0, +1} * alpha

    Per-group alpha chosen to cover asymmetric range [-2, +1]:
        alpha_g[g] = max(max(x_group_g), max(-x_group_g)/2)

    注意：这里 alpha 改为“全局按组”，保存形状为 (1, G, 1)，并在量化/反量化时广播到所有行。
    Returns:
        packed: uint8, shape [N, ceil(D_pad/4)] where N=prod(x.shape[:-1])
        alpha:  x.dtype, shape [1, G, 1]
        meta:   (orig_shape, group_size, d_pad)
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    orig_shape = tuple(x.shape)
    if x.numel() == 0:
        packed = torch.empty((0, 0), device=x.device, dtype=torch.uint8)
        alpha = torch.empty((1, 0, 1), device=x.device, dtype=x.dtype)
        meta = (orig_shape, group_size, 0)
        return packed, alpha, meta

    d = orig_shape[-1]
    n = int(x.numel() // d)
    x2d = x.reshape(n, d)

    d_pad = int(math.ceil(d / group_size) * group_size)
    x2d_pad = _pad_last_dim(x2d, d_pad)  # [N, D_pad]

    G = d_pad // group_size
    xg = x2d_pad.view(n, G, group_size)  # [N, G, group_size]

    # 全局(跨N与组内元素)每组一个 alpha_g: (G,)
    pos_max = torch.amax(xg, dim=(0, 2))  # [G]
    neg_max = torch.amax(torch.clamp(-xg, min=0), dim=(0, 2))  # [G]
    alpha_g = torch.maximum(pos_max, neg_max / 2.0)  # [G]

    if eps > 0:
        alpha_g = torch.where(alpha_g > eps, alpha_g, alpha_g.new_zeros(()).expand_as(alpha_g))

    alpha = alpha_g.view(1, G, 1)  # (1, G, 1)
    inv_alpha = torch.where(alpha > 0, 1.0 / alpha, torch.zeros_like(alpha))

    # quantize to integers in {-2,-1,0,1}
    q = torch.round(xg * inv_alpha).to(torch.int16)
    q = torch.clamp(q, min=-2, max=1)

    # map q -> code in {0,1,2,3}: (-2->0, -1->1, 0->2, 1->3)
    codes = (q + 2).to(torch.uint8)  # [N,G,group_size]
    codes2d = codes.view(n, d_pad)

    packed = _pack_2bit(codes2d)
    meta = (orig_shape, group_size, d_pad)
    return packed, alpha.to(dtype=x.dtype), meta


def dequantize_2bit_group(
    packed: Tensor,
    alpha: Tensor,
    meta: Any,
) -> Tensor:
    """
    Inverse of quantize_2bit_group.

    packed: uint8, shape [N, ceil(D_pad/4)]
    alpha:  float tensor, shape [1, G, 1] (broadcasted)
    meta:   (orig_shape, group_size, d_pad)
    """
    orig_shape, group_size, d_pad = meta
    orig_shape = tuple(orig_shape)
    d = orig_shape[-1]
    if d_pad == 0:
        return torch.empty(orig_shape, device=packed.device, dtype=alpha.dtype)

    n = int(math.prod(orig_shape[:-1])) if len(orig_shape) > 1 else 1
    codes2d = _unpack_2bit(packed, d_pad).view(n, d_pad)  # [N, D_pad]

    # code -> q in {-2,-1,0,1}
    q = (codes2d.to(torch.int16) - 2).to(alpha.dtype)  # [N, D_pad]

    G = d_pad // group_size

    # 兼容：允许 alpha 传 (G,) 或 (1,G) 或 (1,G,1)，最终规范成 (1,G,1)
    if alpha.dim() == 1:
        if alpha.numel() != G:
            raise ValueError(f"alpha numel mismatch: got {alpha.numel()}, expected {G}")
        alpha = alpha.view(1, G, 1)
    elif alpha.dim() == 2:
        if alpha.shape != (1, G):
            raise ValueError(f"alpha shape mismatch: got {tuple(alpha.shape)}, expected (1,{G})")
        alpha = alpha.view(1, G, 1)
    elif alpha.dim() == 3:
        if alpha.shape != (1, G, 1):
            raise ValueError(f"alpha shape mismatch: got {tuple(alpha.shape)}, expected (1,{G},1)")
    else:
        raise ValueError(f"alpha dim must be 1/2/3, got {alpha.dim()}")

    alpha = alpha.to(device=packed.device)

    # 展开到每个元素: (N, D_pad)
    alpha_e = alpha.expand(n, G, 1).repeat(1, 1, group_size).reshape(n, d_pad)
    x2d_pad = q * alpha_e
    x2d = x2d_pad[:, :d]
    return x2d.reshape(orig_shape)