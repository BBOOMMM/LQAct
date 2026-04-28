import math

import torch
import torch.nn.functional as F


_SHIFT8 = (7, 6, 5, 4, 3, 2, 1, 0)
_BIT_PACK_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_bit_pack_weights(device: torch.device) -> torch.Tensor:
    key = (device, torch.uint8)
    weights = _BIT_PACK_CACHE.get(key)
    if weights is None:
        weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=device)
        _BIT_PACK_CACHE[key] = weights
    return weights


def _get_bit_shifts(device: torch.device) -> torch.Tensor:
    key = (device, torch.int64)
    shifts = _BIT_PACK_CACHE.get(key)
    if shifts is None:
        shifts = torch.tensor(_SHIFT8, dtype=torch.int64, device=device)
        _BIT_PACK_CACHE[key] = shifts
    return shifts


def _pack_bits(bits: torch.Tensor) -> torch.Tensor:
    bits = bits.reshape(-1).to(torch.uint8)
    pad_bits = (-bits.numel()) % 8
    if pad_bits:
        bits = F.pad(bits, (0, pad_bits), value=0)
    bits8 = bits.view(-1, 8)
    weights = _get_bit_pack_weights(bits.device)
    return (bits8 * weights).sum(dim=1, dtype=torch.uint8)


def _unpack_bits(packed: torch.Tensor) -> torch.Tensor:
    shifts = _get_bit_shifts(packed.device)
    return (((packed.unsqueeze(1) >> shifts) & 1) != 0).reshape(-1)


def _sample_bits_from_prob(prob: torch.Tensor, stochastic: bool) -> torch.Tensor:
    if stochastic:
        return torch.rand_like(prob).lt_(prob)
    return prob.ge_(0.5)


def quantize_1bit(x: torch.Tensor, stochastic: bool = True, eps: float = 1e-8):
    alpha = x.abs().to(torch.float32).mean().clamp_min(eps)
    a = alpha.to(device=x.device, dtype=torch.float32)
    x_fp32 = x.to(torch.float32)
    x_clipped = x_fp32.clamp(min=-a, max=a)
    prob = (x_clipped + a) / (2 * a)
    bits = _sample_bits_from_prob(prob, stochastic)
    packed_tensor = _pack_bits(bits)
    return packed_tensor, alpha, x.shape


def dequantize_1bit(packed_tensor: torch.Tensor, alpha: torch.Tensor, original_shape: torch.Size):
    flat_bits = _unpack_bits(packed_tensor)
    original_numel = math.prod(list(original_shape))
    flat_bits = flat_bits[:original_numel]
    binary_tensor = flat_bits.to(dtype=torch.float32).mul_(2.0).sub_(1.0)
    reconstructed_x = binary_tensor.view(original_shape) * alpha.to(device=packed_tensor.device, dtype=torch.float32)
    return reconstructed_x


def quantize_1bit_group(
    x: torch.Tensor,
    group_size: int = 1,
    eps: float = 1e-8,
    stochastic: bool = True,
):
    assert x.is_floating_point(), "x 需要是浮点张量"
    assert group_size >= 1

    orig_shape = tuple(int(s) for s in x.shape)
    C = orig_shape[-1]

    x2d = x.reshape(-1, C)
    N = int(x2d.shape[0])

    G = (C + group_size - 1) // group_size
    Cp = G * group_size
    pad_c = Cp - C
    if pad_c:
        x2d = F.pad(x2d, (0, pad_c), value=0.0)

    xg = x2d.view(N, G, group_size)
    xg_fp32 = xg.to(torch.float32)
    alpha_g = xg_fp32.abs().mean(dim=(0, 2)).clamp_min(eps)

    a = alpha_g.view(1, G, 1)
    x_clipped = xg_fp32.clamp(min=-a, max=a)
    prob = (x_clipped + a) / (2 * a)
    bits = _sample_bits_from_prob(prob, stochastic)
    packed = _pack_bits(bits)

    meta = {
        "orig_shape": orig_shape,
        "dtype": x.dtype,
        "C": C,
        "group_size": int(group_size),
        "G": int(G),
        "Cp": int(Cp),
        "pad_c": int(pad_c),
        "pad_bits": int((-bits.numel()) % 8),
        "N": N,
        "stochastic": bool(stochastic),
    }
    return packed, alpha_g, meta


def dequantize_1bit_group(packed: torch.Tensor, alpha_g: torch.Tensor, meta: dict):
    orig_shape = tuple(int(s) for s in meta["orig_shape"])
    C = int(meta["C"])
    group_size = int(meta["group_size"])
    Cp = int(meta["Cp"])
    N = int(meta["N"])
    out_dtype = meta.get("dtype", torch.float32)

    unpacked = _unpack_bits(packed)
    need = N * Cp
    unpacked = unpacked[:need]

    binary = unpacked.to(torch.float32).mul_(2.0).sub_(1.0).view(N, Cp)
    alpha_cols = alpha_g.to(device=packed.device, dtype=torch.float32).repeat_interleave(group_size)[:Cp]
    xrec = binary * alpha_cols.view(1, Cp)
    xrec = xrec[:, :C].view(*orig_shape)
    return xrec.to(dtype=out_dtype)
