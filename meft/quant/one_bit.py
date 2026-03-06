import torch

# def quantize_1bit(x: torch.Tensor):
#     alpha = x.abs().mean().item()
    
#     # x_clipped = torch.clamp(x, -alpha, alpha)
#     # p = (x_clipped + alpha) / (2 * alpha)
#     # rand_matrix = torch.rand_like(p)
#     # binary_tensor = torch.where(rand_matrix < p, 1.0, -1.0)
    
#     binary_tensor = torch.where(x > 0, 1.0, -1.0)
    
#     bit_tensor = (binary_tensor > 0).to(torch.uint8)
    
#     flat_bits = bit_tensor.flatten()
#     original_numel = flat_bits.numel()
    
#     pad_len = (8 - original_numel % 8) % 8
#     if pad_len > 0:
#         flat_bits = torch.nn.functional.pad(flat_bits, (0, pad_len), value=0)
    
#     flat_bits = flat_bits.view(-1, 8)
    
#     powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=x.device)
    
#     packed_tensor = (flat_bits * powers).sum(dim=1, dtype=torch.uint8)
    
#     return packed_tensor, alpha, x.shape


# def dequantize_1bit(packed_tensor: torch.Tensor, alpha: float, original_shape: torch.Size):
#     powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=packed_tensor.device)
    
#     unpacked_bits = (packed_tensor.unsqueeze(1) & powers) > 0

#     flat_bits = unpacked_bits.flatten()
#     original_numel = original_shape[0] * original_shape[1]
#     flat_bits = flat_bits[:original_numel]

#     binary_tensor = torch.where(flat_bits, 1.0, -1.0)

#     reconstructed_x = binary_tensor.view(original_shape) * alpha
    
#     return reconstructed_x


def quantize_1bit(x: torch.Tensor):
    # alpha 保持为 Tensor（便于save_for_backward/设备一致）
    alpha = x.abs().mean().to(dtype=torch.float32)

    # 如果你要无偏随机量化，用下面的（并可选clip）：
    x_clipped = torch.clamp(x, -alpha, alpha)
    p = (x_clipped + alpha) / (2 * alpha)
    binary_tensor = torch.where(torch.rand_like(p) < p, 1.0, -1.0)

    # binary_tensor = torch.where(x > 0, 1.0, -1.0)
    
    bit_tensor = (binary_tensor > 0).to(torch.uint8)

    flat_bits = bit_tensor.flatten()
    original_numel = flat_bits.numel()

    pad_len = (8 - original_numel % 8) % 8
    if pad_len > 0:
        flat_bits = F.pad(flat_bits, (0, pad_len), value=0)

    flat_bits = flat_bits.view(-1, 8)
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=x.device)
    packed_tensor = (flat_bits * powers).sum(dim=1, dtype=torch.uint8)

    # shape 仍返回 torch.Size/tuple 都行，但不要假设二维
    return packed_tensor, alpha, x.shape


def dequantize_1bit(packed_tensor: torch.Tensor, alpha: torch.Tensor, original_shape: torch.Size):
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=packed_tensor.device)
    unpacked_bits = (packed_tensor.unsqueeze(1) & powers) > 0

    flat_bits = unpacked_bits.flatten()

    # 修复：支持任意维度shape
    original_numel = math.prod(list(original_shape))
    flat_bits = flat_bits[:original_numel]

    binary_tensor = torch.where(flat_bits, 1.0, -1.0).to(dtype=torch.float32, device=packed_tensor.device)
    reconstructed_x = binary_tensor.view(original_shape) * alpha.to(device=packed_tensor.device, dtype=torch.float32)
    return reconstructed_x



# import math
# import torch
# import torch.nn.functional as F

# def quantize_1bit_group(x: torch.Tensor, group_size: int = 1, eps: float = 1e-8):
#     """
#     按通道分组(最后一维C分组)的1-bit随机量化 + bit-pack
#     返回:
#       packed:  uint8 打包后的比特流 (一维)
#       alpha_g: (G,) 每个通道组的缩放系数
#       meta:    dict，包含反量化所需的形状/分组信息
#     """
#     assert x.is_floating_point(), "x 需要是浮点张量"
#     assert group_size >= 1

#     orig_shape = tuple(x.shape)
#     C = x.shape[-1]
#     X2d = x.reshape(-1, C)  # (N, C)
#     N = X2d.shape[0]

#     # 1) 按组对C做padding，保证能整除group_size
#     G = math.ceil(C / group_size)
#     Cp = G * group_size
#     pad_c = Cp - C
#     if pad_c > 0:
#         X2d = F.pad(X2d, (0, pad_c), value=0.0)  # pad last dim: (left=0,right=pad_c)

#     # 2) reshape成 (N, G, group_size)
#     Xg = X2d.view(N, G, group_size)

#     # 3) 每组一个alpha：在 (N, group_size) 上求 mean(|x|)
#     alpha_g = Xg.abs().mean(dim=(0, 2))  # (G,)
#     alpha_g = alpha_g.clamp_min(eps)     # 防止除0

#     # # 4) clamp + 概率p + 随机二值化（按组广播alpha）
#     # a = alpha_g.view(1, G, 1)  # broadcast到(N,G,group)
#     # Xc = torch.clamp(Xg, -a, a)
#     # p = (Xc + a) / (2 * a)     # in [0,1]
#     # binary = torch.where(torch.rand_like(p) < p, 1.0, -1.0)
    
#     binary = torch.where(Xg > 0, 1.0, -1.0)

#     # 5) bit-pack：{-1,+1}->{0,1} 然后按8个bit打包到uint8
#     bits = (binary > 0).to(torch.uint8).reshape(-1)  # 展平

#     pad_bits = (8 - (bits.numel() % 8)) % 8
#     if pad_bits > 0:
#         bits = F.pad(bits, (0, pad_bits), value=0)

#     bits8 = bits.view(-1, 8)
#     powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=x.device)
#     packed = (bits8 * powers).sum(dim=1, dtype=torch.uint8)

#     meta = {
#         "orig_shape": orig_shape,
#         "C": C,
#         "group_size": group_size,
#         "Cp": Cp,
#         "pad_bits": pad_bits,
#     }
#     return packed, alpha_g, meta


# def dequantize_1bit_group(packed: torch.Tensor, alpha_g: torch.Tensor, meta: dict):
#     """
#     对 stochastic_1bit_group_quantize 的反量化
#     """
#     orig_shape = meta["orig_shape"]
#     C = meta["C"]
#     group_size = meta["group_size"]
#     Cp = meta["Cp"]

#     # 1) unpack bits
#     powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=packed.device)
#     unpacked = ((packed.unsqueeze(1) & powers) > 0).reshape(-1)  # bool

#     # 2) 截断到需要的元素数：N*Cp
#     N = int(torch.tensor(orig_shape[:-1]).prod().item()) if len(orig_shape) > 1 else 1
#     need = N * Cp
#     unpacked = unpacked[:need]

#     # 3) {0,1}->{-1,+1}
#     binary = torch.where(unpacked, 1.0, -1.0).view(N, Cp)

#     # 4) 组alpha展开到每个通道列
#     # alpha_g: (G,) -> (Cp,)
#     alpha_cols = alpha_g.repeat_interleave(group_size)[:Cp].to(dtype=binary.dtype, device=binary.device)
#     Xrec = binary * alpha_cols.view(1, Cp)

#     # 5) 去掉C方向padding，并reshape回原形状
#     Xrec = Xrec[:, :C].view(*orig_shape)
#     return Xrec




_SHIFT8 = [7, 6, 5, 4, 3, 2, 1, 0]


def quantize_1bit_group(x: torch.Tensor, group_size: int = 1, eps: float = 1e-8):
    """
    按最后一维 C 分组的 1-bit 量化 + bit-pack（确定性符号量化）
    返回:
      packed:  (num_bytes,) uint8
      alpha_g: (G,) float32，每组缩放
      meta:    dict，反量化所需信息（含N，避免反量化阶段 .item()）
    """
    assert x.is_floating_point(), "x 需要是浮点张量"
    assert group_size >= 1

    orig_shape = tuple(int(s) for s in x.shape)
    C = orig_shape[-1]

    x2d = x.reshape(-1, C)  # (N, C)
    N = x2d.shape[0]        # SymInt in general, but stored as Python int via int() below (not used under torch.compile)
    N_int = int(N)

    # pad channels to multiple of group_size
    G = (C + group_size - 1) // group_size
    Cp = G * group_size
    pad_c = Cp - C
    if pad_c:
        x2d = F.pad(x2d, (0, pad_c), value=0.0)

    xg = x2d.view(N_int, G, group_size)  # (N, G, group_size)

    # alpha per group: mean(|x|)
    alpha_g = xg.abs().to(torch.float32).mean(dim=(0, 2)).clamp_min(eps)  # (G,)
    
    
    a = alpha_g.view(1, G, 1)  # broadcast
    x_clipped = torch.clamp(xg.to(torch.float32), -a, a)
    p = (x_clipped + a) / (2 * a)  # in [0,1]
    r = torch.rand(p.shape, device=p.device, dtype=p.dtype)
    bits = (r < p).to(torch.uint8).reshape(-1)  # {0,1}

    # 1-bit sign (deterministic)
    # bits = (xg > 0).to(torch.uint8).reshape(-1)  # {0,1}

    # pad bits to multiple of 8
    pad_bits = (-bits.numel()) % 8
    if pad_bits:
        bits = F.pad(bits, (0, pad_bits), value=0)

    bits8 = bits.view(-1, 8)  # (num_bytes, 8)

    shift = torch.tensor(_SHIFT8, device=x.device, dtype=torch.int64)  # (8,)
    packed = ((bits8.to(torch.uint8) << shift).sum(dim=1)).to(torch.uint8)  # (num_bytes,)

    meta = {
        "orig_shape": orig_shape,
        "dtype": x.dtype,
        "C": C,
        "group_size": int(group_size),
        "G": int(G),
        "Cp": int(Cp),
        "pad_c": int(pad_c),
        "pad_bits": int(pad_bits),
        "N": int(N_int),  # 关键：反量化直接用它，避免 prod().item()
    }
    return packed, alpha_g, meta


def dequantize_1bit_group(packed: torch.Tensor, alpha_g: torch.Tensor, meta: dict):
    """
    quantize_1bit_group 的反量化
    """
    orig_shape = tuple(int(s) for s in meta["orig_shape"])
    C = int(meta["C"])
    group_size = int(meta["group_size"])
    Cp = int(meta["Cp"])
    N = int(meta["N"])
    out_dtype = meta.get("dtype", torch.float32)

    # unpack bits: packed(uint8) -> (num_bits,) bool
    shift = torch.tensor(_SHIFT8, device=packed.device, dtype=torch.int64)  # (8,)
    unpacked = (((packed.unsqueeze(1) >> shift) & 1) != 0).reshape(-1)      # bool

    need = N * Cp
    unpacked = unpacked[:need]

    # {0,1} -> {-1,+1}
    binary = torch.where(unpacked, 1.0, -1.0).to(torch.float32).view(N, Cp)

    # expand group alpha to columns
    alpha_cols = (
        alpha_g.to(device=packed.device, dtype=torch.float32)
        .repeat_interleave(group_size)[:Cp]
    )  # (Cp,)

    xrec = binary * alpha_cols.view(1, Cp)        # (N, Cp)
    xrec = xrec[:, :C].view(*orig_shape)          # remove channel padding
    return xrec.to(dtype=out_dtype)






import math
import torch
import torch.nn.functional as F

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
        b = y[..., h:2*h]
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
    Cp2 = _next_pow2(C)

    x2d = x.reshape(-1, C)  # (N, C)
    if Cp2 != C:
        x2d = F.pad(x2d, (0, Cp2 - C), value=0.0)  # pad channels

    # random Rademacher signs d in {+1,-1}
    # 用int8生成再转float，减少随机开销
    if generator is None:
        d = torch.empty((Cp2,), device=x.device).bernoulli_(0.5).mul_(2).sub_(1)
    else:
        d = torch.empty((Cp2,), device=x.device).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)

    x_signed = x2d * d.view(1, Cp2)
    x_rot = fwht_lastdim(x_signed, normalize=True)  # (N, Cp2)

    meta = {"orig_shape": orig_shape, "C": C, "Cp2": Cp2, "d": d}
    return x_rot.view(*orig_shape[:-1], Cp2), meta

def srht_unrotate_channels(x_rot: torch.Tensor, meta: dict):
    """
    inverse SRHT: x_rot -> H @ x_rot -> * d
    注意：fwht在normalize=True时自逆（同一个fwht调用两次就是逆）
    """
    C = meta["C"]
    Cp2 = meta["Cp2"]
    d = meta["d"]

    y2d = x_rot.reshape(-1, Cp2)
    y = fwht_lastdim(y2d, normalize=True)
    y = y * d.view(1, Cp2)
    y = y[:, :C]  # unpad channels
    return y.view(*meta["orig_shape"])

def quantize_1bit_with_srht(R: torch.Tensor, group_size: int = 1, eps: float = 1e-8, generator: torch.Generator | None = None):
    """
    R --SRHT--> R_rot --1bit group quant--> packed, alpha_g, meta_all
    """
    R_rot, meta_h = srht_rotate_channels(R.to(torch.float32), generator=generator)
    packed, alpha_g, meta_q = quantize_1bit_group(R_rot, group_size=group_size, eps=eps)
    meta_all = {"hadamard": meta_h, "quant": meta_q}
    return packed, alpha_g, meta_all

def dequantize_1bit_with_srht(packed: torch.Tensor, alpha_g: torch.Tensor, meta_all: dict):
    """
    packed --dequant--> R_rot_hat --invSRHT--> R_hat
    """
    R_rot_hat = dequantize_1bit_group(packed, alpha_g, meta_all["quant"])
    R_hat = srht_unrotate_channels(R_rot_hat, meta_all["hadamard"])
    return R_hat