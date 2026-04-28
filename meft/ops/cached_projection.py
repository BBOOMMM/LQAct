from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..compressed import get_quant_cache
from ..quant.one_bit import (
    quantize_1bit,
    dequantize_1bit,
    quantize_1bit_group,
    dequantize_1bit_group,
)
from ..quant.ternary import quantize_ternary_group_lastdim, dequantize_ternary_group_lastdim
from ..quant.two_bit import quantize_2bit_group, dequantize_2bit_group


CACHED_PROJECTION_METHOD = "cached_projection_lowrank_dynamic_quantization"


@dataclass
class CachedProjectionState:
    rank: int | float
    T_cycle: int = 1
    niter: int = 0
    oversample: int = 0
    projection: Tensor | None = None
    projection_rank: int | None = None
    step: int = 0

    def resolve_rank(self, hidden_size: int, rows: int) -> int:
        if isinstance(self.rank, float):
            rank = int(hidden_size * self.rank)
        elif isinstance(self.rank, int):
            rank = self.rank
        else:
            raise TypeError("rank must be int or float")
        return max(1, min(rank, hidden_size, rows))

    def should_update(self, tensor_2d: Tensor, rank: int) -> bool:
        if self.projection is None:
            return True
        if self.projection.device != tensor_2d.device or self.projection.dtype != tensor_2d.dtype:
            return True
        if self.projection.shape != (tensor_2d.shape[-1], rank):
            return True
        return self.step % max(1, int(self.T_cycle)) == 0

    @torch.no_grad()
    def get_projection(self, tensor_2d: Tensor) -> Tensor:
        rows, hidden_size = tensor_2d.shape
        rank = self.resolve_rank(hidden_size, rows)
        if self.should_update(tensor_2d, rank):
            q = min(hidden_size, rows, rank + max(0, int(self.oversample)))
            _, _, v = torch.svd_lowrank(
                tensor_2d.detach().to(torch.float32),
                q=q,
                niter=max(0, int(self.niter)),
            )
            self.projection = v[:, :rank].contiguous().to(device=tensor_2d.device, dtype=tensor_2d.dtype)
            self.projection_rank = rank
        self.step += 1
        return self.projection


def _quant_cache_key(quant_method: str | None, state: CachedProjectionState) -> tuple:
    return (CACHED_PROJECTION_METHOD, quant_method, id(state))


def quantize_residual(tensor: Tensor, quant_method: str | None):
    if quant_method == "1bit_pertensor":
        return quantize_1bit(tensor)
    if quant_method == "1bit_pergroupchannel":
        return quantize_1bit_group(tensor, group_size=1)
    if quant_method == "ternary":
        return quantize_ternary_group_lastdim(tensor)
    if quant_method == "two_bit_group":
        return quantize_2bit_group(tensor, group_size=1)
    raise ValueError(f"Unsupported quant_method: {quant_method}")


def dequantize_residual(packed: Tensor, alpha: Tensor, meta: Any, quant_method: str | None) -> Tensor:
    if quant_method == "1bit_pertensor":
        return dequantize_1bit(packed, alpha, meta)
    if quant_method == "1bit_pergroupchannel":
        return dequantize_1bit_group(packed, alpha, meta)
    if quant_method == "ternary":
        return dequantize_ternary_group_lastdim(packed, alpha, meta)
    if quant_method == "two_bit_group":
        return dequantize_2bit_group(packed, alpha, meta)
    raise ValueError(f"Unsupported quant_method: {quant_method}")


@torch.no_grad()
def compress_cached_projection(
    tensor: Tensor,
    compress_kwargs: dict,
    quant_method: str | None,
    cache_tensor: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Any]:
    state = compress_kwargs.get("projection_state")
    if not isinstance(state, CachedProjectionState):
        raise TypeError("compress_kwargs['projection_state'] must be a CachedProjectionState")

    cache_tensor = tensor if cache_tensor is None else cache_tensor
    quant_cache = get_quant_cache()
    cache_key = _quant_cache_key(quant_method, state)
    if cache_tensor in quant_cache[cache_key]:
        return quant_cache[cache_key][cache_tensor]

    original_shape = tensor.shape
    tensor_2d = tensor.flatten(0, -2)
    projection = state.get_projection(tensor_2d)
    coefficients = tensor_2d @ projection
    lowrank = (coefficients @ projection.mT).reshape(original_shape)
    residual = tensor - lowrank
    packed_R, alpha, meta = quantize_residual(residual, quant_method)
    compressed = (coefficients, projection, packed_R, alpha, meta)
    quant_cache[cache_key][cache_tensor] = compressed
    return compressed


@torch.no_grad()
def reconstruct_cached_projection(
    coefficients: Tensor,
    projection: Tensor,
    residual: Tensor,
    original_shape: torch.Size | tuple[int, ...],
    requires_grad: bool,
) -> Tensor:
    tensor = (coefficients @ projection.mT).reshape(original_shape) + residual
    tensor.requires_grad = requires_grad
    return tensor
