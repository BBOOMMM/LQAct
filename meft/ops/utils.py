from enum import Enum

import torch
from torch import Tensor


class CastingMode(Enum):
    NONE = "none"
    ALIGN = "align"
    INPUT = "input"
    ALL = "all"


_floating_dtypes = tuple(
    value for value in vars(torch).values()
    if isinstance(value, torch.dtype) and value.is_floating_point
)

# 兼容量化/自定义 dtype，跳过不支持 torch.finfo 的类型
_floating_bits = {}
_floating_eps = {}
for dtype in _floating_dtypes:
    try:
        _floating_bits[dtype] = torch.finfo(dtype).bits
        _floating_eps[dtype] = torch.finfo(dtype).eps
    except (TypeError, NotImplementedError, ValueError):
        # 某些量化/自定义 dtype 不支持 eps/bits，跳过
        pass


def get_floating_bits(dtype: torch.dtype) -> int:
    return _floating_bits[dtype]


def get_floating_eps(dtype: torch.dtype) -> float:
    return _floating_eps[dtype]


def convert_dtype(*args, dtype: torch.dtype):
    return tuple(x.to(dtype) if isinstance(x, Tensor) else x for x in args)


def promote_dtype(*args, dtype: torch.dtype):
    return tuple(x.to(dtype) if isinstance(x, Tensor) and get_floating_bits(x.dtype) < get_floating_bits(dtype) else x for x in args)
