import warnings
from collections.abc import Callable
from types import MethodType
from functools import partial

from torch import nn
from transformers.utils.import_utils import is_peft_available

from .functions import checkpoint, nn_linear_forward, checkpoint_lowrank_plus_quantization


def _checkpoint_module(
    module: nn.Module,
    compress_method: str | None = None,
    compress_kwargs: dict | None = None,
    quant_method: str | None = None,
) -> None:
    requires_grad = any(param.requires_grad for param in module.parameters())
    if compress_method == "lowrank_plus_quantization":
        module.forward = MethodType(partial(checkpoint_lowrank_plus_quantization, module.forward.__func__, requires_grad=requires_grad, compress_method=compress_method,compress_kwargs=compress_kwargs, quant_method=quant_method,), module)
    else:
        module.forward = MethodType(partial(checkpoint, module.forward.__func__, requires_grad=requires_grad, compress_kwargs=compress_kwargs), module)


def _patch_module(
    module: nn.Module,
    forward: Callable,
    compress_method: str | None = None,
    compress_kwargs: dict | None = None,
    quant_method: str | None = None,
) -> None:
    if is_peft_available():
        from peft.tuners.tuners_utils import BaseTunerLayer
        if isinstance(module, BaseTunerLayer):
            _patch_module(module.get_base_layer(), forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            _patch_peft_module(module, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            return
        elif hasattr(module, "__module__") and module.__module__.startswith("peft."):
            warnings.warn(f"No patch supported for module type: {type(module)}.")
            return
    if compress_method == 'lowrank_plus_quantization':
        module.forward = MethodType(partial(forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method), module)
    else:
        module.forward = MethodType(partial(forward, compress_kwargs=compress_kwargs), module)


def _patch_peft_module(
    module: nn.Module,
    compress_method: str | None = None,
    compress_kwargs: dict | None = None,
    quant_method: str | None = None,
) -> None:
    from peft.tuners import lora
    if isinstance(module, lora.Linear):
        for adapter in module.lora_A.values():
            _patch_module(adapter, nn_linear_forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
    else:
        warnings.warn(f"No patch supported for module type: {type(module)}.")
