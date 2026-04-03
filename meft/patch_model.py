from .trainer import MeftConfig, MeftTrainer
import warnings
from collections.abc import Iterable

import torch
from transformers.trainer import Trainer
from transformers.modeling_utils import PreTrainedModel

from .compressed import get_compress_processor
from .config import MeftConfig
from .patch import apply_patch_to_model

def patch_model(meft_config: MeftConfig | None = None, model: PreTrainedModel | None = None):
    if meft_config is None:
        meft_config = MeftConfig()
    if isinstance(meft_config.patch_locations, str):
        if meft_config.patch_locations == "layer":
            patch_locations = ("ckpt_layer",)
        elif meft_config.patch_locations == "sublayer":
            patch_locations = ("norm", "ckpt_attn", "ckpt_mlp")
        else:
            raise ValueError("Invalid value of `meft_config.patch_locations`, must be `'layer'` or `'sublayer'`.")
    elif isinstance(meft_config.patch_locations, Iterable | None):
        patch_locations = meft_config.patch_locations
    else:
        raise TypeError("Invalid type of `meft_config.patch_locations`, must be `str`, `Iterable`, or `None`.")
    
    if isinstance(meft_config.compress_method, str | None):
        compress_method = meft_config.compress_method
    else:
        raise TypeError("Invalid type of `meft_config.compress_method`, must be `str` or `None`.")

    if isinstance(meft_config.compress_kwargs, dict | None):
        compress_kwargs = meft_config.compress_kwargs
    else:
        raise TypeError("Invalid type of `meft_config.compress_kwargs`, must be `dict` or `None`.")

    if isinstance(meft_config.compress_workers, int | None):
        compress_workers = meft_config.compress_workers
    else:
        raise TypeError("Invalid type of `meft_config.compress_workers`, must be `int` or `None`.")
    
    if isinstance(meft_config.quant_method, str | None):
        quant_method = meft_config.quant_method
    else:
        raise TypeError("Invalid type of `meft_config.compress_kwargs`, must be `dict` or `None`.")
    

    if patch_locations:
        if isinstance(model, PreTrainedModel):
            apply_patch_to_model(
                model,
                patch_locations=patch_locations,
                compress_method=compress_method,
                compress_kwargs=compress_kwargs,
                quant_method=quant_method,
            )
        elif hasattr(model, "get_base_model") and isinstance(model.get_base_model(), PreTrainedModel):
            apply_patch_to_model(
                model.get_base_model(),
                patch_locations=patch_locations,
                compress_method=compress_method,
                compress_kwargs=compress_kwargs,
                quant_method=quant_method,
            )
        else:
            warnings.warn("The model is not an instance of PreTrainedModel. No patch will be applied.")