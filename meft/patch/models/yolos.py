import warnings

from torch.nn import ModuleList

from ..functions import nn_layer_norm_forward, nn_layer_norm_forward_lowrank_plus_quantization
from ..patch import _checkpoint_module, _patch_module

import copy


def _get_yolos_base_model(model):
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "encoder"):
        return base_model

    vit = getattr(model, "vit", None)
    if vit is not None and hasattr(vit, "encoder"):
        return vit

    raise ValueError("Unable to locate YOLOS encoder on the provided model.")


def apply_patch_to_yolos_model(
    model,
    norm: bool = False,
    attn_in: bool = False,
    attn_out: bool = False,
    mlp_in: bool = False,
    mlp_out: bool = False,
    act_fn: bool = False,
    ckpt_attn: bool = False,
    ckpt_mlp: bool = False,
    ckpt_layer: bool = False,
    compress_method: str | None = None,
    compress_kwargs: dict | None = None,
    quant_method: str | None = None,
) -> None:
    if attn_in or attn_out or mlp_in or mlp_out or act_fn:
        raise NotImplementedError("YOLOS patch only supports norm/ckpt_* in the first version.")

    base_model = _get_yolos_base_model(model)

    class CheckpointYolosMLPWarning(UserWarning): ...

    warnings.simplefilter("once", CheckpointYolosMLPWarning)

    layers: ModuleList = base_model.encoder.layer
    norm_forward = (
        nn_layer_norm_forward_lowrank_plus_quantization
        if compress_method == "dynamic_fixed_rank_dynamic_quantization"
        else nn_layer_norm_forward
    )

    if type(compress_kwargs["rank"]) is not dict:
        for layer in layers:
            if norm:
                _patch_module(
                    layer.layernorm_before,
                    norm_forward,
                    compress_method=compress_method,
                    compress_kwargs=compress_kwargs,
                    quant_method=quant_method,
                )
                _patch_module(
                    layer.layernorm_after,
                    norm_forward,
                    compress_method=compress_method,
                    compress_kwargs=compress_kwargs,
                    quant_method=quant_method,
                )
            if ckpt_attn:
                _checkpoint_module(
                    layer.attention,
                    compress_method=compress_method,
                    compress_kwargs=compress_kwargs,
                    quant_method=quant_method,
                )
            if ckpt_mlp:
                warnings.warn(
                    "YOLOS only supports checkpointing the first layer of MLP.",
                    CheckpointYolosMLPWarning,
                )
                _checkpoint_module(
                    layer.intermediate,
                    compress_method=compress_method,
                    compress_kwargs=compress_kwargs,
                    quant_method=quant_method,
                )
            if ckpt_layer:
                _checkpoint_module(
                    layer,
                    compress_method=compress_method,
                    compress_kwargs=compress_kwargs,
                    quant_method=quant_method,
                )
        return

    for i, layer in enumerate(layers):
        compress_kwargs_layer = copy.deepcopy(compress_kwargs)
        compress_kwargs_layer["rank"] = None
        if norm:
            kwargs_before = copy.deepcopy(compress_kwargs_layer)
            kwargs_before["rank"] = compress_kwargs["rank"][f"layer_{i}.layernorm_before"]["output"]
            _patch_module(
                layer.layernorm_before,
                norm_forward,
                compress_method=compress_method,
                compress_kwargs=kwargs_before,
                quant_method=quant_method,
            )

            kwargs_after = copy.deepcopy(compress_kwargs_layer)
            kwargs_after["rank"] = compress_kwargs["rank"][f"layer_{i}.layernorm_after"]["output"]
            _patch_module(
                layer.layernorm_after,
                norm_forward,
                compress_method=compress_method,
                compress_kwargs=kwargs_after,
                quant_method=quant_method,
            )
        if ckpt_attn:
            kwargs_attention = copy.deepcopy(compress_kwargs_layer)
            kwargs_attention["rank"] = compress_kwargs["rank"][f"layer_{i}.attention"]["input"]
            _checkpoint_module(
                layer.attention,
                compress_method=compress_method,
                compress_kwargs=kwargs_attention,
                quant_method=quant_method,
            )
        if ckpt_mlp:
            warnings.warn(
                "YOLOS only supports checkpointing the first layer of MLP.",
                CheckpointYolosMLPWarning,
            )
            kwargs_mlp = copy.deepcopy(compress_kwargs_layer)
            kwargs_mlp["rank"] = compress_kwargs["rank"][f"layer_{i}.intermediate"]["input"]
            _checkpoint_module(
                layer.intermediate,
                compress_method=compress_method,
                compress_kwargs=kwargs_mlp,
                quant_method=quant_method,
            )
        if ckpt_layer:
            kwargs_layer = copy.deepcopy(compress_kwargs_layer)
            kwargs_layer["rank"] = compress_kwargs["rank"][f"layer_{i}"]["input"]
            _checkpoint_module(
                layer,
                compress_method=compress_method,
                compress_kwargs=kwargs_layer,
                quant_method=quant_method,
            )
