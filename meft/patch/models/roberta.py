import transformers
import warnings

from ..functions import nn_layer_norm_forward, nn_linear_forward,nn_linear_forward_lowrank_plus_quantization, gelu_forward, gelu_forward_lowrank_plus_quantization, nn_layer_norm_forward_lowrank_plus_quantization
from ..patch import _checkpoint_module, _patch_module


def apply_patch_to_roberta_model(
    model: "transformers.models.roberta.modeling_roberta.RobertaPreTrainedModel",
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
    from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaLayer
    base_model: RobertaModel = model.base_model
    
    
    if compress_method == "lowrank_plus_quantization":
        for layer in base_model.encoder.layer:
            layer: RobertaLayer
            if norm:
                _patch_module(layer.attention.output.LayerNorm, nn_layer_norm_forward_lowrank_plus_quantization, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
                _patch_module(layer.output.LayerNorm, nn_layer_norm_forward_lowrank_plus_quantization, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if attn_in:
                _patch_module(layer.attention.self.query, nn_linear_forward_lowrank_plus_quantization, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
                _patch_module(layer.attention.self.key, nn_linear_forward_lowrank_plus_quantization, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
                _patch_module(layer.attention.self.value, nn_linear_forward_lowrank_plus_quantization, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if attn_out:
                _patch_module(layer.attention.output.dense, nn_linear_forward_lowrank_plus_quantization, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if mlp_in:
                _patch_module(layer.intermediate.dense, nn_linear_forward_lowrank_plus_quantization, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if mlp_out:
                _patch_module(layer.output.dense, nn_linear_forward_lowrank_plus_quantization, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if act_fn:
                _patch_module(layer.intermediate.intermediate_act_fn, gelu_forward_lowrank_plus_quantization, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if ckpt_attn:
                _checkpoint_module(layer.attention, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if ckpt_mlp:
                warnings.warn("RoBerTA only supports checkpointing the first layer of MLP.")
                _checkpoint_module(layer.intermediate, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if ckpt_layer:
                _checkpoint_module(layer, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            
        return
    

    for layer in base_model.encoder.layer:
        layer: RobertaLayer
        if norm:
            _patch_module(layer.attention.output.LayerNorm, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.output.LayerNorm, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
        if attn_in:
            _patch_module(layer.attention.self.query, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.attention.self.key, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.attention.self.value, nn_linear_forward, compress_kwargs=compress_kwargs)
        if attn_out:
            _patch_module(layer.attention.output.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_in:
            _patch_module(layer.intermediate.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_out:
            _patch_module(layer.output.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
        if act_fn:
            _patch_module(layer.intermediate.intermediate_act_fn, gelu_forward, compress_kwargs=compress_kwargs)
        if ckpt_attn:
            _checkpoint_module(layer.attention, compress_kwargs=compress_kwargs)
        if ckpt_mlp:
            warnings.warn("RoBerTA only supports checkpointing the first layer of MLP.")
            _checkpoint_module(layer.intermediate, compress_kwargs=compress_kwargs)
        if ckpt_layer:
            _checkpoint_module(layer, compress_kwargs=compress_kwargs)