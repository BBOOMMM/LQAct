import warnings

import transformers
from torch.nn import ModuleList

from ..functions import nn_layer_norm_forward, nn_linear_forward, gelu_forward, nn_layer_norm_forward_lowrank_plus_quantization
from ..patch import _checkpoint_module, _patch_module

import copy


def apply_patch_to_dinov2_model(
    model: "transformers.models.dinov2.modeling_dinov2.Dinov2PreTrainedModel",
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
    from transformers.models.dinov2.modeling_dinov2 import Dinov2Model, Dinov2Layer
    base_model: Dinov2Model = model.base_model

    class CheckpointDinov2MLPWarning(UserWarning): ...
    warnings.simplefilter("once", CheckpointDinov2MLPWarning)

    if compress_method == "dynamic_fixed_rank_dynamic_quantization":
        if type(compress_kwargs["rank"]) is not dict:
            for layer in base_model.encoder.layer:
                layer: Dinov2Layer
                if norm:
                    _patch_module(
                        layer.norm1,
                        nn_layer_norm_forward_lowrank_plus_quantization,
                        compress_method=compress_method,
                        compress_kwargs=compress_kwargs,
                        quant_method=quant_method,
                    )
                    _patch_module(
                        layer.norm2,
                        nn_layer_norm_forward_lowrank_plus_quantization,
                        compress_method=compress_method,
                        compress_kwargs=compress_kwargs,
                        quant_method=quant_method,
                    )
                if attn_in:
                    _patch_module(layer.attention.attention.query, nn_linear_forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
                    _patch_module(layer.attention.attention.key, nn_linear_forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
                    _patch_module(layer.attention.attention.value, nn_linear_forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
                if attn_out:
                    _patch_module(layer.attention.output.dense, nn_linear_forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
                if mlp_in:
                    _patch_module(layer.mlp.fc1, nn_linear_forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
                if mlp_out:
                    _patch_module(layer.mlp.fc2, nn_linear_forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
                if act_fn:
                    _patch_module(layer.mlp.activation, gelu_forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
                if ckpt_attn:
                    _checkpoint_module(layer.attention, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
                if ckpt_mlp:
                    warnings.warn("Dinov2 only supports checkpointing the first layer of MLP.", CheckpointDinov2MLPWarning)
                    _checkpoint_module(layer.mlp, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
                if ckpt_layer:
                    _checkpoint_module(layer, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            return

        for i in range(len(base_model.encoder.layer)):
            base_model.encoder.layer: ModuleList
            layer: Dinov2Layer = base_model.encoder.layer[i]
            compress_kwargs_layer = copy.deepcopy(compress_kwargs)
            compress_kwargs_layer["rank"] = None
            if norm:
                kwargs_norm1 = copy.deepcopy(compress_kwargs_layer)
                kwargs_norm1["rank"] = compress_kwargs["rank"][f"layer_{i}.norm1"]["output"]
                _patch_module(
                    layer.norm1,
                    nn_layer_norm_forward_lowrank_plus_quantization,
                    compress_method=compress_method,
                    compress_kwargs=kwargs_norm1,
                    quant_method=quant_method,
                )
                kwargs_norm2 = copy.deepcopy(compress_kwargs_layer)
                kwargs_norm2["rank"] = compress_kwargs["rank"][f"layer_{i}.norm2"]["output"]
                _patch_module(
                    layer.norm2,
                    nn_layer_norm_forward_lowrank_plus_quantization,
                    compress_method=compress_method,
                    compress_kwargs=kwargs_norm2,
                    quant_method=quant_method,
                )
            if attn_in:
                kwargs_attn_q = copy.deepcopy(compress_kwargs_layer)
                kwargs_attn_q["rank"] = compress_kwargs["rank"][f"layer_{i}.attention_query"]["input"]
                _patch_module(layer.attention.attention.query, nn_linear_forward, compress_method=compress_method, compress_kwargs=kwargs_attn_q, quant_method=quant_method)
                kwargs_attn_k = copy.deepcopy(compress_kwargs_layer)
                kwargs_attn_k["rank"] = compress_kwargs["rank"][f"layer_{i}.attention_key"]["input"]
                _patch_module(layer.attention.attention.key, nn_linear_forward, compress_method=compress_method, compress_kwargs=kwargs_attn_k, quant_method=quant_method)
                kwargs_attn_v = copy.deepcopy(compress_kwargs_layer)
                kwargs_attn_v["rank"] = compress_kwargs["rank"][f"layer_{i}.attention_value"]["input"]
                _patch_module(layer.attention.attention.value, nn_linear_forward, compress_method=compress_method, compress_kwargs=kwargs_attn_v, quant_method=quant_method)
            if attn_out:
                kwargs_attn_out = copy.deepcopy(compress_kwargs_layer)
                kwargs_attn_out["rank"] = compress_kwargs["rank"][f"layer_{i}.attention_output_dense"]["input"]
                _patch_module(layer.attention.output.dense, nn_linear_forward, compress_method=compress_method, compress_kwargs=kwargs_attn_out, quant_method=quant_method)
            if mlp_in:
                kwargs_mlp_in = copy.deepcopy(compress_kwargs_layer)
                kwargs_mlp_in["rank"] = compress_kwargs["rank"][f"layer_{i}.mlp_fc1"]["input"]
                _patch_module(layer.mlp.fc1, nn_linear_forward, compress_method=compress_method, compress_kwargs=kwargs_mlp_in, quant_method=quant_method)
            if mlp_out:
                kwargs_mlp_out = copy.deepcopy(compress_kwargs_layer)
                kwargs_mlp_out["rank"] = compress_kwargs["rank"][f"layer_{i}.mlp_fc2"]["input"]
                _patch_module(layer.mlp.fc2, nn_linear_forward, compress_method=compress_method, compress_kwargs=kwargs_mlp_out, quant_method=quant_method)
            if act_fn:
                _patch_module(layer.mlp.activation, gelu_forward, compress_method=compress_method, compress_kwargs=compress_kwargs_layer, quant_method=quant_method)
            if ckpt_attn:
                kwargs_attention = copy.deepcopy(compress_kwargs_layer)
                kwargs_attention["rank"] = compress_kwargs["rank"][f"layer_{i}.attention"]["input"]
                _checkpoint_module(layer.attention, compress_method=compress_method, compress_kwargs=kwargs_attention, quant_method=quant_method)
            if ckpt_mlp:
                warnings.warn("Dinov2 only supports checkpointing the first layer of MLP.", CheckpointDinov2MLPWarning)
                kwargs_mlp = copy.deepcopy(compress_kwargs_layer)
                kwargs_mlp["rank"] = compress_kwargs["rank"][f"layer_{i}.mlp"]["input"]
                _checkpoint_module(layer.mlp, compress_method=compress_method, compress_kwargs=kwargs_mlp, quant_method=quant_method)
            if ckpt_layer:
                kwargs_layer = copy.deepcopy(compress_kwargs_layer)
                kwargs_layer["rank"] = compress_kwargs["rank"][f"layer_{i}"]["input"]
                _checkpoint_module(layer, compress_method=compress_method, compress_kwargs=kwargs_layer, quant_method=quant_method)
        return

    if type(compress_kwargs['rank']) is not dict:
        for layer in base_model.encoder.layer:
            layer: Dinov2Layer
            if norm:
                _patch_module(layer.norm1, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
                _patch_module(layer.norm2, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
            if attn_in:
                _patch_module(layer.attention.attention.query, nn_linear_forward, compress_kwargs=compress_kwargs)
                _patch_module(layer.attention.attention.key, nn_linear_forward, compress_kwargs=compress_kwargs)
                _patch_module(layer.attention.attention.value, nn_linear_forward, compress_kwargs=compress_kwargs)
            if attn_out:
                _patch_module(layer.attention.output.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
            if mlp_in:
                _patch_module(layer.mlp.fc1, nn_linear_forward, compress_kwargs=compress_kwargs)
            if mlp_out:
                _patch_module(layer.mlp.fc2, nn_linear_forward, compress_kwargs=compress_kwargs)
            if act_fn:
                _patch_module(layer.mlp.activation, gelu_forward, compress_kwargs=compress_kwargs)
            if ckpt_attn:
                _checkpoint_module(layer.attention, compress_kwargs=compress_kwargs)
            if ckpt_mlp:
                warnings.warn("Dinov2 only supports checkpointing the first layer of MLP.", CheckpointDinov2MLPWarning)
                _checkpoint_module(layer.mlp, compress_kwargs=compress_kwargs)
            if ckpt_layer:
                _checkpoint_module(layer, compress_kwargs=compress_kwargs)
    
    else:
        for i in range(len(base_model.encoder.layer)):
            base_model.encoder.layer : ModuleList
            layer: Dinov2Layer = base_model.encoder.layer[i]
            compress_kwargs_layer_tocopy = copy.deepcopy(compress_kwargs)
            compress_kwargs_layer_tocopy['rank'] = None
            if norm:
                kwargs_layernorm_before = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_layernorm_before['rank'] = compress_kwargs['rank'][f'layer_{i}.norm1']['output']
                _patch_module(layer.norm1, nn_layer_norm_forward, compress_kwargs=kwargs_layernorm_before)
                kwargs_layernorm_after = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_layernorm_after['rank'] = compress_kwargs['rank'][f'layer_{i}.norm2']['output']
                _patch_module(layer.norm2, nn_layer_norm_forward, compress_kwargs=kwargs_layernorm_after)
            if attn_in:
                kwargs_attn_q = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_attn_q['rank'] = compress_kwargs['rank'][f'layer_{i}.attention_query']['input']
                _patch_module(layer.attention.attention.query, nn_linear_forward, compress_kwargs=kwargs_attn_q)
                kwargs_attn_k = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_attn_k['rank'] = compress_kwargs['rank'][f'layer_{i}.attention_key']['input']
                _patch_module(layer.attention.attention.key, nn_linear_forward, compress_kwargs=kwargs_attn_k)
                kwargs_attn_v = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_attn_v['rank'] = compress_kwargs['rank'][f'layer_{i}.attention_value']['input']
                _patch_module(layer.attention.attention.value, nn_linear_forward, compress_kwargs=kwargs_attn_v)
            if attn_out:
                kwargs_attn_out = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_attn_out['rank'] = compress_kwargs['rank'][f'layer_{i}.attention_output_dense']['input']
                _patch_module(layer.attention.output.dense, nn_linear_forward, compress_kwargs=kwargs_attn_out)
            if mlp_in:
                kwargs_mlp_in = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_mlp_in['rank'] = compress_kwargs['rank'][f'layer_{i}.mlp_fc1']['input']
                _patch_module(layer.mlp.fc1, nn_linear_forward, compress_kwargs=kwargs_mlp_in)
            if mlp_out:
                kwargs_mlp_out = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_mlp_out['rank'] = compress_kwargs['rank'][f'layer_{i}.mlp_fc2']['input']
                _patch_module(layer.mlp.fc2, nn_linear_forward, compress_kwargs=kwargs_mlp_out)
            if act_fn:
                raise NotImplementedError
                # _patch_module(layer.intermediate.intermediate_act_fn, gelu_forward, compress_kwargs=compress_kwargs)
            if ckpt_attn:
                kwargs_attention = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_attention['rank'] = compress_kwargs['rank'][f'layer_{i}.attention']['input']
                _checkpoint_module(layer.attention, compress_kwargs=kwargs_attention)
            if ckpt_mlp:
                warnings.warn("Dinov2 only supports checkpointing the first layer of MLP.", CheckpointDinov2MLPWarning)
                kwargs_mlp = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_mlp['rank'] = compress_kwargs['rank'][f'layer_{i}.mlp']['input']
                _checkpoint_module(layer.mlp, compress_kwargs=kwargs_mlp)
            if ckpt_layer:
                kwargs_layer = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_layer['rank'] = compress_kwargs['rank'][f'layer_{i}']['input']
                _checkpoint_module(layer, compress_kwargs=kwargs_layer)
