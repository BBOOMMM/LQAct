import warnings

import transformers
from torch.nn import ModuleList

from ..functions import nn_layer_norm_forward, nn_linear_forward, gelu_forward, nn_layer_norm_forward_lowrank_plus_quantization
from ..patch import _checkpoint_module, _patch_module

import copy


# def apply_patch_to_vit_model(
#     model: "transformers.models.vit.modeling_vit.ViTPreTrainedModel",
#     norm: bool = False,
#     attn_in: bool = False,
#     attn_out: bool = False,
#     mlp_in: bool = False,
#     mlp_out: bool = False,
#     act_fn: bool = False,
#     ckpt_attn: bool = False,
#     ckpt_mlp: bool = False,
#     ckpt_layer: bool = False,
#     compress_kwargs: dict | None = None,
# ) -> None:
#     from transformers.models.vit.modeling_vit import ViTModel, ViTLayer
#     base_model: ViTModel = model.base_model

#     class CheckpointViTMLPWarning(UserWarning): ...
#     warnings.simplefilter("once", CheckpointViTMLPWarning)

#     for layer in base_model.encoder.layer:
#         layer: ViTLayer
#         if not compress_kwargs.get("lowrank_plus_quantization", False):
#             if norm:
#                 _patch_module(layer.layernorm_before, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
#                 _patch_module(layer.layernorm_after, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
#             if attn_in:
#                 _patch_module(layer.attention.attention.query, nn_linear_forward, compress_kwargs=compress_kwargs)
#                 _patch_module(layer.attention.attention.key, nn_linear_forward, compress_kwargs=compress_kwargs)
#                 _patch_module(layer.attention.attention.value, nn_linear_forward, compress_kwargs=compress_kwargs)
#             if attn_out:
#                 _patch_module(layer.attention.output.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
#             if mlp_in:
#                 _patch_module(layer.intermediate.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
#             if mlp_out:
#                 _patch_module(layer.output.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
#             if act_fn:
#                 _patch_module(layer.intermediate.intermediate_act_fn, gelu_forward, compress_kwargs=compress_kwargs)
#             if ckpt_attn:
#                 _checkpoint_module(layer.attention, compress_kwargs=compress_kwargs)
#             if ckpt_mlp:
#                 warnings.warn("ViT only supports checkpointing the first layer of MLP.", CheckpointViTMLPWarning)
#                 _checkpoint_module(layer.intermediate, compress_kwargs=compress_kwargs)
#             if ckpt_layer:
#                 _checkpoint_module(layer, compress_kwargs=compress_kwargs)
#         else:
#             if norm:
#                 _patch_module(layer.layernorm_before, nn_layer_norm_forward_lowrank_plus_quantization, compress_kwargs=compress_kwargs)
#                 _patch_module(layer.layernorm_after, nn_layer_norm_forward_lowrank_plus_quantization, compress_kwargs=compress_kwargs)
#             if attn_in:
#                 _patch_module(layer.attention.attention.query, nn_linear_forward, compress_kwargs=compress_kwargs)
#                 _patch_module(layer.attention.attention.key, nn_linear_forward, compress_kwargs=compress_kwargs)
#                 _patch_module(layer.attention.attention.value, nn_linear_forward, compress_kwargs=compress_kwargs)
#             if attn_out:
#                 _patch_module(layer.attention.output.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
#             if mlp_in:
#                 _patch_module(layer.intermediate.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
#             if mlp_out:
#                 _patch_module(layer.output.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
#             if act_fn:
#                 _patch_module(layer.intermediate.intermediate_act_fn, gelu_forward, compress_kwargs=compress_kwargs)
#             if ckpt_attn:
#                 _checkpoint_module(layer.attention, compress_kwargs=compress_kwargs)
#             if ckpt_mlp:
#                 warnings.warn("ViT only supports checkpointing the first layer of MLP.", CheckpointViTMLPWarning)
#                 _checkpoint_module(layer.intermediate, compress_kwargs=compress_kwargs)
#             if ckpt_layer:
#                 _checkpoint_module(layer, compress_kwargs=compress_kwargs)


def apply_patch_to_vit_model(
    model: "transformers.models.vit.modeling_vit.ViTPreTrainedModel",
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
    from transformers.models.vit.modeling_vit import ViTModel, ViTLayer
    base_model: ViTModel = model.base_model

    class CheckpointViTMLPWarning(UserWarning): ...
    warnings.simplefilter("once", CheckpointViTMLPWarning)
    
    if "project_matrixes" in compress_kwargs:
        print("Applying patch to ViT with project matrixes...")
        for i in range(len(base_model.encoder.layer)):
            base_model.encoder.layer : ModuleList
            layer: ViTLayer = base_model.encoder.layer[i]
            compress_kwargs_layer_tocopy = {}
            compress_kwargs_layer_tocopy['project_matrix'] = None
            if norm:
                kwargs_layernorm_before = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_layernorm_before['project_matrix'] = compress_kwargs['project_matrixes'][f'layer_{i}.layernorm_before']['output']
                _patch_module(layer.layernorm_before, nn_layer_norm_forward, compress_kwargs=kwargs_layernorm_before)
                kwargs_layernorm_after = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_layernorm_after['project_matrix'] = compress_kwargs['project_matrixes'][f'layer_{i}.layernorm_after']['output']
                _patch_module(layer.layernorm_after, nn_layer_norm_forward, compress_kwargs=kwargs_layernorm_after)
            if attn_in:
                raise NotImplementedError
                # _patch_module(layer.attention.attention.query, nn_linear_forward, compress_kwargs=compress_kwargs)
                # _patch_module(layer.attention.attention.key, nn_linear_forward, compress_kwargs=compress_kwargs)
                # _patch_module(layer.attention.attention.value, nn_linear_forward, compress_kwargs=compress_kwargs)
            if attn_out:
                raise NotImplementedError
                # _patch_module(layer.attention.output.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
            if mlp_in:
                raise NotImplementedError
                # _patch_module(layer.intermediate.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
            if mlp_out:
                raise NotImplementedError
                # _patch_module(layer.output.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
            if act_fn:
                raise NotImplementedError
                # _patch_module(layer.intermediate.intermediate_act_fn, gelu_forward, compress_kwargs=compress_kwargs)
            if ckpt_attn:
                kwargs_attention = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_attention['project_matrix'] = compress_kwargs['project_matrixes'][f'layer_{i}.attention']['input']
                _checkpoint_module(layer.attention, compress_kwargs=kwargs_attention)
            if ckpt_mlp:
                warnings.warn("ViT only supports checkpointing the first layer of MLP.", CheckpointViTMLPWarning)
                kwargs_mlp = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_mlp['project_matrix'] = compress_kwargs['project_matrixes'][f'layer_{i}.intermediate']['input']
                _checkpoint_module(layer.intermediate, compress_kwargs=kwargs_mlp)
            if ckpt_layer:
                kwargs_layer = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_layer['project_matrix'] = compress_kwargs['project_matrixes'][f'layer_{i}']['input']
                _checkpoint_module(layer, compress_kwargs=kwargs_layer)
        return

    
    # if compress_kwargs.get("lowrank_plus_quantization", False):
    if compress_method == "lowrank_plus_quantization":
        for i in range(len(base_model.encoder.layer)):
            base_model.encoder.layer : ModuleList
            layer: ViTLayer = base_model.encoder.layer[i]
            if norm:
                _patch_module(layer.layernorm_before, nn_layer_norm_forward_lowrank_plus_quantization, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
                _patch_module(layer.layernorm_after, nn_layer_norm_forward_lowrank_plus_quantization, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if attn_in:
                _patch_module(layer.attention.attention.query, nn_linear_forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
                _patch_module(layer.attention.attention.key, nn_linear_forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
                _patch_module(layer.attention.attention.value, nn_linear_forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if attn_out:
                _patch_module(layer.attention.output.dense, nn_linear_forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if mlp_in:
                _patch_module(layer.intermediate.dense, nn_linear_forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if mlp_out:
                _patch_module(layer.output.dense, nn_linear_forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if act_fn:
                _patch_module(layer.intermediate.intermediate_act_fn, gelu_forward, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if ckpt_attn:
                _checkpoint_module(layer.attention, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if ckpt_mlp:
                warnings.warn("ViT only supports checkpointing the first layer of MLP.", CheckpointViTMLPWarning)
                _checkpoint_module(layer.intermediate, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            if ckpt_layer:
                _checkpoint_module(layer, compress_method=compress_method, compress_kwargs=compress_kwargs, quant_method=quant_method)
            
        return
    
    
    if type(compress_kwargs['rank']) is not dict:
        for layer in base_model.encoder.layer:
            layer: ViTLayer
            if norm:
                _patch_module(layer.layernorm_before, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
                _patch_module(layer.layernorm_after, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
            if attn_in:
                _patch_module(layer.attention.attention.query, nn_linear_forward, compress_kwargs=compress_kwargs)
                _patch_module(layer.attention.attention.key, nn_linear_forward, compress_kwargs=compress_kwargs)
                _patch_module(layer.attention.attention.value, nn_linear_forward, compress_kwargs=compress_kwargs)
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
                warnings.warn("ViT only supports checkpointing the first layer of MLP.", CheckpointViTMLPWarning)
                _checkpoint_module(layer.intermediate, compress_kwargs=compress_kwargs)
            if ckpt_layer:
                _checkpoint_module(layer, compress_kwargs=compress_kwargs)
    
    else:
        for i in range(len(base_model.encoder.layer)):
            base_model.encoder.layer : ModuleList
            layer: ViTLayer = base_model.encoder.layer[i]
            compress_kwargs_layer_tocopy = copy.deepcopy(compress_kwargs)
            compress_kwargs_layer_tocopy['rank'] = None
            if norm:
                kwargs_layernorm_before = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_layernorm_before['rank'] = compress_kwargs['rank'][f'layer_{i}.layernorm_before']['output']
                _patch_module(layer.layernorm_before, nn_layer_norm_forward, compress_kwargs=kwargs_layernorm_before)
                kwargs_layernorm_after = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_layernorm_after['rank'] = compress_kwargs['rank'][f'layer_{i}.layernorm_after']['output']
                _patch_module(layer.layernorm_after, nn_layer_norm_forward, compress_kwargs=kwargs_layernorm_after)
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
                kwargs_mlp_in['rank'] = compress_kwargs['rank'][f'layer_{i}.intermediate_dense']['input']
                _patch_module(layer.intermediate.dense, nn_linear_forward, compress_kwargs=kwargs_mlp_in)
            if mlp_out:
                kwargs_mlp_out = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_mlp_out['rank'] = compress_kwargs['rank'][f'layer_{i}.output_dense']['input']
                _patch_module(layer.output.dense, nn_linear_forward, compress_kwargs=kwargs_mlp_out)
            if act_fn:
                raise NotImplementedError
                # _patch_module(layer.intermediate.intermediate_act_fn, gelu_forward, compress_kwargs=compress_kwargs)
            if ckpt_attn:
                kwargs_attention = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_attention['rank'] = compress_kwargs['rank'][f'layer_{i}.attention']['input']
                _checkpoint_module(layer.attention, compress_kwargs=kwargs_attention)
            if ckpt_mlp:
                warnings.warn("ViT only supports checkpointing the first layer of MLP.", CheckpointViTMLPWarning)
                kwargs_mlp = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_mlp['rank'] = compress_kwargs['rank'][f'layer_{i}.intermediate']['input']
                _checkpoint_module(layer.intermediate, compress_kwargs=kwargs_mlp)
            if ckpt_layer:
                kwargs_layer = copy.deepcopy(compress_kwargs_layer_tocopy)
                kwargs_layer['rank'] = compress_kwargs['rank'][f'layer_{i}']['input']
                _checkpoint_module(layer, compress_kwargs=kwargs_layer)