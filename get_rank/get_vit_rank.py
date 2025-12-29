import torch
from torch.utils.data import DataLoader
from collections import defaultdict

def get_vit_rank(model, dataset, batch_size, patch_locations):
    device = next(model.parameters()).device
    
    eval_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )  # [bs, c, h, w]   [16, 3, 224, 224]
    one_batch = next(iter(eval_dataloader))
    one_batch = {k: v.to(device) for k, v in one_batch.items()}
    
    model.eval()
    
    activations = defaultdict(lambda: defaultdict(list))
    handles = []

    def register_hooks():
        def get_hook(name):
            def hook(module, input, output):
                x = input[0] if isinstance(input, tuple) else input
                # y = output[0] if isinstance(output, tuple) else output
                
                # activations[name].append(input.detach().to("cpu"), output.detach().to("cpu"))
                activations[name]['input'].append(x.detach().to("cpu"))
                # activations[name]['output'].append(y.detach().to("cpu"))
            return hook

        for i, layer in enumerate(model.vit.encoder.layer):
            if patch_locations == 1:
                raise NotImplementedError("Only support patch_locations 2 for ViT")
            
            elif patch_locations == 2:
                pass
            
            else:
                raise ValueError("Only support patch_locations 1 or 2")
            
            # attn
            handles.append(layer.attention.attention.query.register_forward_hook(get_hook(f"layer_{i}.self_attn_q_proj")))
            handles.append(layer.attention.attention.key.register_forward_hook(get_hook(f"layer_{i}.self_attn_k_proj")))
            handles.append(layer.attention.attention.value.register_forward_hook(get_hook(f"layer_{i}.self_attn_v_proj")))
            handles.append(layer.attention.output.dense.register_forward_hook(get_hook(f"layer_{i}.self_attn_o_proj")))
            
            # mlp
            handles.append(layer.intermediate.dense.register_forward_hook(get_hook(f"layer_{i}.mlp_gate_proj")))
            handles.append(layer.intermediate.intermediate_act_fn.register_forward_hook(get_hook(f"layer_{i}.mlp_up_proj")))
            handles.append(layer.output.dense.register_forward_hook(get_hook(f"layer_{i}.mlp_down_proj")))
            
            # input_layernorm
            handles.append(layer.layernorm_before.register_forward_hook(get_hook(f"layer_{i}.input_layernorm")))
            
            # post_attention_layernorm
            handles.append(layer.layernorm_after.register_forward_hook(get_hook(f"layer_{i}.post_attention_layernorm")))

    register_hooks()
    
    with torch.no_grad():
        # for step, batch in enumerate(eval_dataloader):
        #     if step >= 1:
        #         break
        #     batch = {k: v.to(device) for k, v in batch.items()}
        #     outputs = model(**batch)
        outputs = model(**one_batch)
    
    for h in handles:
        h.remove()
        
    activations_to_save = {
        layer_name: {
            io_type: tensors   # 这里还是 list[Tensor]，后面读取时再自己处理
            for io_type, tensors in io_dict.items()
        }
        for layer_name, io_dict in activations.items()
    }