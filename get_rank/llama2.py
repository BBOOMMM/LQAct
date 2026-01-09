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
        def get_hook(name, type_save):
            def hook(module, input, output):
                if type_save == 'input':
                    x = input[0] if isinstance(input, tuple) else input
                    activations[name]['input'].append(x.detach().to("cpu"))
                elif type_save == 'output':
                    y = output[0] if isinstance(output, tuple) else output
                    activations[name]['output'].append(y.detach().to("cpu"))
                else:
                    raise ValueError("type_save can only be input or output")
            return hook

        for i, layer in enumerate(model.model.layers):
            if patch_locations == 1:
                # meft_patch_locations = ("ckpt_layer",)
                # 存储整个layer前的输入
                handles.append(layer.register_forward_hook(get_hook(f"layer_{i}", "input")))
            
            elif patch_locations == 2:
                # meft_patch_locations = ("norm", "ckpt_attn", "ckpt_mlp",)
                # 存储 RMSNorm 的输出
                handles.append(layer.input_layernorm.register_forward_hook(get_hook(f"layer_{i}.input_layernorm", "output")))
                handles.append(layer.post_attention_layernorm.register_forward_hook(get_hook(f"layer_{i}.post_attention_layernorm", "output")))
                
                # 存储 attention 块的输入
                handles.append(layer.self_attn.register_forward_hook(get_hook(f"layer_{i}.self_attn", "input")))
                
                # 存储 mlp 块的输入
                handles.append(layer.mlp.register_forward_hook(get_hook(f"layer_{i}.mlp", "input")))
            
            else:
                raise ValueError("Only support patch_locations 1 or 2")

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