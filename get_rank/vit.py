import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from meft.linalg import *

import time

def get_vit_activations(model, dataset, batch_size, patch_locations):
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

        for i, layer in enumerate(model.vit.encoder.layer):
            if patch_locations == 1:
                # meft_patch_locations = ("ckpt_layer",)
                # 存储整个layer前的输入
                handles.append(layer.register_forward_hook(get_hook(f"layer_{i}", "input")))
            
            elif patch_locations == 2:
                # meft_patch_locations = ("norm", "ckpt_attn", "ckpt_mlp",)
                # 存储 RMSNorm 的输出
                handles.append(layer.layernorm_before.register_forward_hook(get_hook(f"layer_{i}.layernorm_before", "output")))
                handles.append(layer.layernorm_after.register_forward_hook(get_hook(f"layer_{i}.layernorm_after", "output")))
                
                # 存储 attention 块的输入
                handles.append(layer.attention.register_forward_hook(get_hook(f"layer_{i}.attention", "input")))
                
                # 存储 mlp 块的输入
                handles.append(layer.intermediate.register_forward_hook(get_hook(f"layer_{i}.intermediate", "input")))
            
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
        
    activations = {
        layer_name: {
            io_type: tensors   # 这里还是 list[Tensor]，后面读取时再自己处理
            for io_type, tensors in io_dict.items()
        }
        for layer_name, io_dict in activations.items()
    }
    
    return activations


def get_vit_rank(model, dataset, batch_size, patch_locations):
    activations = get_vit_activations(model, dataset, batch_size, patch_locations)
    rank_dict_0p1 = {}
    rank_dict_0p9 = {}
    rank_dict_0p5 = {}
    rank_dict_0p3 = {}
    rank_dict_0p2 = {}
    for layer_name, io_dict in activations.items():
        rank_dict_0p9[layer_name] = {}
        rank_dict_0p1[layer_name] = {}
        rank_dict_0p5[layer_name] = {}
        rank_dict_0p3[layer_name] = {}
        rank_dict_0p2[layer_name] = {}
        for io_type, tensors in io_dict.items():
            act = torch.cat(tensors, dim=0)  # [B, S, D]
            act = act.reshape(-1, act.shape[-1]).to(torch.float32).to("cuda")  # [B*S, D]
            singular_values = torch.linalg.svdvals(act)
            cumsum = torch.cumsum(singular_values, dim=0)
            total = cumsum[-1]
            ratio = cumsum / total
            rank_0p9 = torch.sum(ratio < 0.9).item() + 1  # 保留90%能量所需的秩
            rank_dict_0p9[layer_name][io_type] = rank_0p9
            rank_0p1 = torch.sum(ratio < 0.1).item() + 1  # 保留10%能量所需的秩
            rank_dict_0p1[layer_name][io_type] = rank_0p1
            rank_0p5 = torch.sum(ratio < 0.5).item() + 1  # 保留50%能量所需的秩
            rank_dict_0p5[layer_name][io_type] = rank_0p5
            rank_0p3 = torch.sum(ratio < 0.3).item() + 1  # 保留30%能量所需的秩
            rank_dict_0p3[layer_name][io_type] = rank_0p3
            rank_0p2 = torch.sum(ratio < 0.2).item() + 1  # 保留30%能量所需的秩
            rank_dict_0p2[layer_name][io_type] = rank_0p2

    return activations, rank_dict_0p2


# TODO: 为了rank_ratio进行比较，需要实现各层按比例分配
def get_vit_rank_ratio(model, dataset, batch_size, patch_locations, base_ratio=1.0/8.0):
    activations, rank_dict = get_vit_rank(model, dataset, batch_size, patch_locations)
    # breakpoint()
    num_layers = model.vit.config.num_hidden_layers
    hidden_size = model.vit.config.hidden_size
    
    # 不妨先假设 hidden_size <= batch_size * seq_len
    # base_ratio 对应 rank = hidden_size / 16
    
    # 假定原来统一设置秩取1/16，那么每一个layer的layernorm、attn、mlp的秩均为 hidden_size / 16
    # 为了公平比较，需要确保总秩一样  总秩 = num_layers * (hidden_size / 16)
    # 对不同layer的相同功能块，按rank_dict比例分配秩

    # 按“功能块类型 + io_type”分组，保证相同功能块在不同 layer 间按比例分配
    # group_key: (block_type, io_type)
    groups: dict[tuple[str, str], list[tuple[str, str, int]]] = {}

    for layer_name, io_dict in rank_dict.items():
        for io_type, r in io_dict.items():
            # 根据 patch_locations 和 layer_name 推出功能块类型
            if patch_locations == 1:
                # 只有整层输入: "layer_i" + "input"
                block_type = "layer"
            elif patch_locations == 2:
                # 例如:
                # layer_name: "layer_0.layernorm_before" / "layer_0.layernorm_after"
                #             "layer_0.attention" / "layer_0.intermediate"
                if "." in layer_name:
                    suffix = layer_name.split(".", 1)[1]
                else:
                    suffix = "layer"
                if suffix == "intermediate":
                    block_type = "mlp"  # MLP 块
                else:
                    block_type = suffix  # layernorm_before / layernorm_after / attention
            else:
                raise ValueError("Only support patch_locations 1 or 2")

            key = (block_type, io_type)
            groups.setdefault(key, []).append((layer_name, io_type, r))
    
    # breakpoint()

    # 计算每个块的 rank_ratio
    rank_ratio_dict: dict[str, dict[str, float]] = {}

    for (block_type, io_type), entries in groups.items():
        # entries: list of (layer_name, io_type, importance_rank)
        importance_sum = sum(r for _, _, r in entries)
        n = len(entries)

        if importance_sum == 0:
            # 极端情况：所有 rank 都是 0，退化为均匀分配
            for layer_name, io_type_, _ in entries:
                rank_ratio_dict.setdefault(layer_name, {})[io_type_] = base_ratio
            continue

        # 目标：这一类(block_type, io_type)的平均 rank_ratio 仍为 base_ratio，
        # 即 sum_j ratio_j = n * base_ratio
        for layer_name, io_type_, r in entries:
            rank_ratio = (r / importance_sum) * (base_ratio * n * hidden_size)
            rank_ratio_dict.setdefault(layer_name, {})[io_type_] = int(rank_ratio)
    
    # breakpoint()

    return activations, rank_ratio_dict


def get_vit_project_matrix(model, dataset, batch_size, patch_locations, base_ratio=1.0/8.0):
    activations, rank_dict = get_vit_rank_ratio(model, dataset, batch_size, patch_locations, base_ratio)
    
    start_time = time.time()
    
    project_matrixes = {}
    for layer_name, io_dict in activations.items():
        project_matrixes[layer_name] = {}
        for io_type, tensors in io_dict.items():
            act = torch.cat(tensors, dim=0)  # [B, S, D]
            act = act.reshape(-1, act.shape[-1]).to(torch.float32).to("cuda")  # [B*S, D]
            rank = rank_dict[layer_name][io_type]
            
            # 使用自己的算法
            Q, _ = randomized_qb(act, rank, left=True)
            
            # 或者使用 SVD 得到正交矩阵
            # U, S, Vh = torch.linalg.svd(act, full_matrices=False)
            # Q = U[:, :rank]  # [D, rank]
            
            project_matrixes[layer_name][io_type] = Q.to(torch.bfloat16)
    
    end_time = time.time()
    print(f"Computed project matrixes for ViT in {end_time - start_time:.2f} seconds.")
    
    # breakpoint()
    
    return project_matrixes




def get_vit_project_matrix(model, dataset, batch_size, patch_locations, base_ratio=1.0/8.0):
    activations, rank_dict = get_vit_rank_ratio(model, dataset, batch_size, patch_locations, base_ratio)
    # rank_dict = {}
    # for layer_name, io_dict in activations.items():
    #     rank_dict[layer_name] = {}
    #     for io_type, tensors in io_dict.items():
    #         rank_dict[layer_name][io_type] = base_ratio
    
    start_time = time.time()
    
    project_matrixes = {}
    for layer_name, io_dict in activations.items():
        project_matrixes[layer_name] = {}
        for io_type, tensors in io_dict.items():
            act = torch.cat(tensors, dim=0)  # [B, S, D]
            act = act.reshape(-1, act.shape[-1]).to(torch.float32).to("cuda")  # [B*S, D]
            rank = rank_dict[layer_name][io_type]
            
            # 使用自己的算法
            # Q, _ = randomized_qb(act, rank, left=True)
            
            # 或者使用 SVD 得到正交矩阵
            U, S, Vh = torch.linalg.svd(act, full_matrices=False)
            Q = U[:, :rank]  # [D, rank]
            
            project_matrixes[layer_name][io_type] = Q.to(torch.bfloat16)
    
    end_time = time.time()
    print(f"Computed project matrixes for ViT in {end_time - start_time:.2f} seconds.")
    
    # breakpoint()
    
    return project_matrixes
