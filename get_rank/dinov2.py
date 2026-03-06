import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from meft.linalg import *

import time

def get_dinov2_activations(model, dataset, batch_size, patch_locations):
    device = next(model.parameters()).device
    
    eval_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )  # [bs, c, h, w]   [16, 3, 224, 224]
    one_batch = next(iter(eval_dataloader))
    one_batch = {k: v.to(device) for k, v in one_batch.items()}
    
    # print(one_batch["labels"].min().item(), one_batch["labels"].max().item())
    
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

        for i, layer in enumerate(model.dinov2.encoder.layer):
            if patch_locations == 1:
                # meft_patch_locations = ("ckpt_layer",)
                # 存储整个layer前的输入
                handles.append(layer.register_forward_hook(get_hook(f"layer_{i}", "input")))
            
            elif patch_locations == 2:
                # meft_patch_locations = ("norm", "ckpt_attn", "ckpt_mlp",)
                # 存储 RMSNorm 的输出
                handles.append(layer.norm1.register_forward_hook(get_hook(f"layer_{i}.norm1", "output")))
                handles.append(layer.norm2.register_forward_hook(get_hook(f"layer_{i}.norm2", "output")))
                
                # 存储 attention 块的输入
                handles.append(layer.attention.register_forward_hook(get_hook(f"layer_{i}.attention", "input")))
                
                # 存储 mlp 块的输入
                handles.append(layer.mlp.register_forward_hook(get_hook(f"layer_{i}.mlp", "input")))
            
            elif patch_locations == 3:
                # meft_patch_locations = ("norm", "attn_in", "attn_out", "mlp_in", "mlp_out",)
                # 存储 RMSNorm 的输出
                handles.append(layer.norm1.register_forward_hook(get_hook(f"layer_{i}.norm1", "output")))
                handles.append(layer.norm2.register_forward_hook(get_hook(f"layer_{i}.norm2", "output")))
                
                # 存储 attention in 的输入
                handles.append(layer.attention.attention.query.register_forward_hook(get_hook(f"layer_{i}.attention_query", "input")))
                handles.append(layer.attention.attention.key.register_forward_hook(get_hook(f"layer_{i}.attention_key", "input")))
                handles.append(layer.attention.attention.value.register_forward_hook(get_hook(f"layer_{i}.attention_value", "input")))
                
                # 存储 attention out 的输入
                handles.append(layer.attention.output.dense.register_forward_hook(get_hook(f"layer_{i}.attention_output_dense", "input")))
                
                # 存储 mlp in 块的输入
                handles.append(layer.mlp.fc1.register_forward_hook(get_hook(f"layer_{i}.mlp_fc1", "input")))
                
                # 存储 mlp out 块的输入
                handles.append(layer.mlp.fc2.register_forward_hook(get_hook(f"layer_{i}.mlp_fc2", "input")))
            
            else:
                raise ValueError("Only support patch_locations 1 or 2")


    register_hooks()
    
    steps = max(1, 1024 // batch_size)
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            if step >= steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
        # outputs = model(**one_batch)
    
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


def get_dinov2_rank(model, dataset, batch_size, patch_locations, energy_ratio=0.5):
    activations = get_dinov2_activations(model, dataset, batch_size, patch_locations)
    rank_dict = {}
    for layer_name, io_dict in activations.items():
        rank_dict[layer_name] = {}
        for io_type, tensors in io_dict.items():
            # act = torch.cat(tensors, dim=0)  # [B, S, D]
            # act = act.reshape(-1, act.shape[-1]).to(torch.float32).to("cuda")  # [B*S, D]
            # singular_values = torch.linalg.svdvals(act)
            # cumsum = torch.cumsum(singular_values, dim=0)
            # total = cumsum[-1]
            # ratio = cumsum / total
            # rank = torch.sum(ratio < energy_ratio).item() + 1  # 保留30%能量所需的秩
            # rank_dict[layer_name][io_type] = rank
            
            act = torch.cat(tensors, dim=0)  # [B, S, D]
            act = act.reshape(-1, act.shape[-1]).to(torch.float32).to("cuda")  # [B*S, D]
            with torch.no_grad():
                singular_values = torch.linalg.svdvals(act)  # [D]
                cumsum = torch.cumsum(singular_values, dim=0)
                total = cumsum[-1]
                ratio = (cumsum / total).cpu()       # 放回 CPU，节省显存
                rank = torch.sum(ratio < energy_ratio).item() + 1     
            rank_dict[layer_name][io_type] = rank

            # 及时释放中间张量，避免占用过多显存
            del act, singular_values, cumsum, total
            torch.cuda.empty_cache()

    return activations, rank_dict


# 二分搜索总秩相同的 energy_ratio
def get_dinov2_rank_binary_search_energy_ratio(model, dataset, batch_size, patch_locations, rank_ratio=0.125):
    activations = get_dinov2_activations(model, dataset, batch_size, patch_locations)
    num_layers = len(activations)
    hidden_size = model.dinov2.config.hidden_size
    total_rank = num_layers * int(hidden_size * rank_ratio)
    print(f"rank_ratio : {rank_ratio}, total rank: {total_rank}")

    # 1) 预先对每个激活矩阵做一次 SVD，并缓存累计能量比例 ratio
    #    ratio: shape [D]，ratio[k] = 累计能量 / 总能量
    ratio_dict = {}
    for layer_name, io_dict in activations.items():
        ratio_dict[layer_name] = {}
        for io_type, tensors in io_dict.items():
            act = torch.cat(tensors, dim=0)          # [B, S, D]
            act = act.reshape(-1, act.shape[-1])     # [B*S, D]
            act = act.to(torch.float32).to("cuda")
            with torch.no_grad():
                singular_values = torch.linalg.svdvals(act)  # [D]
                cumsum = torch.cumsum(singular_values, dim=0)
                total = cumsum[-1]
                ratio = (cumsum / total).cpu()       # 放回 CPU，节省显存
            ratio_dict[layer_name][io_type] = ratio

            # 及时释放中间张量，避免占用过多显存
            del act, singular_values, cumsum, total
            torch.cuda.empty_cache()
    
    
    # 2) 二分 + 以 current_total_rank 为早停条件
    left_energy = 0.0
    right_energy = 1.0
    energy_ratio = None
    tolerance = 5  # 允许的 rank 误差

    while True:
        energy_ratio = (left_energy + right_energy) / 2.0
        current_total_rank = 0
        for layer_name, io_dict in ratio_dict.items():
            for io_type, ratio in io_dict.items():
                rank = int((ratio < energy_ratio).sum().item()) + 1
                current_total_rank += rank
        print(f"energy_ratio: {energy_ratio:.4f}, current total rank: {current_total_rank}")

        # 误差足够小就停
        if abs(current_total_rank - total_rank) <= tolerance:
            break

        # 否则继续二分
        if current_total_rank > total_rank:
            right_energy = energy_ratio
        else:
            left_energy = energy_ratio

        # 如果能量区间已经很小，也强制停止，避免死循环
        if right_energy - left_energy < 1e-4:
            break

    # 3) 用最终确定的 energy_ratio 构造 rank_dict（同样只用 ratio）
    rank_dict = {}
    for layer_name, io_dict in ratio_dict.items():
        rank_dict[layer_name] = {}
        for io_type, ratio in io_dict.items():
            rank = int((ratio < energy_ratio).sum().item()) + 1
            rank_dict[layer_name][io_type] = rank
    
    
    print("Adjusted rank ratio dict:")
    for layer_name, io_dict in rank_dict.items():
        print(f"  {layer_name}: {io_dict}")

    return activations, rank_dict



# 为了rank_ratio进行比较，需要实现各层按比例分配
def get_dinov2_rank_ratio(model, dataset, batch_size, patch_locations, base_ratio=1.0/8.0, energy_ratio=0.5):
    activations, rank_dict = get_dinov2_rank(model, dataset, batch_size, patch_locations, energy_ratio)
    # breakpoint()
    num_layers = model.dinov2.config.num_hidden_layers
    hidden_size = model.dinov2.config.hidden_size
    
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
                if suffix == "intermediate" or suffix == "mlp":
                    block_type = "mlp"  # MLP 块
                else:
                    block_type = suffix  # layernorm_before / layernorm_after / attention
            elif patch_locations == 3:
                # 例如:
                #   layer_0.layernorm_before
                #   layer_0.layernorm_after
                #   layer_0.attention_query / attention_key / attention_value
                #   layer_0.attention_output_dense
                #   layer_0.intermediate_dense
                #   layer_0.output_dense
                if "." in layer_name:
                    suffix = layer_name.split(".", 1)[1]
                else:
                    suffix = "layer"

                if suffix in ("norm1", "norm2"):
                    block_type = suffix
                elif suffix in ("attention_query", "attention_key", "attention_value"):
                    # 三个 attn_in 共享一类，用同一组重要性分配规则
                    block_type = "attn_in"
                elif suffix == "attention_output_dense":
                    block_type = "attn_out"
                elif suffix == "mlp_fc1":
                    block_type = "mlp_in"
                elif suffix == "mlp_fc2":
                    block_type = "mlp_out"
                else:
                    # 未知后缀，直接用原始名字分组（防御性处理）
                    block_type = suffix
            else:
                raise ValueError("Only support patch_locations 1 or 2")

            key = (block_type, io_type)
            groups.setdefault(key, []).append((layer_name, io_type, r))
    
    # breakpoint()

    # 计算每个块的 rank_ratio
    rank_ratio_dict: dict[str, dict[str, float]] = {}

    for (block_type, io_type), entries in groups.items():
        # # entries: list of (layer_name, io_type, importance_rank)
        # importance_sum = sum(r for _, _, r in entries)
        # n = len(entries)

        # if importance_sum == 0:
        #     # 极端情况：所有 rank 都是 0，退化为均匀分配
        #     for layer_name, io_type_, _ in entries:
        #         rank_ratio_dict.setdefault(layer_name, {})[io_type_] = base_ratio
        #     continue

        # # 目标：这一类(block_type, io_type)的平均 rank_ratio 仍为 base_ratio，
        # # 即 sum_j ratio_j = n * base_ratio
        # for layer_name, io_type_, r in entries:
        #     rank_ratio = (r / importance_sum) * (base_ratio * n * hidden_size)
        #     rank_ratio_dict.setdefault(layer_name, {})[io_type_] = int(rank_ratio + 1)
        
        # entries: list of (layer_name, io_type, importance_rank)
        importance_sum = sum(r for _, _, r in entries)
        n = len(entries)

        if importance_sum == 0:
            # 极端情况：所有 rank 都是 0，退化为均匀分配
            for layer_name, io_type_, _ in entries:
                rank_ratio_dict.setdefault(layer_name, {})[io_type_] = base_ratio
            continue

        # 本组目标总秩：sum_j rank_j = base_ratio * n * hidden_size
        target_total_rank = int(base_ratio * n * hidden_size)

        # 1) 连续秩（按重要性比例分配）
        cont_ranks = [
            (r / importance_sum) * target_total_rank
            for _, _, r in entries
        ]

        # 2) 先取整并保证每层至少 1
        int_ranks = [max(1, int(x)) for x in cont_ranks]
        current_total = sum(int_ranks)

        # 3) 调整使整数秩之和尽量等于 target_total_rank
        if current_total > target_total_rank:
            # 有多余的秩，需要减掉
            surplus = current_total - target_total_rank
            # 优先从“比连续值超得多”的层上减
            order = sorted(
                range(n),
                key=lambda i: (int_ranks[i] - cont_ranks[i]),
                reverse=True,
            )
            for idx in order:
                if surplus == 0:
                    break
                if int_ranks[idx] > 1:
                    int_ranks[idx] -= 1
                    surplus -= 1

        elif current_total < target_total_rank:
            # 不足，需要补一些
            deficit = target_total_rank - current_total
            # 优先给“小数部分最大”的层加 1
            order = sorted(
                range(n),
                key=lambda i: (cont_ranks[i] - int_ranks[i]),
                reverse=True,
            )
            for idx in order:
                if deficit == 0:
                    break
                int_ranks[idx] += 1
                deficit -= 1

        # 4) 写回（这里存的实际上是「每层的 rank」，名字沿用 rank_ratio_dict）
        for (layer_name, io_type_, _), rank_int in zip(entries, int_ranks):
            rank_ratio_dict.setdefault(layer_name, {})[io_type_] = rank_int
    
    print("Adjusted rank ratio dict:")
    for layer_name, io_dict in rank_ratio_dict.items():
        print(f"  {layer_name}: {io_dict}")
    
    # breakpoint()

    return activations, rank_ratio_dict


# 为了rank_ratio进行比较，需要实现各层按比例分配
def get_dinov2_rank_ratio_gentle(model, dataset, batch_size, patch_locations, base_ratio=1.0/8.0, energy_ratio=0.5):
    activations, rank_dict = get_dinov2_rank(model, dataset, batch_size, patch_locations, energy_ratio)
    print("Original rank dict:")
    for layer_name, io_dict in rank_dict.items():
        print(f"  {layer_name}: {io_dict}")
    num_layers = model.dinov2.config.num_hidden_layers
    hidden_size = model.dinov2.config.hidden_size
    
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
                if suffix == "intermediate" or suffix == "mlp":
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

    base_rank = int(hidden_size * base_ratio)

    for (block_type, io_type), entries in groups.items():
        # entries: list of (layer_name, io_type, importance_rank)
        raw_ranks = [r for _, _, r in entries]
        n = len(entries)

        # 1) 用幂次 γ<1 平滑一下，避免过于极端
        gamma = 0.8  # 0.5 相当于开根号，拉近大 r 和小 r 的差距
        scores = [max(r, 1.0) ** gamma for r in raw_ranks]
        score_sum = sum(scores)

        # 2) 本组目标总秩
        target_total_rank = base_rank * n

        # 3) 先按平滑后的 scores 分配连续 rank
        cont_ranks = [s / score_sum * target_total_rank for s in scores]

        # # 4) 可选：对单层 rank 进行上下界裁剪，再重新归一化
        # r_min = base_rank * 0.5   # 每层至少 0.25×base_rank
        # r_max = base_rank * 2.0    # 每层最多 4×base_rank

        # cont_ranks = [min(max(r, r_min), r_max) for r in cont_ranks]
        # # 再缩放一次，使总和仍为 target_total_rank
        # scale = target_total_rank / sum(cont_ranks)
        # cont_ranks = [r * scale for r in cont_ranks]

        # 5) 写回 rank_ratio_dict（这里其实是「每层的 rank」）
        for (layer_name, io_type_, _), r_cont in zip(entries, cont_ranks):
            rank = int(round(r_cont))
            rank_ratio_dict.setdefault(layer_name, {})[io_type_] = max(rank, 1)
    
    # breakpoint()
    print("Adjusted rank ratio dict:")
    for layer_name, io_dict in rank_ratio_dict.items():
        print(f"  {layer_name}: {io_dict}")
    
    
    # breakpoint()

    return activations, rank_ratio_dict


