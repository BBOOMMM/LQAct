import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from meft.linalg import *

import time

def get_swin_activations(model, dataset, batch_size, patch_locations):
    device = next(model.parameters()).device
    
    eval_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )  # [bs, c, h, w]   [16, 3, 224, 224]
    one_batch = next(iter(eval_dataloader))
    one_batch = {k: v.to(device) for k, v in one_batch.items()}
    
    model_dtype = next(model.parameters()).dtype
    one_batch = {
        k: (
            v.to(device=device, dtype=model_dtype)
            if torch.is_floating_point(v)
            else v.to(device)
        )
        for k, v in one_batch.items()
    }
    
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
        
        for i, stage in enumerate(model.swin.encoder.layers):
            for j, layer in enumerate(stage.blocks):
                if patch_locations == 1:
                    # meft_patch_locations = ("ckpt_layer",)
                    # 存储整个layer前的输入
                    handles.append(layer.register_forward_hook(get_hook(f"stage_{i}_layer_{j}", "input")))
                
                elif patch_locations == 2:
                    # meft_patch_locations = ("norm", "ckpt_attn", "ckpt_mlp",)
                    # 存储 RMSNorm 的输出
                    handles.append(layer.layernorm_before.register_forward_hook(get_hook(f"stage_{i}_layer_{j}.layernorm_before", "output")))
                    handles.append(layer.layernorm_after.register_forward_hook(get_hook(f"stage_{i}_layer_{j}.layernorm_after", "output")))
                    
                    # 存储 attention 块的输入
                    handles.append(layer.attention.register_forward_hook(get_hook(f"stage_{i}_layer_{j}.attention", "input")))
                    
                    # 存储 mlp 块的输入
                    handles.append(layer.intermediate.register_forward_hook(get_hook(f"stage_{i}_layer_{j}.intermediate", "input")))
                
                else:
                    raise ValueError("Only support patch_locations 1 or 2")


    register_hooks()
    
    with torch.no_grad():
        # for step, batch in enumerate(eval_dataloader):
        #     if step >= 5:
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


def get_swin_rank(model, dataset, batch_size, patch_locations, energy_ratio=0.5):
    activations = get_swin_activations(model, dataset, batch_size, patch_locations)
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
def get_swin_rank_binary_search_energy_ratio(model, dataset, batch_size, patch_locations, rank_ratio=0.125):
    activations = get_swin_activations(model, dataset, batch_size, patch_locations)

    # 1) 对每个激活矩阵做一次 SVD，缓存累计能量比例 ratio
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
                ratio = (cumsum / total).cpu()       # [D]，放回 CPU，节省显存
            ratio_dict[layer_name][io_type] = ratio

            # 及时释放中间张量，避免占用过多显存
            del act, singular_values, cumsum, total
            torch.cuda.empty_cache()

    # 2) 按 stage 分组
    def _parse_stage_id(layer_name: str) -> int:
        # 兼容:
        #   "stage_0_layer_0"
        #   "stage_0_layer_0.layernorm_before"
        prefix = layer_name.split(".", 1)[0]   # "stage_0_layer_0"
        parts = prefix.split("_")              # ["stage", "0", "layer", "0"]
        return int(parts[1])

    # stage_id -> { layer_name -> { io_type -> ratio } }
    stage_to_layers: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for layer_name, io_dict in ratio_dict.items():
        stage_id = _parse_stage_id(layer_name)
        stage_to_layers.setdefault(stage_id, {})[layer_name] = io_dict

    # 3) 对每个 stage 单独二分搜索 energy_ratio
    rank_dict: dict[str, dict[str, int]] = {}
    tolerance = 5  # 每个 stage 允许的 rank 误差

    for stage_id, layer_ratio_dict in stage_to_layers.items():
        print(f"\n===== Stage {stage_id} =====")

        # 3.1 计算该 stage 的目标总秩:
        #     对于该 stage 下的每个激活矩阵，目标 rank ≈ D * rank_ratio
        target_total_rank = 0
        stage_hidden_size = None
        for layer_name, io_dict in layer_ratio_dict.items():
            for io_type, ratio in io_dict.items():
                D = ratio.numel()
                stage_hidden_size = D  # 同一 stage 的 hidden_size 一样，随便取一个
                base_rank = max(int(D * rank_ratio), 1)
                target_total_rank += base_rank

        print(f"  stage hidden_size ≈ {stage_hidden_size}, "
              f"rank_ratio = {rank_ratio}, "
              f"target_total_rank(stage {stage_id}) = {target_total_rank}")

        # 3.2 在该 stage 上做二分搜索 energy_ratio
        left_energy = 0.0
        right_energy = 1.0
        energy_ratio_stage = 1.0

        # 防止极端情况下死循环，设置最多迭代次数
        max_iter = 50
        for _ in range(max_iter):
            energy_ratio_stage = (left_energy + right_energy) / 2.0
            current_total_rank = 0

            for layer_name, io_dict in layer_ratio_dict.items():
                for io_type, ratio in io_dict.items():
                    rank = int((ratio < energy_ratio_stage).sum().item()) + 1
                    current_total_rank += rank

            print(f"  [stage {stage_id}] energy_ratio: {energy_ratio_stage:.4f}, "
                  f"current_total_rank: {current_total_rank}")

            # 满足误差要求就停
            if abs(current_total_rank - target_total_rank) <= tolerance:
                break

            # 根据当前总秩调整搜索区间
            if current_total_rank > target_total_rank:
                right_energy = energy_ratio_stage
            else:
                left_energy = energy_ratio_stage

            # 区间足够小也停止
            if right_energy - left_energy < 1e-3:
                break

        print(f"  ==> final energy_ratio(stage {stage_id}) = {energy_ratio_stage:.4f}")

        # 3.3 用该 stage 的 energy_ratio_stage 回写每个激活矩阵的 rank
        for layer_name, io_dict in layer_ratio_dict.items():
            rank_dict.setdefault(layer_name, {})
            for io_type, ratio in io_dict.items():
                rank = int((ratio < energy_ratio_stage).sum().item()) + 1
                rank_dict[layer_name][io_type] = rank

    # 4) 打印结果
    print("\nAdjusted rank dict (per stage):")
    for layer_name, io_dict in rank_dict.items():
        print(f"  {layer_name}: {io_dict}")

    return activations, rank_dict



# 为了rank_ratio进行比较，需要实现各层按比例分配
def get_swin_rank_ratio(model, dataset, batch_size, patch_locations, base_ratio=1.0/8.0, energy_ratio=0.5):
    activations, rank_dict = get_swin_rank(model, dataset, batch_size, patch_locations, energy_ratio)
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
            rank_ratio_dict.setdefault(layer_name, {})[io_type_] = int(rank_ratio + 1)
    
    print("Adjusted rank ratio dict:")
    for layer_name, io_dict in rank_ratio_dict.items():
        print(f"  {layer_name}: {io_dict}")
    
    # breakpoint()

    return activations, rank_ratio_dict


# 为了rank_ratio进行比较，需要实现各层按比例分配
def get_vit_rank_ratio_gentle(model, dataset, batch_size, patch_locations, base_ratio=1.0/8.0, energy_ratio=0.5):
    activations, rank_dict = get_swin_rank(model, dataset, batch_size, patch_locations, energy_ratio)
    print("Original rank dict:")
    for layer_name, io_dict in rank_dict.items():
        print(f"  {layer_name}: {io_dict}")
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


def get_swin_project_matrix(model, dataset, batch_size, patch_locations, base_ratio=1.0/8.0):
    activations, rank_dict = get_swin_rank_ratio(model, dataset, batch_size, patch_locations, base_ratio)
    
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




