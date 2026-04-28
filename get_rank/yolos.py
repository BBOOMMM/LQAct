from collections import defaultdict

import torch
from torch.utils.data import DataLoader


def _get_yolos_backbone(model):
    for candidate in (
        getattr(model, "yolos", None),
        getattr(model, "base_model", None),
    ):
        if candidate is not None and hasattr(candidate, "encoder"):
            return candidate

    if hasattr(model, "get_base_model"):
        base = model.get_base_model()
        for candidate in (
            getattr(base, "yolos", None),
            getattr(base, "base_model", None),
            base,
        ):
            if candidate is not None and hasattr(candidate, "encoder"):
                return candidate

    raise ValueError("Unable to locate YOLOS encoder for rank estimation.")


def get_yolos_activations(model, dataset, batch_size, patch_locations):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: batch)
    backbone = _get_yolos_backbone(model)

    model.eval()
    activations = defaultdict(lambda: defaultdict(list))
    handles = []

    def get_hook(name, type_save):
        def hook(module, input, output):
            if type_save == "input":
                value = input[0] if isinstance(input, tuple) else input
                activations[name]["input"].append(value.detach().to("cpu"))
            elif type_save == "output":
                value = output[0] if isinstance(output, tuple) else output
                activations[name]["output"].append(value.detach().to("cpu"))
            else:
                raise ValueError("type_save can only be input or output")

        return hook

    for i, layer in enumerate(backbone.encoder.layer):
        if patch_locations == 1:
            handles.append(layer.register_forward_hook(get_hook(f"layer_{i}", "input")))
        elif patch_locations == 2:
            handles.append(layer.layernorm_before.register_forward_hook(get_hook(f"layer_{i}.layernorm_before", "output")))
            handles.append(layer.layernorm_after.register_forward_hook(get_hook(f"layer_{i}.layernorm_after", "output")))
            handles.append(layer.attention.register_forward_hook(get_hook(f"layer_{i}.attention", "input")))
            handles.append(layer.intermediate.register_forward_hook(get_hook(f"layer_{i}.intermediate", "input")))
        else:
            raise ValueError("YOLOS rank estimation only supports patch_locations 1 or 2.")

    steps = max(1, 1024 // batch_size)
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            if step >= steps:
                break
            pixel_values = torch.stack([item["pixel_values"] for item in batch]).to(device=device, dtype=dtype)
            labels = []
            for item in batch:
                labels.append(
                    {
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in item["labels"].items()
                    }
                )
            model(pixel_values=pixel_values, labels=labels)

    for handle in handles:
        handle.remove()

    return {
        layer_name: {io_type: tensors for io_type, tensors in io_dict.items()}
        for layer_name, io_dict in activations.items()
    }


def get_yolos_rank(model, dataset, batch_size, patch_locations, energy_ratio=0.5):
    activations = get_yolos_activations(model, dataset, batch_size, patch_locations)
    rank_dict = {}
    for layer_name, io_dict in activations.items():
        rank_dict[layer_name] = {}
        for io_type, tensors in io_dict.items():
            act = torch.cat(tensors, dim=0).reshape(-1, tensors[0].shape[-1]).to(torch.float32).to("cuda")
            with torch.no_grad():
                singular_values = torch.linalg.svdvals(act)
                cumsum = torch.cumsum(singular_values, dim=0)
                total = cumsum[-1]
                ratio = (cumsum / total).cpu()
                rank = torch.sum(ratio < energy_ratio).item() + 1
            rank_dict[layer_name][io_type] = rank
            del act, singular_values, cumsum, total
            torch.cuda.empty_cache()
    return activations, rank_dict


def get_yolos_rank_binary_search_energy_ratio(model, dataset, batch_size, patch_locations, rank_ratio=0.125):
    activations = get_yolos_activations(model, dataset, batch_size, patch_locations)
    backbone = _get_yolos_backbone(model)
    num_layers = len(activations)
    hidden_size = backbone.config.hidden_size
    total_rank = num_layers * int(hidden_size * rank_ratio)

    ratio_dict = {}
    for layer_name, io_dict in activations.items():
        ratio_dict[layer_name] = {}
        for io_type, tensors in io_dict.items():
            act = torch.cat(tensors, dim=0).reshape(-1, tensors[0].shape[-1]).to(torch.float32).to("cuda")
            with torch.no_grad():
                singular_values = torch.linalg.svdvals(act)
                cumsum = torch.cumsum(singular_values, dim=0)
                total = cumsum[-1]
                ratio = (cumsum / total).cpu()
            ratio_dict[layer_name][io_type] = ratio
            del act, singular_values, cumsum, total
            torch.cuda.empty_cache()

    left_energy = 0.0
    right_energy = 1.0
    tolerance = 5
    while True:
        energy_ratio = (left_energy + right_energy) / 2.0
        current_total_rank = 0
        for io_dict in ratio_dict.values():
            for ratio in io_dict.values():
                current_total_rank += int((ratio < energy_ratio).sum().item()) + 1
        if abs(current_total_rank - total_rank) <= tolerance:
            break
        if current_total_rank > total_rank:
            right_energy = energy_ratio
        else:
            left_energy = energy_ratio
        if right_energy - left_energy < 1e-4:
            break

    rank_dict = {}
    for layer_name, io_dict in ratio_dict.items():
        rank_dict[layer_name] = {}
        for io_type, ratio in io_dict.items():
            rank_dict[layer_name][io_type] = int((ratio < energy_ratio).sum().item()) + 1
    return activations, rank_dict


def get_yolos_rank_ratio(model, dataset, batch_size, patch_locations, base_ratio=1.0 / 8.0, energy_ratio=0.5):
    activations, rank_dict = get_yolos_rank(model, dataset, batch_size, patch_locations, energy_ratio)
    backbone = _get_yolos_backbone(model)
    hidden_size = backbone.config.hidden_size

    groups = {}
    for layer_name, io_dict in rank_dict.items():
        for io_type, rank in io_dict.items():
            suffix = layer_name.split(".", 1)[1] if "." in layer_name else "layer"
            block_type = "mlp" if suffix == "intermediate" else suffix
            groups.setdefault((block_type, io_type), []).append((layer_name, io_type, rank))

    rank_ratio_dict = {}
    for entries in groups.values():
        importance_sum = sum(rank for _, _, rank in entries)
        n = len(entries)
        if importance_sum == 0:
            for layer_name, io_type, _ in entries:
                rank_ratio_dict.setdefault(layer_name, {})[io_type] = base_ratio
            continue
        target_total_rank = int(base_ratio * n * hidden_size)
        cont_ranks = [(rank / importance_sum) * target_total_rank for _, _, rank in entries]
        int_ranks = [max(1, int(value)) for value in cont_ranks]
        current_total = sum(int_ranks)
        if current_total > target_total_rank:
            surplus = current_total - target_total_rank
            order = sorted(range(n), key=lambda i: int_ranks[i] - cont_ranks[i], reverse=True)
            for idx in order:
                if surplus == 0:
                    break
                if int_ranks[idx] > 1:
                    int_ranks[idx] -= 1
                    surplus -= 1
        elif current_total < target_total_rank:
            deficit = target_total_rank - current_total
            order = sorted(range(n), key=lambda i: cont_ranks[i] - int_ranks[i], reverse=True)
            for idx in order:
                if deficit == 0:
                    break
                int_ranks[idx] += 1
                deficit -= 1
        for (layer_name, io_type, _), rank in zip(entries, int_ranks):
            rank_ratio_dict.setdefault(layer_name, {})[io_type] = rank

    return activations, rank_ratio_dict
