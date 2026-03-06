import torch
from torch import Tensor

from ..compressed import CompressedTensor

import math

# 压缩的是 input
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        compress_kwargs: dict | None = None,
    ) -> Tensor:
        return torch._C._nn.linear(input, weight, bias)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias, compress_kwargs = inputs
        
        ctx.device_type = input.device.type
        ctx.autocast_kwargs = {
            "dtype": torch.get_autocast_dtype(ctx.device_type),
            "enabled": torch.is_autocast_enabled(ctx.device_type),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        if compress_kwargs is not None:
            if compress_kwargs.get('RandomGaussion', False):
                assert 'rank' in compress_kwargs
                r = compress_kwargs['rank']
                hidden_size = input.shape[-1]
                if isinstance(r, float):
                    r = int(r * hidden_size)
                seed = torch.randint(0, 10000, (1,)).item()
                g = torch.Generator(device=input.device)
                g.manual_seed(seed)
                P = torch.randn(hidden_size, r, device=input.device,
                                dtype=input.dtype, generator=g) / math.sqrt(r)
                z = input.reshape(-1, hidden_size) @ P
                z.requires_grad = input.requires_grad
                ctx.save_for_backward(z, weight, bias)
                ctx.seed = seed
                ctx.org_shape = input.shape
                ctx.hidden_size = hidden_size
                ctx.rank = r
            else:
                ctx.save_for_backward(CompressedTensor(input, **compress_kwargs), weight, bias)
        else:
            ctx.save_for_backward(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        input, weight, bias = ctx.saved_tensors

        if hasattr(ctx, 'seed'):
            z = input  # (B*, r)
            seed = ctx.seed
            g = torch.Generator(device=z.device)
            g.manual_seed(seed)
            P = torch.randn(ctx.hidden_size, ctx.rank, device=z.device,
                            dtype=z.dtype, generator=g) / math.sqrt(ctx.rank)
            input = (z @ P.T).reshape(ctx.org_shape)
        elif isinstance(input, CompressedTensor):
            input = input.reconstruct()

        with torch.autocast(ctx.device_type, **ctx.autocast_kwargs):
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
            input_2d = input.reshape(-1, input.shape[-1])

            if ctx.needs_input_grad[0]:
                grad_input = grad_output @ weight
            else:
                grad_input = None

            if ctx.needs_input_grad[1]:
                grad_weight = grad_output_2d.T @ input_2d
            else:
                grad_weight = None

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output_2d.sum(dim=0)
            else:
                grad_bias = None

        return grad_input, grad_weight, grad_bias, None
