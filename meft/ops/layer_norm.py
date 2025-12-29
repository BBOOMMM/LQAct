import torch
from torch import Tensor

from ..compressed import CompressedTensor
from .utils import CastingMode, get_floating_eps, convert_dtype, promote_dtype


import bitsandbytes

class LayerNormFunction(torch.autograd.Function):
    @torch.compile
    @staticmethod
    def forward(
        input: Tensor,
        normalized_shape: list[int],
        weight: Tensor | None = None,
        bias: Tensor | None = None,
        eps: float | None = None,
        casting_mode: CastingMode = CastingMode.NONE,
        compress_kwargs: dict | None = None,
    ) -> tuple[Tensor, Tensor]:
        if normalized_shape is not None:
            assert input.shape[-len(normalized_shape):] == tuple(normalized_shape)
            reduction_dim = list(range(-len(normalized_shape), 0))
        else:
            reduction_dim = -1

        input_dtype = input.dtype

        if casting_mode == CastingMode.INPUT:
            input, = promote_dtype(input, dtype=torch.float32)

        if casting_mode == CastingMode.ALL:
            input, weight, bias = promote_dtype(input, weight, bias, dtype=torch.float32)

        if eps is None:
            eps = get_floating_eps(input.dtype)

        mean = input.mean(dim=reduction_dim, keepdim=True)
        var = input.var(dim=reduction_dim, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + eps)
        output = (input - mean) * rstd

        if casting_mode == CastingMode.INPUT:
            output, = convert_dtype(output, dtype=input_dtype)

        if weight is not None:
            output = output * weight

        if bias is not None:
            output = output + bias

        if casting_mode == CastingMode.ALL:
            output, = convert_dtype(output, dtype=input_dtype)
        return output, rstd

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        input, normalized_shape, weight, bias, eps, casting_mode, compress_kwargs = inputs
        output, rstd = output
        ctx.normalized_shape = normalized_shape
        ctx.casting_mode = casting_mode
        if compress_kwargs is not None:
            ctx.save_for_backward(CompressedTensor(output, **compress_kwargs), weight, bias, rstd)
        else:
            ctx.save_for_backward(output, weight, bias, rstd)

    @torch.compile
    @staticmethod
    def backward(ctx, grad_output: Tensor, _1) -> tuple[Tensor | None, ...]:
        if ctx.normalized_shape is not None:
            reduction_dim = list(range(-len(ctx.normalized_shape), 0))
        else:
            reduction_dim = -1
        output, weight, bias, rstd, = ctx.saved_tensors

        if isinstance(output, CompressedTensor):
            output = output.reconstruct()

        input_normalized = output
        if bias is not None:
            input_normalized = input_normalized - bias
        if weight is not None:
            input_normalized = input_normalized / weight

        if ctx.needs_input_grad[0]:
            input_dtype = output.dtype

            if ctx.casting_mode == CastingMode.ALL:
                grad_output, output = promote_dtype(grad_output, output, dtype=torch.float32)

            if weight is not None:
                grad_input_normalized = grad_output * weight
            else:
                grad_input_normalized = grad_output

            if ctx.casting_mode == CastingMode.INPUT:
                grad_input_normalized, input_normalized = promote_dtype(grad_input_normalized, input_normalized, dtype=torch.float32)

            grad_input = rstd * (
                grad_input_normalized
                - grad_input_normalized.mean(dim=reduction_dim, keepdim=True)
                - input_normalized * (grad_input_normalized * input_normalized).mean(dim=reduction_dim, keepdim=True)
            )
            grad_input, = convert_dtype(grad_input, dtype=input_dtype)
        else:
            grad_input = None

        if weight is not None and ctx.needs_input_grad[2]:
            grad_weight = (grad_output * input_normalized).view(-1, *weight.shape).sum(dim=0)
        else:
            grad_weight = None

        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.view(-1, *bias.shape).sum(dim=0)
        else:
            grad_bias = None

        return grad_input, None, grad_weight, grad_bias, None, None, None


class LayerNormFunction_LowrankPlusQuantization(torch.autograd.Function):
    @torch.compile
    @staticmethod
    def forward(
        input: Tensor,
        normalized_shape: list[int],
        weight: Tensor | None = None,
        bias: Tensor | None = None,
        eps: float | None = None,
        casting_mode: CastingMode = CastingMode.NONE,
        compress_kwargs: dict | None = None,
    ) -> tuple[Tensor, Tensor]:
        if normalized_shape is not None:
            assert input.shape[-len(normalized_shape):] == tuple(normalized_shape)
            reduction_dim = list(range(-len(normalized_shape), 0))
        else:
            reduction_dim = -1

        input_dtype = input.dtype

        if casting_mode == CastingMode.INPUT:
            input, = promote_dtype(input, dtype=torch.float32)

        if casting_mode == CastingMode.ALL:
            input, weight, bias = promote_dtype(input, weight, bias, dtype=torch.float32)

        if eps is None:
            eps = get_floating_eps(input.dtype)

        mean = input.mean(dim=reduction_dim, keepdim=True)
        var = input.var(dim=reduction_dim, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + eps)
        output = (input - mean) * rstd

        if casting_mode == CastingMode.INPUT:
            output, = convert_dtype(output, dtype=input_dtype)

        if weight is not None:
            output = output * weight

        if bias is not None:
            output = output + bias

        if casting_mode == CastingMode.ALL:
            output, = convert_dtype(output, dtype=input_dtype)
        return output, rstd

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        input, normalized_shape, weight, bias, eps, casting_mode, compress_kwargs = inputs
        output, rstd = output
        ctx.normalized_shape = normalized_shape
        ctx.casting_mode = casting_mode
        if compress_kwargs is not None:
            # LowRank = CompressedTensor(output, **compress_kwargs)
            # # Q, B = LowRank.factors
            # # R = output - (Q @ B)
            # R = output - LowRank.reconstruct()
            # quant_state = bitsandbytes.functional.quantize_4bit(
            #     R,
            #     quant_type="nf4",  # 指定 NF4 格式（适配正态分布的 R）
            #     blocksize=128,      # 必须为 32/64/128/256
            #     compress_statistics=True
            # )
            # LowRank.quant_state = quant_state
            # ctx.save_for_backward(LowRank, weight, bias, rstd)
            
            to_save = torch.empty((), device=output.device, dtype=output.dtype)  # 占位符，实际不保存 hidden_states
            quant_state = bitsandbytes.functional.quantize_4bit(
                output,
                quant_type="fp4",  # 指定 NF4 格式（适配正态分布的 R）
                blocksize=128,      # 必须为 32/64/128/256
                compress_statistics=True
            )
            to_save.quant_state = quant_state
            ctx.save_for_backward(to_save, weight, bias, rstd)
        else:
            ctx.save_for_backward(output, weight, bias, rstd)

    @torch.compile
    @staticmethod
    def backward(ctx, grad_output: Tensor, _1) -> tuple[Tensor | None, ...]:
        if ctx.normalized_shape is not None:
            reduction_dim = list(range(-len(ctx.normalized_shape), 0))
        else:
            reduction_dim = -1

        output, weight, bias, rstd, = ctx.saved_tensors

        # if isinstance(output, CompressedTensor):
        #     LowRank = output.reconstruct()
        #     quant_state = output.quant_state
        #     dequant = bitsandbytes.functional.dequantize_4bit(*quant_state)
        #     output = LowRank + dequant
        #     del LowRank, quant_state, dequant
        if isinstance(output, torch.Tensor) and hasattr(output, "quant_state"):
            quant_state = output.quant_state
            output = bitsandbytes.functional.dequantize_4bit(*quant_state)
            del quant_state

        input_normalized = output
        if bias is not None:
            input_normalized = input_normalized - bias
        if weight is not None:
            input_normalized = input_normalized / weight

        if ctx.needs_input_grad[0]:
            input_dtype = output.dtype

            if ctx.casting_mode == CastingMode.ALL:
                grad_output, output = promote_dtype(grad_output, output, dtype=torch.float32)

            if weight is not None:
                grad_input_normalized = grad_output * weight
            else:
                grad_input_normalized = grad_output

            if ctx.casting_mode == CastingMode.INPUT:
                grad_input_normalized, input_normalized = promote_dtype(grad_input_normalized, input_normalized, dtype=torch.float32)

            grad_input = rstd * (
                grad_input_normalized
                - grad_input_normalized.mean(dim=reduction_dim, keepdim=True)
                - input_normalized * (grad_input_normalized * input_normalized).mean(dim=reduction_dim, keepdim=True)
            )
            grad_input, = convert_dtype(grad_input, dtype=input_dtype)
        else:
            grad_input = None

        if weight is not None and ctx.needs_input_grad[2]:
            grad_weight = (grad_output * input_normalized).view(-1, *weight.shape).sum(dim=0)
        else:
            grad_weight = None

        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.view(-1, *bias.shape).sum(dim=0)
        else:
            grad_bias = None

        return grad_input, None, grad_weight, grad_bias, None, None, None