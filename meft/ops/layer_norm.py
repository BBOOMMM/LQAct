import torch
from torch import Tensor

from ..compressed import CompressedTensor
from .utils import CastingMode, get_floating_eps, convert_dtype, promote_dtype

from ..quant.one_bit import quantize_1bit, dequantize_1bit, quantize_1bit_group, dequantize_1bit_group, quantize_1bit_with_srht, dequantize_1bit_with_srht
from ..quant.ternary import quantize_ternary_group_lastdim, dequantize_ternary_group_lastdim, quantize_ternary_with_srht, dequantize_ternary_with_srht
from ..quant.two_bit import quantize_2bit_group, dequantize_2bit_group

import bitsandbytes
import math

class LayerNormFunction(torch.autograd.Function):
    # @torch.compile
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
            if 'project_matrix' in compress_kwargs:
                Q = compress_kwargs['project_matrix']
                to_save = Q.T @ output.reshape((-1, output.shape[-1]))
                to_save.requires_grad = output.requires_grad
                ctx.save_for_backward(to_save, weight, bias, rstd)
                ctx.project_matrix = Q
                ctx.org_shape = output.shape
            elif compress_kwargs.get('RandomGaussion', False):
                assert 'rank' in compress_kwargs
                r = compress_kwargs['rank']
                hidden_size = output.shape[-1]
                if isinstance(r, float):
                    r = int(r * hidden_size)
                seed = torch.randint(0, 10000, (1,)).item()
                g = torch.Generator(device=output.device)
                g.manual_seed(seed)
                P = torch.randn(hidden_size, r, device=output.device,
                                dtype=output.dtype, generator=g) / math.sqrt(r)
                z = output.reshape(-1, hidden_size) @ P
                ctx.save_for_backward(z, weight, bias, rstd)
                ctx.seed = seed
                ctx.org_shape = output.shape
                ctx.hidden_size = hidden_size
                ctx.rank = r
                breakpoint()
            else:
                ctx.save_for_backward(CompressedTensor(output, **compress_kwargs), weight, bias, rstd)
        else:
            ctx.save_for_backward(output, weight, bias, rstd)

    # @torch.compile
    @staticmethod
    def backward(ctx, grad_output: Tensor, _1) -> tuple[Tensor | None, ...]:
        if ctx.normalized_shape is not None:
            reduction_dim = list(range(-len(ctx.normalized_shape), 0))
        else:
            reduction_dim = -1
        output, weight, bias, rstd, = ctx.saved_tensors

        if hasattr(ctx, 'project_matrix'):
            Q = ctx.project_matrix
            output = (Q @ output).reshape(ctx.org_shape)
        elif hasattr(ctx, 'seed'):
            z = output  # (B*, r)
            seed = ctx.seed
            g = torch.Generator(device=z.device)
            g.manual_seed(seed)
            P = torch.randn(ctx.hidden_size, ctx.rank, device=z.device,
                            dtype=z.dtype, generator=g) / math.sqrt(ctx.rank)
            output = (z @ P.T).reshape(ctx.org_shape)
        elif isinstance(output, CompressedTensor):
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
        compress_method: str | None = None,
        compress_kwargs: dict | None = None,
        quant_method: str | None = None,
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
        input, normalized_shape, weight, bias, eps, casting_mode, compress_method, compress_kwargs, quant_method = inputs
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
            
            # to_save = torch.empty((), device=output.device, dtype=output.dtype)  # 占位符，实际不保存 hidden_states
            # quant_state = bitsandbytes.functional.quantize_4bit(
            #     output,
            #     quant_type="fp4",  # 指定 NF4 格式（适配正态分布的 R）
            #     blocksize=128,      # 必须为 32/64/128/256
            #     compress_statistics=True
            # )
            # to_save.quant_state = quant_state
            # ctx.save_for_backward(to_save, weight, bias, rstd)
            
            
            if quant_method == 'two_bit_group':
                packed_R, alpha, shape = quantize_2bit_group(output, group_size=1)
                ctx.save_for_backward(weight, bias, rstd)
                ctx.packed_R = packed_R
                ctx.alpha = alpha
                ctx.shape = shape
                ctx.quant_method = quant_method
            else:
                LowRank = CompressedTensor(output, **compress_kwargs)
                R = output - LowRank.reconstruct()
                # if compress_kwargs['quant_method'] == 'uniform':
                if quant_method == 'uniform':
                    packed_R, alpha, shape = quantize_1bit(R)
                # elif compress_kwargs['quant_method'] == 'group':
                elif quant_method == 'group':
                    packed_R, alpha, shape = quantize_1bit_group(R, group_size=1)
                # elif compress_kwargs['quant_method'] == 'srht+group':
                elif quant_method == 'srht+group':
                    packed_R, alpha, shape = quantize_1bit_with_srht(R)
                elif quant_method == 'ternary':
                    packed_R, alpha, shape = quantize_ternary_group_lastdim(R)
                elif quant_method == 'ternary+srht':
                    packed_R, alpha, shape = quantize_ternary_with_srht(R)
            
                ctx.save_for_backward(LowRank, weight, bias, rstd)
                ctx.packed_R = packed_R
                ctx.alpha = alpha
                ctx.shape = shape
                ctx.quant_method = quant_method
        else:
            ctx.save_for_backward(output, weight, bias, rstd)

    @torch.compile
    @staticmethod
    def backward(ctx, grad_output: Tensor, _1) -> tuple[Tensor | None, ...]:
        if ctx.normalized_shape is not None:
            reduction_dim = list(range(-len(ctx.normalized_shape), 0))
        else:
            reduction_dim = -1

        # if isinstance(output, CompressedTensor):
        #     LowRank = output.reconstruct()
        #     quant_state = output.quant_state
        #     dequant = bitsandbytes.functional.dequantize_4bit(*quant_state)
        #     output = LowRank + dequant
        #     del LowRank, quant_state, dequant

        if ctx.quant_method == 'two_bit_group':
            weight, bias, rstd, = ctx.saved_tensors
            reconstructed_R = dequantize_2bit_group(ctx.packed_R, ctx.alpha, ctx.shape)
            output = reconstructed_R
        else:
            output, weight, bias, rstd, = ctx.saved_tensors
            if ctx.quant_method == 'uniform':
                reconstructed_R = dequantize_1bit(ctx.packed_R, ctx.alpha, ctx.shape)
            elif ctx.quant_method == 'group':
                reconstructed_R = dequantize_1bit_group(ctx.packed_R, ctx.alpha, ctx.shape)
            elif ctx.quant_method == 'srht+group':
                reconstructed_R = dequantize_1bit_with_srht(ctx.packed_R, ctx.alpha, ctx.shape)
            elif ctx.quant_method == 'ternary':
                reconstructed_R = dequantize_ternary_group_lastdim(ctx.packed_R, ctx.alpha, ctx.shape)
            elif ctx.quant_method == 'ternary+srht':
                reconstructed_R = dequantize_ternary_with_srht(ctx.packed_R, ctx.alpha, ctx.shape)
            # reconstructed_R = 0
            output = output.reconstruct() + reconstructed_R
            
        # if isinstance(output, torch.Tensor) and hasattr(output, "quant_state"):
        #     quant_state = output.quant_state
        #     output = bitsandbytes.functional.dequantize_4bit(*quant_state)
        #     del quant_state

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

        return grad_input, None, grad_weight, grad_bias, None, None, None, None, None