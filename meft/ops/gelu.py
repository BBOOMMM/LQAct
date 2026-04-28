import math

import torch
from torch import Tensor

from ..compressed import CompressedTensor

from ..quant.one_bit import quantize_1bit, dequantize_1bit, quantize_1bit_group, dequantize_1bit_group
from ..quant.ternary import quantize_ternary_group_lastdim, dequantize_ternary_group_lastdim
from ..quant.two_bit import quantize_2bit_group, dequantize_2bit_group


class GELUFunction(torch.autograd.Function):
    @torch.compile
    @staticmethod
    def forward(
        input: Tensor,
        approximate: str = "none",
        compress_kwargs: dict | None = None,
    ) -> Tensor:
        if approximate == "none":
            return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
        elif approximate == "tanh":
            return input * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))
        elif approximate == "sigmoid":
            return input * torch.sigmoid(1.702 * input)
        else:
            raise ValueError("Unexpected value of argument `approximate`, must be `'none'`, `'tanh'` or `'sigmoid'`.")

    @staticmethod
    def setup_context(ctx, inputs: tuple, output: Tensor) -> None:
        input, approximate, compress_kwargs = inputs
        ctx.approximate = approximate
        if compress_kwargs is not None:
            ctx.save_for_backward(CompressedTensor(input, **compress_kwargs))
        else:
            ctx.save_for_backward(input)

    @torch.compile
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        input, = ctx.saved_tensors

        if isinstance(input, CompressedTensor):
            input = input.reconstruct()

        if ctx.needs_input_grad[0]:
            if ctx.approximate == "none":
                cdf = 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
                deriv_cdf = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * torch.pow(input, 2))
            elif ctx.approximate == "tanh":
                cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))
                deriv_cdf = 2.0 * cdf * (1.0 - cdf) * math.sqrt(2.0 / math.pi) * (1.0 + 0.134145 * torch.pow(input, 2))
            elif ctx.approximate == "sigmoid":
                cdf = torch.sigmoid(1.702 * input)
                deriv_cdf = 1.702 * cdf * (1.0 - cdf)
            else:
                raise ValueError("Unexpected value of argument `approximate`, must be `'none'`, `'tanh'` or `'sigmoid'`.")
            grad_input = grad_output * (cdf + input * deriv_cdf)
        else:
            grad_input = None

        return grad_input, None, None


class GELUFunction_LowrankPlusQuantization(torch.autograd.Function):
    @torch.compile
    @staticmethod
    def forward(
        input: Tensor,
        approximate: str = "none",
        compress_method: str | None = None,
        compress_kwargs: dict | None = None,
        quant_method: str | None = None
    ) -> Tensor:
        if approximate == "none":
            return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
        elif approximate == "tanh":
            return input * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))
        elif approximate == "sigmoid":
            return input * torch.sigmoid(1.702 * input)
        else:
            raise ValueError("Unexpected value of argument `approximate`, must be `'none'`, `'tanh'` or `'sigmoid'`.")

    @staticmethod
    def setup_context(ctx, inputs: tuple, output: Tensor) -> None:
        input, approximate, compress_method, compress_kwargs, quant_method = inputs
        ctx.approximate = approximate
        if compress_kwargs is not None:
            if quant_method == 'two_bit_group':
                packed_R, alpha, shape = quantize_2bit_group(input, group_size=1)
                ctx.packed_R = packed_R
                ctx.alpha = alpha
                ctx.shape = shape
                ctx.quant_method = quant_method
            else:
                LowRank = CompressedTensor(input, **compress_kwargs)
                R = input - LowRank.reconstruct()
                # if compress_kwargs['quant_method'] == '1bit_pertensor':
                if quant_method == '1bit_pertensor':
                    packed_R, alpha, shape = quantize_1bit(R)
                # elif compress_kwargs['quant_method'] == '1bit_pergroupchannel':
                elif quant_method == '1bit_pergroupchannel':
                    packed_R, alpha, shape = quantize_1bit_group(R, group_size=1)
                elif quant_method == 'ternary':
                    packed_R, alpha, shape = quantize_ternary_group_lastdim(R)
            
                ctx.save_for_backward(LowRank)
                ctx.packed_R = packed_R
                ctx.alpha = alpha
                ctx.shape = shape
                ctx.quant_method = quant_method
        else:
            ctx.save_for_backward(input)

    @torch.compile
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:

        # if isinstance(input, CompressedTensor):
        #     input = input.reconstruct()
        
        if ctx.quant_method == 'two_bit_group':
            reconstructed_R = dequantize_2bit_group(ctx.packed_R, ctx.alpha, ctx.shape)
            input = reconstructed_R
        else:
            input, = ctx.saved_tensors
            if ctx.quant_method == '1bit_pertensor':
                reconstructed_R = dequantize_1bit(ctx.packed_R, ctx.alpha, ctx.shape)
            elif ctx.quant_method == '1bit_pergroupchannel':
                reconstructed_R = dequantize_1bit_group(ctx.packed_R, ctx.alpha, ctx.shape)
            elif ctx.quant_method == 'ternary':
                reconstructed_R = dequantize_ternary_group_lastdim(ctx.packed_R, ctx.alpha, ctx.shape)
            # reconstructed_R = 0
            input = input.reconstruct() + reconstructed_R

        if ctx.needs_input_grad[0]:
            if ctx.approximate == "none":
                cdf = 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
                deriv_cdf = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * torch.pow(input, 2))
            elif ctx.approximate == "tanh":
                cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))
                deriv_cdf = 2.0 * cdf * (1.0 - cdf) * math.sqrt(2.0 / math.pi) * (1.0 + 0.134145 * torch.pow(input, 2))
            elif ctx.approximate == "sigmoid":
                cdf = torch.sigmoid(1.702 * input)
                deriv_cdf = 1.702 * cdf * (1.0 - cdf)
            else:
                raise ValueError("Unexpected value of argument `approximate`, must be `'none'`, `'tanh'` or `'sigmoid'`.")
            grad_input = grad_output * (cdf + input * deriv_cdf)
        else:
            grad_input = None

        return grad_input, None, None, None, None
