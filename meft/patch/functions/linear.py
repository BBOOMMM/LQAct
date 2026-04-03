from torch import nn, Tensor

from ...ops.linear import LinearFunction, LinearFunction_LowrankPlusQuantization


def nn_linear_forward(
    self: "nn.Linear",
    input: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    output = LinearFunction.apply(
        input,
        self.weight,
        self.bias,
        compress_kwargs if self.training else None,
    )
    return output



def nn_linear_forward_lowrank_plus_quantization(
    self: "nn.Linear",
    input: Tensor,
    compress_method: str | None = None,
    compress_kwargs: dict | None = None,
    quant_method: str | None = None,
) -> Tensor:
    output = LinearFunction_LowrankPlusQuantization.apply(
        input,
        self.weight,
        self.bias,
        compress_method if self.training else None,
        compress_kwargs if self.training else None,
        quant_method if self.training else None,
    )
    return output