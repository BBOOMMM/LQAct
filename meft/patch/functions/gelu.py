from torch import Tensor
import transformers.activations

from ...ops.gelu import GELUFunction, GELUFunction_LowrankPlusQuantization


def gelu_forward(
    self: "transformers.activations.GELUActivation",
    input: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    output = GELUFunction.apply(
        input,
        "none",
        compress_kwargs if self.training else None,
    )
    return output


def gelu_forward_lowrank_plus_quantization(
    self: "transformers.activations.GELUActivation",
    input: Tensor,
    compress_method: str | None = None,
    compress_kwargs: dict | None = None,
    quant_method: str | None = None,
) -> Tensor:
    output = GELUFunction_LowrankPlusQuantization.apply(
        input,
        "none",
        compress_method if self.training else None,
        compress_kwargs if self.training else None,
        quant_method if self.training else None,
    )
    return output


def gelu_new_forward(
    self: "transformers.activations.NewGELUActivation",
    input: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    output = GELUFunction.apply(
        input,
        "tanh",
        compress_kwargs if self.training else None,
    )
    return output


def gelu_pytorch_tanh_forward(
    self: "transformers.activations.PytorchGELUTanh",
    input: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    output = GELUFunction.apply(
        input,
        "tanh",
        compress_kwargs if self.training else None,
    )
    return output


def quick_gelu_forward(
    self: "transformers.activations.QuickGELUActivation",
    input: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    output = GELUFunction.apply(
        input,
        "sigmoid",
        compress_kwargs if self.training else None,
    )
    return output
