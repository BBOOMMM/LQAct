import contextlib
from collections.abc import Callable

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import (
    _get_device_module, _infer_device_type, get_device_states, set_device_states, _get_autocast_kwargs
)

from ..compressed import CompressedTensor, get_quant_cache

import math

import bitsandbytes

from ..quant.one_bit import quantize_1bit, dequantize_1bit, quantize_1bit_group, dequantize_1bit_group
from ..quant.ternary import quantize_ternary_group_lastdim, dequantize_ternary_group_lastdim
from ..quant.two_bit import quantize_2bit_group, dequantize_2bit_group

# CheckpointFunction 存储的是每层的 输入


def _freeze_cache_value(value):
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_cache_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_cache_value(item) for item in value))
    return value


def _quant_cache_key(quant_method: str | None, compress_kwargs: dict | None) -> tuple:
    kwargs_items = tuple(
        sorted((key, _freeze_cache_value(value)) for key, value in (compress_kwargs or {}).items())
    )
    return quant_method, kwargs_items


def _quantize_residual(tensor: Tensor, quant_method: str | None):
    if quant_method == "1bit_pertensor":
        return quantize_1bit(tensor)
    if quant_method == "1bit_pergroupchannel":
        return quantize_1bit_group(tensor, group_size=1)
    if quant_method == "ternary":
        return quantize_ternary_group_lastdim(tensor)
    if quant_method == "two_bit_group":
        return quantize_2bit_group(tensor, group_size=1)
    raise ValueError(f"Unsupported quant_method: {quant_method}")


def detach_variable(
    ctx,
    hidden_states: Tensor,
    args: tuple,
    kwargs: dict,
) -> tuple[Tensor, tuple, dict]:

    if hasattr(ctx, 'project_matrix'):
        Q = ctx.project_matrix
        detached_hidden_states = (Q @ hidden_states).reshape(ctx.org_shape).detach()
    elif hasattr(ctx, 'seed'):
        z = hidden_states
        seed = ctx.seed
        g = torch.Generator(device=z.device)
        g.manual_seed(seed)
        P = torch.randn(ctx.hidden_size, ctx.rank, device=z.device,
                        dtype=z.dtype, generator=g) / math.sqrt(ctx.rank)
        detached_hidden_states = (z @ P.T).reshape(ctx.org_shape).detach()
    elif isinstance(hidden_states, CompressedTensor):
        detached_hidden_states = hidden_states.reconstruct().detach()
    else:
        detached_hidden_states = hidden_states.detach()
    detached_hidden_states.requires_grad = hidden_states.requires_grad

    detached_args = []
    for arg in args:
        if not isinstance(arg, Tensor):
            detached_args.append(arg)
        else:
            x = arg.detach()
            x.requires_grad = arg.requires_grad
            detached_args.append(x)
    detached_args = tuple(detached_args)

    detached_kwargs = {}
    for key, val in kwargs.items():
        if not isinstance(val, Tensor):
            detached_kwargs[key] = val
        else:
            x = val.detach()
            x.requires_grad = val.requires_grad
            detached_kwargs[key] = x
        
    return detached_hidden_states, detached_args, detached_kwargs


def detach_variable_LowrankPlusQuantization(
    ctx,
    hidden_states: Tensor,
    args: tuple,
    kwargs: dict,
) -> tuple[Tensor, tuple, dict]:

    # if isinstance(hidden_states, CompressedTensor):
    #     LowRank = hidden_states.reconstruct().detach()
    #     quant_state = hidden_states.quant_state
    #     dequant = bitsandbytes.functional.dequantize_4bit(*quant_state)
    #     detached_hidden_states = LowRank + dequant
    #     del LowRank, quant_state, dequant
    # if isinstance(hidden_states, torch.Tensor) and hasattr(hidden_states, "quant_state"):
    #     quant_state = hidden_states.quant_state
    #     detached_hidden_states = bitsandbytes.functional.dequantize_4bit(*quant_state)
    #     del quant_state
    # else:
    #     detached_hidden_states = hidden_states.detach()
    
    if ctx.quant_method == 'two_bit_group':
        reconstructed_R = dequantize_2bit_group(ctx.packed_R, ctx.alpha, ctx.shape)
        detached_hidden_states = reconstructed_R.detach()
        detached_hidden_states.requires_grad = True
    else:
        if ctx.quant_method == '1bit_pertensor':
            reconstructed_R = dequantize_1bit(ctx.packed_R, ctx.alpha, ctx.shape)
        elif ctx.quant_method == '1bit_pergroupchannel':
            reconstructed_R = dequantize_1bit_group(ctx.packed_R, ctx.alpha, ctx.shape)
        elif ctx.quant_method == 'ternary':
            reconstructed_R = dequantize_ternary_group_lastdim(ctx.packed_R, ctx.alpha, ctx.shape)
        # reconstructed_R = 0
        detached_hidden_states = (hidden_states.reconstruct() + reconstructed_R).detach()
        detached_hidden_states.requires_grad = hidden_states.requires_grad

    detached_args = []
    for arg in args:
        if not isinstance(arg, Tensor):
            detached_args.append(arg)
        else:
            x = arg.detach()
            x.requires_grad = arg.requires_grad
            detached_args.append(x)
    detached_args = tuple(detached_args)

    detached_kwargs = {}
    for key, val in kwargs.items():
        if not isinstance(val, Tensor):
            detached_kwargs[key] = val
        else:
            x = val.detach()
            x.requires_grad = val.requires_grad
            detached_kwargs[key] = x
        
    return detached_hidden_states, detached_args, detached_kwargs


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        run_function: Callable,
        self: nn.Module,
        hidden_states: Tensor,
        preserve_rng_state: bool,
        dummy: Tensor | None,
        compress_kwargs: dict | None,
        n_args: int,
        n_kwargs: int,
        *args_kwargs,
    ):
        args, kwargs_keys, kwargs_vals = \
            args_kwargs[:n_args], args_kwargs[n_args:-n_kwargs], args_kwargs[-n_kwargs:]
        kwargs = dict(zip(kwargs_keys, kwargs_vals))

        outputs = run_function(self, hidden_states, *args, **kwargs)
        return outputs

    @staticmethod
    def setup_context(ctx, inputs, output):
        run_function, self, hidden_states, preserve_rng_state, dummy, compress_kwargs, n_args, n_kwargs, *args_kwargs = inputs
        args, kwargs_keys, kwargs_vals = \
            args_kwargs[:n_args], args_kwargs[n_args:-n_kwargs], args_kwargs[-n_kwargs:]
        kwargs = dict(zip(kwargs_keys, kwargs_vals))

        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.device_type = _infer_device_type(hidden_states, *args, *kwargs_vals)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device_type
        )
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(hidden_states, *args, *kwargs_vals)

        ctx.run_function = run_function
        ctx.self = self

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.input_args = []
        ctx.input_kwargs = {}
        ctx.tensor_keys = []
        saved_tensors = []

        ctx.tensor_keys.append(None)
        if compress_kwargs is not None:
            if 'project_matrix' in compress_kwargs:
                Q = compress_kwargs['project_matrix']
                to_save = Q.T @ hidden_states.reshape((-1, hidden_states.shape[-1]))
                to_save.requires_grad = hidden_states.requires_grad
                saved_tensors.append(to_save)
                ctx.project_matrix = Q
                ctx.org_shape = hidden_states.shape
            elif compress_kwargs.get('RandomGaussion', False):
                assert 'rank' in compress_kwargs
                r = compress_kwargs['rank']
                hidden_size = hidden_states.shape[-1]
                if isinstance(r, float):
                    r = int(r * hidden_size)
                seed = torch.randint(0, 10000, (1,)).item()
                g = torch.Generator(device=hidden_states.device)
                g.manual_seed(seed)
                P = torch.randn(hidden_size, r, device=hidden_states.device,
                                dtype=hidden_states.dtype, generator=g) / math.sqrt(r)
                z = hidden_states.reshape(-1, hidden_size) @ P
                z.requires_grad = hidden_states.requires_grad
                saved_tensors.append(z)
                ctx.seed = seed
                ctx.org_shape = hidden_states.shape
                ctx.hidden_size = hidden_size
                ctx.rank = r
            else:
                saved_tensors.append(CompressedTensor(hidden_states, **compress_kwargs))
        else:
            saved_tensors.append(hidden_states)

        for key, val in enumerate(args):
            if not torch.is_tensor(val):
                ctx.input_args.append(val)
            else:
                ctx.input_args.append(None)
                ctx.tensor_keys.append(key)  # int
                saved_tensors.append(val)

        for key, val in kwargs.items():
            if not torch.is_tensor(val):
                ctx.input_kwargs[key] = val
            else:
                ctx.input_kwargs[key] = None
                ctx.tensor_keys.append(key)  # str
                saved_tensors.append(val)

        ctx.save_for_backward(*saved_tensors)

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> tuple[Tensor | None, ...]:
        # Copy the list to avoid modifying original list.
        input_args = ctx.input_args
        input_kwargs = ctx.input_kwargs

        # Fill in inputs with appropriate saved tensors.
        for key, tensor in zip(ctx.tensor_keys, ctx.saved_tensors):
            if isinstance(key, int):
                input_args[key] = tensor
            elif isinstance(key, str):
                input_kwargs[key] = tensor
            else:
                hidden_states = tensor

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device_type
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states, device_type=ctx.device_type)
            detached_hidden_states, detached_args, detached_kwargs = detach_variable(ctx, hidden_states, input_args, input_kwargs)

            device_autocast_ctx = torch.autocast(
                device_type=ctx.device_type, **ctx.device_autocast_kwargs
            ) if torch.amp.autocast_mode.is_autocast_available(ctx.device_type) else contextlib.nullcontext()
            with torch.enable_grad(), device_autocast_ctx, torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
                outputs = ctx.run_function(ctx.self, detached_hidden_states, *detached_args, **detached_kwargs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        grad_outputs_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                grad_outputs_with_grad.append(grad_outputs[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "None of output has requires_grad=True, this checkpoint() is not necessary."
            )
        torch.autograd.backward(outputs_with_grad, grad_outputs_with_grad)

        grad_hidden_states = detached_hidden_states.grad
        grads_args = tuple(
            arg.grad if isinstance(arg, torch.Tensor) else None
            for arg in detached_args
        )
        grads_kwargs_keys = tuple(
            None for _ in detached_kwargs.keys()
        )
        grads_kwargs_vals = tuple(
            val.grad if isinstance(val, torch.Tensor) else None
            for val in detached_kwargs.values()
        )
        return (None, None, grad_hidden_states, None, None, None, None, None) + grads_args + grads_kwargs_keys + grads_kwargs_vals


class CheckpointFunction_LowrankPlusQuantization(torch.autograd.Function):
    @staticmethod
    def forward(
        run_function: Callable,
        self: nn.Module,
        hidden_states: Tensor,
        preserve_rng_state: bool,
        dummy: Tensor | None,
        compress_method: str | None,
        compress_kwargs: dict | None,
        quant_method: str | None,
        n_args: int,
        n_kwargs: int,
        *args_kwargs,
    ):
        args, kwargs_keys, kwargs_vals = \
            args_kwargs[:n_args], args_kwargs[n_args:-n_kwargs], args_kwargs[-n_kwargs:]
        kwargs = dict(zip(kwargs_keys, kwargs_vals))

        outputs = run_function(self, hidden_states, *args, **kwargs)
        return outputs

    @staticmethod
    def setup_context(ctx, inputs, output):
        run_function, self, hidden_states, preserve_rng_state, dummy, compress_method, compress_kwargs, quant_method, n_args, n_kwargs, *args_kwargs = inputs
        args, kwargs_keys, kwargs_vals = \
            args_kwargs[:n_args], args_kwargs[n_args:-n_kwargs], args_kwargs[-n_kwargs:]
        kwargs = dict(zip(kwargs_keys, kwargs_vals))

        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.device_type = _infer_device_type(hidden_states, *args, *kwargs_vals)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device_type
        )
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(hidden_states, *args, *kwargs_vals)

        ctx.run_function = run_function
        ctx.self = self

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.input_args = []
        ctx.input_kwargs = {}
        ctx.tensor_keys = []
        saved_tensors = []

        ctx.tensor_keys.append(None)
        if compress_kwargs is not None:
            quant_cache = get_quant_cache()
            cache_key = _quant_cache_key(quant_method, compress_kwargs)
            # LowRank = CompressedTensor(hidden_states, **compress_kwargs)
            # # Q, B = LowRank.factors
            # # R = hidden_states - (Q @ B)
            # R = hidden_states - LowRank.reconstruct()
            # quant_state = bitsandbytes.functional.quantize_4bit(
            #     R,
            #     quant_type="nf4",  # 指定 NF4 格式（适配正态分布的 R）
            #     blocksize=128,      # 必须为 32/64/128/256
            #     compress_statistics=True
            # )
            # LowRank.quant_state = quant_state
            # saved_tensors.append(LowRank)
            
            # to_save = torch.empty((), device=hidden_states.device, dtype=hidden_states.dtype)  # 占位符，实际不保存 hidden_states
            # to_save.requires_grad = hidden_states.requires_grad
            # quant_state = bitsandbytes.functional.quantize_4bit(
            #     hidden_states,
            #     quant_type="fp4",  # 指定 NF4 格式（适配正态分布的 R）
            #     blocksize=128,      # 必须为 32/64/128/256
            #     compress_statistics=True
            # )
            # to_save.quant_state = quant_state
            # saved_tensors.append(to_save)

            if hidden_states in quant_cache[cache_key]:
                packed_R, alpha, shape = quant_cache[cache_key][hidden_states]
                if quant_method != 'two_bit_group':
                    saved_tensors.append(CompressedTensor(hidden_states, **compress_kwargs))
            elif quant_method == 'two_bit_group':
                packed_R, alpha, shape = _quantize_residual(hidden_states, quant_method)
                quant_cache[cache_key][hidden_states] = (packed_R, alpha, shape)
            else:
                lowrank = CompressedTensor(hidden_states, **compress_kwargs)
                R = hidden_states - lowrank.reconstruct()
                packed_R, alpha, shape = _quantize_residual(R, quant_method)
                quant_cache[cache_key][hidden_states] = (packed_R, alpha, shape)
                saved_tensors.append(lowrank)

            ctx.packed_R = packed_R
            ctx.alpha = alpha
            ctx.shape = shape
            ctx.quant_method = quant_method
        else:
            saved_tensors.append(hidden_states)

        for key, val in enumerate(args):
            if not torch.is_tensor(val):
                ctx.input_args.append(val)
            else:
                ctx.input_args.append(None)
                ctx.tensor_keys.append(key)  # int
                saved_tensors.append(val)

        for key, val in kwargs.items():
            if not torch.is_tensor(val):
                ctx.input_kwargs[key] = val
            else:
                ctx.input_kwargs[key] = None
                ctx.tensor_keys.append(key)  # str
                saved_tensors.append(val)

        ctx.save_for_backward(*saved_tensors)

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> tuple[Tensor | None, ...]:
        # Copy the list to avoid modifying original list.
        input_args = ctx.input_args
        input_kwargs = ctx.input_kwargs

        # Fill in inputs with appropriate saved tensors.
        hidden_states = None
        for key, tensor in zip(ctx.tensor_keys, ctx.saved_tensors):
            if isinstance(key, int):
                input_args[key] = tensor
            elif isinstance(key, str):
                input_kwargs[key] = tensor
            else:
                hidden_states = tensor

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device_type
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states, device_type=ctx.device_type)
            detached_hidden_states, detached_args, detached_kwargs = detach_variable_LowrankPlusQuantization(ctx, hidden_states, input_args, input_kwargs)

            device_autocast_ctx = torch.autocast(
                device_type=ctx.device_type, **ctx.device_autocast_kwargs
            ) if torch.amp.autocast_mode.is_autocast_available(ctx.device_type) else contextlib.nullcontext()
            with torch.enable_grad(), device_autocast_ctx, torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
                outputs = ctx.run_function(ctx.self, detached_hidden_states, *detached_args, **detached_kwargs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        grad_outputs_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                grad_outputs_with_grad.append(grad_outputs[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "None of output has requires_grad=True, this checkpoint() is not necessary."
            )
        torch.autograd.backward(outputs_with_grad, grad_outputs_with_grad)

        grad_hidden_states = detached_hidden_states.grad
        grads_args = tuple(
            arg.grad if isinstance(arg, torch.Tensor) else None
            for arg in detached_args
        )
        grads_kwargs_keys = tuple(
            None for _ in detached_kwargs.keys()
        )
        grads_kwargs_vals = tuple(
            val.grad if isinstance(val, torch.Tensor) else None
            for val in detached_kwargs.values()
        )
        return (None, None, grad_hidden_states, None, None, None, None, None, None, None) + grads_args + grads_kwargs_keys + grads_kwargs_vals
