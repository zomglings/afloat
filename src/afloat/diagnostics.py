"""Observe activation regions on identical probes with and without rounded weights."""

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.func import functional_call


def capture_inputs(
    model: nn.Module, parameters: dict[str, Tensor], x: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    captured: dict[str, Tensor] = {}

    def make_hook(name: str) -> Callable[[nn.Module, tuple[Tensor, ...]], None]:
        def capture(module: nn.Module, args: tuple[Tensor, ...]) -> None:
            captured[name] = args[0].detach().clone()

        return capture

    handles = [
        module.register_forward_pre_hook(make_hook(name))
        for name, module in model.named_modules()
        if isinstance(module, (nn.ReLU, nn.Tanh, nn.SiLU))
    ]
    try:
        with torch.no_grad():
            prediction: Tensor = functional_call(model, parameters, (x,), strict=True)
    finally:
        for handle in handles:
            handle.remove()
    return prediction, captured


def activation_statistics(
    model: nn.Module, parameters: dict[str, Tensor], x: Tensor
) -> list[dict[str, object]]:
    _, reference = capture_inputs(model, dict(model.named_parameters()), x)
    _, rounded = capture_inputs(model, parameters, x)
    rows: list[dict[str, object]] = []
    for name, z in rounded.items():
        base = reference[name]
        family = name.rsplit(".", 1)[1]
        if family == "relu":
            slope = (z > 0).float()
            output, original = z.relu(), base.relu()
        elif family == "tanh":
            output, original = z.tanh(), base.tanh()
            slope = 1 - output.square()
        else:
            sigmoid = z.sigmoid()
            slope = sigmoid * (1 + z * (1 - sigmoid))
            output, original = z * sigmoid, base * base.sigmoid()
        quantiles = torch.quantile(z, torch.tensor([0.01, 0.5, 0.99]))
        rows.append(
            {
                "activation": name,
                "family": family,
                "input_p01": float(quantiles[0]),
                "input_p50": float(quantiles[1]),
                "input_p99": float(quantiles[2]),
                "input_abs_max": float(z.abs().max()),
                "negative_fraction": float((z < 0).float().mean()),
                "near_zero_fraction": float((z.abs() < 0.1).float().mean()),
                "small_slope_fraction": float((slope.abs() < 0.01).float().mean()),
                "mean_abs_slope": float(slope.abs().mean()),
                "relu_gate_flip_fraction": (
                    float(((z > 0) != (base > 0)).float().mean())
                    if family == "relu"
                    else None
                ),
                "nonlinear_output_mse": float((output - original).square().mean()),
            }
        )
    return rows
