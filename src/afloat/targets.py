"""Analytic functions and measurements of features hidden by an overall loss."""

import math

import torch
from torch import Tensor

TARGETS = ("linear", "kink", "plateau", "detail", "bump", "composition")


def target_values(name: str, x: Tensor) -> Tensor:
    a, b = x[:, 0], x[:, 1]
    if name == "linear":
        y = a + 0.5 * b
    elif name == "kink":
        y = a.relu() + 0.3 * b.abs()
    elif name == "plateau":
        y = torch.tanh(6 * (a + 0.5 * b))
    elif name == "detail":
        y = torch.sin(2 * math.pi * a) + 0.01 * torch.sin(8 * math.pi * b)
    elif name == "bump":
        y = torch.exp(-80 * ((a - 0.6).square() + (b + 0.3).square()))
    elif name == "composition":
        y = torch.tanh(3 * torch.sin(4 * a) + 2 * b)
    else:
        raise ValueError(f"Unknown target: {name}")
    return y.unsqueeze(1)


def sample_inputs(seed: int, count: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return 2 * torch.rand(count, 2, generator=generator) - 1


def validation_inputs() -> Tensor:
    # The grid resolves the small oscillation and localized bump for every seed.
    axis = torch.linspace(-1, 1, 65)
    return torch.cartesian_prod(axis, axis)


def feature_metrics(name: str, x: Tensor, prediction: Tensor) -> dict[str, float]:
    errors = (prediction - target_values(name, x)).square().flatten()
    a, b = x[:, 0], x[:, 1]
    metrics = {"validation_mse": float(errors.mean())}
    if name == "kink":
        mask = (a.abs() <= 0.1) | (b.abs() <= 0.1)
        metrics["transition_mse"] = float(errors[mask].mean())
    elif name == "plateau":
        mask = (a + 0.5 * b).abs() <= 0.2
        metrics["transition_mse"] = float(errors[mask].mean())
        metrics["flat_region_mse"] = float(errors[~mask].mean())
    elif name == "bump":
        mask = (a - 0.6).square() + (b + 0.3).square() <= 0.15**2
        metrics["bump_region_mse"] = float(errors[mask].mean())
        metrics["outside_bump_mse"] = float(errors[~mask].mean())
    elif name == "detail":
        component = torch.sin(8 * math.pi * b)
        residual = prediction.flatten() - torch.sin(2 * math.pi * a)
        amplitude = float((residual * component).sum() / component.square().sum())
        metrics["detail_amplitude"] = amplitude
        metrics["detail_amplitude_error"] = abs(amplitude - 0.01)
    return metrics
