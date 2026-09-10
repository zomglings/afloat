"""Finite minifloat grids and per-array power-of-two scaling."""

import math
from dataclasses import dataclass
from functools import cache
from time import perf_counter

import torch
from torch import Tensor


@dataclass(frozen=True)
class FloatFormat:
    name: str
    exponent_bits: int
    fraction_bits: int
    finite_only: bool = False

    def levels(self) -> Tensor:
        return positive_levels(self)


@cache
def positive_levels(fmt: FloatFormat) -> Tensor:
    bias = 2 ** (fmt.exponent_bits - 1) - 1
    fractions = 2**fmt.fraction_bits
    exponents = 2**fmt.exponent_bits
    values = []
    for exponent in range(exponents if fmt.finite_only else exponents - 1):
        for fraction in range(fractions):
            if fmt.finite_only and (exponent, fraction) == (
                exponents - 1,
                fractions - 1,
            ):
                continue
            value = (
                fraction / fractions * 2.0 ** (1 - bias)
                if exponent == 0
                else (1 + fraction / fractions) * 2.0 ** (exponent - bias)
            )
            values.append(value)
    return torch.tensor(values, dtype=torch.float32)


FORMATS = {
    "e4m3": FloatFormat("e4m3", 4, 3, finite_only=True),
    "e5m2": FloatFormat("e5m2", 5, 2),
    "e3m4": FloatFormat("e3m4", 3, 4),
}
MODES = ("fp32", "fixed-e4m3", "fixed-e5m2", "fixed-hybrid", "calibrated", "adaptive")


def array_scale(x: Tensor, fmt: FloatFormat) -> float:
    if x.device.type != "cpu" or x.dtype != torch.float32:
        raise ValueError("The initial simulator requires CPU float32 arrays")
    if x.numel() == 0:
        raise ValueError("Quantization requires a nonempty array")
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError("Quantization requires finite values")
    maximum = float(x.detach().abs().max())
    if maximum == 0:
        return 1.0
    exponent = max(-149, math.ceil(math.log2(maximum / float(fmt.levels()[-1]))))
    return math.ldexp(1.0, exponent)


def quantize(x: Tensor, fmt: FloatFormat, scale: float | None = None) -> Tensor:
    """Round to nearest finite code; midpoint ties choose the even code."""
    natural_scale = array_scale(x, fmt)
    scale = natural_scale if scale is None else scale
    if not math.isfinite(scale) or scale <= 0 or float(torch.tensor(scale)) != scale:
        raise ValueError("Scale must be positive and exactly representable in FP32")
    levels = fmt.levels().double()
    # FP64 intermediates preserve FP32 subnormals when applying very small scales.
    magnitudes = x.detach().double().abs() / scale
    right = torch.searchsorted(levels, magnitudes.contiguous()).clamp_max(
        len(levels) - 1
    )
    left = (right - 1).clamp_min(0)
    left_error = (magnitudes - levels[left]).abs()
    right_error = (magnitudes - levels[right]).abs()
    choose_right = (right_error < left_error) | (
        (right_error == left_error) & (right.remainder(2) == 0)
    )
    codes = torch.where(choose_right, right, left)
    result = torch.copysign(levels[codes] * scale, x.detach()).float()
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Scaled quantized value exceeds FP32 simulation range")
    return result


def choose_format(x: Tensor, sample_size: int = 2048) -> str:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    flat = x.detach().flatten()
    sample = flat[:: max(1, math.ceil(flat.numel() / sample_size))]
    scores = {}
    for name, fmt in FORMATS.items():
        rounded = quantize(sample, fmt, array_scale(x, fmt))
        scores[name] = float((sample.double() - rounded.double()).square().mean())
    return min(scores, key=lambda name: scores[name])


class FormatPolicy:
    def __init__(
        self, mode: str, initial: dict[str, str], interval: int, sample_size: int
    ) -> None:
        if mode not in MODES or interval <= 0 or sample_size <= 0:
            raise ValueError("Invalid format policy")
        self.mode = mode
        self.choices = initial.copy()
        self.interval = interval
        self.sample_size = sample_size
        self.events: list[dict[str, object]] = []
        self.selection_seconds = 0.0

    def apply(self, x: Tensor, name: str, kind: str, step: int) -> Tensor:
        if kind not in ("weight", "gradient"):
            raise ValueError("kind must be weight or gradient")
        if self.mode == "fp32":
            return x.detach()
        key = f"{kind}:{name}"
        changed = False
        if self.mode.startswith("fixed-"):
            choice = self.mode.removeprefix("fixed-")
            if choice == "hybrid":
                choice = "e4m3" if kind == "weight" else "e5m2"
        else:
            choice = self.choices[key]
            if self.mode == "adaptive" and step > 0 and step % self.interval == 0:
                started = perf_counter()
                selected = choose_format(x, self.sample_size)
                self.selection_seconds += perf_counter() - started
                changed = selected != choice
                choice = selected
                self.choices[key] = choice
        fmt = FORMATS[choice]
        scale = array_scale(x, fmt)
        rounded = quantize(x, fmt, scale)
        self.events.append(
            {
                "step": step,
                "array": key,
                "format": choice,
                "changed": changed,
                "scale": scale,
                "mse": float((x.detach().double() - rounded.double()).square().mean()),
                "zeroed": int(((x.detach() != 0) & (rounded == 0)).sum()),
                "clipped": int((x.detach().abs() / scale > fmt.levels()[-1]).sum()),
                "elements": x.numel(),
            }
        )
        return rounded

    def evaluate(self, x: Tensor, name: str) -> Tensor:
        """Apply current weight choices without changing training or its logs."""
        if self.mode == "fp32":
            return x.detach()
        if self.mode.startswith("fixed-"):
            choice = self.mode.removeprefix("fixed-")
            choice = "e4m3" if choice == "hybrid" else choice
        else:
            choice = self.choices[f"weight:{name}"]
        return quantize(x, FORMATS[choice])
