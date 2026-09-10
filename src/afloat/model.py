"""Small residual feed-forward blocks with three activation families."""

import math

import torch
from torch import Tensor, nn


class MixedBlock(nn.Module):
    def __init__(self, activation_gain: float = 1.0) -> None:
        super().__init__()
        self.activation_gain = activation_gain
        self.expand = nn.Linear(32, 48)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.silu = nn.SiLU()
        self.project = nn.Linear(48, 32)

    def forward(self, h: Tensor) -> Tensor:
        a, b, c = (self.expand(h) * self.activation_gain).chunk(3, dim=-1)
        nonlinear = torch.cat((self.relu(a), self.tanh(b), self.silu(c)), dim=-1)
        result: Tensor = h + self.project(nonlinear)
        return result


class FourierInputs(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        angles = x.unsqueeze(-1) * x.new_tensor([1, 2, 4, 8]) * math.pi
        return torch.cat((x, angles.sin().flatten(1), angles.cos().flatten(1)), dim=1)


def make_model(
    seed: int, activation_gain: float = 1.0, input_features: str = "raw"
) -> nn.Sequential:
    if input_features not in ("raw", "fourier"):
        raise ValueError("Unknown input features")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        layers = nn.Sequential(
            nn.Linear(2 if input_features == "raw" else 18, 32),
            MixedBlock(activation_gain),
            MixedBlock(activation_gain),
            MixedBlock(activation_gain),
            nn.Linear(32, 1),
        )
        return (
            layers
            if input_features == "raw"
            else nn.Sequential(FourierInputs(), *layers)
        )
