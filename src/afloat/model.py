"""Small residual feed-forward blocks with three activation families."""

import torch
from torch import Tensor, nn


class MixedBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.expand = nn.Linear(32, 48)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.silu = nn.SiLU()
        self.project = nn.Linear(48, 32)

    def forward(self, h: Tensor) -> Tensor:
        a, b, c = self.expand(h).chunk(3, dim=-1)
        nonlinear = torch.cat((self.relu(a), self.tanh(b), self.silu(c)), dim=-1)
        result: Tensor = h + self.project(nonlinear)
        return result


def make_model(seed: int) -> nn.Sequential:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return nn.Sequential(
            nn.Linear(2, 32), MixedBlock(), MixedBlock(), MixedBlock(), nn.Linear(32, 1)
        )
