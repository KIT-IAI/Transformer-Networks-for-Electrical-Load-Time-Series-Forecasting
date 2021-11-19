import torch
from torch import nn
from torch.autograd.grad_mode import F


class Generator(nn.Module):

    def __init__(self, d_model: int, target_features: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, target_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.proj(x))
