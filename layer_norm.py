import torch
import torch.nn as nn

class LayerNorm(nn.Module):

    def __init__(self, shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False ,keepdim=True) + self.eps) + self.beta
    