import torch
import torch.nn as nn

class LayerNorm(nn.Module):

    def __init__(self, shape, eps=10e-10):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.randn(shape))
        self.beta = nn.Parameter(torch.randn(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * (x - x.mean(-1, keepdim=True)) / (torch.sqrt(x.var(-1, unbiased=True ,keepdim=True)) + self.eps) + self.beta
        
