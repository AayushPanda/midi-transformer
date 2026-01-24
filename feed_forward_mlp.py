from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, shape: List[int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        depth = len(shape)
        self.layers = nn.Sequential(*[
            nn.Linear(shape[i], shape[j]) for i,j in zip(range(depth-1), range(1, depth+1))
        ])

    def forward(self, x):
        return self.layers(x)

