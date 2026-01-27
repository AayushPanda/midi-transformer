from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, shape: List[int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        depth = len(shape)
        layers = []
        for i in range(len(shape)-1):
            layers.append(nn.Linear(shape[i], shape[i+1]))
            if i < len(shape)-2:
                layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)
        
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.layers(x)

