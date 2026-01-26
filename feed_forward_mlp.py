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
            self.layers.append(nn.Linear(shape[i], shape[i+1]))
            if i < len(shape)-2:
                self.layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)
        

    def forward(self, x):
        return self.layers(x)

