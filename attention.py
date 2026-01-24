import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, block_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        head_dims = block_size / n_heads

        kp

