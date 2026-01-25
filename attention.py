import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, in_dim, n_heads, attention_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = attention_dim // n_heads
        self.attention_dim = self.block_size * n_heads
        self.n_heads = n_heads
        self.wq = nn.Parameter(torch.randn(n_heads, in_dim, self.block_size))
        self.wk = nn.Parameter(torch.rand_like(self.wq))
        self.wv = nn.Parameter(torch.rand_like(self.wq))
        self.reproj = nn.Linear(attention_dim, in_dim)
    
    def forward(self, x: torch.Tensor, mask=False):
        # gives batches x heads x n x attn_dim
        n_batches = x.shape[0]

        k = torch.einsum("bni,hid->bhnd", x, self.wk)
        q = torch.einsum("bni,hid->bhnd", x, self.wq)
        v = torch.einsum("bni,hid->bhnd", x, self.wv)

        # gives batches x n_heads x n x n
        if mask:
            w = q @ torch.transpose(k, -2, -1) / math.sqrt(self.block_size)

            tril = torch.tril(torch.ones_like(w))
            w = torch.softmax(torch.masked_fill(w, tril==0, float("-inf")), -1)


        else:
            w = torch.softmax((q @ torch.transpose(k, -2, -1)) / math.sqrt(self.block_size), -1)

        o = w @ v  # n_heads x n x block_size
        o = o.permute(0,2,1,3).reshape(n_batches,-1, self.attention_dim)  # n x attention_dim

        return self.reproj(o)
