import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, in_dim, n_heads, attention_dim, context_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = attention_dim // n_heads
        self.attention_dim = self.block_size * n_heads
        self.n_heads = n_heads
        scale = math.sqrt(2.0 / (in_dim + self.block_size))
        self.wq = nn.Parameter(torch.randn(n_heads, in_dim, self.block_size)*scale)
        self.wk = nn.Parameter(torch.randn_like(self.wq)*scale)
        self.wv = nn.Parameter(torch.randn_like(self.wq)*scale)
        self.reproj = nn.Linear(attention_dim, in_dim)
        self.register_buffer("tril", torch.tril(torch.ones(context_length, context_length)))
    
    def forward(self, x: torch.Tensor, mask=False):
        # gives batches x heads x n x attn_dim
        n_batches = x.shape[0]

        k = torch.einsum("bni,hid->bhnd", x, self.wk)
        q = torch.einsum("bni,hid->bhnd", x, self.wq)
        v = torch.einsum("bni,hid->bhnd", x, self.wv)

        # gives batches x n_heads x n x n
        if mask:
            w = q @ torch.transpose(k, -2, -1) / math.sqrt(self.block_size)

            tril = self.get_buffer("tril")[:w.size(2), :w.size(2)]
            w = torch.softmax(torch.masked_fill(w, tril==0, float("-inf")), -1)


        else:
            w = torch.softmax((q @ torch.transpose(k, -2, -1)) / math.sqrt(self.block_size), -1)

        o = w @ v  # n_heads x n x block_size
        o = o.permute(0,2,1,3).reshape(n_batches,-1, self.attention_dim)  # n x attention_dim

        return self.reproj(o)

    def forward_kv_cached(self, x, mask=False, keys: Optional[torch.Tensor] =None, values: Optional[torch.Tensor]=None):
        n_batches = x.shape[0]
        if not (keys or values):

            k = torch.einsum("bni,hid->bhnd", x, self.wk)
            q = torch.einsum("bni,hid->bhnd", x, self.wq)
            v = torch.einsum("bni,hid->bhnd", x, self.wv)

            # gives batches x n_heads x n x n
            if mask:
                w = q @ torch.transpose(k, -2, -1) / math.sqrt(self.block_size)

                tril = self.get_buffer("tril")[:w.size(2), :w.size(2)]
                w = torch.softmax(torch.masked_fill(w, tril==0, float("-inf")), -1)


            else:
                w = torch.softmax((q @ torch.transpose(k, -2, -1)) / math.sqrt(self.block_size), -1)

            o = w @ v  # n_heads x n x block_size
            o = o.permute(0,2,1,3).reshape(n_batches,-1, self.attention_dim)  # n x attention_dim

            return self.reproj(o), k, v
        else:
            k: torch.Tensor = keys
            v: torch.Tensor = values

            k_new = torch.einsum("bi,hid->bhd", x, self.wk)
            v_new = torch.einsum("bi,hid->bhd", x, self.wk)
            q = torch.einsum("bi,hid->bhd", x, self.wk)
            k = torch.cat([k, k_new], dim=1)
            v = torch.cat([v, v_new], dim=1)

            q = torch.einsum("bi,hid->bhd")

            # gives batches x n_heads x n
            w = q @ torch.transpose(k, -2, -1) / math.sqrt(self.block_size)
            w = torch.softmax(w, -1)
            o = w @ v   # bhd
            o = o.permute().reshape(n_batches, -1, self.attention_dim)
            return self.reproj(o), k, v
            