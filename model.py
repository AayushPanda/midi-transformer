import torch
import torch.nn as nn

from attention import MultiHeadAttention
from layer_norm import LayerNorm
from feed_forward_mlp import MLP

class Block(nn.Module):
    def __init__(self, in_dims, attention_dims, n_attn_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attn = MultiHeadAttention(in_dims, n_attn_heads, attention_dims)
        self.layer_norm_1 = LayerNorm(in_dims)
        self.layer_norm_2 = LayerNorm(in_dims)
        self.ff = MLP([in_dims, in_dims*4, in_dims*4, in_dims])
    
    def forward(self, x):
        x1 = self.attn(x, mask=True)
        x = self.layer_norm_1(x+x1)

        x1 = self.ff(x)
        x = self.layer_norm_2(x + x1)

        return x

class Transformer(nn.Module):
    def __init__(self, embedding_dims, n_blocks, attention_dims, n_attn_heads, context_length, vocab_length, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embeddings = nn.Parameter(torch.randn(vocab_length, embedding_dims))
        
        # figure this one out
        self.pos_embeddings = nn.Parameter(torch.zeros(context_length, embedding_dims))

        self.blocks = nn.Sequential(
                *[Block(embedding_dims, attention_dims, n_attn_heads)
            for _ in range(n_blocks)]
        )

        self.reproj = nn.Linear(embedding_dims, vocab_length)

    def forward(self, x):
        tok_e = self.embeddings[x]
        pos_e = self.pos_embeddings[x]
        e = tok_e+pos_e

        return self.reproj(self.blocks(e))
