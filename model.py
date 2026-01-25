from typing import Literal
import torch
import torch.nn as nn

from attention import MultiHeadAttention
from layer_norm import LayerNorm
from feed_forward_mlp import MLP
from tokeniser import BPETokeniser

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
    def __init__(self, embedding_dims, n_blocks, attention_dims, n_attn_heads, context_length, vocab_length, tokeniser, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokeniser: BPETokeniser = tokeniser
        self.context_length = self.tokeniser
        self.embeddings = nn.Parameter(torch.randn(vocab_length, embedding_dims))
        
        # figure this one out
        self.pos_embeddings = nn.Parameter(torch.zeros(context_length, embedding_dims))

        self.blocks = nn.Sequential(
                *[Block(embedding_dims, attention_dims, n_attn_heads)
            for _ in range(n_blocks)]
        )

        self.reproj = nn.Linear(embedding_dims, vocab_length)

    def forward(self, x) -> torch.Tensor:
        tok_e = self.embeddings[x]
        pos_e = self.pos_embeddings[x]
        e = tok_e+pos_e

        return self.reproj(self.blocks(e))

    def inference(self, prompt: str, max_tokens: int = 500, decoding_mode: Literal["greedy"] | Literal["beam"] | Literal["nucleus"] = "greedy"):
        tokens = self.tokeniser.encode(prompt)
        if len(tokens) > max_tokens: raise ValueError(f"Context window exceeded: max {self.max_tokens}, got {len(tokens)}")
        output = []
        for _ in range(max_tokens):
            token_probs = torch.softmax(self.forward(tokens), 0)
            if decoding_mode=="greedy":
                output.append(self.tokeniser.decode(torch.argmax(token_probs)))

        return "".join(output)
    
    def inference_stream(self, prompt: str, max_tokens: int = 500, decoding_mode: Literal["greedy"] | Literal["beam"] | Literal["nucleus"] = "greedy"):
        tokens = self.tokeniser.encode(prompt)
        if len(tokens) > max_tokens: raise ValueError(f"Context window exceeded: max {self.max_tokens}, got {len(tokens)}")

        for _ in range(max_tokens):
            token_probs = torch.softmax(self.forward(tokens), 0)
            if decoding_mode=="greedy":
                yield self.tokeniser.decode(torch.argmax(token_probs))


            