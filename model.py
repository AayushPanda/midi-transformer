from typing import List, Literal
import torch
import torch.nn as nn
import numpy

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
        self.context_length = context_length
        self.embeddings = nn.Parameter(torch.randn(vocab_length, embedding_dims))
        
        # figure this one out
        self.pos_embeddings = nn.Parameter(torch.randn(context_length, embedding_dims))

        self.blocks = nn.Sequential(
                *[Block(embedding_dims, attention_dims, n_attn_heads)
            for _ in range(n_blocks)]
        )

        self.reproj = nn.Linear(embedding_dims, vocab_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok_e = self.embeddings[x]
        pos_e = self.pos_embeddings[:x.shape[-1]]
        e = tok_e+pos_e

        return self.reproj(self.blocks(e))

    def inference(self, prompts: List[str], max_tokens: int = 30, decoding_mode: Literal["greedy"] | Literal["beam"] | Literal["nucleus"] = "greedy"):
        self.eval()
        with torch.no_grad():
            tokens = [self.tokeniser.encode(prompt) for prompt in prompts]
            print(tokens)
            token_ends = torch.tensor([len(prompt) for prompt in tokens], dtype=torch.long)
            print(token_ends)
            tokens = [prompt + [0]*(self.context_length - token_ends[i]) for i, prompt in enumerate(tokens)]
            tokens = torch.tensor(tokens, dtype=torch.long)
            outputs = []
            for _ in range(max_tokens):
                if decoding_mode == "greedy":
                    token_probs = torch.softmax(self.forward(tokens), -1)
                    print(torch.argmax(token_probs, -1).int())
                    token_probs = token_probs[torch.arange(tokens.size(0)), token_ends, :]

                    print(torch.topk(token_probs, 5, -1))
                    out_tokens = torch.multinomial(token_probs, 1).squeeze(-1)
                    # out_tokens = torch.argmax(token_probs,-1).int()
                    outputs.append(list(out_tokens))
                    tokens[torch.arange(tokens.size(0)), token_ends] = out_tokens
                    token_ends += 1
                    token_ends = torch.clamp(token_ends, max=self.context_length-1)
                    # tokens_c = tokens.clone()
                    # tokens[:, :-1] = tokens_c[:, 1:]
            
            outputs = torch.tensor(outputs)
            outputs = outputs.permute(1,0)
            outputs_strings = []
            for seq in outputs:
                outputs_strings.append(self.tokeniser.decode(seq.tolist()))
            
            return outputs_strings
