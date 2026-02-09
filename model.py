import math
from typing import List, Literal
import torch
import torch.nn as nn
import numpy

from attention import MultiHeadAttention
from layer_norm import LayerNorm
from feed_forward_mlp import MLP
from tokeniser import BPETokeniser


class KVCache():
    def __init__(self, context_len: int):
        self.context_len = context_len
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """
        Docstring for update
        
        :param k: keys (format (batch, head, n, dim))
        :type k: torch.Tensor
        :param v: values (format (batch, head, n, dim))
        :type v: torch.Tensor
        """
        
        self.k = k[:,:,(k.size(2) - self.context_len):,:]
        self.v = v[:,:,(v.size(2) - self.context_len):,:]

    def clear(self):
        self.k = self.v = None

    def get_cache(self):
        return (self.k, self.v)

class Block(nn.Module):
    def __init__(self, in_dims, attention_dims, n_attn_heads, context_length, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kv_cache = KVCache(context_length)
        self.attn = MultiHeadAttention(in_dims, n_attn_heads, attention_dims, context_length)
        self.layer_norm_1 = LayerNorm(in_dims)
        self.layer_norm_2 = LayerNorm(in_dims)
        self.ff = MLP([in_dims, in_dims*4, in_dims])
    
    def forward(self, x):
        x1 = self.attn(self.layer_norm_1(x), mask=True)
        x = x+x1

        x1 = self.ff(self.layer_norm_2(x))
        x = x + x1

        return x
    
    def forward_kv_cached(self, x, keys=None, values=None):
        x1, k ,v = self.attn.forward_kv_cached(self.layer_norm_1(x), True, self.kv_cache.k, self.kv_cache.v)
        self.kv_cache.update(k,v)

        x = x+x1

        x1 = self.ff(self.layer_norm_2(x))
        x = x + x1

        return x

class Transformer(nn.Module):
    def __init__(self, embedding_dims, n_blocks, attention_dims, n_attn_heads, context_length, vocab_length, tokeniser, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_dims = embedding_dims
        self.tokeniser: BPETokeniser = tokeniser
        self.context_length = context_length
        self.embeddings = nn.Parameter(torch.randn(vocab_length, embedding_dims)*0.02)
        
        # figure this one out
        self.pos_embeddings = nn.Parameter(torch.randn(context_length, embedding_dims)*0.02)

        self.blocks = nn.Sequential(
                *[Block(embedding_dims, attention_dims, n_attn_heads, context_length)
            for _ in range(n_blocks)]
        )

        self.final_norm = LayerNorm(embedding_dims)

        self.reproj = nn.Linear(embedding_dims, vocab_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok_e = self.embeddings[x] * math.sqrt(self.embedding_dims)
        pos_e = self.pos_embeddings[:x.shape[-1]]
        e = tok_e+pos_e

        return self.reproj(self.final_norm(self.blocks(e)))

    def forward_inference(self, x: torch.Tensor):
        tok_e = self.embeddings[x] * math.sqrt(self.embedding_dims)
        pos_e = self.pos_embeddings[:x.shape[-1]]
        e = tok_e+pos_e
        
        for block in self.blocks:
            e = block.forward_kv_cached(e) # type: ignore

        return self.reproj(self.final_norm(e))


    def inference(self, prompts: List[str], max_tokens: int = 30, temperature=0.2, decoding_mode: Literal["multinomial"] | Literal["greedy"] | Literal["beam"] | Literal["nucleus"] = "greedy"):
        self.eval()
        with torch.no_grad():
            tokens = [self.tokeniser.encode(prompt) for prompt in prompts]
            # print(tokens)
            token_ends = torch.tensor([len(prompt) - 1 for prompt in tokens], dtype=torch.long)
            # print(token_ends)
            
            tokens = [prompt + [0]*(self.context_length - token_ends[i] - 1) for i, prompt in enumerate(tokens)]
            tokens = torch.tensor(tokens, dtype=torch.long)
            if decoding_mode == "greedy":
                outputs = torch.zeros(max_tokens, tokens.size(0))
                for i in range(max_tokens):
                    token_probs = torch.softmax(self.forward(tokens), -1)
                    print(token_probs.shape)
                    token_probs = token_probs[torch.arange(tokens.size(0)), token_ends, :]
                    print(f"After: {token_probs.shape}")
                    out_tokens = torch.argmax(token_probs,-1).long()
                    outputs[i] = out_tokens
                    full = token_ends == self.context_length - 1
                    if full.any():
                        tokens[full, :-1] = tokens[full, 1:]
                    write_locations = torch.where(
                        full,
                        torch.full_like(token_ends, self.context_length - 1),
                        token_ends + 1
                    )
                    tokens[torch.arange(tokens.size(0)), write_locations] = out_tokens
                    token_ends = write_locations
                outputs = outputs.permute(1,0)
                outputs_strings = []
                for seq in outputs:
                    outputs_strings.append(self.tokeniser.decode(seq.tolist()))
                
                return outputs_strings
            elif decoding_mode == "multinomial":
                outputs = torch.zeros(max_tokens, tokens.size(0))
                for i in range(max_tokens):
                    logits = self.forward(tokens)
                    token_probs = torch.softmax(logits[torch.arange(tokens.size(0)), token_ends, :]/temperature, -1)

                    # print(torch.topk(token_probs, 5, -1))
                    out_tokens = torch.multinomial(token_probs, 1).squeeze(-1)
                    outputs[i] = out_tokens
                    full = token_ends == self.context_length - 1
                    if full.any():
                        tokens[full, :-1] = tokens[full, 1:]
                    write_locations = torch.where(
                        full,
                        torch.full_like(token_ends, self.context_length - 1),
                        token_ends + 1
                    )
                    tokens[torch.arange(tokens.size(0)), write_locations] = out_tokens
                    token_ends = write_locations

                outputs = outputs.permute(1,0)
                outputs_strings = []
                for seq in outputs:
                    outputs_strings.append(self.tokeniser.decode(seq.tolist(), show_boundaries=False))
                
                return outputs_strings
            elif decoding_mode == "nucleus":
                pass
            elif decoding_mode == "beam":
                pass
            

    def inference_cached(self, prompts: List[str], max_tokens: int = 30, temperature=0.2, decoding_mode: Literal["multinomial"] | Literal["greedy"] | Literal["beam"] | Literal["nucleus"] = "greedy"):
        self.eval()
        with torch.no_grad():
            tokens = [self.tokeniser.encode(prompt) for prompt in prompts]
            # print(tokens)
            token_ends = torch.tensor([len(prompt) - 1 for prompt in tokens], dtype=torch.long)
            # print(token_ends)
            
            tokens = torch.tensor(tokens, dtype=torch.long)
            if decoding_mode == "greedy":
                outputs = torch.zeros(max_tokens, tokens.size(0))
                for i in range(max_tokens):
                    token_probs = torch.softmax(self.forward_inference(tokens), -1)
                    print(token_probs.shape)
                    token_probs = token_probs[:, -1:, :]
                    print(f"After: {token_probs.shape}")
                    out_tokens = torch.argmax(token_probs,-1).long()
                    outputs[i] = out_tokens
                    full = token_ends == self.context_length - 1
                    if full.any():
                        tokens[full, :-1] = tokens[full, 1:]
                    write_locations = torch.where(
                        full,
                        torch.full_like(token_ends, self.context_length - 1),
                        token_ends + 1
                    )
                 
                    tokens = torch.cat((tokens, out_tokens), dim=1)
                    token_ends = write_locations
                outputs = outputs.permute(1,0)
                outputs_strings = []
                for seq in outputs:
                    outputs_strings.append(self.tokeniser.decode(seq.tolist()))
                
                return outputs_strings
            elif decoding_mode == "multinomial":
                pass
            elif decoding_mode == "nucleus":
                pass
            elif decoding_mode == "beam":
                pass
            
