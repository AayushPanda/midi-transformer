import torch
from model import Transformer
from tokeniser import BPETokeniser

tokeniser = BPETokeniser()
tokeniser.load("tokeniser_vocab.pkl")

model = Transformer(
    embedding_dims=768,
    n_blocks=12,
    attention_dims=768,
    n_attn_heads=12,
    context_length=512,
    vocab_length=301,
    tokeniser=tokeniser
).to("cuda")

model_state_dict = torch.load("checkpoint.pth")
model.load_state_dict(model_state_dict)
model.eval()

print(model.inference(["""Lord of my love, to whom in vassalage
  Thy merit hath my duty strongly knit;
  To thee I send this written embassage
  To witness duty, not to"""]))