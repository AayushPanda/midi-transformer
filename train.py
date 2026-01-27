import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as F
import logging
from model import Transformer
from tokeniser import BPETokeniser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

class TextDataLoader():
    def __init__(self, in_path: str, tokeniser: BPETokeniser) -> None:
        self.file_path = in_path
        self.tokeniser = tokeniser
        logging.info(f"TextDataLoader initialized with file: {self.file_path}")

    def get_train_val_splits(self, context_length: int, ratio: float = 0.7):
        logging.info(f"Loading data from {self.file_path}")
        with open(self.file_path, "r") as f:
            content = f.read()
        logging.info(f"File loaded, length: {len(content)} characters")

        logging.info("Encoding content with tokeniser...")
        tokens = self.tokeniser.encode(content)
        logging.info(f"Content encoded into {len(tokens)} tokens")

        examples = []
        for i in range(0, len(tokens), context_length + 1):
            substring = tokens[i: i + context_length + 1]
            if len(substring) < context_length:
                logging.debug(f"Skipping short substring at index {i}, length {len(substring)}")
                continue
            examples.append(substring)
        logging.info(f"Generated {len(examples)} examples of length {context_length}")

        random.shuffle(examples)
        logging.info("Shuffled examples")

        n_train = int(ratio * len(examples))
        logging.info(f"Train/val split: {n_train}/{len(examples) - n_train}")
        return examples[:n_train], examples[n_train:]


def log_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return f"GPU memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB"
    else:
        return ""


def train(model: Transformer, device="cuda" if torch.cuda.is_available() else "cpu"):
    logging.info(f"Starting training on device: {device}")
    model.to(device)

    epochs = 10
    batch_size = 64
    learning_rate = 1e-5

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=10e-8)
    logging.info(f"Optimizer initialized: Adam with lr={learning_rate}")

    dataloader = TextDataLoader("shakespeare.txt", model.tokeniser)
    train_data, val_data = dataloader.get_train_val_splits(model.context_length, 0.9)
    logging.info(f"Training data: {len(train_data)} examples, Validation data: {len(val_data)} examples")
    losses = []
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs} starting")
        for i in range(0, len(train_data), batch_size):
            batch_examples = train_data[i: i + batch_size]
            batch = torch.tensor(batch_examples, dtype=torch.long, device=device)
            x = batch[:,:-1]
            y = batch[:,1:]

            optimizer.zero_grad()
            logits = model(x)
            B, T, D = logits.shape
            logits = logits.view(B*T, D)
            y = y.reshape(B*T)
            loss = F.cross_entropy(logits, y.long())
            losses.append(loss.item())

            if torch.isnan(loss):
                logging.error(f"NaN loss detected at epoch {epoch+1}, batch {i // batch_size + 1}")
                break

            loss.backward()
            # Log gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            optimizer.step()
            torch.cuda.synchronize()
            logging.debug(f"Optimizer step completed for batch {i // batch_size + 1}")
            logging.debug("")
            logging.info(
                f"Epoch {epoch+1}/{epochs} | Batch {i//batch_size}/{len(train_data)//batch_size} | "
                f"Loss: {loss.item():.4f} | Logits std: {logits.std().item():.4f} | "
                f"Grad norm: {total_norm:.4f} | {log_gpu_memory()}"
            )
            log_gpu_memory()

    torch.save(model.state_dict(), "checkpoint.pth")
    logging.info("Model checkpoint saved to 'checkpoint.pth'")
    plt.plot(losses)
    plt.show()

if __name__ == "__main__":
    logging.info("Loading tokeniser...")
    tokeniser = BPETokeniser()
    tokeniser.load("tokeniser_vocab.pkl")
    logging.info("Tokeniser loaded")

    # model = Transformer(
    #     embedding_dims=768,
    #     n_blocks=12,
    #     attention_dims=768,
    #     n_attn_heads=12,
    #     context_length=512,
    #     vocab_length=tokeniser.vocab_size,
    #     tokeniser=tokeniser
    # )
    model = Transformer(
        embedding_dims=256,       # Smaller embedding, still expressive
        n_blocks=4,               # 4 transformer blocks instead of 12
        attention_dims=256,       # Matches embedding dims
        n_attn_heads=4,           # 4 attention heads
        context_length=256,       # Can process 256 tokens at a time
        vocab_length=tokeniser.vocab_size,
        tokeniser=tokeniser
    )
    model_state_dict = torch.load("checkpoint.pth")
    model.load_state_dict(model_state_dict)

    model.train()

    train(model)
    logging.info("Training complete")
