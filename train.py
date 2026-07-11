"""Train Quinn from scratch on your data, then save a checkpoint.

    python train.py

Point `Config.data_path` at your own text (formatted like data/sample.txt) and
bump the model size once you have enough of it.
"""
import os

import torch

from config import Config
from data import get_batch, load_text
from model import GPT
from tokenizer import CharTokenizer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, data, cfg, device, batches=20):
    model.eval()
    losses = torch.zeros(batches)
    for i in range(batches):
        xb, yb = get_batch(data, cfg.block_size, cfg.batch_size, device)
        _, loss = model(xb, yb)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = get_device()
    print(f"training on {device}")

    text = load_text(cfg.data_path)
    tokenizer = CharTokenizer.from_text(text)
    cfg.vocab_size = tokenizer.vocab_size

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(ids))
    train_data, val_data = ids[:split], ids[split:]

    model = GPT(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"model has {params / 1e6:.2f}M parameters, vocab {cfg.vocab_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    for step in range(cfg.max_steps):
        xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            val = estimate_loss(model, val_data, cfg, device)
            print(f"step {step:5d} | train {loss.item():.3f} | val {val:.3f}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    tokenizer.save(os.path.join(cfg.out_dir, "tokenizer.json"))
    torch.save(
        {"model": model.state_dict(), "config": cfg.__dict__},
        os.path.join(cfg.out_dir, "model.pt"),
    )
    print(f"saved checkpoint to {cfg.out_dir}/")


if __name__ == "__main__":
    main()
