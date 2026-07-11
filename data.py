"""Load the training text and serve random batches of (context, next-char)."""
import torch


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Sample `batch_size` random chunks; y is x shifted one character right."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
