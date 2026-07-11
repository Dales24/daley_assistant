"""Hyperparameters for the model and training. Edit these to scale up."""
from dataclasses import dataclass


@dataclass
class Config:
    # data
    data_path: str = "data/sample.txt"
    out_dir: str = "checkpoints"

    # model size (start small; grow n_layer/n_embd once you have more data)
    block_size: int = 128     # context length in characters
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    # training
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 2000
    eval_interval: int = 200
    seed: int = 1337

    # filled in at train time from the tokenizer
    vocab_size: int = 0
