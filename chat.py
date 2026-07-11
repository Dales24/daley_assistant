"""Quinn Bot — chat with your trained model.

    python chat.py

Loads the checkpoint written by train.py and talks turn by turn. Since this is a
character-level model trained from scratch, replies are only as good as your
data and training — expect gibberish until you've trained on enough text.
"""
import os

import torch

from config import Config
from model import GPT
from tokenizer import CharTokenizer

USER_TAG = "User:"
BOT_TAG = "Quinn:"
MAX_NEW_TOKENS = 200


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(out_dir: str, device: str):
    ckpt_path = os.path.join(out_dir, "model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path} — run train.py first.")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = CharTokenizer.load(os.path.join(out_dir, "tokenizer.json"))
    return model, tokenizer, cfg


def reply(model, tokenizer, cfg, device, prompt: str) -> str:
    """Generate one Quinn turn and stop at the next 'User:'."""
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, MAX_NEW_TOKENS, temperature=0.8, top_k=40)
    text = tokenizer.decode(out[0].tolist())[len(prompt):]
    return text.split(USER_TAG)[0].strip()


def main():
    device = get_device()
    model, tokenizer, cfg = load_model(Config().out_dir, device)

    print("Quinn Bot — type your message, or 'quit' to exit.\n")
    transcript = ""
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"quit", "exit"}:
            break

        transcript += f"{USER_TAG} {user}\n{BOT_TAG}"
        answer = reply(model, tokenizer, cfg, device, transcript)
        print(f"Quinn: {answer}\n")
        transcript += f" {answer}\n"


if __name__ == "__main__":
    main()
