# daley_assistant — Quinn Bot

A small **language model built from scratch** (a character-level GPT in PyTorch)
that you train yourself, with **Quinn Bot** as the chat interface. No API, no
pretrained weights — the model learns entirely from the text you give it.

This is a learning/starter scaffold: a clean, minimal transformer you can train
on a laptop, then grow.

## Architecture (and why)

- **Decoder-only transformer (GPT-style)** — the right family for *generating*
  replies. (Encoder models like BERT are for *understanding* text, not writing
  it; RNN/LSTMs also work but are older and harder to scale.)
- **Learned token embeddings**, not one-hot vectors — an embedding is the
  efficient, trainable version of the same idea.
- **Character-level tokenizer** — the simplest thing that works. Swap in a
  subword (BPE) tokenizer later; only `tokenizer.py` changes.

## Files

```
daley_assistant/
├─ config.py       hyperparameters (model size, training steps)
├─ tokenizer.py    character-level encode/decode
├─ model.py        the GPT (attention + MLP blocks)
├─ data.py         load text, sample training batches
├─ train.py        training loop → writes checkpoints/
├─ chat.py         Quinn Bot — chat with the trained model
└─ data/sample.txt starter conversation data (User: / Quinn:)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

This trains on `data/sample.txt` and writes `checkpoints/model.pt` +
`checkpoints/tokenizer.json`. On CPU the tiny default config trains in a few
minutes; it uses CUDA or Apple MPS automatically if available.

## Chat

```bash
python chat.py
```

```
Quinn Bot — type your message, or 'quit' to exit.

You: help me plan my day
Quinn: ...
```

> ⚠️ Trained from scratch on a **tiny** sample, Quinn will mostly produce
> gibberish. That's expected — the point is the pipeline. It gets good only with
> much more data and training.

## Train it on your own data

1. Replace/extend `data/sample.txt` with your text. Keep the turn format so Quinn
   learns to reply:
   ```
   User: <a message>
   Quinn: <the reply>
   ```
   More data is the single biggest lever — aim for a lot of it.
2. Scale the model in `config.py` (`n_layer`, `n_embd`, `block_size`) as your
   dataset grows, and raise `max_steps`.
3. Re-run `python train.py`, then `python chat.py`.

## Next steps

- [ ] Subword (BPE) tokenizer for real vocabulary.
- [ ] Bigger model + more data; train on a GPU.
- [ ] Learning-rate schedule + gradient clipping.
- [ ] Separate pretraining (raw text) from fine-tuning (chat pairs).
