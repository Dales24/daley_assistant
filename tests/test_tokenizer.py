"""Tests for the character tokenizer (pure Python — no torch needed)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenizer import CharTokenizer  # noqa: E402


def test_roundtrip():
    tok = CharTokenizer.from_text("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"


def test_vocab_is_unique_sorted_chars():
    tok = CharTokenizer.from_text("bca")
    assert tok.vocab_size == 3
    assert tok.chars == ["a", "b", "c"]


def test_encode_drops_unknown_characters():
    tok = CharTokenizer.from_text("abc")
    assert tok.encode("abcz") == tok.encode("abc")   # 'z' unseen -> dropped


def test_save_and_load(tmp_path):
    tok = CharTokenizer.from_text("hello")
    path = tmp_path / "tok.json"
    tok.save(str(path))
    restored = CharTokenizer.load(str(path))
    assert restored.chars == tok.chars
    assert restored.encode("hell") == tok.encode("hell")
