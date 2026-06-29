"""Tests for the byte-level BPE tokenizer."""

from __future__ import annotations

import pytest

from slm.tokenizer import BPETokenizer

CORPUS = "the quick brown fox. the lazy dog. hello world! 123 456 the the the\n" * 50


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train([CORPUS], vocab_size=400, special_tokens=["<|endoftext|>"])


@pytest.mark.parametrize(
    "text",
    [
        "the quick brown fox",
        "hello world! 123",
        "",
        "你好，世界",  # multi-byte CJK  # noqa: RUF001 - intentional fullwidth comma
        "emoji 🌍🚀 test",
        "  leading and  internal   spaces ",
        "MixedCASE_with-symbols!@#$%",
    ],
)
def test_roundtrip(tok: BPETokenizer, text: str):
    assert tok.decode(tok.encode(text)) == text


def test_vocab_includes_all_bytes(tok: BPETokenizer):
    # Every one of the 256 base byte tokens must exist for losslessness.
    assert sum(1 for v in tok.vocab.values() if v < 256) == 256


def test_special_token_atomic(tok: BPETokenizer):
    ids = tok.encode("a<|endoftext|>b")
    assert tok.eot_id in ids
    assert ids.count(tok.eot_id) == 1
    assert tok.decode(ids) == "a<|endoftext|>b"


def test_special_token_disabled_is_bytes(tok: BPETokenizer):
    ids = tok.encode("<|endoftext|>", allowed_special=False)
    assert tok.eot_id not in ids
    assert tok.decode(ids) == "<|endoftext|>"


def test_id_to_bytes_roundtrip(tok: BPETokenizer):
    ids = tok.encode("hello")
    joined = b"".join(tok.id_to_bytes(i) for i in ids)
    assert joined.decode("utf-8") == "hello"
    assert tok.id_to_bytes(tok.eot_id) == b"<|endoftext|>"


def test_save_load_roundtrip(tok: BPETokenizer, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.merges == tok.merges
    assert loaded.encode("the quick fox") == tok.encode("the quick fox")


def test_training_is_deterministic():
    a = BPETokenizer.train([CORPUS], vocab_size=400)
    b = BPETokenizer.train([CORPUS], vocab_size=400)
    assert a.merges == b.merges


def test_vocab_size_too_small_raises():
    with pytest.raises(ValueError, match="too small"):
        BPETokenizer.train([CORPUS], vocab_size=100, special_tokens=["<|endoftext|>"])


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        BPETokenizer.load(tmp_path / "absent.json")


def test_load_bad_version(tmp_path):
    import json

    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {"version": 999, "vocab": {}, "merges": [], "special_tokens": {}, "pattern": "x"}
        )
    )
    with pytest.raises(ValueError, match="version"):
        BPETokenizer.load(path)


def test_token_to_id(tok: BPETokenizer):
    assert tok.token_to_id("<|endoftext|>") == tok.eot_id
    assert tok.token_to_id("definitely-not-real") is None


def test_eot_id_absent_when_no_special():
    plain = BPETokenizer.train([CORPUS], vocab_size=300, special_tokens=[])
    assert plain.eot_id is None


def test_single_character_encode(tok: BPETokenizer):
    # Single-byte words hit the short-circuit branch in the BPE merge routine.
    assert tok.decode(tok.encode("a")) == "a"


def test_decode_ignores_unknown_ids(tok: BPETokenizer):
    # Out-of-range ids are skipped rather than crashing the decoder.
    assert tok.decode([10_000_000]) == ""


def test_id_to_bytes_unknown_id(tok: BPETokenizer):
    assert tok.id_to_bytes(10_000_000) == b""
