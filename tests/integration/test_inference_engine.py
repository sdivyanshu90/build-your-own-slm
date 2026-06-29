"""Integration tests for the inference engine."""

from __future__ import annotations

import pytest

from slm.generation.sampler import GenerationConfig
from slm.inference.engine import InferenceEngine, _incremental_decode

pytestmark = pytest.mark.integration


def test_metadata(inference_engine: InferenceEngine):
    meta = inference_engine.metadata
    assert meta.num_parameters > 0
    assert meta.block_size == 32
    assert meta.device == "cpu"
    assert meta.n_layer == 2


def test_generate_returns_result(inference_engine: InferenceEngine):
    result = inference_engine.generate("the quick", GenerationConfig(max_new_tokens=12, seed=1))
    assert isinstance(result.text, str)
    assert 1 <= result.completion_tokens <= 12
    assert result.finish_reason in {"length", "stop"}
    assert result.prompt_tokens >= 1


def test_generate_is_reproducible_with_seed(inference_engine: InferenceEngine):
    cfg = GenerationConfig(max_new_tokens=10, temperature=0.9, seed=7)
    a = inference_engine.generate("the dog", cfg)
    b = inference_engine.generate("the dog", cfg)
    assert a.text == b.text


def test_stream_matches_buffered(inference_engine: InferenceEngine):
    cfg = GenerationConfig(max_new_tokens=10, temperature=0.0, seed=3)
    streamed = "".join(inference_engine.stream("the cat", cfg))
    buffered = inference_engine.generate("the cat", cfg).text
    assert streamed == buffered


def test_empty_prompt_uses_eot_seed(inference_engine: InferenceEngine):
    result = inference_engine.generate("", GenerationConfig(max_new_tokens=5, seed=1))
    assert result.prompt_tokens == 1  # seeded with the eot token


def test_prompt_too_long_raises(trained_model_dir):
    engine = InferenceEngine.from_pretrained(trained_model_dir, device="cpu", max_prompt_tokens=2)
    with pytest.raises(ValueError, match="exceeding the limit"):
        engine.generate("this prompt is clearly longer than two tokens", GenerationConfig())


def test_max_new_tokens_clamped(trained_model_dir):
    engine = InferenceEngine.from_pretrained(
        trained_model_dir, device="cpu", max_new_tokens_limit=4
    )
    result = engine.generate("the", GenerationConfig(max_new_tokens=1000, seed=1))
    assert result.completion_tokens <= 4


def test_stop_sequence_truncation(inference_engine: InferenceEngine):
    text, hit = InferenceEngine._truncate_at_stop("hello world stop here", ["stop"])
    assert hit is True
    assert text == "hello world "


@pytest.mark.parametrize(
    "buffer,expected_text,expected_consumed",
    [
        (bytearray(b"abc"), "abc", 3),
        (bytearray("é".encode()[:1]), "", 0),  # incomplete 2-byte sequence
        (bytearray("世".encode()), "世", 3),
    ],
)
def test_incremental_decode(buffer, expected_text, expected_consumed):
    text, consumed = _incremental_decode(buffer)
    assert text == expected_text
    assert consumed == expected_consumed


def test_incremental_decode_invalid_start_byte():
    # A lone continuation byte cannot start a sequence -> replacement, consume 1.
    text, consumed = _incremental_decode(bytearray(b"\x80abc"))
    assert consumed == 1
    assert text == "�"
