"""CPU-only tests for the Qwen3 configuration support guard.

These tests construct config objects directly and never load a model, so they
run without a GPU.
"""

import pytest
from transformers import Qwen3Config

from nanovllm.config import validate_supported_config


def make_qwen3_config(**overrides) -> Qwen3Config:
    """Qwen3-0.6B-like config; individual fields may be overridden."""
    base = dict(
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        torch_dtype="bfloat16",
        # Qwen3Config defaults sliding_window to 4096, but the validated
        # Qwen3-0.6B snapshot has sliding_window=None; the engine only
        # supports the non-sliding configuration.
        sliding_window=None,
    )
    base.update(overrides)
    return Qwen3Config(**base)


def test_known_supported_qwen3_config_accepted():
    validate_supported_config(make_qwen3_config())


def test_non_qwen3_rejected():
    from transformers import LlamaConfig

    with pytest.raises(ValueError, match="only for Qwen3"):
        validate_supported_config(LlamaConfig())


def test_attention_bias_true_rejected():
    with pytest.raises(ValueError, match="attention_bias=True"):
        validate_supported_config(make_qwen3_config(attention_bias=True))


def test_sliding_window_enabled_rejected():
    with pytest.raises(ValueError, match="sliding-window"):
        validate_supported_config(make_qwen3_config(sliding_window=4096))


def test_rope_scaling_non_none_rejected():
    with pytest.raises(ValueError, match="RoPE scaling"):
        validate_supported_config(
            make_qwen3_config(rope_scaling={"type": "yarn", "factor": 2.0})
        )


def test_absent_optional_fields_treated_as_unsupported_off():
    """Fields the engine does not read must default to the safe value even if
    a config version omits them entirely."""
    cfg = make_qwen3_config()
    delattr(cfg, "attention_bias")
    validate_supported_config(cfg)  # getattr default False -> accepted
