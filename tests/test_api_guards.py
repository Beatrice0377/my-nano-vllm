"""CPU-only unit tests for the public API fail-fast guards."""

import pytest

from nanovllm.sampling_params import SamplingParams


def test_max_tokens_zero_rejected():
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        SamplingParams(max_tokens=0)


def test_max_tokens_negative_rejected():
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        SamplingParams(max_tokens=-1)


def test_max_tokens_one_accepted():
    assert SamplingParams(max_tokens=1).max_tokens == 1


def test_generate_list_mismatch_rejected():
    # The length check runs in LLMEngine.generate before any request is added
    # or any model work happens, so an engine instance can be created without
    # __init__ (no model snapshot needed) and the guard still fires.
    from nanovllm.engine.llm_engine import LLMEngine

    engine = object.__new__(LLMEngine)
    with pytest.raises(ValueError, match="sampling_params list length"):
        engine.generate(
            ["a", "b", "c"],
            [SamplingParams(), SamplingParams()],
            use_tqdm=False,
        )
