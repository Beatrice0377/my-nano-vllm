"""Phase 5 end-to-end test: mixed scheduler on the real Qwen3-0.6B model.

Deterministic via a test-only greedy sampler (SamplingParams forbids
temperature=0 by design; we do not modify the production sampler).

Requires a GPU with the Qwen3-0.6B-local snapshot.
"""

import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pytest
import torch

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

MODEL = "/home/beatrice/huggingface/Qwen3-0.6B-local"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isdir(MODEL),
    reason="requires GPU + Qwen3-0.6B-local snapshot",
)


class GreedySampler:
    def __call__(self, logits, temperatures):
        return logits.argmax(dim=-1)


@pytest.fixture(scope="module")
def llm():
    engine = LLM(MODEL, enforce_eager=True)
    engine.model_runner.sampler = GreedySampler()
    yield engine


def drive(llm, n_steps=200):
    outputs = []
    n_mixed = 0
    for _ in range(n_steps):
        if llm.is_finished():
            break
        finished, n_prefill, n_decode = llm.step()
        if n_prefill > 0 and n_decode > 0:
            n_mixed += 1
        outputs.extend(finished)
    return n_mixed, outputs


def test_pure_decode(llm):
    llm.add_request(
        "The capital of France is", SamplingParams(max_tokens=8, ignore_eos=True)
    )
    _, outs = drive(llm)
    assert outs, "no output produced"
    seq_id, completion = outs[0]
    assert len(completion) == 8
    text = llm.tokenizer.decode(completion)
    assert "Paris" in text, f"unexpected decode: {text!r}"


def test_pure_prefill(llm):
    llm.add_request(
        "The capital of Japan is", SamplingParams(max_tokens=8, ignore_eos=True)
    )
    _, outs = drive(llm)
    assert outs
    text = llm.tokenizer.decode(outs[0][1])
    assert "Tokyo" in text, f"unexpected prefill: {text!r}"


def test_mixed_batch(llm):
    # two decodes running, then inject a prefill mid-flight -> mixed step(s)
    llm.add_request(
        "The capital of Germany is", SamplingParams(max_tokens=20, ignore_eos=True)
    )
    llm.add_request(
        "The capital of Italy is", SamplingParams(max_tokens=20, ignore_eos=True)
    )
    for _ in range(3):
        if llm.is_finished():
            break
        llm.step()
    llm.add_request(
        "The capital of Spain is", SamplingParams(max_tokens=10, ignore_eos=True)
    )
    n_mixed, outs = drive(llm)
    assert n_mixed >= 1, (
        "expected at least one mixed step (decode + prefill co-resident)"
    )
    assert len(outs) == 3, f"expected 3 completions, got {len(outs)}"
    texts = [llm.tokenizer.decode(comp) for _, comp in outs]
    assert any("Berlin" in t for t in texts)
    assert any("Rome" in t for t in texts)
    assert any("Madrid" in t for t in texts)
