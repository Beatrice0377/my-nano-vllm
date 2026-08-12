"""Phase 5 end-to-end test: mixed scheduler on the real Qwen3-0.6B model.

Deterministic via a test-only greedy sampler (SamplingParams forbids
temperature=0 by design; we do not modify the production sampler).

Requires a GPU with a Qwen3-0.6B snapshot (set NANOVLLM_MODEL to the local
snapshot path).
"""

import sys
import os
import atexit

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pytest
import torch

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

MODEL = os.environ.get("NANOVLLM_MODEL")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not (MODEL and os.path.isdir(MODEL)),
    reason="requires GPU + Qwen3-0.6B snapshot (NANOVLLM_MODEL)",
)


class GreedySampler:
    def __call__(self, logits, temperatures):
        return logits.argmax(dim=-1)


# module-scope engine holder: the 3 fixture-based tests share one engine; the
# parity test (which needs its own two engines) releases it first, because two
# ModelRunners cannot coexist in one process (each calls dist.init_process_group)
_MODULE_ENGINE = [None]


@pytest.fixture(scope="module")
def llm():
    engine = LLM(MODEL, enforce_eager=True)
    engine.model_runner.sampler = GreedySampler()
    _MODULE_ENGINE[0] = engine
    yield engine
    if _MODULE_ENGINE[0] is engine:
        _release_engine(engine)


def _release_engine(engine):
    atexit.unregister(engine.exit)
    engine.exit()
    if _MODULE_ENGINE[0] is engine:
        _MODULE_ENGINE[0] = None
    torch.cuda.empty_cache()


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


class _GraphReplayObserver:
    """Test-only counter for CUDA-graph replay calls (run_model is otherwise
    not observable without instrumenting the runtime)."""

    def __init__(self, model_runner):
        self.count = 0
        self.mixed_seen = False
        self._orig_run_model = model_runner.run_model
        self._mr = model_runner

    def install(self):
        self._mr.run_model = self._wrapped
        return self

    def _wrapped(self, input_ids, positions, is_pure_decode):
        if (
            not self._mr.enforce_eager
            and is_pure_decode
            and input_ids.size(0) <= 512
            and self._mr.graph_vars is not None
        ):
            self.count += 1
        return self._orig_run_model(input_ids, positions, is_pure_decode)

    def uninstall(self):
        self._mr.run_model = self._orig_run_model


def _run_scripted(engine, n_steps=300):
    """Two decodes, inject a prefill mid-flight (forces >=1 mixed step), then
    keep decoding (re-enters pure decode once the prefill completes)."""
    engine.add_request(
        "The capital of Germany is", SamplingParams(max_tokens=20, ignore_eos=True)
    )
    engine.add_request(
        "The capital of Italy is", SamplingParams(max_tokens=20, ignore_eos=True)
    )
    for _ in range(3):
        if engine.is_finished():
            break
        engine.step()
    engine.add_request(
        "The capital of Spain is", SamplingParams(max_tokens=10, ignore_eos=True)
    )
    outs = []
    n_mixed = 0
    for _ in range(n_steps):
        if engine.is_finished():
            break
        finished, n_prefill, n_decode = engine.step()
        if n_prefill > 0 and n_decode > 0:
            n_mixed += 1
        outs.extend(finished)
    return n_mixed, sorted(comp for _, comp in outs)


def test_mixed_to_cuda_graph_parity():
    """A batch that goes through a mixed (eager) step must re-enter the pure
    decode CUDA-graph path and produce byte-identical completion tokens.

    Two engines cannot coexist in one process (each ModelRunner calls
    dist.init_process_group), so the module-scoped engine is released first,
    then the eager engine is run and torn down (destroying the process group)
    before the CUDA-graph engine is built, with atexit unregistered so teardown
    happens exactly once.
    """
    if _MODULE_ENGINE[0] is not None:
        _release_engine(_MODULE_ENGINE[0])

    eager_comps = None
    eager = LLM(MODEL, enforce_eager=True)
    atexit.unregister(eager.exit)
    try:
        eager.model_runner.sampler = GreedySampler()
        _, eager_comps = _run_scripted(eager)
    finally:
        eager.exit()
        # the first engine's model is still resident in the caching allocator;
        # release it so the second engine can size its own KV pool
        torch.cuda.empty_cache()

    graph = LLM(MODEL, enforce_eager=False)
    atexit.unregister(graph.exit)
    try:
        graph.model_runner.sampler = GreedySampler()
        observer = _GraphReplayObserver(graph.model_runner).install()
        try:
            n_mixed, graph_comps = _run_scripted(graph)
        finally:
            observer.uninstall()
    finally:
        graph.exit()

    assert n_mixed >= 1, "expected at least one mixed step in the graph engine"
    assert observer.count >= 1, (
        "expected the graph engine to re-enter pure-decode CUDA-graph replay "
        "after the mixed steps"
    )
    assert len(eager_comps) == len(graph_comps) == 3
    assert graph_comps == eager_comps, (
        "completion token IDs differ between eager and CUDA-graph engines:\n"
        f"eager: {eager_comps}\ngraph: {graph_comps}"
    )
