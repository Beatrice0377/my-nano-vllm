<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM Enhanced

A lightweight research inference engine derived from
[GeeeekExplorer/nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm).

This repository is **not a from-scratch implementation**. The initial code was
imported from the upstream `bb823b3e` baseline commit and the local Git history
was then reinitialized. The original MIT license and copyright attribution are
retained in [`LICENSE`](LICENSE).

## Project scope

The project keeps nano-vLLM's small, readable style while adding a small number
of measurable GPU-kernel and serving-system improvements. It is an educational
and research codebase, not a production serving framework. The current model
path is the Qwen3 path already supported by the baseline.

The current checked-in implementation is still the imported baseline. In
particular, the planned Triton kernels and mixed scheduling are not implemented
yet and must not be inferred from the roadmap below.

## Roadmap

1. Triton fused add + RMSNorm, and SiLU × gate.
2. Triton PagedAttention decode and FlashAttention prefill integration.
3. Unified mixed scheduling: decode-first, a shared token budget, chunked
   prefill, and decode/prefill requests in one model batch.
4. Prefix-cache-affinity scheduling with a bounded candidate window and aging.
5. Persistent decode metadata to reduce repeated Python/CPU-tensor/H2D setup.
6. Correctness and performance ablations covering kernel latency, TTFT, TPOT,
   P50/P95, throughput, prefix-cache hits, prefill work, and input preparation.

Non-goals for this phase are speculative decoding, quantization, LoRA, MoE,
CPU KV offload, distributed serving, an HTTP server, async scheduling, complex
KV eviction, arbitrary model support, and training kernels.

## Installation

Install the dependencies in `pyproject.toml` in a CUDA-enabled environment. The
FlashAttention dependency is platform and CUDA-version dependent, so the exact
installation command is intentionally left to the target environment.

```bash
pip install -e .
```

## Quick start

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-0.6B", enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(
    ["Hello, Nano-vLLM."],
    SamplingParams(temperature=0.6, max_tokens=256),
)
print(outputs[0]["text"])
```

See [`example.py`](example.py) for a chat-template example. The model argument
must point to a local Hugging Face model directory containing its config and
weight files, not merely the root of a cache layout.

## Baseline benchmark

[`bench.py`](bench.py) is a small offline harness over the existing
`LLMEngine.step()` loop; it does not add a server or alter scheduling. It has
three fixed-seed workloads: `balanced`, `decode-heavy`, and `prefix-sharing`.
Each successful run writes a machine-readable JSON result under
`benchmarks/results/` with synchronized step rates, token-level TTFT/TPOT,
request latency P50/P95, block-level prefix-cache counters,
input-preparation call time, and peak allocated GPU memory.

```bash
python3 bench.py --workload balanced --model /path/to/Qwen3-0.6B
python3 bench.py --workload decode-heavy --model /path/to/Qwen3-0.6B
python3 bench.py --workload prefix-sharing --model /path/to/Qwen3-0.6B
```

See [`docs/phase1-benchmark.md`](docs/phase1-benchmark.md) for metric
definitions and [`docs/environment.md`](docs/environment.md) for the pinned
validation target. No result file is written when CUDA is unavailable or the
model directory is incomplete. Results from the old upstream README are not
treated as this project's baseline until reproduced in the same environment.

## License and attribution

This project retains the upstream MIT license and attribution. See
[`LICENSE`](LICENSE) for the complete text.
