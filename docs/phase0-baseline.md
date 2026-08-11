# Phase 0 baseline notes

This document records the baseline architecture and the measurement boundary
before any inference feature is changed.

## Provenance

The repository is derived from `GeeeekExplorer/nano-vLLM`, imported at upstream
commit `bb823b3e` before the local history was reinitialized. The original MIT
license and copyright notice remain in `LICENSE`.

## Data flow

```text
LLMEngine
  -> Scheduler
       -> Sequence / BlockManager (waiting, running, KV block table, prefix hash)
  -> ModelRunner
       -> CPU lists -> pinned CPU tensors -> CUDA tensors
       -> Qwen3ForCausalLM
            -> embedding -> decoder layers
                 -> Qwen3Attention -> Attention
                      -> store KV -> FlashAttention / paged KV attention
            -> ParallelLMHead
  -> Sampler
  -> Scheduler.postprocess
       -> hash completed KV blocks, append sampled token, finish/deallocate
```

### Prefill

`Scheduler.schedule()` first drains `waiting`. For a sequence without a block
table it calls `BlockManager.can_allocate()`, which walks the hash table over
the reusable full prefix blocks and returns the number of cached blocks. The
new blocks are allocated once with `allocate()`. For a sequence already being
chunk-prefilled, scheduling starts at `seq.num_cached_tokens` and limits the
new tokens by the shared `max_num_batched_tokens` budget. The current scheduler
only permits chunking for the first waiting sequence and returns a batch-level
`is_prefill=True`.

`ModelRunner.prepare_prefill()` concatenates the new token ranges, positions,
ragged query/key cumulative lengths, and KV `slot_mapping`. If cached K/V is
used, it also prepares `block_tables`. `set_context(True, ...)` publishes these
values to the model. Each attention layer writes new K/V to its layer cache and
calls `flash_attn_varlen_func`; the LM head selects the last query position of
each sequence using `cu_seqlens_q`. The sampler produces one token per sequence.
`postprocess()` hashes newly completed blocks, advances
`num_cached_tokens`, and appends the sampled token only after the full prompt
has been prefetched. A partial chunk therefore produces no completion token.

### Decode

When no waiting sequence can be prefetched, the scheduler takes running
sequences, ensures a new KV block is available when a block boundary is
crossed, sets one scheduled token per sequence, and returns
`is_prefill=False`. `prepare_decode()` sends each sequence's last token and
position, its write slot, context length, and padded block table. The context
selects `flash_attn_with_kvcache`; the LM head consumes every row because each
row is one decode query. The sampler appends one token per sequence, and a
finished sequence releases its blocks.

### `is_prefill` and CUDA Graph

The current batch-level flag is propagated as follows:

```text
Scheduler.schedule() -> LLMEngine.step()
  -> ModelRunner.call("run", seqs, is_prefill)
  -> ModelRunner.run(..., is_prefill)
       -> prepare_prefill / prepare_decode
       -> set_context(is_prefill, ...)
       -> run_model(..., is_prefill)
            -> Attention.context.is_prefill
            -> ParallelLMHead.context.is_prefill
```

`ModelRunner` warms the model, allocates one flat KV tensor per rank, and—when
`enforce_eager=False`—captures decode-only CUDA graphs for a fixed set of batch
sizes. Decode metadata is copied into persistent graph buffers before replay.
Prefill always runs the eager model path. A future mixed batch cannot reuse this
single boolean or the current decode graph contract unchanged.

## Current benchmark coverage

The original imported `bench.py` measured only total requested output tokens,
wall time, and output tokens/s. Phase 1 replaces that harness with three fixed
seed workloads and a machine-readable JSON result. The complete audit and
metric definitions are in [`phase1-benchmark.md`](phase1-benchmark.md).

The Phase 1 harness keeps the same direct offline engine and adds synchronized
phase rates, end-to-end throughput, request-level TTFT, real token-level TPOT,
block-level prefix-cache lookup counters, scheduled/executed prefill tokens,
and input-preparation call timing. The only engine change is an optional
benchmark observer in `BlockManager.can_allocate()`; it is inactive by default.

## Baseline run status in the Codex execution sandbox

On 2026-08-10 the source passed `python3 -m compileall -q .`, but the actual
benchmark could not start:

- Python: 3.12.3 (`/usr/bin/python3`)
- PyTorch: 2.13.0+cu132; `torch.cuda.is_available()` was false
- Triton: 3.7.1
- `tqdm`, `transformers`, `flash-attn`, `xxhash`, and `safetensors` were not installed
- `nvidia-smi` could not access NVML because GPU access was blocked
- `/home/beatrice/huggingface/Qwen3-0.6B` was a cache root with tokenizer links,
  not yet a confirmed local model snapshot containing config and weights

Consequently this sandbox has no GPU, throughput, or peak-memory numbers. This
does not establish that the user's WSL/GPU host lacks a GPU. Run `bench.py`
there after installing compatible dependencies and preparing the model snapshot.

## Future dependency map

| Planned feature | Primary files | Reason |
| --- | --- | --- |
| Fused add + RMSNorm | `nanovllm/layers/layernorm.py`, new Triton kernel module, model call sites | Replace the compiled PyTorch residual-add/normalization path while preserving residual semantics. |
| Fused SiLU × gate | `nanovllm/layers/activation.py`, Qwen3 MLP call site | Fuse the split, activation, and multiply without changing the packed projection layout. |
| PagedAttention decode | `nanovllm/layers/attention.py`, context metadata, benchmark correctness path | Replace or dispatch the decode attention kernel using existing block tables and slot mappings. |
| FlashAttention prefill | `nanovllm/layers/attention.py`, `model_runner.py` prefill metadata | Keep ragged prefill and prefix-cache semantics while changing the kernel dispatch. |
| Unified mixed scheduling | `scheduler.py`, `llm_engine.py`, `model_runner.py`, `utils/context.py`, `attention.py`, `embed_head.py` | Replace the batch-level phase flag with per-row metadata and dispatch both attention modes in one model call. |
| Prefix-cache affinity | `scheduler.py`, small BlockManager inspection API, benchmark scenario | Rank only a bounded waiting-window candidate set by reusable prefix blocks, with aging. |
| Persistent decode metadata | `sequence.py`, `block_manager.py`, `model_runner.py`, possibly a small gather kernel | Give stable requests/state slots and avoid rebuilding CPU lists and H2D metadata every decode step. |
| Ablation metrics | `bench.py` and benchmark utilities | Keep each optimization comparable to the unchanged direct-engine baseline. |

## Risks to resolve before implementation

- Mixed prefill/decode breaks the current `Context.is_prefill` branch in both
  attention and LM head. Per-row output selection and two attention metadata
  layouts are required.
- CUDA Graph capture assumes decode-only shapes, fixed graph batch sizes, and
  graph-compatible metadata buffers. A mixed path may need separate graphs or
  an eager fallback.
- Chunked prefill must not hash or expose a partially filled shared block as a
  reusable prefix. Cache ownership and block boundaries need explicit tests.
- The LM head currently selects the last row of each ragged prefill sequence,
  but all rows of decode are logits rows. Mixed batches need an explicit row
  mapping rather than a single `is_prefill` branch.
- Triton attention integration must preserve the existing slot mapping,
  block-table layout, GQA head mapping, causal mask, and prefix-cache behavior.
