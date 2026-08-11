# Phase 3 — Triton PagedAttention Decode (v1)

Status: **Accepted — standalone implementation, no runtime integration.**

Standalone deliverable in `nanovllm/layers/attention.py`:

* `_paged_attention_kernel` (Triton, forward-only decode attention)
* `paged_attention(...)` wrapper — not wired into `Attention.forward()`

Current decode runtime path (`flash_attn_with_kvcache` in `Attention.forward`) is unchanged.

## 1. KV cache layout

Per layer (assigned from the flat pool in `ModelRunner.allocate_kv_cache`):

```
k_cache / v_cache : [num_blocks, BLOCK_SIZE=256, num_kv_heads, head_dim=128]
```

Token-level addressing via `slot_mapping`: `slot = physical_block * 256 + offset_in_block`.
`store_kvcache` scatters the current step's K/V into these slots with a one-program-per-token
Triton kernel (`slot == -1` entries skipped).

## 2. Logical → physical addressing

`block_tables` is `[batch, max_blocks]` int32, padded with -1 to the current batch's widest
table (eager) or the CUDA-graph static width (graph path).

The kernel derives the needed block count from the context length and never assumes a fixed
table width:

```
num_blocks = (context_len + 256 - 1) // 256
for lb in range(num_blocks):
    phys = block_tables[seq, lb]            # logical -> physical
    ...                                     # one physical block may span multiple tiles
```

Stale physical content that is not reachable via `block_table`/`context_len` never participates
in attention.

## 3. GQA mapping

Not hard-coded to Qwen3 (16/8):

```
GROUP_SIZE = NUM_HEADS // NUM_KV_HEADS
kvh = qh // GROUP_SIZE
```

`paged_attention` asserts `num_heads % num_kv_heads == 0`; head counts stay constexpr/runtime
params so a TP local-head layout (halved heads at TP=2) is not broken.

*Note:* v1 maps one program per query head, so two query-head programs of the same GQA group
produce duplicated *logical* KV reads, with no explicit/cooperative reuse between them.
The same KV data may still be served by hardware caches (e.g. L2), so logical duplicate reads
must **not** be equated with 2x actual DRAM traffic. This is why only logical/effective KV
bandwidth is reported, never actual DRAM bandwidth.

## 4. Online softmax

FP32 running state: `m` (max), `l` (denominator), `acc` (accumulator). Per KV tile:

```
tile_max = max(scores)
m_new    = max(m, tile_max)
alpha    = exp(m - m_new)
p        = exp(scores - m_new)
acc      = acc * alpha + sum(p[:, None] * V, axis=ctx)
l        = l   * alpha + sum(p, axis=ctx)
m        = m_new

out = acc / l
```

Uses `tl.exp`; `exp2`/log2 rescaling deferred until profiling justifies it. Masked positions
get `scores = -inf` → `p = 0`, so partial blocks contribute nothing.

## 5. Triton program mapping

```
grid = (batch, num_query_heads)        # program <-> (sequence, query head)
```

One program: load one q vector (fp32) → map to KV head → walk paged KV blocks
(inner loop over `BLOCK_N`-sized tiles inside each 256-token physical block) → online softmax →
store one head output. `BLOCK_N ∈ {32, 64}`; v1 selected config: **BLOCK_N=64, num_warps=4**
(ptxas: 248 regs, 0 spills; ~50% occupancy at 2×128-thread blocks/SM).

## 6. Correctness strategy

Two independent references + Triton, three-way check:

* **PyTorch explicit paged reference** (`tests/kernels/test_paged_attention.py`): per sequence,
  `logical_block = pos // 256`, `block_offset = pos % 256`, `physical_block = bt[seq, lb]`,
  GQA gather, fp32 `scores = q @ K.T * scale` → softmax → `p @ V`. Deliberately simple, test-only.
* **flash_attn_with_kvcache** (`q.unsqueeze(1)` + `block_table`).

24 tests, all passing: deterministic non-contiguous mapping (logical 0..3 → physical 7,2,11,4),
block-boundary contexts {1, 255, 256, 257, 512, 1024, 2048, 4096} × fp16/bf16, permuted shared
physical blocks, batch=128 vs FA2, both BLOCK_N configs, all 16 query heads (GQA mapping).

## 7. Benchmark result

`benchmarks/bench_kernels.py` — `bench_paged_attention()` (bf16, `do_bench` w25/r100, µs):

| ctx | batch | FA2 | triton64 | 64/FA2 |
|---|---|---|---|---|
| 256 | 1 | 17.3 | 19.5 | 1.02x |
| 256 | 16 | 66.2 | 71.1 | 0.90x |
| 256 | 128 | 371.7 | 487.4 | 0.77x |
| 1024 | 1 | 31.3 | 47.4 | 0.66x |
| 1024 | 16 | 199.9 | 249.5 | 0.79x |
| 1024 | 128 | 1241 | 1940 | 0.64x |
| 2048 | 16 | 383.3 | 501.5 | 0.77x |
| 4096 | 1 | 67.5 | 136.7 | 0.49x |
| 4096 | 16 | 682.0 | 886.9 | 0.82x |
| 4096 | 128 | 2347 | 3576 | ~0.66x |

BLOCK_N=64 beats BLOCK_N=32 everywhere. Logical KV bandwidth at batch=128: FA2 722–915 GB/s,
Triton ~480–600 GB/s.

*Timing caveat:* absolute latency showed sensitivity to GPU clock/power/runtime state across
processes (fresh-process runs measured ~2x slower than the same config inside the full bench
run), while the relative ordering and Triton/FA2 ratios stayed stable across repeated
measurements. Root cause was not pinned to a specific mechanism (no clock/profiler evidence);
a short GPU warmup was added before this section to stabilize independent runs.

## 8. Why no runtime integration

v1 proves correctness (three-way check) but benchmarks at 0.49–0.82x of FA2; forcing it into
the decode path would be a real regression. Project principle is benchmark-driven engineering:
a self-written kernel is not required to become the default backend. Kept as a standalone,
reviewable implementation of the PagedAttention mechanism.

## 9. Potential future v2 (recorded, not implemented)

* One program per GQA group (i.e. per KV head), processing all query heads in the group
  while sharing a single K/V load — explicit KV reuse.
* Trade-off: higher register pressure per program (multiple q vectors and accumulators),
  which would further cut occupancy; requires re-benchmarking, especially long-context
  batch=1 where the v1 GQA duplication is most visible.
* Not pursued now: PagedAttention v1 already showed the theoretical KV-sharing optimization
  is not obviously worth a more complex mapping.
