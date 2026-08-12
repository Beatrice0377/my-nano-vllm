# Phase 4 — Triton FlashAttention Prefill

Custom Triton FlashAttention prefill kernel: a standalone educational /
reference implementation. **Not integrated** into the runtime decode or
prefill path — see [Why Not Integrated](#9-why-not-integrated).

## 1. Scope

Supported (and only these):

- dense / non-prefix prefill (`block_tables is None`)
- varlen (cu_seqlens)
- causal
- GQA
- fp16 / bf16
- head_dim = 128
- forward only

Explicitly out of scope: prefix-cache prefill, arbitrary masking, TMA /
descriptor loads, split-K, cooperative GQA, multi-stage autotuning.

## 2. Data Layout

```text
q   [total_tokens, num_heads, D]
k/v [total_tokens, num_kv_heads, D]
cu_seqlens [batch + 1] int32
```

`total_k == total_q` (dense prefill; no prefix cache). Output `[total_tokens,
num_heads, D]`.

## 3. Program Mapping

```text
grid = (batch, num_query_heads, ceil(max_seqlen_q / BLOCK_M))
program -> (sequence, query_head, query_tile)
```

`max_seqlen` is passed by the caller; deriving it from the GPU tensor via
`.max()` forces a device-to-host sync on every call, which dominates
small-batch latency.

## 4. Causal K-loop Pruning

The K loop does **not** scan the full sequence and then mask future tokens.
Each Q tile only reads the K/V positions it can causally attend to:

```text
q_local_start = tile * BLOCK_M
q_local_end   = min(q_local_start + BLOCK_M, seq_len)
num_k_tiles   = ceil(q_local_end / BLOCK_N)
```

```text
L = 4096, BLOCK_M = 64
Q tile 0  -> K[0:64]
Q tile 1  -> K[0:128]
Q tile 2  -> K[0:192]
...
Q tile 63 -> K[0:4096]
```

This is the core causal optimization of the first version.

## 5. Online Softmax

FP32 running state per query row:

```text
m   running max
l   running denominator
acc running output accumulator
```

Update per K tile:

```text
m_new    = max(m, tile_max)
alpha    = exp(m - m_new)
p        = exp(scores - m_new)
l        = l * alpha + sum(p)
acc      = acc * alpha + p @ V
```

`tl.dot` inputs stay in storage dtype (fp16/bf16) with
`out_dtype=tl.float32`; the softmax math runs in fp32; the result is cast
back to the storage dtype on store.

### Implementation bugs found by tests (worth keeping)

Two bugs were actually caught by the correctness suite during Phase 4. Both
are silent-wrong-answer bugs — tests pass for short sequences and fail once
the sequence spans multiple K tiles.

**Bug 1 — broadcast dimension.** `p = exp(scores - m_new)` with `m_new` a 1-D
tensor of shape `[BLOCK_M]`: without `[:, None]`, the 1-D tensor aligns with
the last dimension (`BLOCK_N`), and the same row vector is subtracted from
every query row, corrupting the per-row softmax state. Correct form:
`p = exp(scores - m_new[:, None])`.

**Bug 2 — accumulator rescaling.** When the running max increases
(`m_old -> m_new`), `alpha = exp(m_old - m_new)` must rescale **both** the
denominator `l` and the output accumulator `acc`:

```text
acc = acc * alpha[:, None] + p @ V
```

`tl.dot(acc=...)` only accumulates (adds), it never applies the `alpha`
rescale. A single-K-tile sequence is unaffected; multi-tile sequences
produce wrong outputs. Tests must therefore cover `sequence length >
BLOCK_N`.

## 6. GQA Mapping

```text
GROUP_SIZE = NUM_HEADS // NUM_KV_HEADS
kv_head = query_head // GROUP_SIZE
```

Each program reads its group's KV head. Two query heads sharing one KV head
issue duplicated logical KV reads; hardware cache may absorb part of this.
No cooperative KV sharing in v1.

## 7. Correctness Strategy

Three-way check:

```text
PyTorch explicit reference
          ↕
        Triton
          ↕
FlashAttention (flash_attn_varlen_func, causal=True)
```

Key cases (all in `tests/kernels/test_flash_attention.py`, 28 tests):

- partial Q tile: `L = BLOCK_M - 1` / `BLOCK_M + 1` (invalid rows must not
  produce NaN; they are loaded as zero so their scores are finite, and the
  store is masked)
- multi-K-tile (regression for the two bugs above)
- causal boundary: `L = 1024` with future K poisoned (`k[256:] += 100`),
  check tokens 0/1/63/64 never see future K
- sequence isolation: seq 0 K/V biased +5, seq 1 biased -5 (varlen indexing
  errors must fail loudly)
- ragged varlen (13 sequences, lengths 1..4096)
- GQA: every query head checked per-head
- FA2 parity across lengths and both dtypes

## 8. Benchmark

bf16, `do_bench(warmup=25, rep=100)`, microseconds, RTX 5060 Laptop GPU.
Raw timings are the committed artifact
`benchmarks/results/kernels-benchmark-raw-10db113.txt` (single re-run of
`benchmarks/bench_kernels.py` in the clean commit-1 worktree); ratios =
FA2 / Triton, computed by script from that artifact, > 1 means Triton faster:

| seq/batch | FA2 | Triton 64x64 | Triton 32x64 | FA2 / Triton (best) |
| --- | ---: | ---: | ---: | ---: |
| L=128 | 18.0 | 19.9 | 19.8 | 0.91 |
| L=256 | 35.2 | 35.8 | 36.7 | 0.98 |
| L=512 | 76.8 | 84.9 | 91.2 | 0.91 |
| L=1024 | 201.0 | 259.4 | 277.0 | 0.77 |
| L=2048 | 646.4 | 893.4 | 968.5 | 0.72 |
| L=4096 | 2300.2 | 3388.8 | 3607.6 | 0.68 |
| varlen (7872 tok) | 3097.5 | 4552.4 | 4962.6 | 0.68 |
| ragged (6080 tok) | 2532.7 | 3735.5 | 4052.4 | 0.68 |

The negative result is real and is not hidden: the custom kernel is at parity
for short sequences and 0.68–0.98x of FA2 for long-prefill workloads.
Logical causal-attention TFLOP/s (computed as `2*D*H*Σ L_s*(L_s+1)`, not a
hardware tensor-core utilization claim) at L=4096, from the same raw
latencies: FLOPs = `2*128*16*4096*4097` = 68,736,253,952 → **FA2 ≈ 29.9 TF/s,
Triton ≈ 20.3 TF/s** (`68,736,253,952 / 2300.16e-6 / 1e12` and
`68,736,253,952 / 3388.85e-6 / 1e12`). The kernel remains slower than
FlashAttention 2 as sequence length grows; this project did not profile the
hardware bottleneck deeply enough to attribute the gap to a single limiting
factor.

## 9. Why Not Integrated

FA2 remains faster for long-prefill workloads, so the custom implementation
is intentionally kept standalone. The project's Phase 4 kernel work is a
benchmark-driven decision: correctness is complete and the implementation is
readable, but there is no end-to-end benefit from replacing
`flash_attn_varlen_func` with it.

## 10. Limitations / Future Work

Recorded, not implemented:

- cooperative GQA reuse (one program per KV head, shared across the group)
- more parallel K scheduling (e.g. split-K)
- advanced production tiling (TMA, multi-stage pipelines)
