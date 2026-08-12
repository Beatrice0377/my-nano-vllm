# Phase 7 — Persistent Decode Metadata: Profiling and Decision

Status: **Profiled — not integrated (final decision: DO NOT IMPLEMENT)**

This report records a focused profiling and a minimal write-path prototype that
evaluated replacing the per-step construction of pure-decode model metadata with
persistent pinned staging buffers. The measured gain did not justify the added
runtime state, so the feature was intentionally not integrated.

## Original hypothesis

Each decode step rebuilds five model metadata tensors from Python `Sequence`
objects:

```text
input_ids       int64 [bs]
positions       int64 [bs]
slot_mapping    int32 [bs]
context_lens    int32 [bs]
block_tables    int32 [bs, max_blocks]
```

They are constructed as pinned CPU tensors, moved to CUDA, and copied into the
CUDA-Graph static buffers (`graph_vars`). The hypothesis was that a large,
fixed fraction of this per-step cost comes from repeated allocation and
transfer machinery (pinned allocation, intermediate CUDA tensors, D2D copies),
and that writing directly into persistent pinned staging buffers would remove
it.

## Current pure-decode metadata path

```text
Python Sequence traversal / list construction
-> torch.tensor(..., pin_memory=True)        x5
-> .cuda(non_blocking=True)                  x5
-> graph_vars[:bs] = gpu_tmp                 x5  (D2D, incl. fill/zero)
```

`temperatures` is built separately in `prepare_sample()` and is out of scope
for this phase.

## Focused profile (steady-state pure decode, CUDA Graph enabled)

Measured per-step metadata preparation, split into Python collection, pinned
tensor construction, H2D enqueue (CPU wall time), and D2D enqueue:

| phase (bs=1) | median |
|---|---|
| Python list construction | ~3 µs |
| pinned tensor construction | ~29 µs |
| H2D enqueue | ~69 µs |
| D2D to graph_vars | ~106 µs |
| **total** | **~214 µs** |

The D2D and H2D enqueue costs are largely batch-size independent (fixed
dispatch/allocation overheads dominate at small batch).

## Candidate A: direct persistent pinned storage

Persistent pinned CPU tensors (initialized once) with a shared NumPy view.
Each step writes the active rows directly while iterating the sequences, then:

```python
graph_vars["input_ids"][:bs].copy_(
    pinned_buf["input_ids"][:bs],
    non_blocking=True,
)
```

This removes repeated pinned allocation, intermediate CUDA tensors, and the
D2D copies, at the cost of a new CPU population path and a required direct H2D
enqueue per step.

## Candidate B: pageable temporary -> persistent pinned

Fallback that keeps the Python list construction and only removes repeated pin
allocation, CUDA temporaries, and D2D:

```text
Python list -> torch.tensor(list)           # pageable
-> persistent_pinned.copy_()                # pageable -> pinned
-> graph_vars[:bs].copy_(pinned, non_blocking=True)
```

## Batch-size dependent results

Median total metadata preparation time (correctness verified element-wise
against the baseline graph_vars for all batch sizes):

| bs | Baseline | Candidate A | Candidate B |
|---|---|---|---|
| 1 | 214 µs | 110 µs | 151 µs |
| 8 | 222 µs | 149 µs | 185 µs |
| 32 | 246 µs | 184 µs | 210 µs |
| 128 | 337 µs | 342 µs | 297 µs |

Saving and projected end-to-end impact (denominator = full decode-step wall):

```text
Candidate A:
bs=1    ~105 us saved, ~2.2% projected E2E
bs=8     ~73 us saved, ~1.0%
bs=32    ~62 us saved, ~0.5%
bs=128   ~0

Candidate B:
~36-63 us saved
representative decode-heavy projected gain ~0.3%
```

## Why small-bs benefits but representative decode-heavy workload does not

At small batch sizes the fixed per-step overheads (pinned allocation, CUDA
temporary creation, D2D enqueue) dominate, so removing them is visible
(~105 µs at bs=1). At larger batches the CPU population of the persistent
storage grows linearly (NumPy-style element-wise writes), and the required
direct H2D enqueue remains costly; the gains shrink to zero at bs=128. The
project's representative `decode-heavy` workload runs at an average batch of
roughly 40 sequences, where the projected gain is below ~1%.

## Final decision: DO NOT IMPLEMENT

> Persistent staging successfully removes repeated pinned allocation,
> intermediate CUDA tensors, and the subsequent D2D copies. However, the
> required direct H2D enqueue remains costly, while populating persistent CPU
> storage introduces its own overhead. The resulting net saving is too small at
> the batch sizes representative of this project.

Persistent staging would require additional runtime state and a second
metadata-population path; this was not justified by the measured gain.

No production code was modified during this evaluation. All profiling scripts
and microprototypes lived in temporary, untracked paths.
