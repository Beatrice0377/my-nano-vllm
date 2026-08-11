# Phase 6 — Prefix Cache-Affinity Scheduling

## Status

Implemented, tested, and benchmarked against a FIFO-only scheduler baseline
(git-HEAD scheduler without the affinity reorder) under the same prefix-sharing
workload at three token-budget settings. Under the no-pressure workload (the
KV pool, even at the default `gpu_memory_utilization`, leaves ~160 blocks —
more than the ~100 the workload needs — so warmup cache blocks are never
evicted), the affinity reorder does **not** reduce executed prefill tokens or
improve aggregate throughput, because cache hits are reachable regardless of
scheduling order. It does shift request completion order — shared-prefix
requests are prefilled earlier and get a lower TTFT (see Benchmark section) —
at the cost of slightly later cold requests. Under explicit cache pressure
(small KV pool, see Cache-pressure experiment), the affinity reorder does
reduce executed prefill tokens by scheduling shared hits before the warm
blocks are recycled. State the workload/config conditions, not an
unconditional win.

## Design (as implemented)

A small, bounded affinity reorder inside `Scheduler._select_prefill()`, applied
only when the scheduler must pick a prefill from `waiting`:

- `AFFINITY_WINDOW = 8` — scan only `waiting[1:min(8, len)]`; the FIFO head is
  never skipped silently.
- `MIN_CACHED_BLOCKS = 2` — reorder only when the best candidate can reuse
  `>= 2` full cached blocks (a single full block is not a strong signal).
- `MAX_BYPASS = 8` — aging guard: once the FIFO head has been bypassed this
  many times, it is force-scheduled to prevent starvation.
- Chunked-prefill continuation (a sequence whose `block_table` is already
  non-empty) always keeps top priority — it is never displaced by the affinity
  scan.
- Feasibility is request-specific: `probe_allocate(seq)` returns
  `(num_cached_blocks, num_required_free_blocks)` without mutating
  observer/hash/refcount state; if the candidate needs more free blocks than
  are available it is not considered, and if the FIFO head itself is not
  feasible the scheduler stops (never jumps past an infeasible head).
- A selected candidate still goes through the normal `can_allocate(selected)`
  path, so the Phase 1 prefix-cache observer semantics are unchanged (the
  prefix hash may legitimately be consulted twice for the same sequence).
- Aging state is scalar: `affinity_head_seq_id` + `affinity_head_bypasses`,
  reset whenever the FIFO head changes. A bypass is counted only when a
  non-head candidate is actually scheduled into this step's prefill batch
  (`waiting.remove` + `affinity_head_bypasses += 1` in `schedule()`, after
  budget feasibility is confirmed). Budget-limited candidates are never removed
  from `waiting`, so the FIFO head does not change and no false bypass is
  possible; a chunked-continuation head takes the early-return path without
  touching aging state.

`BlockManager.probe_allocate()` is a read-only twin of `can_allocate()`: it
walks the same recursive-block-hash prefix loop and reports how many blocks
could be reused and how many new blocks the sequence would need, with zero
side effects.

## Correctness

- `tests/engine/test_affinity.py` (15 tests): head feasibility break, no
  jumping past an infeasible head, candidate outside the window not considered,
  `probe_allocate` does not touch the observer counts, aging forces the head
  after `MAX_BYPASS`, chunked continuation keeps priority, tie-breaks stay
  FIFO, prefix reuse returns the expected cached-block counts (incl. the
  `range(num_blocks - 1)` baseline quirk: with exactly 512 shared tokens = 2
  blocks, only the first block participates in reuse accounting because the
  loop walks `range(num_blocks - 1)`; `MIN_CACHED_BLOCKS = 2` therefore needs
  a prompt longer than the shared prefix, e.g. 560 tokens / 3 blocks with the
  512-token shared prefix as 2 reusable full blocks).
- Full suite: **137 passed** (was 122 before this phase).
- GPU E2E smoke: a second request sharing a 512-token prefix inside a
  560-token prompt (3 blocks) reuses 2/2 shared blocks — prefill ~0.31 s vs
  ~2.75 s cold — the prefix cache works end to end under the new scheduler.

## Benchmark (prefix-affinity workload, seed 0)

Workload: `bench.py --workload prefix-affinity` = a warmup stage (one request
with a 512-token shared prefix, `n` empty tokens) followed by 24 measured
requests alternating cold / shared-hit: 12 cold with unique 768–1024-token
prompts, 12 sharing the 512-token prefix (even request indices cold, odd
shared-hit). The FIFO baseline is the identical code with the affinity scan
statically bypassed (a separate worktree copy at git HEAD). All runs warm.
Each JSON records the actual KV pool size at runtime:
`metrics.num_kvcache_blocks` and `metrics.gpu_memory_utilization` (the pool is
computed by the model runner from device memory, weights, and the requested
utilization — it is not fixed and varies slightly between runs, e.g. 160 vs
174/175 blocks, so results must be read against the recorded pool size).
JSON: `benchmarks/results/phase6-{fifo,affinity}-prefix-affinity[-{2048,1024}].json`.

| budget | variant  | KV pool | elapsed | steps | prefill exec | hit rate | TTFT p50/p95 | TPOT p50 | out tok/s |
| ------ | -------- | ------: | ------: | ----: | -----------: | -------: | -----------: | -------: | --------: |
| 16384  | FIFO     | 160     | 3.47s   | 254   | 15453        | 0.500    | 653/653 ms   | 12.44    | 1332      |
| 16384  | affinity | 160     | 3.50s   | 255   | 15453        | 0.500    | 653/653 ms   | 12.62    | 1321      |
| 2048   | FIFO     | 174     | 3.65s   | 260   | 15453        | 0.459    | 537/843 ms   | 12.71    | 1266      |
| 2048   | affinity | 174     | 3.78s   | 263   | 15453        | 0.459    | 494/780 ms   | 13.05    | 1223      |
| 1024   | FIFO     | 175     | 4.27s   | 270   | 15453        | 0.505    | 858/1432 ms  | 13.12    | 1082      |
| 1024   | affinity | 175     | 4.11s   | 271   | 15453        | 0.476    | 701/1373 ms  | 12.73    | 1124      |

Interpretation:

- `prefill_tokens_executed` is **identical across all six runs (15453)** — the
  affinity reorder does not reduce the prefill compute. Cache hits happen
  regardless of scheduling order because the KV block pool is sufficient for
  this measured workload: the actual pool is ~160–175 blocks at the default
  `gpu_memory_utilization` (measured at runtime, not configured), versus ~100
  blocks the workload needs, so the warmup-prefix blocks are never reallocated
  and their hash entries survive for every later request.
- The reorder only changes *completion order*: shared-prefix requests are
  prefilled earlier, which lowers their TTFT. At budget 2048 TTFT p50 drops
  537→494 ms (−8 %); at 1024, 858→701 ms (−18 %), and TTFT p95 drops
  1432→1373 ms. Cold requests are pushed later; this is a request-ordering /
  prioritization trade-off, not a compute saving.
- Aggregate throughput/elapsed/TPOT are within noise (±1 %). At the default
  budget (16384) the whole workload prefills in one step, so the scan is never
  meaningfully exercised and both schedulers are identical.
- Scheduler ordering changes observer-visible lookup history. Because executed
  prefill tokens remain identical, the hit-rate difference at budget 1024
  (0.505 → 0.476) does not correspond to a change in total prefill compute in
  this workload.

## Cache-pressure experiment (added in review round 2)

The no-pressure result above is the expected regime: with the free-block pool
sufficient for this measured workload, the warmup-prefix blocks are never
reallocated and cache hits are reachable regardless of scheduling order. To
test the mechanism where it could matter, the KV block pool is shrunk via the
existing `gpu_memory_utilization` knob (no new config option) so that cold
allocations actually approach recycling the warm hashed-but-deallocated blocks:

- util 0.50 → 44 blocks, util 0.45 → 29, util 0.40 → 15 (model weights leave
  only ~160 blocks even at the default util 0.90).
- Workload: Stage 0 warms a 512-token shared prefix (2 full blocks, physical
  ids recorded); Stage 1 (canonical case: cold-first) runs 24 measured
  requests — 10 cold with unique prompts, then 14 shared-hit requests each
  reusing the 512-token prefix; a debug-only observer records when the warm
  physical blocks are recycled (`hash_to_block_id` entry removed by
  `_allocate_block`) and how many shared-hit requests still reuse the warm
  physical blocks.
- At util 0.45 / cold-first, the FIFO baseline recycles the warm blocks early
  (step ~1/32), so the first shared-hit request prefills from scratch, while
  affinity schedules the shared hits before the pool is exhausted (recycled at
  step ~452, all hits reuse 2/2 warm blocks): executed prefill tokens 16379
  (FIFO) vs 14319 (affinity), a −12.6 % reduction. Elapsed time is within noise
  (27.1 s vs 26.6 s): the saved prefill work is real but too small to change
  end-to-end throughput on this model/GPU/workload.
- At util 0.40 (15 blocks) the pool is so small that even the affinity scan
  cannot get a shared hit in front of the cold batch; both schedulers behave
  identically. At util 0.50 (44 blocks) the pool is large enough that recycling
  happens late or not at all, and again no executed-token difference appears.
- Caveat: these pressure runs use the same model/seed/prompts as the main
  workload, but a debug-only instrumentation harness rather than `bench.py`'s
  full metric pipeline; the FIFO baseline is a separate worktree at git HEAD
  (no affinity code), sharing the identical KV block pool sizing.

## Cache-pressure attribution (util 0.45 / cold-first, seed 0)

Per-request attribution for the canonical pressure case (29 KV blocks, 24
requests: 10 cold first, then 14 shared-hit; each shared-hit request has a
512-token shared prefix = 2 warm full blocks):

| metric | FIFO | affinity |
|---|---|---|
| shared-hit requests | 14 | 14 |
| possible warm-block reuse events (14 × 2) | 28 | 28 |
| warm-block reuse events (physical ids match) | 0 | 28 |
| shared requests with any warm reuse | 0 | 14 |
| shared requests with full 2-block reuse | 0 | 14 |
| shared-hit prefill tokens, first prefill | 5236 | 4724 |
| shared-hit prefill tokens, executed (incl. re-prefill) | 6719 | 5253 |
| shared-hit re-prefill work (executed − first) | 1483 | 529 |
| shared-hit requests preempted | 4 (20,21,22,25) | 2 (20,24) |
| cold requests preempted | 1 (11) | 1 (11) |
| total prefill tokens executed | 16379 | 14319 |

The net difference (−2060 tokens) decomposes as:

- **−512 tokens from preserved warm-prefix reuse**: the only first-prefill
  difference is request 15 (868 tokens in FIFO vs 356 in affinity, one full
  2-block reuse). In FIFO every later shared-hit request still finds the
  shared-prefix hash (re-created by request 15's own prefill), so it also skips
  512 tokens — but against the *re-warmed* blocks, not the original warm
  physical blocks. This is why physical warm reuse is 0/28 while first-prefill
  hits 2/2 blocks for requests 17–28 (request 16's first `can_allocate` probe
  already sees the warm hash gone in FIFO, but its actual prefill step runs
  after request 15's prefill has re-created the shared-prefix hash).
- **−954 tokens from different preemption/recompute behavior**: FIFO preempts
  four shared-hit requests (re-prefill 1483 tokens), affinity only two
  (529 tokens).
- **−594 tokens from cold-side scheduling differences**: cold requests execute
  9660 tokens in FIFO vs 9066 in affinity (the reorder changes which requests
  are preempted and how much of their prefill must be re-executed).

512 tokens of the net reduction are directly attributable to preserving the
original two-block shared prefix for the first shared-hit request. The
remaining 1,548 tokens come from indirect changes in preemption and re-prefill
work caused by the reordered schedule under tight KV-cache capacity. The
cold-side −594 tokens reflect a different preemption/recompute allocation
across cold requests under the reordered schedule; they are not a direct
consequence of preserving the warm blocks.

## CPU overhead

`_select_prefill()` adds a bounded probe scan over up to 7 candidates. Measured
in a CPU-only microbenchmark (`bench_cpu_overhead.py`, 6 cold + 6 shared-hit
sequences, median over repeated `schedule()` calls): FIFO ≈ 27–34 µs per
schedule call, affinity ≈ 69–83 µs. Bounded affinity probing adds roughly
40–50 µs of scheduler CPU work per relevant scheduling call in the measured
workloads. This is small compared with millisecond-scale model execution in
the tested configuration, and it is paid only when a prefill decision is made.

## Files changed (Phase 6)

- `nanovllm/engine/scheduler.py` — `AFFINITY_WINDOW`/`MIN_CACHED_BLOCKS`/
  `MAX_BYPASS`, `_select_prefill()` affinity scan, scalar aging
- `nanovllm/engine/block_manager.py` — read-only `probe_allocate()` twin of
  `can_allocate()`
- `bench.py` — `prefix-affinity` workload (warmup stage + cold/hot interleave)
- `tests/engine/test_affinity.py` — 15 unit tests
- `benchmarks/results/phase6-{fifo,affinity}-prefix-affinity[-{2048,1024}].json`
