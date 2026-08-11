# Phase 5 — Unified Mixed Scheduling (V1-inspired)

## Status

Implemented, tested, and benchmarked against a hot-state prefill-first baseline.
Verdict: functionally correct (mixed batches execute and produce identical text),
but on the current Qwen3-0.6B / RTX 5060 setup with the default token budget it
shows **no measurable throughput or TPOT win over the prefill-first baseline**.
A real TTFT/throughput win appears only when the shared token budget is small
enough to force chunked prefill. Details below — do not cite this feature as a
performance win without these caveats.

## Design (as implemented)

One model batch per step, laid out as `[decode rows][prefill rows]`:

```
Scheduler.schedule() -> ScheduleOutput(decode_seqs, prefill_seqs)
    decode-first: running queue drained first, 1 token budget per decode seq,
                  seqs restored to running in original order
    prefill: waiting[0] FIFO with remaining shared budget, chunked prefill allowed
             only for the first scheduled prefill seq

ModelRunner.run(output)
    decode only  -> prepare_decode            (baseline path, CUDA graph)
    prefill only -> prepare_prefill           (baseline path)
    mixed        -> prepare_mixed: concat [decode data][prefill data]
                    single forward over the concatenated batch

Attention
    store_kvcache once with the concatenated slot_mapping
    decode slice  -> flash_attn_with_kvcache
    prefill slice -> flash_attn_varlen_func (k/v from k_cache when prefix cache active)

ParallelLMHead
    pure decode  -> logits_indices=None  (all rows, baseline CUDA-graph path)
    prefill/mixed-> logits_indices = [decode rows] + [last row of completed prefill seqs]

Sampling
    sample_seqs = decode_seqs + completed_prefill_seqs
    token_ids   = sampled decode tokens + sampled first completion tokens
                  (chunked prefill seqs carry a placeholder, postprocess skips them)
```

## Correctness

- `tests/engine/test_scheduler.py` (8 tests): running-order preservation, budget
  bound (`decode_count > token_budget` stays within budget), mixed budget sharing,
  chunked prefill stays in waiting, partial blocks never enter the prefix hash,
  full-block prefix reuse.
- `tests/engine/test_runner.py` (8 tests): completed-prefill-seq detection, token
  mapping for pure decode / pure prefill / mixed with chunked placeholder.
- `tests/engine/test_e2e_mixed.py` (3 tests, GPU): identical text for pure decode
  ("Paris..."), pure prefill ("Tokyo..."), and a mixed batch (2 decoding + 1
  prefill, 1 mixed step observed, correct completions).
- Full suite: **122 passed**.

## Benchmark (contention workload, seed 0, hot compile state)

Both variants were measured with the same contention workload (stage 1: 16 short
prompt/long output; stage 2: 4 long prompts). The baseline variant is the
prefill-first scheduler (git-HEAD semantics) measured *after* warmup so the
comparison is compile-fair — the raw `phase5-baseline-contention.json` TTFT
(~1569 ms) was polluted by first-step torch.compile of the prefill shape and must
not be compared against the mixed result.

| max_batched_tokens | variant    | elapsed | out tok/s | TTFT p50/p95 | TPOT p50/p95 | lat p50/p95 |
| ------------------ | ---------- | ------: | --------: | -----------: | -----------: | ----------: |
| 16384 (default)    | baseline   | 4.51s   | 1522      | 222/263 ms   | 9.6/10.4 ms  | 3619/4457   |
| 16384              | mixed      | 4.52s   | 1517      | 220/274 ms   | 9.7/10.5 ms  | ~3700/4564  |
| 2048               | baseline   | 4.56s   | 1505      | 227/229 ms   | 9.6/10.5 ms  | 3658/4504   |
| 2048               | mixed      | 4.52s   | 1517      | 215/219 ms   | 9.7/10.6 ms  | 3628/4471   |
| 1024               | baseline   | 5.06s   | 1355      | 231/664 ms   | 9.6/10.5 ms  | 4155/5010   |
| 1024               | mixed      | 4.69s   | 1462      | 210/349 ms   | 9.8/10.6 ms  | 3808/4636   |

Interpretation:

- At the default budget (16384) the stage-2 prefill (6495 tokens) fits in one
  step, so the baseline pauses decode for only ~0.3 s — mixed has nothing to win.
  Throughput/TTFT/TPOT are identical within run-to-run noise.
- When the budget forces chunked prefill (1024), the baseline stalls all decodes
  for every prefill chunk (10 pure-prefill steps); mixed runs 9 mixed steps and
  keeps decodes moving: elapsed −7.4 %, output tok/s +7.9 %, TTFT p95 −47 %
  (664→349 ms), latency p95 −7.5 %.
- TPOT p50/p95 never differ: a stall shows up as a gap in only ~16 of 6841
  inter-token samples (0.23 %), below the P95 quantile. P95 TPOT is therefore
  insensitive to this stall pattern; TTFT p95 and end-to-end latency capture it.

## Limitation

The default scheduler config here effectively never chunks the given workloads
(max_num_batched_tokens=16384 ≫ typical prompts), so the mixed path is not
exercised by the default contention benchmark. The win is real but conditional on
budget pressure. Pure-decode CUDA graphs are preserved unchanged (mixed steps run
eager).

## Files changed (Phase 5)

- `nanovllm/engine/scheduler.py` — ScheduleOutput, decode-first shared-budget
  schedule(), per-seq postprocess
- `nanovllm/engine/llm_engine.py` — step() returns (outputs, num_prefill_tokens,
  num_decode_tokens)
- `nanovllm/engine/model_runner.py` — _prefill_data/_decode_data/prepare_mixed,
  run(output), is_pure_decode naming
- `nanovllm/utils/context.py` — num_decode_tokens, prefill_block_tables,
  logits_indices fields; keyword-only set_context
- `nanovllm/layers/attention.py` — mixed branch (decode slice + prefill slice)
- `nanovllm/layers/embed_head.py` — logits_indices-based row selection
- `bench.py` — mixed step accounting (wall time recorded once, not split)
- `tests/engine/test_scheduler.py`, `test_runner.py`, `test_e2e_mixed.py`
- `benchmarks/results/phase5-baseline-contention.json`,
  `phase5-mixed-contention.json`
