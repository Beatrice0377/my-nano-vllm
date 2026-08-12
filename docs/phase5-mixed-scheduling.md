# Phase 5 — Unified Mixed Scheduling (V1-inspired)

## Status

Implemented, tested, and benchmarked against the real Phase 4 prefill-first
code (commit `5c030cc`) under the same workload/config, all in a warm compile
state. Verdict: functionally correct (mixed batches execute and produce
deterministic expected completions; the E2E tests assert expected texts, not
baseline-vs-mixed token equality), and aggregate throughput/TTFT/TPOT were
similar to the prefill-first baseline at every tested budget in the recorded
runs (no repeated-run variance estimate was collected). The mixed scheduler
removes prefill-induced decode stalls when the shared token budget forces
chunked prefill (see the incumbent-gap metric below). Details in the Benchmark
section — do not cite this feature as an unconditional performance win; state
the workload/config conditions it was measured under.

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
                  (incomplete prefill chunks produce no token — postprocess skips them)
```

## Correctness

- `tests/engine/test_scheduler.py` (8 tests): running-order preservation, budget
  bound (`decode_count > token_budget` stays within budget), mixed budget sharing,
  chunked prefill stays in waiting, partial blocks never enter the prefix hash,
  full-block prefix reuse.
- `tests/engine/test_runner.py` (4 tests): completed-prefill-seq detection.
- `tests/engine/test_e2e_mixed.py` (GPU): deterministic expected completions for
  pure decode ("Paris..."), pure prefill ("Tokyo..."), and a mixed batch
  (2 decoding + 1 prefill, >=1 mixed step observed), plus eager-vs-CUDA-graph
  completion-token parity after mixed steps.
- Full suite: **137 passed** (post-Phase-6 head).

## Benchmark (contention workload, seed 0, warm compile state)

Both variants were measured with the same contention workload (stage 1: 16 short
prompt/long output; stage 2: 4 long prompts, 6495 tokens) on the real code — the
baseline is the Phase 4 prefill-first scheduler at commit `5c030cc`, the mixed
variant is this Phase 5 implementation — at three `max_num_batched_tokens`
values. All runs are warm (first-step torch.compile happens during warmup).
JSON:
`benchmarks/results/phase5-{baseline,mixed}-contention-{16384,2048,1024}.json`.

| budget | variant  | elapsed | out tok/s | TTFT p50/p95 | TPOT p50/p95 | lat p50/p95 |
| ------ | -------- | ------: | --------: | -----------: | -----------: | ----------: |
| 16384  | baseline | 4.56s   | 1505      | 216/266 ms   | 9.79/10.60   | —           |
| 16384  | mixed    | 4.62s   | 1484      | 241/281 ms   | 9.75/10.76   | —           |
| 2048   | baseline | 4.57s   | 1501      | 220/223 ms   | 9.75/10.62   | —           |
| 2048   | mixed    | 4.63s   | 1483      | 225/229 ms   | 9.88/10.99   | —           |
| 1024   | baseline | 4.82s   | 1425      | 217/329 ms   | 9.89/10.84   | —           |
| 1024   | mixed    | 4.68s   | 1466      | 211/331 ms   | 9.81/10.47   | —           |

### Incumbent decode stall metric

The pooled TPOT distribution cannot see a rare stall (a stall affects only one
gap per stalled request — ~16 gaps in ~6841 samples — so it sits below P95).
The added metric measures all inter-token gaps of the 16 stage-1 requests that
occur *after* stage-2 injection (`later >= stage2_injected_at`), then reports
the P99 / P99.9 / max of that distribution:

| budget | variant  | gaps after injection | P99 | P99.9 | max gap |
| ------ | -------- | -------------------: | --: | ----: | ------: |
| 16384  | baseline | 5972                 | 10.9 ms | 275.5 ms | 275.5 ms |
| 16384  | mixed    | 5962                 | 11.3 ms | 280.7 ms | 280.7 ms |
| 2048   | baseline | 5972                 | 11.1 ms | 290.0 ms | 290.0 ms |
| 2048   | mixed    | 5962                 | 64.1 ms | 84.9 ms  | 84.9 ms  |
| 1024   | baseline | 5972                 | 11.1 ms | 435.5 ms | 435.5 ms |
| 1024   | mixed    | 5962                 | 52.6 ms | 58.4 ms  | 58.4 ms  |

Interpretation:

- At the default budget (16384) the stage-2 prefill (6495 tokens) fits in one
  step, so the baseline pauses decode for one ~0.28 s step and mixed cannot do
  better — throughput/TTFT/TPOT and even the max incumbent gap (~280 ms) are
  similar in the recorded runs. The mixed path is essentially not exercised.
- When the budget forces chunked prefill (2048, 1024), the baseline stalls all
  decodes for every prefill chunk (max incumbent gap 290→435 ms); mixed keeps
  decodes moving and bounds the max incumbent gap at ~85 ms (2048) / ~58 ms
  (1024). End-to-end numbers differed by ~1–3% in the recorded runs because
  decode is only ~4 s of the run and prefill work is identical — the win is in
  *latency variance of incumbent decodes*, not aggregate throughput.
- TPOT p50/p95 never differ (9.8/10.5 ms everywhere): a stall shows up as a gap
  in only ~16 of 6841 inter-token samples (0.23 %), below the P95 quantile. The
  incumbent-gap distribution above keeps those rare gaps visible; P99.9 and max
  are the relevant tail statistics.

### Decode-heavy regression check

The same decode-heavy workload run on the Phase 4 code (commit `5c030cc`) vs
this implementation shows no pure-decode regression (both with CUDA graphs,
warm): elapsed 39.14 s → 38.94 s, output 2528 → 2541 tok/s, TTFT p50/p95
626/909 → 640/945 ms, TPOT p50/p95 17.86/20.81 → 18.15/21.29 ms — similar in
the recorded runs (no repeated-run variance estimate was collected).

### Note on the logits_indices sync bug

During this check an earlier mixed implementation built `logits_indices` as a
CPU tensor; indexing the GPU logits tensor with it forced an implicit device
sync every prefill step (~110 ms/step). Fixed by constructing the index on the
GPU (`(cu_seqlens_q[1:] - 1)[completed]`, `torch.arange(nd, device="cuda")`).
This is a real performance bug found by measuring, not by inspection.

## Limitation

The default scheduler config here effectively never chunks the given workloads
(max_num_batched_tokens=16384 ≫ typical prompts), so the mixed path is not
exercised by the default contention benchmark. The latency-variance win is real
but conditional on budget pressure. Pure-decode CUDA graphs are preserved
unchanged (mixed steps run eager).

## Files changed (Phase 5)

- `nanovllm/engine/scheduler.py` — ScheduleOutput, decode-first shared-budget
  schedule(), per-seq postprocess (index-advanced token consumption)
- `nanovllm/engine/llm_engine.py` — step() returns (outputs, num_prefill_tokens,
  num_decode_tokens)
- `nanovllm/engine/model_runner.py` — _prefill_data/_decode_data/prepare_mixed,
  run(output), is_pure_decode naming, GPU-built logits_indices
- `nanovllm/utils/context.py` — num_decode_tokens, prefill_block_tables,
  logits_indices fields; keyword-only set_context
- `nanovllm/layers/attention.py` — mixed branch (decode slice + prefill slice)
- `nanovllm/layers/embed_head.py` — logits_indices-based row selection
- `bench.py` — mixed step accounting (wall time recorded once, not split);
  incumbent decode stall metrics (P99/P99.9/max gap)
- `tests/engine/test_scheduler.py`, `test_runner.py`, `test_e2e_mixed.py`
- `benchmarks/results/phase5-{baseline,mixed}-contention-{16384,2048,1024}.json`
