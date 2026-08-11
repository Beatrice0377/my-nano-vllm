# Phase 1 benchmark audit

The benchmark is an offline measurement loop over the existing engine. It does
not simulate arrival rates, add a server, or change scheduling.

## Metric definitions

| Metric | Numerator | Denominator and scope | Synchronization |
| --- | --- | --- | --- |
| Prefill tokens/s | Positive `LLMEngine.step()` token count, i.e. newly scheduled query tokens | Sum of prefill step wall times | `torch.cuda.synchronize()` after every measured step; includes scheduler, input preparation, H2D enqueue, model, sampling, postprocess, and the sync wait |
| Decode tokens/s | Decode query tokens, one per scheduled running sequence | Sum of decode step wall times | Same as prefill |
| Output tokens/s | Completion tokens appended to requests | End-to-end run time from first request arrival to the final appended token | Includes request admission, scheduling, input preparation, H2D enqueue, model, sampling, postprocess, and synchronization |
| Total tokens/s | Logical prompt tokens plus completion tokens | Same end-to-end run time | Cached prompt tokens remain in this numerator; this is workload throughput, not executed GPU-token throughput |
| Requests/s | Number of requests | Same end-to-end run time | Same as output tokens/s |

The prefill/decode rates are phase-local rates, while output/total/request rates
are end-to-end rates. The explicit synchronization prevents CUDA asynchronous
kernel launches from making the step rates artificially high. The separate
`prepare_prefill`/`prepare_decode` timers intentionally measure only CPU-side
method call wall time and do not force a second synchronization. Model loading,
KV-cache allocation, CUDA-graph capture, and the explicit warmup request are
outside the measured run.

## TTFT

For every request, the benchmark records an arrival timestamp immediately
before `LLMEngine.add_request()`. After each existing `LLMEngine.step()` returns
and the benchmark synchronizes CUDA, it compares `Sequence.num_tokens` with the
previous step. When the length increases, it records the token append
timestamp. The first such token is the request's first generated token:

```text
TTFT = first token append timestamp - request arrival timestamp
```

This is not the end of a prefill step: partial chunked-prefill steps do not
append completion tokens and therefore do not create a TTFT sample. The
timestamp is taken after append/postprocess, after the existing step returns,
and after the benchmark's CUDA synchronization.

## TPOT

The observer records every appended completion-token timestamp per request:

```text
token_timestamps = [t1, t2, t3, ...]
inter_token_latencies = [t2 - t1, t3 - t2, ...]
```

`tpot_p50_ms` and `tpot_p95_ms` are percentiles over all real inter-token
latency samples. There is no request-level finish-time approximation. The
benchmark checks that the timestamp count equals the final completion-token
count before producing results.

## Prefix cache and prefill work

The baseline `BlockManager.can_allocate()` checks blocks sequentially from the
start of a prompt and stops at the first miss. Its `range(seq.num_blocks - 1)`
rule excludes the last logical block, including a partial block, from the
reusable prefix. A benchmark-only observer records each `can_allocate()` event:

- `prefix_cache_cached_blocks`: successful sequential block hits;
- `prefix_cache_blocks_looked_up`: actual block lookup iterations, including the
  first miss;
- `prefix_cache_hit_rate`: cached blocks divided by looked-up blocks;
- `prefix_cache_lookup_events`: number of allocation lookup attempts.

If a sequence is preempted and allocated again, each real lookup event is
counted. A partial final block cannot be a hit because the baseline never
examines it. The block-level denominator avoids mixing request hit rate with
block hit rate.

This metric is not a request-level shared-prefix hit rate. The hash cache keeps
entries for deallocated blocks, so a preempted sequence that is re-prefilled
can re-link its own previously written blocks, and those self-reuse hits are
included in `prefix_cache_cached_blocks`. Under KV capacity pressure this can
inflate the absolute rate even for workloads with no shared prefix. Interpret
the rate together with `prefill_tokens_executed` and the cached-block counts
rather than as a standalone shared-prefix ratio.

For every prefill step, the positive `LLMEngine.step()` count is recorded as
both `prefill_tokens_scheduled` and `prefill_tokens_executed`. They are equal
in this baseline because every scheduled non-cached query token is executed.
Chunked prefill does not double-count a token: each step advances
`seq.num_cached_tokens`, and the next step starts at that offset.

## Reproducible workloads

All workloads use random seed `0` by default and generate token IDs directly,
so tokenizer formatting does not change the length distribution.

| Workload | Requests | Prompt tokens | Output tokens | Prefix pattern | Purpose |
| --- | ---: | --- | --- | --- | --- |
| `balanced` | 64 | 256–1024 | 128–256 | All cold/random | Overall throughput |
| `decode-heavy` | 128 | 128–256 | 512–1024 | All cold/random | Decode throughput, TPOT, future PagedAttention/metadata work |
| `prefix-sharing` | 64 | 768–1024 | 128–256 | 512-token shared prefix for 75% of requests; 25% cold/random | Prefix cache and future cache-affinity scheduling |

The default model limits are `max_model_len=4096`,
`max_num_batched_tokens=16384`, and `max_num_seqs=512`. Results are saved as:

```text
benchmarks/results/baseline-balanced.json
benchmarks/results/baseline-decode-heavy.json
benchmarks/results/baseline-prefix-sharing.json
```

## Measurement limitations

- There are no per-kernel GPU timings yet; those belong to Phase 2 kernel
  microbenchmarks.
- Input-preparation timing ends when the non-blocking H2D calls are enqueued;
  the surrounding step timing includes the later synchronization.
- Token timestamps are observed after scheduler postprocess, not inside the
  sampler CUDA kernel. They measure when the token is appended and becomes
  visible to the engine loop.
- `total_tokens_per_second` counts logical cached prompt tokens, so it should
  not be compared directly with executed prefill tokens.
- The JSONs under `benchmarks/results/` were generated on the local RTX 5060
  (8 GB, sm_120); rerun `python bench.py --workload <name> --model <dir>` to
  regenerate.
