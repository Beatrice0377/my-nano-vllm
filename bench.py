"""Small, reproducible offline benchmark for the nano-vLLM baseline."""

import argparse
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from random import Random
from time import perf_counter


WORKLOADS = {
    "balanced": dict(
        requests=64,
        min_input=256,
        max_input=1024,
        min_output=128,
        max_output=256,
        shared_prefix_tokens=0,
        cold_fraction=0.0,
    ),
    "decode-heavy": dict(
        requests=128,
        min_input=128,
        max_input=256,
        min_output=512,
        max_output=1024,
        shared_prefix_tokens=0,
        cold_fraction=0.0,
    ),
    "prefix-sharing": dict(
        requests=64,
        min_input=768,
        max_input=1024,
        min_output=128,
        max_output=256,
        shared_prefix_tokens=512,
        cold_fraction=0.25,
    ),
    "contention": dict(staged=True),
    "prefix-affinity": dict(staged=True),
}

# Deterministic two-stage arrival pattern for scheduler contention.
# Stage 1: short-prompt / long-output requests enter decode first.
# Stage 2: long-prompt requests are injected once Stage 1 is decoding.
CONTENTION_STAGES = [
    dict(requests=16, min_input=64, max_input=128, min_output=256, max_output=512),
    dict(requests=4, min_input=1024, max_input=2048, min_output=128, max_output=256),
]

# Deterministic two-stage arrival pattern for prefix-cache affinity.
# Stage 0: a single shared-prefix request warms the prefix cache so the
# 512-token shared prefix (2 full blocks) is present; its TTFT/throughput is
# NOT included in the measured result. Stage 1: measured requests alternate
# cold / shared-hit so both schedulers see the same queue: cold prompts are
# fully random, shared-hit prompts reuse the same 512-token prefix from the
# warmup stage, i.e. exactly MIN_CACHED_BLOCKS (2) full cached blocks.
PREFIX_AFFINITY_STAGES = [
    dict(
        requests=1,
        shared_prefix_tokens=512,
        min_input=768,
        max_input=1024,
        min_output=8,
        max_output=8,
    ),
    dict(
        requests=24,
        shared_prefix_tokens=512,
        min_input=768,
        max_input=1024,
        min_output=128,
        max_output=256,
    ),
]


@dataclass(slots=True)
class RequestSpec:
    prompt_token_ids: list[int]
    max_tokens: int


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile in the same unit as values."""
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def build_request_specs(args: argparse.Namespace) -> list[RequestSpec]:
    rng = Random(args.seed)
    shared_prefix = [rng.randrange(10_000) for _ in range(args.shared_prefix_tokens)]
    shared_request_count = (
        round(args.requests * (1.0 - args.cold_fraction)) if shared_prefix else 0
    )
    shared_request_ids = set(rng.sample(range(args.requests), shared_request_count))
    specs = []
    for request_id in range(args.requests):
        input_len = rng.randint(args.min_input, args.max_input)
        output_len = rng.randint(args.min_output, args.max_output)
        use_shared_prefix = request_id in shared_request_ids
        if use_shared_prefix:
            suffix = [
                rng.randrange(10_000) for _ in range(input_len - len(shared_prefix))
            ]
            prompt_token_ids = shared_prefix + suffix
        else:
            prompt_token_ids = [rng.randrange(10_000) for _ in range(input_len)]
        specs.append(RequestSpec(prompt_token_ids, output_len))
    return specs


def build_contention_specs(seed: int) -> list[list[RequestSpec]]:
    """Generate the two contention stages with a fixed seed.

    Stage 1: 16 short-prompt / long-output requests (enter decode quickly).
    Stage 2: 4 long-prompt requests injected while Stage 1 is decoding.
    """
    stages = []
    for stage in CONTENTION_STAGES:
        rng = Random(seed)
        specs = []
        for _ in range(stage["requests"]):
            input_len = rng.randint(stage["min_input"], stage["max_input"])
            output_len = rng.randint(stage["min_output"], stage["max_output"])
            specs.append(
                RequestSpec(
                    [rng.randrange(10_000) for _ in range(input_len)], output_len
                )
            )
        stages.append(specs)
    return stages


def build_prefix_affinity_specs(
    seed: int,
) -> tuple[list[RequestSpec], list[RequestSpec]]:
    """Generate (warmup_specs, measured_specs) sharing one fixed 512-token prefix.

    Warmup: 1 shared-prefix request that fills the prefix cache. Measured:
    cold / shared-hit alternating requests (cold at even indices, shared-hit at
    odd indices), both drawn from the same length range so the two schedulers
    see identical queues. The shared prefix is generated once and reused.
    """
    rng = Random(seed)
    shared_prefix = [rng.randrange(10_000) for _ in range(512)]
    warmup_specs = []
    for _ in range(PREFIX_AFFINITY_STAGES[0]["requests"]):
        input_len = rng.randint(
            PREFIX_AFFINITY_STAGES[0]["min_input"],
            PREFIX_AFFINITY_STAGES[0]["max_input"],
        )
        output_len = rng.randint(
            PREFIX_AFFINITY_STAGES[0]["min_output"],
            PREFIX_AFFINITY_STAGES[0]["max_output"],
        )
        suffix = [rng.randrange(10_000) for _ in range(input_len - len(shared_prefix))]
        warmup_specs.append(RequestSpec(shared_prefix + suffix, output_len))
    measured_specs = []
    for i in range(PREFIX_AFFINITY_STAGES[1]["requests"]):
        input_len = rng.randint(
            PREFIX_AFFINITY_STAGES[1]["min_input"],
            PREFIX_AFFINITY_STAGES[1]["max_input"],
        )
        output_len = rng.randint(
            PREFIX_AFFINITY_STAGES[1]["min_output"],
            PREFIX_AFFINITY_STAGES[1]["max_output"],
        )
        if i % 2 == 0:
            measured_specs.append(
                RequestSpec(
                    [rng.randrange(10_000) for _ in range(input_len)], output_len
                )
            )
        else:
            suffix = [
                rng.randrange(10_000) for _ in range(input_len - len(shared_prefix))
            ]
            measured_specs.append(RequestSpec(shared_prefix + suffix, output_len))
    return warmup_specs, measured_specs


def install_observers(llm):
    """Install benchmark-only observers without changing engine decisions."""
    prepare_seconds = {"prefill": [], "decode": []}
    for mode in prepare_seconds:
        method_name = f"prepare_{mode}"
        original = getattr(llm.model_runner, method_name)

        def timed(seqs, original=original, mode=mode):
            started = perf_counter()
            result = original(seqs)
            prepare_seconds[mode].append(perf_counter() - started)
            return result

        setattr(llm.model_runner, method_name, timed)

    cache_lookup_events = []

    def observe_cache_lookup(seq, cached_blocks, looked_up_blocks):
        cache_lookup_events.append(
            {
                "seq_id": seq.seq_id,
                "cached_blocks": cached_blocks,
                "looked_up_blocks": looked_up_blocks,
            }
        )

    llm.scheduler.block_manager.cache_lookup_observer = observe_cache_lookup
    return prepare_seconds, cache_lookup_events


def parse_args() -> argparse.Namespace:
    default_model = os.path.expanduser(
        os.environ.get("NANOVLLM_MODEL", "~/models/Qwen3-0.6B")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=tuple(WORKLOADS), default="balanced")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--min-input", type=int)
    parser.add_argument("--max-input", type=int)
    parser.add_argument("--min-output", type=int)
    parser.add_argument("--max-output", type=int)
    parser.add_argument("--shared-prefix-tokens", type=int)
    parser.add_argument("--cold-fraction", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-seqs", type=int, default=512)
    parser.add_argument("--warmup-tokens", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--output")
    args = parser.parse_args()

    workload = WORKLOADS[args.workload].copy()
    if not workload.get("staged"):
        for name in workload:
            value = getattr(args, name)
            if value is not None:
                workload[name] = value
            setattr(args, name, workload[name])
        if args.requests < 1:
            parser.error("--requests must be positive")
        if not 1 <= args.min_input <= args.max_input:
            parser.error("input length range is invalid")
        if not 1 <= args.min_output <= args.max_output:
            parser.error("output length range is invalid")
        if args.shared_prefix_tokens < 0 or args.shared_prefix_tokens > args.min_input:
            parser.error("--shared-prefix-tokens must be in [0, --min-input]")
        if not 0.0 <= args.cold_fraction <= 1.0:
            parser.error("--cold-fraction must be in [0, 1]")
    if args.output is None:
        args.output = f"benchmarks/results/baseline-{args.workload}.json"
    if args.warmup_tokens < 1:
        parser.error("--warmup-tokens must be positive")
    return args


def validate_model_path(model: str):
    model_path = Path(model)
    if not model_path.is_dir():
        raise SystemExit(f"model directory does not exist: {model_path}")
    required = [model_path / "config.json", model_path / "tokenizer.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if not any(model_path.glob("*.safetensors")):
        missing.append(f"{model_path}/*.safetensors")
    if missing:
        raise SystemExit(
            "model directory is not a complete local snapshot; missing: "
            + ", ".join(missing)
        )


def setup_llm(args: argparse.Namespace):
    """Import dependencies, check CUDA, construct and warm up the LLM."""
    try:
        import torch
        import triton
        from nanovllm import LLM, SamplingParams
    except ImportError as exc:
        missing = getattr(exc, "name", "a dependency")
        raise SystemExit(
            f"Cannot start benchmark: missing {missing}. "
            "Install the dependencies from pyproject.toml in a CUDA-enabled environment."
        ) from exc

    cuda_available = torch.cuda.is_available()
    environment = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": getattr(triton, "__version__", "unknown"),
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name() if cuda_available else None,
    }
    if not cuda_available:
        raise SystemExit(
            "CUDA is unavailable in this process; no benchmark result was written.\n"
            + json.dumps(environment, indent=2, sort_keys=True)
        )
    validate_model_path(args.model)

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=args.max_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    llm.generate(
        ["Benchmark warmup"],
        SamplingParams(max_tokens=args.warmup_tokens),
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    return llm, environment


def admit_request(
    llm, spec: RequestSpec, sequences, arrival_timestamps, lengths_before_step, started
):
    """Admit one request; returns the (possibly first) start timestamp."""
    from nanovllm import SamplingParams

    arrival = perf_counter()
    if not sequences:
        started = arrival
    llm.add_request(
        spec.prompt_token_ids,
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=spec.max_tokens),
    )
    seq = llm.scheduler.waiting[-1]
    sequences.append(seq)
    arrival_timestamps[seq.seq_id] = arrival
    lengths_before_step[seq.seq_id] = seq.num_tokens
    return started


def step_once(llm, sequences, lengths_before_step, token_timestamps, stats):
    """Run one engine step and record its per-sequence token timestamps.

    A mixed scheduler step performs decode and prefill work together, so a
    single step wall time is recorded once and attributed to the phase the
    step's work belongs to; it is never split into additive per-phase times.
    """
    step_started = perf_counter()
    _, num_prefill, num_decode = llm.step()
    import torch

    torch.cuda.synchronize()
    timestamp = perf_counter()
    step_seconds = perf_counter() - step_started
    for seq in sequences:
        delta = seq.num_tokens - lengths_before_step[seq.seq_id]
        if delta:
            if delta != 1:
                raise RuntimeError(
                    f"unexpected token delta for sequence {seq.seq_id}: {delta}"
                )
            token_timestamps.setdefault(seq.seq_id, []).append(timestamp)
            lengths_before_step[seq.seq_id] = seq.num_tokens
    stats["steps"] += 1
    stats["prefill_tokens"] += num_prefill
    stats["decode_tokens"] += num_decode
    if num_prefill > 0 and num_decode > 0:
        # Mixed step: decode and prefill share one GPU execution; the wall
        # time is recorded once (never split into additive phase times).
        stats["mixed_steps"] += 1
        stats["mixed_seconds"] += step_seconds
    elif num_prefill > 0:
        stats["prefill_steps"] += 1
        stats["prefill_seconds"] += step_seconds
    else:
        stats["decode_steps"] += 1
        stats["decode_seconds"] += step_seconds


def compute_metrics(
    args,
    llm,
    sequences,
    arrival_timestamps,
    token_timestamps,
    lengths_before_step,
    stats,
    started,
    finished_at,
    prepare_seconds,
    cache_lookup_events,
) -> dict:
    prompt_tokens = sum(seq.num_prompt_tokens for seq in sequences)
    completion_tokens = sum(seq.num_completion_tokens for seq in sequences)
    missing_timestamps = [
        seq.seq_id
        for seq in sequences
        if len(token_timestamps.get(seq.seq_id, [])) != seq.num_completion_tokens
    ]
    if missing_timestamps:
        raise RuntimeError(
            f"missing token timestamps for sequences: {missing_timestamps}"
        )

    ttfts = []
    latencies = []
    inter_token_latencies = []
    for seq in sequences:
        timestamps = token_timestamps[seq.seq_id]
        arrival = arrival_timestamps[seq.seq_id]
        ttfts.append(timestamps[0] - arrival)
        latencies.append(timestamps[-1] - arrival)
        inter_token_latencies.extend(
            later - earlier for earlier, later in zip(timestamps, timestamps[1:])
        )

    cached_blocks = sum(event["cached_blocks"] for event in cache_lookup_events)
    looked_up_blocks = sum(event["looked_up_blocks"] for event in cache_lookup_events)
    elapsed = finished_at - started

    def rate(tokens: int, seconds: float) -> float:
        return tokens / seconds if seconds else 0.0

    import torch

    return {
        "elapsed_seconds": elapsed,
        "steps": stats["steps"],
        "prefill_steps": stats["prefill_steps"],
        "decode_steps": stats["decode_steps"],
        "mixed_steps": stats["mixed_steps"],
        "prefill_tokens_scheduled": stats["prefill_tokens"],
        "prefill_tokens_executed": stats["prefill_tokens"],
        "decode_query_tokens_executed": stats["decode_tokens"],
        "prefill_wall_seconds": stats["prefill_seconds"],
        "decode_wall_seconds": stats["decode_seconds"],
        "mixed_wall_seconds": stats["mixed_seconds"],
        # mixed-step wall time overlaps both phases; each rate below uses the
        # union of its own pure-phase time and all mixed-step time, so the two
        # rates are NOT additive (one mixed step is counted in both).
        "prefill_throughput_tokens_per_second": rate(
            stats["prefill_tokens"],
            stats["prefill_seconds"] + stats["mixed_seconds"],
        ),
        "decode_throughput_tokens_per_second": rate(
            stats["decode_tokens"],
            stats["decode_seconds"] + stats["mixed_seconds"],
        ),
        "output_tokens": completion_tokens,
        "output_tokens_per_second": rate(completion_tokens, elapsed),
        "total_tokens_per_second": rate(prompt_tokens + completion_tokens, elapsed),
        "request_throughput_per_second": rate(len(sequences), elapsed),
        "ttft_p50_ms": percentile(ttfts, 0.50) * 1000,
        "ttft_p95_ms": percentile(ttfts, 0.95) * 1000,
        "tpot_p50_ms": percentile(inter_token_latencies, 0.50) * 1000,
        "tpot_p95_ms": percentile(inter_token_latencies, 0.95) * 1000,
        "tpot_sample_count": len(inter_token_latencies),
        "request_latency_p50_ms": percentile(latencies, 0.50) * 1000,
        "request_latency_p95_ms": percentile(latencies, 0.95) * 1000,
        "timestamped_output_tokens": sum(
            len(tokens) for tokens in token_timestamps.values()
        ),
        "prefix_cache_lookup_events": len(cache_lookup_events),
        "prefix_cache_blocks_looked_up": looked_up_blocks,
        "prefix_cache_cached_blocks": cached_blocks,
        "prefix_cache_hit_rate": cached_blocks / looked_up_blocks
        if looked_up_blocks
        else 0.0,
        "prefix_cache_looked_up_tokens": looked_up_blocks * llm.scheduler.block_size,
        "prefix_cache_cached_tokens": cached_blocks * llm.scheduler.block_size,
        "input_prepare_prefill_cpu_ms": sum(prepare_seconds["prefill"]) * 1000,
        "input_prepare_decode_cpu_ms": sum(prepare_seconds["decode"]) * 1000,
        "input_prepare_prefill_calls": len(prepare_seconds["prefill"]),
        "input_prepare_decode_calls": len(prepare_seconds["decode"]),
        "peak_gpu_memory_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    import torch
    from nanovllm import SamplingParams

    llm, environment = setup_llm(args)

    if WORKLOADS[args.workload].get("staged"):
        if args.workload == "prefix-affinity":
            return run_prefix_affinity_benchmark(args, llm, environment)
        return run_contention_benchmark(args, llm, environment)

    specs = build_request_specs(args)
    if (
        max(len(spec.prompt_token_ids) + spec.max_tokens for spec in specs)
        > args.max_model_len
    ):
        raise SystemExit("generated prompt plus output length exceeds --max-model-len")

    workload = {
        "name": args.workload,
        "model": os.path.abspath(args.model),
        "requests": args.requests,
        "seed": args.seed,
        "input_tokens_min": min(len(spec.prompt_token_ids) for spec in specs),
        "input_tokens_max": max(len(spec.prompt_token_ids) for spec in specs),
        "input_tokens_total": sum(len(spec.prompt_token_ids) for spec in specs),
        "output_tokens_min": min(spec.max_tokens for spec in specs),
        "output_tokens_max": max(spec.max_tokens for spec in specs),
        "output_tokens_requested": sum(spec.max_tokens for spec in specs),
        "shared_prefix_tokens": args.shared_prefix_tokens,
        "cold_fraction": args.cold_fraction,
        "shared_request_count": round(args.requests * (1.0 - args.cold_fraction))
        if args.shared_prefix_tokens
        else 0,
        "cold_request_count": args.requests
        - (
            round(args.requests * (1.0 - args.cold_fraction))
            if args.shared_prefix_tokens
            else 0
        ),
        "max_model_len": args.max_model_len,
        "max_batched_tokens": args.max_batched_tokens,
        "max_seqs": args.max_seqs,
        "enforce_eager": args.enforce_eager,
        "warmup_tokens": args.warmup_tokens,
        "result_path": args.output,
    }

    prepare_seconds, cache_lookup_events = install_observers(llm)
    sequences = []
    arrival_timestamps = {}
    lengths_before_step = {}
    token_timestamps = {}
    stats = {
        "steps": 0,
        "prefill_steps": 0,
        "decode_steps": 0,
        "mixed_steps": 0,
        "prefill_tokens": 0,
        "decode_tokens": 0,
        "prefill_seconds": 0.0,
        "decode_seconds": 0.0,
        "mixed_seconds": 0.0,
    }
    started = 0.0
    for spec in specs:
        started = admit_request(
            llm, spec, sequences, arrival_timestamps, lengths_before_step, started
        )

    while not llm.is_finished():
        step_once(llm, sequences, lengths_before_step, token_timestamps, stats)
    finished_at = perf_counter()

    metrics = compute_metrics(
        args,
        llm,
        sequences,
        arrival_timestamps,
        token_timestamps,
        lengths_before_step,
        stats,
        started,
        finished_at,
        prepare_seconds,
        cache_lookup_events,
    )
    return {"environment": environment, "workload": workload, "metrics": metrics}


def run_contention_benchmark(args, llm, environment) -> dict:
    """Two-stage contention workload run with the current scheduler.

    Stage 1 requests are admitted and driven into decode; only then are the
    Stage 2 long-prompt requests injected. This creates decode + long-prefill
    contention without a load generator.
    """
    import torch
    from nanovllm import SamplingParams

    stages = build_contention_specs(args.seed)
    if any(
        len(spec.prompt_token_ids) + spec.max_tokens > args.max_model_len
        for stage in stages
        for spec in stage
    ):
        raise SystemExit("generated prompt plus output length exceeds --max-model-len")

    workload = {
        "name": "contention",
        "model": os.path.abspath(args.model),
        "stages": [
            {
                **stage,
                "prompt_tokens_total": sum(
                    len(spec.prompt_token_ids) for spec in specs
                ),
                "output_tokens_requested": sum(spec.max_tokens for spec in specs),
            }
            for stage, specs in zip(CONTENTION_STAGES, stages)
        ],
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "max_batched_tokens": args.max_batched_tokens,
        "max_seqs": args.max_seqs,
        "enforce_eager": args.enforce_eager,
        "warmup_tokens": args.warmup_tokens,
        "result_path": args.output,
    }

    prepare_seconds, cache_lookup_events = install_observers(llm)
    sequences = []
    arrival_timestamps = {}
    lengths_before_step = {}
    token_timestamps = {}
    stats = {
        "steps": 0,
        "prefill_steps": 0,
        "decode_steps": 0,
        "mixed_steps": 0,
        "prefill_tokens": 0,
        "decode_tokens": 0,
        "prefill_seconds": 0.0,
        "decode_seconds": 0.0,
        "mixed_seconds": 0.0,
    }

    started = 0.0
    for spec in stages[0]:
        started = admit_request(
            llm, spec, sequences, arrival_timestamps, lengths_before_step, started
        )

    # Drive Stage 1 into decode (prompt fully processed) before Stage 2 arrival.
    while not all(not seq.is_prefill for seq in sequences):
        step_once(llm, sequences, lengths_before_step, token_timestamps, stats)

    # Incumbent requests: Stage 1 sequences already decoding when the Stage 2
    # long-prefill burst is injected. Record the injection wall time so we can
    # isolate their post-injection inter-token latencies (rare stall gaps that
    # P95 TPOT dilutes away across the full sample pool).
    incumbent_seq_ids = [seq.seq_id for seq in sequences]
    stage2_injected_at = perf_counter()
    for spec in stages[1]:
        started = admit_request(
            llm, spec, sequences, arrival_timestamps, lengths_before_step, started
        )

    while not llm.is_finished():
        step_once(llm, sequences, lengths_before_step, token_timestamps, stats)
    finished_at = perf_counter()

    metrics = compute_metrics(
        args,
        llm,
        sequences,
        arrival_timestamps,
        token_timestamps,
        lengths_before_step,
        stats,
        started,
        finished_at,
        prepare_seconds,
        cache_lookup_events,
    )
    # Stall-sensitive metrics on incumbent decodes: gaps that end at/after
    # stage-2 injection (a prefill stall spans the injection boundary, so the
    # gap's earlier timestamp is typically before injection). P99.9 + max
    # capture the extreme inter-token stall from the injected prefill burst.
    incumbent_gaps = []
    for seq_id in incumbent_seq_ids:
        timestamps = token_timestamps.get(seq_id, [])
        incumbent_gaps.extend(
            later - earlier
            for earlier, later in zip(timestamps, timestamps[1:])
            if later >= stage2_injected_at
        )
    if incumbent_gaps:
        metrics["incumbent_decode_gap_count"] = len(incumbent_gaps)
        metrics["incumbent_decode_tpot_p99_ms"] = (
            percentile(incumbent_gaps, 0.99) * 1000
        )
        metrics["incumbent_decode_tpot_p99_9_ms"] = (
            percentile(incumbent_gaps, 0.999) * 1000
        )
        metrics["incumbent_decode_max_gap_ms"] = max(incumbent_gaps) * 1000
    else:
        metrics["incumbent_decode_gap_count"] = 0
        metrics["incumbent_decode_tpot_p99_ms"] = 0.0
        metrics["incumbent_decode_tpot_p99_9_ms"] = 0.0
        metrics["incumbent_decode_max_gap_ms"] = 0.0
    return {"environment": environment, "workload": workload, "metrics": metrics}


def run_prefix_affinity_benchmark(args, llm, environment) -> dict:
    """Cold / shared-hit interleaved workload with a warm prefix cache.

    Stage 0 (warmup): one shared-prefix request fills the prefix cache; its
    latency is NOT part of the measured result. Stage 1 (measured): cold and
    shared-hit requests alternate (cold at even indices, shared-hit at odd),
    so the waiting queue is deterministic and identical across schedulers.
    """
    from nanovllm import SamplingParams
    from nanovllm.engine.block_manager import BlockManager
    import torch

    warmup_specs, measured_specs = build_prefix_affinity_specs(args.seed)
    if any(
        len(spec.prompt_token_ids) + spec.max_tokens > args.max_model_len
        for spec in warmup_specs + measured_specs
    ):
        raise SystemExit("generated prompt plus output length exceeds --max-model-len")

    # Stage 0: warm the prefix cache; drop all per-request bookkeeping.
    for spec in warmup_specs:
        llm.add_request(
            spec.prompt_token_ids,
            SamplingParams(
                temperature=0.6, ignore_eos=True, max_tokens=spec.max_tokens
            ),
        )
    while not llm.is_finished():
        llm.step()
    torch.cuda.synchronize()

    # Verify the warmup actually cached the shared prefix (2 full blocks).
    shared_prefix = warmup_specs[0].prompt_token_ids[:512]
    h = -1
    for i in range(2):
        h = BlockManager.compute_hash(shared_prefix[i * 256 : (i + 1) * 256], h)
    warmup_cached = h in llm.scheduler.block_manager.hash_to_block_id

    workload = {
        "name": "prefix-affinity",
        "model": os.path.abspath(args.model),
        "warmup": {
            "requests": len(warmup_specs),
            "shared_prefix_tokens": 512,
            "measured": False,
        },
        "measured": {
            "requests": len(measured_specs),
            "shared_prefix_tokens": 512,
            "cold": sum(i % 2 == 0 for i in range(len(measured_specs))),
            "shared_hit": sum(i % 2 == 1 for i in range(len(measured_specs))),
            "alternating": True,
        },
        "warmup_verified_cache": warmup_cached,
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "max_batched_tokens": args.max_batched_tokens,
        "max_seqs": args.max_seqs,
        "enforce_eager": args.enforce_eager,
        "warmup_tokens": args.warmup_tokens,
        "result_path": args.output,
    }

    # Stage 1: measured cold/shared-hit interleaved batch.
    prepare_seconds, cache_lookup_events = install_observers(llm)
    sequences = []
    arrival_timestamps = {}
    lengths_before_step = {}
    token_timestamps = {}
    stats = {
        "steps": 0,
        "prefill_steps": 0,
        "decode_steps": 0,
        "mixed_steps": 0,
        "prefill_tokens": 0,
        "decode_tokens": 0,
        "prefill_seconds": 0.0,
        "decode_seconds": 0.0,
        "mixed_seconds": 0.0,
    }

    started = 0.0
    for spec in measured_specs:
        started = admit_request(
            llm, spec, sequences, arrival_timestamps, lengths_before_step, started
        )
    while not llm.is_finished():
        step_once(llm, sequences, lengths_before_step, token_timestamps, stats)
    finished_at = perf_counter()

    metrics = compute_metrics(
        args,
        llm,
        sequences,
        arrival_timestamps,
        token_timestamps,
        lengths_before_step,
        stats,
        started,
        finished_at,
        prepare_seconds,
        cache_lookup_events,
    )
    metrics["warmup_verified_cache"] = warmup_cached
    metrics["num_kvcache_blocks"] = llm.model_runner.config.num_kvcache_blocks
    metrics["gpu_memory_utilization"] = llm.model_runner.config.gpu_memory_utilization
    return {"environment": environment, "workload": workload, "metrics": metrics}


def main():
    args = parse_args()
    result = run_benchmark(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
