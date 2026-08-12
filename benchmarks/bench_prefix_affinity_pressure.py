"""Canonical prefix-cache pressure benchmark for Phase 6 affinity scheduling.

Reproduces the committed Phase 6 pressure result (e.g. 15957 -> 15427
executed prefill tokens, ~3.3% reduction, at the commit that introduced the
affinity scheduler (see the JSON `git_commit` field for the exact SHA), on the
tested RTX 5060 Laptop GPU):
    seed                  0
    gpu_memory_utilization 0.45
    max_num_batched_tokens 16384
    arrangement           coldfirst (10 cold, then 14 shared-prefix requests)
    shared prefix         512 tokens (2 full blocks)

The KV block pool is derived at runtime from the GPU's free memory, so the
recorded `num_kvcache_blocks` may differ across machines. Results can differ;
the JSON records the actual pool size, executed prefill tokens, physical warm
block reuse, and preemption attribution for the run.

How the FIFO baseline is obtained
---------------------------------
Both modes run on the *same* runtime code version (the current checkout), so
the experiment isolates the waiting-queue selection policy and nothing else.
The `fifo` mode monkeypatches `Scheduler._select_prefill()` inside this
benchmark process only:

* a chunked-prefill head (non-empty block_table) still continues first;
* a FIFO head that cannot be allocated (probe_allocate says too many free
  blocks required) stops the prefill loop, as in production;
* no subsequent candidates are scanned, so no affinity reorder ever happens;
* production scheduler source is not modified and no runtime config flag is
  added.

The `affinity` mode uses the production `_select_prefill()` untouched. All
other scheduler / model / block-manager / kernel code is identical between
the two modes.

Usage
-----
    python benchmarks/bench_prefix_affinity_pressure.py --mode fifo \
        --model "$NANOVLLM_MODEL" --output benchmarks/results/phase6-pressure-fifo.json
    python benchmarks/bench_prefix_affinity_pressure.py --mode affinity \
        --model "$NANOVLLM_MODEL" --output benchmarks/results/phase6-pressure-affinity.json
"""

import argparse
import json
import os
import types
from random import Random
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams

BLOCK_SIZE = 256
SHARED_PREFIX_TOKENS = 512


def _fifo_select_prefill(self):
    """Strict FIFO selection policy (benchmark-only).

    Mirrors the production head-handling of `_select_prefill()` without the
    affinity scan: chunked head continues, infeasible head stops the loop,
    otherwise the head itself is always returned. Mutates nothing (production
    aging state is only advanced when a candidate is actually reordered, which
    never happens here because the returned sequence is always `waiting[0]`).
    """
    head = self.waiting[0]
    if head.block_table:
        return head
    head_cached, head_required = self.block_manager.probe_allocate(head)
    if head_required > len(self.block_manager.free_block_ids):
        return None
    return head


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["fifo", "affinity"], required=True)
    p.add_argument("--model", default=os.environ.get("NANOVLLM_MODEL"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--util", type=float, default=0.45)
    p.add_argument("--budget", type=int, default=16384)
    p.add_argument("--cold", type=int, default=10)
    p.add_argument("--shared-hit", type=int, default=14)
    p.add_argument("--output", required=True)
    p.add_argument("--max-model-len", type=int, default=4096)
    return p.parse_args()


def make_specs(args, rng):
    """Cold-first request list: `cold` cold prompts, then `shared-hit` prompts
    sharing the same 512-token prefix. RNG call order is fixed so the queue is
    deterministic across runs."""
    shared_prefix = [rng.randrange(10_000) for _ in range(SHARED_PREFIX_TOKENS)]
    specs = []
    for i in range(args.cold + args.shared_hit):
        input_len = rng.randint(768, 1024)
        output_len = rng.randint(128, 256)
        if i < args.cold:
            prompt = [rng.randrange(10_000) for _ in range(input_len)]
        else:
            prompt = shared_prefix + [
                rng.randrange(10_000) for _ in range(input_len - SHARED_PREFIX_TOKENS)
            ]
        specs.append((prompt, output_len))
    return shared_prefix, specs


def warm_prefix_cache(llm, shared_prefix, rng):
    """Stage 0: one shared-prefix request fills the cache. Returns the physical
    block ids of the two shared blocks, or [-1, -1] if the cache could not be
    warmed (e.g. pool too small)."""
    warm_suffix = [rng.randrange(10_000) for _ in range(256)]
    llm.add_request(
        shared_prefix + warm_suffix,
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8),
    )
    while not llm.is_finished():
        llm.step()
    torch.cuda.synchronize()

    bm = llm.scheduler.block_manager
    ids = []
    h = -1
    for i in range(2):
        h = bm.compute_hash(shared_prefix[i * 256 : (i + 1) * 256], h)
        ids.append(bm.hash_to_block_id.get(h, -1))
    return ids


def run_pressure(args, llm, shared_prefix, shared_prefix_request_ids, warm_block_ids):
    """Stage 1: admit all requests, then step until finished while recording
    per-request prefill accounting and warm-block reuse attribution."""
    bm = llm.scheduler.block_manager

    hit_attr = {}  # shared-hit seq_id -> first can_allocate attribution
    prefill_executed = {}  # all seq_ids -> executed prefill tokens
    prefill_steps = {}  # all seq_ids -> distinct prefill step count
    preempted = set()
    preempt_count = {}

    def observer(seq, num_cached_blocks, num_looked_up_blocks):
        if seq.seq_id not in shared_prefix_request_ids or seq.seq_id in hit_attr:
            return
        phys = []
        h = -1
        for i in range(num_cached_blocks):
            h = bm.compute_hash(seq.block(i), h)
            phys.append(bm.hash_to_block_id.get(h, -1))
        hit_attr[seq.seq_id] = {
            "first_cached_blocks": num_cached_blocks,
            "first_physical_block_ids": phys,
            "first_prefill_tokens": None,
            "prefill_tokens_executed": 0,
            "preempted": False,
        }

    bm.cache_lookup_observer = observer
    orig_preempt = llm.scheduler.preempt

    def counting_preempt(seq):
        preempt_count[seq.seq_id] = preempt_count.get(seq.seq_id, 0) + 1
        preempted.add(seq.seq_id)
        orig_preempt(seq)

    llm.scheduler.preempt = counting_preempt

    started = perf_counter()
    while not llm.is_finished():
        out = llm.scheduler.schedule()
        for s in out.prefill_seqs:
            prefill_executed[s.seq_id] = (
                prefill_executed.get(s.seq_id, 0) + s.num_scheduled_tokens
            )
            prefill_steps[s.seq_id] = prefill_steps.get(s.seq_id, 0) + 1
            if (
                s.seq_id in hit_attr
                and hit_attr[s.seq_id]["first_prefill_tokens"] is None
            ):
                hit_attr[s.seq_id]["first_prefill_tokens"] = s.num_scheduled_tokens
        token_ids = llm.model_runner.run(out)
        llm.scheduler.postprocess(out.decode_seqs + out.prefill_seqs, token_ids)
    torch.cuda.synchronize()
    elapsed = perf_counter() - started

    for sid in hit_attr:
        hit_attr[sid]["prefill_tokens_executed"] = prefill_executed.get(sid, 0)
        hit_attr[sid]["preempted"] = sid in preempted

    per_request = []
    for sid in sorted(hit_attr):
        a = hit_attr[sid]
        phys = a["first_physical_block_ids"]
        per_request.append(
            {
                "seq_id": sid,
                "first_cached_blocks": a["first_cached_blocks"],
                "warm_block0_reused": (len(phys) > 0 and phys[0] == warm_block_ids[0]),
                "warm_block1_reused": (len(phys) > 1 and phys[1] == warm_block_ids[1]),
                "first_prefill_tokens": a["first_prefill_tokens"],
                "prefill_tokens_executed": a["prefill_tokens_executed"],
                "preempted": a["preempted"],
                "preempt_count": preempt_count.get(sid, 0),
                "prefill_steps": prefill_steps.get(sid, 0),
            }
        )

    n_hits = len(hit_attr)
    return {
        "elapsed": round(elapsed, 3),
        "total_prefill_tokens_executed": sum(prefill_executed.values()),
        "shared_hit_request_count": n_hits,
        "possible_warm_block_reuse_events": n_hits * 2,
        "warm_block_reuse_event_count": sum(
            (1 if r["warm_block0_reused"] else 0)
            + (1 if r["warm_block1_reused"] else 0)
            for r in per_request
        ),
        "shared_requests_with_any_warm_reuse": sum(
            1 for r in per_request if r["warm_block0_reused"] or r["warm_block1_reused"]
        ),
        "shared_requests_with_full_2block_reuse": sum(
            1
            for r in per_request
            if r["warm_block0_reused"] and r["warm_block1_reused"]
        ),
        "shared_hits_preempted": sum(1 for r in per_request if r["preempted"]),
        "shared_hits_executed_tokens_total": sum(
            a["prefill_tokens_executed"] for a in hit_attr.values()
        ),
        "shared_hits_reprefill_work": max(
            sum(
                a["prefill_tokens_executed"] - a["first_prefill_tokens"]
                for a in hit_attr.values()
                if a["first_prefill_tokens"] is not None
            ),
            0,
        ),
        "preempted_seqs_all": sorted(preempted),
        "preempt_count_all": preempt_count,
        "per_request": per_request,
    }


def main():
    # Provenance is captured before any output is written so that the
    # benchmark's own output JSON does not itself flip git_dirty.
    provenance_commit = _git("rev-parse", "HEAD")
    provenance_dirty = bool(_git("status", "--porcelain"))
    args = parse_args()
    if not args.model:
        raise SystemExit(
            "--model is required (or set NANOVLLM_MODEL); a local Qwen3-0.6B "
            "snapshot path must point to a directory with config.json and weights"
        )

    rng = Random(args.seed)
    shared_prefix, specs = make_specs(args, rng)

    llm = LLM(
        args.model,
        enforce_eager=True,
        gpu_memory_utilization=args.util,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.budget,
        max_num_seqs=512,
    )
    num_kvcache_blocks = llm.model_runner.config.num_kvcache_blocks

    if args.mode == "fifo":
        llm.scheduler._select_prefill = types.MethodType(
            _fifo_select_prefill, llm.scheduler
        )
    warm_block_ids = warm_prefix_cache(llm, shared_prefix, rng)

    shared_prefix_request_ids = set()
    for i, (prompt, out_len) in enumerate(specs):
        llm.add_request(
            prompt,
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=out_len),
        )
        seq = llm.scheduler.waiting[-1]
        if i >= args.cold:
            shared_prefix_request_ids.add(seq.seq_id)

    result = run_pressure(
        args, llm, shared_prefix, shared_prefix_request_ids, warm_block_ids
    )
    result["mode"] = args.mode
    result["seed"] = args.seed
    result["gpu_memory_utilization"] = args.util
    result["num_kvcache_blocks"] = num_kvcache_blocks
    result["max_num_batched_tokens"] = args.budget
    result["arrangement"] = (
        f"{args.cold} cold + {args.shared_hit} shared-hit (coldfirst)"
    )
    result["shared_prefix_tokens"] = SHARED_PREFIX_TOKENS
    result["warm_block_ids"] = warm_block_ids
    result["warmup_verified_cache"] = warm_block_ids != [-1, -1]
    result["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    result["git_commit"] = provenance_commit
    result["git_dirty"] = provenance_dirty

    out = json.dumps(result, indent=1, sort_keys=True)
    print(out, flush=True)
    path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(out + "\n")


def _git(*cmd):
    import subprocess

    try:
        out = subprocess.check_output(["git", *cmd], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
