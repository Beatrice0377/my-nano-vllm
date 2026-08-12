"""Microbenchmarks: in-repo torch.compile baselines vs Triton fused kernels.

Ops:
  - SiluAndMul        (MLP gate*up, width 2*3072)
  - RMSNorm           (hidden 1024)
  - Add + RMSNorm     (hidden 1024)
  - q/k per-head norm (hidden 128, Qwen3Attention QK-norm path)

All RMSNorm variants benchmark bf16 activations with a bf16 weight parameter,
matching the validated runtime (ModelRunner constructs the model under
torch.set_default_dtype(hf_config.dtype)).

Usage: .venv/bin/python benchmarks/bench_kernels.py
Timing via triton.testing.do_bench (CUDA events, warmup + rep). Results are
mean microseconds over the rep window. Compilation (torch.compile + Triton
JIT) is triggered once per shape outside the timed region.
"""

import torch
import triton.testing
from flash_attn import flash_attn_with_kvcache

from nanovllm.layers.activation import SiluAndMul, TritonSiluAndMul
from nanovllm.layers.attention import paged_attention
from nanovllm.layers.layernorm import RMSNorm, TritonRMSNorm

# Qwen3-0.6B MLP intermediate size -> input width 2 * H.
SILU_SHAPES = [
    (1, 2 * 3072),  # single decode row
    (16, 2 * 3072),  # small decode batch
    (128, 2 * 3072),  # large decode batch
    (1024, 2 * 3072),  # small prefill
    (4096, 2 * 3072),  # prefill-like
]
# Qwen3-0.6B hidden size.
H = 1024
RMS_SHAPES = [
    (1, H),  # single decode row
    (16, H),  # small decode batch
    (32, H),
    (128, H),  # large decode batch
    (512, H),  # CUDA-graph decode batch boundary
    (1024, H),  # small prefill
    (4096, H),  # prefill-like
]
# Per-head QK-norm: [M, num_heads, head_dim].
QK_SHAPES = [(1, 16, 128), (16, 16, 128), (128, 16, 128)]
DTYPE = torch.bfloat16
WARMUP = 25
REP = 100


def bench_row(label, shape, baseline, kernel):
    x = torch.randn(*shape, device="cuda", dtype=DTYPE)
    baseline(x)  # trigger torch.compile once outside the timed region
    kernel(x)
    torch.cuda.synchronize()
    base_us = (
        triton.testing.do_bench(lambda: baseline(x), warmup=WARMUP, rep=REP) * 1000
    )
    kern_us = triton.testing.do_bench(lambda: kernel(x), warmup=WARMUP, rep=REP) * 1000
    print(
        f"{label:>10} {str(shape):>15} {base_us:12.2f} {kern_us:12.2f} {base_us / kern_us:9.2f}x"
    )


def bench_silu():
    baseline = SiluAndMul()
    kernel = TritonSiluAndMul()
    print(f"\nSiluAndMul (torch.compile) vs Triton, dtype={DTYPE}")
    for shape in SILU_SHAPES:
        bench_row("silu", shape, baseline, kernel)


def bench_rms():
    baseline = RMSNorm(H, eps=1e-6).cuda()
    kernel = TritonRMSNorm(H, eps=1e-6).cuda()
    # Match the validated runtime: ModelRunner constructs the model under
    # torch.set_default_dtype(hf_config.dtype), so the weight parameter has
    # the same dtype as the activations (bf16 here), not fp32.
    with torch.no_grad():
        w = torch.nn.Parameter(torch.ones(H, device="cuda", dtype=DTYPE))
        baseline.weight = w
        kernel.weight = w
    print(
        f"\nRMSNorm (torch.compile) vs Triton, input_dtype={DTYPE} weight_dtype={DTYPE}"
    )
    for shape in RMS_SHAPES:
        bench_row("rms", shape, lambda x: baseline.rms_forward(x), lambda x: kernel(x))
    print(
        f"\nAdd+RMSNorm (torch.compile) vs Triton, input_dtype={DTYPE} "
        f"weight_dtype={DTYPE}"
    )
    for shape in RMS_SHAPES:
        r = torch.randn(*shape, device="cuda", dtype=DTYPE)
        bench_row(
            "add_rms",
            shape,
            lambda x: baseline.add_rms_forward(x, r),
            lambda x: kernel(x, r),
        )
    print(
        f"\nQK-norm [M,16,128] (torch.compile) vs Triton, input_dtype={DTYPE} "
        f"weight_dtype={DTYPE}"
    )
    qk_baseline = RMSNorm(128, eps=1e-6).cuda()
    qk_kernel = TritonRMSNorm(128, eps=1e-6).cuda()
    with torch.no_grad():
        qk_w = torch.nn.Parameter(torch.ones(128, device="cuda", dtype=DTYPE))
        qk_baseline.weight = qk_w
        qk_kernel.weight = qk_w
    for shape in QK_SHAPES:
        bench_row(
            "qk",
            shape,
            lambda x: qk_baseline.rms_forward(x),
            lambda x: qk_kernel(x),
        )


def bench_paged_attention():
    """Decode PagedAttention: flash_attn_with_kvcache vs Triton (BLOCK_N 32/64).

    Logical (effective) KV bandwidth = requested K+V bytes / latency. This
    counts GQA-duplicated reads, so it is NOT actual DRAM bandwidth (L2 /
    cache reuse make DRAM traffic unknowable without a profiler).

    A short GPU warmup runs first: absolute latency is sensitive to GPU
    clock/power/runtime state across processes, so warm up before measuring
    to make standalone runs more stable.
    """
    _warm = torch.randn(4096, 4096, device="cuda")
    for _ in range(50):
        _warm = _warm @ _warm
    torch.cuda.synchronize()
    del _warm
    D, BLOCK_SIZE, H, KVH = 128, 256, 16, 8
    SCALE = D**-0.5
    NUM_POOL = 512
    print(f"\nPagedAttention decode (bf16): flash-attn vs Triton")
    print(
        f"{'ctx':>6} {'batch':>6} {'fa2':>9} {'triton32':>9} {'triton64':>9} {'FA2/T32':>7} {'FA2/T64':>7} {'logBW GB/s':>10}"
    )
    print("  relative = FA2_us / Triton_us; >1 means Triton is faster, <1 slower")
    for ctx in (256, 1024, 2048, 4096):
        for batch in (1, 16, 128):
            g = torch.Generator().manual_seed(ctx * 100 + batch)
            q = torch.randn(batch, H, D, device="cuda", dtype=DTYPE)
            k_cache = torch.randn(
                NUM_POOL, BLOCK_SIZE, KVH, D, device="cuda", dtype=DTYPE
            )
            v_cache = torch.randn(
                NUM_POOL, BLOCK_SIZE, KVH, D, device="cuda", dtype=DTYPE
            )
            ctx_lens = torch.full((batch,), ctx, dtype=torch.int32, device="cuda")
            max_b = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
            ids = [
                torch.randperm(NUM_POOL, generator=g)[:max_b].tolist()
                for _ in range(batch)
            ]
            bt = torch.full((batch, max_b), -1, dtype=torch.int32, device="cuda")
            for i, row in enumerate(ids):
                bt[i, :max_b] = torch.tensor(row, dtype=torch.int32, device="cuda")

            def fa():
                return flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=ctx_lens,
                    block_table=bt,
                    softmax_scale=SCALE,
                    causal=True,
                )

            def tr(bn):
                return paged_attention(
                    q, k_cache, v_cache, bt, ctx_lens, H, KVH, SCALE, block_n=bn
                )

            fa()
            tr(32)
            tr(64)
            torch.cuda.synchronize()
            fa_us = triton.testing.do_bench(fa, warmup=WARMUP, rep=REP) * 1000
            t32_us = (
                triton.testing.do_bench(lambda: tr(32), warmup=WARMUP, rep=REP) * 1000
            )
            t64_us = (
                triton.testing.do_bench(lambda: tr(64), warmup=WARMUP, rep=REP) * 1000
            )
            # Logical K+V bytes: per (seq, query head) program, ctx * 2 * D * 2 bytes.
            logical_bytes = batch * H * ctx * 2 * D * 2
            log_bw = logical_bytes / (fa_us * 1e-6) / 1e9
            print(
                f"{ctx:>6} {batch:>6} {fa_us:9.2f} {t32_us:9.2f} {t64_us:9.2f} {fa_us / t32_us:6.2f} {fa_us / t64_us:6.2f} {log_bw:10.1f}"
            )


def bench_flash_attention():
    """Triton FlashAttention (Phase 4 v1) vs flash_attn_varlen_func.

    Rows: single-seq L, varlen batch, and a ragged batch [64,128,...,4096]
    that exposes the T_max grid waste (early-exit programs). Timing is
    triton.testing.do_bench, mean microseconds. The TFLOP/s column is
    logical causal-attention FLOPs only (2 * D * H * sum L*(L+1)), not
    hardware utilization.
    """
    from flash_attn import flash_attn_varlen_func

    from nanovllm.layers.attention import flash_attention

    D, H, KVH, SCALE = 128, 16, 8, 128**-0.5
    lengths_cases = [
        ("L=128", [128]),
        ("L=256", [256]),
        ("L=512", [512]),
        ("L=1024", [1024]),
        ("L=2048", [2048]),
        ("L=4096", [4096]),
        ("varlen", [64, 128, 512, 1024, 2048, 4096]),
        ("ragged", [64, 128, 256, 512, 1024, 4096]),
    ]
    print(
        f"{'case':>8} {'tokens':>7} {'FA2':>10} {'C1 64x64':>10} {'C2 32x64':>10} {'FA2/C1':>7} {'FA2/C2':>7} {'FA2 TF/s':>9}"
    )
    print(
        "  relative = FA2_us / custom_us; >1 means custom Triton is faster, <1 slower"
    )
    for label, lengths in lengths_cases:
        L = sum(lengths)
        q = torch.randn(L, H, D, device="cuda", dtype=DTYPE)
        k = torch.randn(L, KVH, D, device="cuda", dtype=DTYPE)
        v = torch.randn(L, KVH, D, device="cuda", dtype=DTYPE)
        cu = torch.tensor([0] + lengths, dtype=torch.int32, device="cuda")
        cu = torch.cumsum(cu, dim=0).to(torch.int32)

        def fa():
            max_seqlen = max(lengths)
            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=SCALE,
                causal=True,
            )

        def c1():
            return flash_attention(
                q, k, v, cu, SCALE, H, KVH, block_m=64, max_seqlen=max(lengths)
            )

        def c2():
            return flash_attention(
                q, k, v, cu, SCALE, H, KVH, block_m=32, max_seqlen=max(lengths)
            )

        fa()
        c1()
        c2()
        torch.cuda.synchronize()
        fa_us = triton.testing.do_bench(fa, warmup=WARMUP, rep=REP) * 1000
        c1_us = triton.testing.do_bench(c1, warmup=WARMUP, rep=REP) * 1000
        c2_us = triton.testing.do_bench(c2, warmup=WARMUP, rep=REP) * 1000
        # Logical causal-attention FLOPs: 2 * D * H * sum L_s * (L_s + 1).
        flops = 2 * D * H * sum(ls * (ls + 1) for ls in lengths)
        tf = flops / (fa_us * 1e-6) / 1e12
        print(
            f"{label:>8} {L:>7} {fa_us:10.2f} {c1_us:10.2f} {c2_us:10.2f} "
            f"{fa_us / c1_us:7.2f} {fa_us / c2_us:7.2f} {tf:8.2f}"
        )


def main():
    torch.manual_seed(0)
    bench_silu()
    bench_rms()
    bench_paged_attention()
    bench_flash_attention()


if __name__ == "__main__":
    main()
