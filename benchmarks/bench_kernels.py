"""Microbenchmarks: in-repo torch.compile baselines vs Triton fused kernels.

Ops:
  - SiluAndMul        (MLP gate*up, width 2*3072)
  - RMSNorm           (hidden 1024)
  - Add + RMSNorm     (hidden 1024)
  - q/k per-head norm (hidden 128, Qwen3Attention QK-norm path)

Usage: .venv/bin/python benchmarks/bench_kernels.py
Timing via triton.testing.do_bench (CUDA events, warmup + rep). Results are
mean microseconds over the rep window. Compilation (torch.compile + Triton
JIT) is triggered once per shape outside the timed region.
"""

import torch
import triton.testing

from nanovllm.layers.activation import SiluAndMul, TritonSiluAndMul
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
    print(f"\nRMSNorm (torch.compile) vs Triton, dtype={DTYPE}")
    for shape in RMS_SHAPES:
        bench_row("rms", shape, lambda x: baseline.rms_forward(x), lambda x: kernel(x))
    print(f"\nAdd+RMSNorm (torch.compile) vs Triton, dtype={DTYPE}")
    for shape in RMS_SHAPES:
        r = torch.randn(*shape, device="cuda", dtype=DTYPE)
        bench_row(
            "add_rms",
            shape,
            lambda x: baseline.add_rms_forward(x, r),
            lambda x: kernel(x, r),
        )
    print(f"\nQK-norm [M,16,128] (torch.compile) vs Triton, dtype={DTYPE}")
    qk_baseline = RMSNorm(128, eps=1e-6).cuda()
    qk_kernel = TritonRMSNorm(128, eps=1e-6).cuda()
    for shape in QK_SHAPES:
        bench_row(
            "qk",
            shape,
            lambda x: qk_baseline.rms_forward(x),
            lambda x: qk_kernel(x),
        )


def main():
    torch.manual_seed(0)
    bench_silu()
    bench_rms()


if __name__ == "__main__":
    main()
