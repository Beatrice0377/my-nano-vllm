"""Correctness tests for the Triton fused silu(gate) * up kernel.

Reference is the current in-repo baseline `SiluAndMul` (torch.compile path).
Tolerance: both implementations compute in fp32; only the storage dtype
differs. fp16 has ~5e-4 and bf16 ~4e-3 relative precision, so rtol=1e-2 /
atol=1e-2 leaves a 2-3x margin while still catching a wrong formula (which
would produce O(1) errors).
"""

import pytest
import torch

from nanovllm.layers.activation import SiluAndMul, TritonSiluAndMul

# Qwen3-0.6B MLP intermediate size -> input width 2 * H.
SHAPES = [
    (1, 2 * 3072),  # single decode row
    (16, 2 * 3072),  # small decode batch
    (128, 2 * 3072),  # large decode batch
    (1024, 2 * 3072),  # small prefill
    (4096, 2 * 3072),  # prefill-like
]
# Shapes where H or M is not a multiple of the kernel block sizes, to
# exercise the masked load/store path (H=1000 is not divisible by 128).
NON_BLOCK_MULTIPLE = [(37, 2000), (333, 6144)]

DTYPES = [torch.float16, torch.bfloat16]

_REF = SiluAndMul()
_KERNEL = TritonSiluAndMul()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES + NON_BLOCK_MULTIPLE)
def test_matches_baseline(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    expected = _REF(x)
    actual = _KERNEL(x)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_known_value():
    # gate = 1.0 at position 0, up = 2.0 at position H: silu(1.0) * 2.0 = 1.46212...
    h = 3072
    x = torch.zeros((1, 2 * h), device="cuda", dtype=torch.bfloat16)
    x[0, 0] = 1.0
    x[0, h] = 2.0
    out = _KERNEL(x)
    assert out.shape == (1, h)
    assert out[0, 0].item() == pytest.approx(1.4621, rel=1e-2)
    assert out[0, 1].item() == pytest.approx(0.0, abs=1e-2)
