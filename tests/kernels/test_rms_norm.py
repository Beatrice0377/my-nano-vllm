"""Correctness tests for the Triton RMSNorm and fused Add+RMSNorm kernels.

Reference is the in-repo baseline `RMSNorm` (torch.compile path). Both
implementations compute the sum/variance in fp32 and use the same
round-to-storage-dtype then fp32-weight-multiply chain, so differences are
limited to fp32 reduction-order ulps (invisible after fp16/bf16 rounding).
rtol=1e-2 / atol=1e-2 leaves a 2-25x margin over one storage ulp (bf16 ~4e-3,
fp16 ~5e-4) while still catching a wrong formula (O(1) errors).
"""

import pytest
import torch

from nanovllm.layers.layernorm import (
    RMSNorm,
    TritonRMSNorm,
    add_rms_norm,
    rms_norm,
)

H_QWEN3 = 1024
SHAPES = [
    (1, H_QWEN3),  # single decode row
    (16, H_QWEN3),  # small decode batch
    (128, H_QWEN3),  # large decode batch
    (512, H_QWEN3),  # CUDA-graph decode batch boundary
    (1024, H_QWEN3),  # medium prefill
    (4096, H_QWEN3),  # prefill-like
]
# Non-power-of-two H exercises the masked load/store path (wrapper supports it).
NON_POW2 = [(333, H_QWEN3), (128, 1000)]

DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES + NON_POW2)
def test_rms_norm(dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype) * 3.0
    ref = RMSNorm(shape[-1], eps=1e-6).cuda()
    expected = ref.rms_forward(x)
    actual = rms_norm(x, ref.weight, ref.eps)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES + NON_POW2)
def test_add_rms_norm(dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    residual = torch.randn(*shape, device="cuda", dtype=dtype)
    ref = RMSNorm(shape[-1], eps=1e-6).cuda()
    exp_normed, exp_residual = ref.add_rms_forward(x, residual)
    act_normed, act_residual = add_rms_norm(x, residual, ref.weight, ref.eps)
    torch.testing.assert_close(act_normed, exp_normed, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(act_residual, exp_residual, rtol=1e-2, atol=1e-2)


# Validated-runtime semantics: ModelRunner builds the model under
# torch.set_default_dtype(hf_config.dtype), so the weight parameter has the
# model dtype (bf16 for Qwen3-0.6B), not fp32. These cases cover that path.
ADD_RMS_BF16_WEIGHT_SHAPES = [
    (1, H_QWEN3),  # single decode row
    (16, H_QWEN3),
    (128, H_QWEN3),
    (512, H_QWEN3),
    (1024, H_QWEN3),
    (4096, H_QWEN3),
]


@pytest.mark.parametrize("shape", ADD_RMS_BF16_WEIGHT_SHAPES)
def test_add_rms_norm_bf16_weight(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    ref = RMSNorm(shape[-1], eps=1e-6).cuda()
    triton_mod = TritonRMSNorm(shape[-1], eps=1e-6).cuda()
    with torch.no_grad():
        bf16_w = torch.ones(shape[-1], device="cuda", dtype=torch.bfloat16)
        ref.weight = torch.nn.Parameter(bf16_w)
        triton_mod.weight = torch.nn.Parameter(bf16_w)
    assert ref.weight.dtype == torch.bfloat16
    exp_normed, exp_residual = ref.add_rms_forward(x, residual)
    act_normed, act_residual = triton_mod(x, residual)
    torch.testing.assert_close(act_normed, exp_normed, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(act_residual, exp_residual, rtol=1e-2, atol=1e-2)


def test_qk_norm_3d():
    # Qwen3Attention q_norm/k_norm input: [M, num_heads, head_dim].
    torch.manual_seed(0)
    x = torch.randn(128, 16, 128, device="cuda", dtype=torch.bfloat16)
    ref = RMSNorm(128, eps=1e-6).cuda()
    expected = ref.rms_forward(x)
    actual = rms_norm(x, ref.weight, ref.eps)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_add_rms_norm_inputs_not_mutated():
    x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
    x_copy = x.clone()
    residual_copy = residual.clone()
    add_rms_norm(x, residual, torch.ones(1024, device="cuda"), 1e-6)
    assert torch.equal(x, x_copy)
    assert torch.equal(residual, residual_copy)


def test_known_value():
    # mean(x^2) = 12.5 -> scale = rsqrt(12.5) = 0.28284
    x = torch.tensor([[3.0, 4.0]], device="cuda", dtype=torch.bfloat16)
    w = torch.ones(2, device="cuda")
    out = rms_norm(x, w, 1e-6)
    assert out[0, 0].item() == pytest.approx(3.0 * 0.28284, rel=1e-2)
    assert out[0, 1].item() == pytest.approx(4.0 * 0.28284, rel=1e-2)


def test_triton_module_matches_baseline_module():
    torch.manual_seed(0)
    x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    ref = RMSNorm(1024, eps=1e-6).cuda()
    triton_mod = TritonRMSNorm(1024, eps=1e-6).cuda()
    exp_normed, exp_residual = ref(x, residual)
    act_normed, act_residual = triton_mod(x, residual)
    torch.testing.assert_close(act_normed, exp_normed, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(act_residual, exp_residual, rtol=1e-2, atol=1e-2)
