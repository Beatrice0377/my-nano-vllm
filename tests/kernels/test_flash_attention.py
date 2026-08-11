"""Correctness tests for Triton FlashAttention prefill (dense, non-prefix).

Three-way check: explicit PyTorch reference <-> Triton <-> flash_attn_varlen_func.

Covers: fp16/bf16, batch=1, ragged batches, tile-boundary lengths,
partial-Q tiles (no NaN), early causal tokens (never see future K),
sequence isolation (varlen indexing), GQA mapping.
"""

import torch
import pytest
from flash_attn import flash_attn_varlen_func

from nanovllm.layers.attention import flash_attention

D = 128
H = 16
KVH = 8
SCALE = D**-0.5
RTOL = ATOL = 1e-2
BLOCK_M_CONFIGS = [64, 32]  # C1, C2
DTYPES = [torch.float16, torch.bfloat16]


def make_inputs(lengths, dtype, seed=0, k_bias=0.0):
    """Build varlen q/k/v + cu_seqlens. k_bias adds a per-token bias to K to
    make cross-sequence leakage loudly visible (sequence isolation tests)."""
    torch.manual_seed(seed)
    total = sum(lengths)
    q = torch.randn(total, H, D, device="cuda", dtype=dtype)
    k = torch.randn(total, KVH, D, device="cuda", dtype=dtype)
    v = torch.randn(total, KVH, D, device="cuda", dtype=dtype)
    cu_seqlens = torch.tensor(
        [0] + torch.cumsum(torch.tensor(lengths), 0).tolist(),
        dtype=torch.int32,
        device="cuda",
    )
    if k_bias != 0.0:
        k = k + k_bias
    return q, k, v, cu_seqlens


def flash_attention_reference(q, k, v, cu_seqlens, scale):
    """Explicit per-sequence PyTorch reference: per q head, gather its GQA KV
    head, fp32 scores = Q @ K.T * scale, causal mask, softmax, out = p @ V.
    Deliberately simple, test-only."""
    total, H, D = q.shape
    KVH = k.shape[1]
    group = H // KVH
    out = torch.empty_like(q)
    for s in range(cu_seqlens.numel() - 1):
        start, end = cu_seqlens[s].item(), cu_seqlens[s + 1].item()
        L = end - start
        q_s = q[start:end].float()
        k_s = k[start:end].float()
        v_s = v[start:end].float()
        causal = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
        for h in range(H):
            kvh = h // group
            scores = q_s[:, h] @ k_s[:, kvh].T * scale
            scores = scores.masked_fill(~causal, float("-inf"))
            p = torch.softmax(scores, dim=-1)
            out[start:end, h] = (p @ v_s[:, kvh]).to(q.dtype)
    return out


def fa2(q, k, v, cu_seqlens):
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=SCALE,
        causal=True,
    )


def assert_close(a, b, label):
    max_abs = (a.float() - b.float()).abs().max().item()
    denom = b.float().abs() + 1e-8
    max_rel = ((a.float() - b.float()).abs() / denom).max().item()
    assert torch.allclose(a.float(), b.float(), rtol=RTOL, atol=ATOL), (
        f"{label}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )
    return max_abs, max_rel


@pytest.mark.parametrize("block_m", BLOCK_M_CONFIGS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_partial_q_tile_no_nan(block_m, dtype):
    """L = BLOCK_M - 1 and BLOCK_M + 1: invalid rows must not produce NaN and
    must not affect valid-row outputs."""
    for L in (block_m - 1, block_m + 1):
        q, k, v, cu = make_inputs([L], dtype, seed=L)
        out = flash_attention(q, k, v, cu, SCALE, H, KVH, block_m=block_m)
        assert torch.isfinite(out).all(), f"L={L} bm={block_m} dtype={dtype}: NaN"
        ref = flash_attention_reference(q, k, v, cu, SCALE)
        assert_close(out, ref, f"partial L={L} bm={block_m}")


@pytest.mark.parametrize("block_m", BLOCK_M_CONFIGS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tile_boundary_lengths(block_m, dtype):
    """Lengths around BLOCK_M and BLOCK_N boundaries (single sequence)."""
    for L in (
        block_m - 1,
        block_m,
        block_m + 1,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        1023,
        1024,
    ):
        q, k, v, cu = make_inputs([L], dtype, seed=L)
        out = flash_attention(q, k, v, cu, SCALE, H, KVH, block_m=block_m)
        ref = flash_attention_reference(q, k, v, cu, SCALE)
        assert_close(out, ref, f"L={L} bm={block_m}")


@pytest.mark.parametrize("dtype", DTYPES)
def test_ragged_batch(dtype):
    """Highly ragged batch: 13 sequences, lengths from 1 to 4096."""
    lengths = [1, 63, 64, 65, 127, 128, 129, 255, 257, 1000, 1024, 2048, 4096]
    q, k, v, cu = make_inputs(lengths, dtype, seed=7)
    out = flash_attention(q, k, v, cu, SCALE, H, KVH, block_m=64)
    ref = flash_attention_reference(q, k, v, cu, SCALE)
    assert_close(out, ref, "ragged 13-seq batch")


@pytest.mark.parametrize("block_m", BLOCK_M_CONFIGS)
def test_early_causal_tokens_never_see_future(block_m):
    """Poison all K after position 256 with a huge bias: if any early token
    (0/1/63/64/127) attended future K, its output would be dominated by the
    poisoned keys and diverge from the reference."""
    L = 1024
    q, k, v, cu = make_inputs([L], torch.bfloat16, seed=3)
    poison = torch.zeros_like(k)
    poison[256:] = 100.0
    k = k + poison
    out = flash_attention(q, k, v, cu, SCALE, H, KVH, block_m=block_m)
    ref = flash_attention_reference(q, k, v, cu, SCALE)
    for pos in (0, 1, 63, 64, 127):
        assert_close(out[pos : pos + 1], ref[pos : pos + 1], f"early token {pos}")


def test_sequence_isolation():
    """Two sequences with strongly separated K/V distributions (positive vs
    negative bias). Varlen indexing errors would cross-contaminate outputs."""
    q, k, v, cu = make_inputs([512, 512], torch.bfloat16, seed=11, k_bias=0.0)
    # Positive bias for seq 0's KV, negative for seq 1's.
    k[0:512] = k[0:512] + 5.0
    v[0:512] = v[0:512] + 5.0
    k[512:1024] = k[512:1024] - 5.0
    v[512:1024] = v[512:1024] - 5.0
    out = flash_attention(q, k, v, cu, SCALE, H, KVH, block_m=64)
    ref = flash_attention_reference(q, k, v, cu, SCALE)
    assert_close(out, ref, "sequence isolation")


@pytest.mark.parametrize("dtype", DTYPES)
def test_gqa_all_heads(dtype):
    """All 16 query heads must map to the correct KV head (group size 2)."""
    q, k, v, cu = make_inputs([300, 700], dtype, seed=17)
    out = flash_attention(q, k, v, cu, SCALE, H, KVH, block_m=64)
    ref = flash_attention_reference(q, k, v, cu, SCALE)
    assert_close(out, ref, "GQA all heads")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "lengths", [[1], [64], [65], [1024], [64, 128, 512], [1, 63, 65, 129, 1000]]
)
def test_fa2_parity(dtype, lengths):
    """Triton vs flash_attn_varlen_func on the same non-prefix varlen case."""
    q, k, v, cu = make_inputs(lengths, dtype, seed=21)
    out = flash_attention(q, k, v, cu, SCALE, H, KVH, block_m=64)
    ref = fa2(q, k, v, cu)
    assert_close(out, ref, f"FA2 parity {lengths} {dtype}")


def test_batch1_long():
    """Single 4096-token sequence (grid T_max = 64 with BLOCK_M=64)."""
    q, k, v, cu = make_inputs([4096], torch.bfloat16, seed=29)
    out = flash_attention(q, k, v, cu, SCALE, H, KVH, block_m=64)
    ref = flash_attention_reference(q, k, v, cu, SCALE)
    assert_close(out, ref, "batch1 L=4096")
