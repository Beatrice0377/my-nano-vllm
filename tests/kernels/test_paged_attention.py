"""PagedAttention decode correctness: explicit paged PyTorch reference <-> Triton <-> flash-attn.

The PyTorch reference independently re-derives the paged addressing
(logical block -> block_table -> physical block) and the GQA head mapping,
so it validates addressing rather than just kernel parity.
"""

import torch
import pytest
from flash_attn import flash_attn_with_kvcache

from nanovllm.layers.attention import paged_attention

D = 128
BLOCK_SIZE = 256
NUM_HEADS, NUM_KV_HEADS = 16, 8
SCALE = D**-0.5
DTYPES = [torch.float16, torch.bfloat16]
# Context lengths covering block boundaries (BLOCK_SIZE=256).
CTX_LENS = [1, 255, 256, 257, 512, 1024, 2048, 4096]
NUM_BLOCKS_POOL = 512
MAX_BLOCKS = 16  # ceil(4096 / 256)
RTOL, ATOL = 1e-2, 1e-2


def paged_attention_reference(q, k_cache, v_cache, block_tables, context_lens, scale):
    """Explicit per-sequence gather: logical -> physical via block_table.

    Test-only; deliberately kept simple and unoptimized.
    """
    N, H, _ = q.shape
    group = H // NUM_KV_HEADS
    out = torch.empty_like(q)
    for i in range(N):
        ctx = int(context_lens[i])
        positions = torch.arange(ctx, device=q.device)
        logical_blocks = positions // BLOCK_SIZE
        block_offsets = positions % BLOCK_SIZE
        physical_blocks = block_tables[i][logical_blocks]
        for h in range(H):
            kvh = h // group
            k = k_cache[physical_blocks, block_offsets, kvh].float()
            v = v_cache[physical_blocks, block_offsets, kvh].float()
            scores = q[i, h].float() @ k.T * scale
            p = torch.softmax(scores, dim=-1)
            out[i, h] = p @ v
    return out


def flash_decode(q, k_cache, v_cache, block_tables, context_lens):
    return flash_attn_with_kvcache(
        q.unsqueeze(1),
        k_cache,
        v_cache,
        cache_seqlens=context_lens,
        block_table=block_tables,
        softmax_scale=SCALE,
        causal=True,
    ).squeeze(1)


def make_cache(dtype, num_blocks=NUM_BLOCKS_POOL):
    torch.manual_seed(0)
    k_cache = torch.randn(
        num_blocks, BLOCK_SIZE, NUM_KV_HEADS, D, device="cuda", dtype=dtype
    )
    v_cache = torch.randn(
        num_blocks, BLOCK_SIZE, NUM_KV_HEADS, D, device="cuda", dtype=dtype
    )
    return k_cache, v_cache


def make_block_tables(ctx_lens, physical_ids):
    """physical_ids[seq] = list of physical block ids for that seq's logical blocks."""
    widths = [(ctx + BLOCK_SIZE - 1) // BLOCK_SIZE for ctx in ctx_lens]
    max_w = max(widths)
    bt = torch.full((len(ctx_lens), max_w), -1, dtype=torch.int32, device="cuda")
    for i, w in enumerate(widths):
        bt[i, :w] = torch.tensor(physical_ids[i], dtype=torch.int32, device="cuda")
    return bt


def assert_close(triton_out, ref, dtype, tag):
    a = triton_out.float()
    b = ref.float()
    max_abs = (a - b).abs().max().item()
    max_rel = ((a - b).abs() / (b.abs() + 1e-8)).max().item()
    torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)
    return max_abs, max_rel


@pytest.mark.parametrize("dtype", DTYPES)
def test_deterministic_noncontiguous_blocks(dtype):
    """Deterministic physical mapping: logical 0..3 -> physical 7,2,11,4."""
    k_cache, v_cache = make_cache(dtype)
    ctx = 4 * BLOCK_SIZE  # 1024, four full blocks
    q = torch.randn(1, NUM_HEADS, D, device="cuda", dtype=dtype)
    context_lens = torch.tensor([ctx], dtype=torch.int32, device="cuda")
    block_tables = make_block_tables([ctx], [[7, 2, 11, 4]])
    ref = paged_attention_reference(
        q, k_cache, v_cache, block_tables, context_lens, SCALE
    )
    triton_out = paged_attention(
        q, k_cache, v_cache, block_tables, context_lens, NUM_HEADS, NUM_KV_HEADS, SCALE
    )
    fa_out = flash_decode(q, k_cache, v_cache, block_tables, context_lens)
    assert_close(triton_out, ref, dtype, "deterministic/triton-vs-ref")
    assert_close(triton_out, fa_out, dtype, "deterministic/triton-vs-fa2")


@pytest.mark.parametrize("ctx", CTX_LENS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_block_boundary_contexts(ctx, dtype):
    """Block boundaries incl. partial last block, batch=1."""
    k_cache, v_cache = make_cache(dtype)
    q = torch.randn(1, NUM_HEADS, D, device="cuda", dtype=dtype)
    context_lens = torch.tensor([ctx], dtype=torch.int32, device="cuda")
    # Random (but reproducible) physical ids: non-contiguous, non-sequential.
    g = torch.Generator().manual_seed(ctx)
    ids = torch.randperm(NUM_BLOCKS_POOL, generator=g)[
        : (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    ].tolist()
    block_tables = make_block_tables([ctx], [ids])
    ref = paged_attention_reference(
        q, k_cache, v_cache, block_tables, context_lens, SCALE
    )
    triton_out = paged_attention(
        q, k_cache, v_cache, block_tables, context_lens, NUM_HEADS, NUM_KV_HEADS, SCALE
    )
    fa_out = flash_decode(q, k_cache, v_cache, block_tables, context_lens)
    assert_close(triton_out, ref, dtype, f"ctx{ctx}/triton-vs-ref")
    assert_close(triton_out, fa_out, dtype, f"ctx{ctx}/triton-vs-fa2")


def test_permuted_shared_blocks_bf16():
    """Batch of seqs with shuffled, overlapping physical blocks (shared KV content)."""
    dtype = torch.bfloat16
    k_cache, v_cache = make_cache(dtype)
    ctx_lens = [255, 512, 1000, 1, 1024, 333, 4096, 800]
    N = len(ctx_lens)
    q = torch.randn(N, NUM_HEADS, D, device="cuda", dtype=dtype)
    context_lens = torch.tensor(ctx_lens, dtype=torch.int32, device="cuda")
    g = torch.Generator().manual_seed(42)
    ids = []
    for ctx in ctx_lens:
        ids.append(
            torch.randperm(NUM_BLOCKS_POOL, generator=g)[
                : (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
            ].tolist()
        )
    block_tables = make_block_tables(ctx_lens, ids)
    ref = paged_attention_reference(
        q, k_cache, v_cache, block_tables, context_lens, SCALE
    )
    triton_out = paged_attention(
        q, k_cache, v_cache, block_tables, context_lens, NUM_HEADS, NUM_KV_HEADS, SCALE
    )
    fa_out = flash_decode(q, k_cache, v_cache, block_tables, context_lens)
    assert_close(triton_out, ref, dtype, "permuted/triton-vs-ref")
    assert_close(triton_out, fa_out, dtype, "permuted/triton-vs-fa2")


@pytest.mark.parametrize("ctx", [256, 1024, 4096])
def test_large_batch_fa2_parity(ctx):
    """batch=128, Triton vs flash-attn only (PyTorch ref would be too slow)."""
    dtype = torch.bfloat16
    k_cache, v_cache = make_cache(dtype)
    N = 128
    q = torch.randn(N, NUM_HEADS, D, device="cuda", dtype=dtype)
    context_lens = torch.full((N,), ctx, dtype=torch.int32, device="cuda")
    g = torch.Generator().manual_seed(ctx + 7)
    ids = [
        torch.randperm(NUM_BLOCKS_POOL, generator=g)[: ctx // BLOCK_SIZE].tolist()
        for _ in range(N)
    ]
    block_tables = make_block_tables([ctx] * N, ids)
    triton_out = paged_attention(
        q, k_cache, v_cache, block_tables, context_lens, NUM_HEADS, NUM_KV_HEADS, SCALE
    )
    fa_out = flash_decode(q, k_cache, v_cache, block_tables, context_lens)
    assert_close(triton_out, fa_out, dtype, f"batch128/ctx{ctx}/triton-vs-fa2")


def test_block_n_64_config_bf16():
    """The alternative BLOCK_N=64 launch config must give the same result."""
    dtype = torch.bfloat16
    k_cache, v_cache = make_cache(dtype)
    ctx = 1000
    q = torch.randn(3, NUM_HEADS, D, device="cuda", dtype=dtype)
    context_lens = torch.tensor([ctx, 1, 257], dtype=torch.int32, device="cuda")
    ids = [[7, 2, 11, 4], [3], [12, 5]]
    block_tables = make_block_tables([ctx, 1, 257], ids)
    ref = paged_attention_reference(
        q, k_cache, v_cache, block_tables, context_lens, SCALE
    )
    for block_n in (32, 64):
        out = paged_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            NUM_HEADS,
            NUM_KV_HEADS,
            SCALE,
            block_n=block_n,
        )
        assert_close(out, ref, dtype, f"block_n{block_n}/triton-vs-ref")


def test_gqa_head_mapping_bf16():
    """Both query heads sharing one KV head must match the reference."""
    dtype = torch.bfloat16
    k_cache, v_cache = make_cache(dtype)
    q = torch.randn(2, NUM_HEADS, D, device="cuda", dtype=dtype)
    context_lens = torch.tensor([300, 700], dtype=torch.int32, device="cuda")
    ids = [[9, 4], [6, 20, 3]]
    block_tables = make_block_tables([300, 700], ids)
    ref = paged_attention_reference(
        q, k_cache, v_cache, block_tables, context_lens, SCALE
    )
    triton_out = paged_attention(
        q, k_cache, v_cache, block_tables, context_lens, NUM_HEADS, NUM_KV_HEADS, SCALE
    )
    # Group size 2: q head 0 and 1 both map to kv head 0.
    for h in range(NUM_HEADS):
        torch.testing.assert_close(triton_out[0, h], ref[0, h], rtol=RTOL, atol=ATOL)
