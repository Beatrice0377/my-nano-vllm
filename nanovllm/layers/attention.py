import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )



@triton.jit
def _paged_attention_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    out_ptr,
    scale,
    stride_qn,
    stride_qh,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_bt,
    stride_on,
    stride_oh,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # One program per (sequence, query head). GQA: map the query head to its
    # KV head; two query-head programs share one KV head (accepted for v1).
    seq = tl.program_id(0)
    qh = tl.program_id(1)
    kvh = qh // (NUM_HEADS // NUM_KV_HEADS)

    ctx = tl.load(context_lens_ptr + seq)
    if ctx <= 0:
        return

    q = tl.load(q_ptr + seq * stride_qn + qh * stride_qh + tl.arange(0, D)).to(
        tl.float32
    )

    # Online softmax state (fp32): running max m, denominator l, accumulator acc.
    m = tl.full([1], float("-inf"), tl.float32)
    l = tl.zeros([1], tl.float32)
    acc = tl.zeros([D], tl.float32)

    num_blocks = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    for lb in tl.range(0, num_blocks):
        phys = tl.load(block_tables_ptr + seq * stride_bt + lb)
        for toff in tl.static_range(0, BLOCK_SIZE, BLOCK_N):
            token_offsets = toff + tl.arange(0, BLOCK_N)
            abs_pos = lb * BLOCK_SIZE + token_offsets
            mask = abs_pos < ctx
            k = tl.load(
                k_cache_ptr
                + phys * stride_kb
                + token_offsets[:, None] * stride_kt
                + kvh * stride_kh
                + tl.arange(0, D)[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(q[None, :] * k, axis=1) * scale
            scores = tl.where(mask, scores, float("-inf"))
            tile_max = tl.max(scores, axis=0)
            m_new = tl.maximum(m, tile_max)
            alpha = tl.exp(m - m_new)
            p = tl.exp(scores - m_new)
            v = tl.load(
                v_cache_ptr
                + phys * stride_vb
                + token_offsets[:, None] * stride_vt
                + kvh * stride_vh
                + tl.arange(0, D)[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l = l * alpha + tl.sum(p, axis=0)
            m = m_new

    out = acc / l
    tl.store(
        out_ptr + seq * stride_on + qh * stride_oh + tl.arange(0, D), out.to(DTYPE)
    )


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    block_n: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    """Decode PagedAttention, standalone (not yet wired into Attention.forward).

    q is [N, num_heads, head_dim]; k/v caches are
    [num_blocks, BLOCK_SIZE, num_kv_heads, head_dim] (the nano-vLLM layout);
    block_tables is [N, max_blocks] int32 (padded -1, width only needs to
    cover the actual context lengths); context_lens is [N] int32.
    Matches flash_attn_with_kvcache(q.unsqueeze(1), ...) semantics.
    """
    N, H, D = q.shape
    assert num_heads % num_kv_heads == 0
    assert D == 128
    assert k_cache.shape[1] == 256
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.dtype == k_cache.dtype == v_cache.dtype
    assert block_n in (32, 64)
    DTYPE = tl.float16 if q.dtype == torch.float16 else tl.bfloat16
    out = torch.empty_like(q)
    _paged_attention_kernel[(N, H)](
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        out,
        scale,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        block_tables.stride(0),
        out.stride(0),
        out.stride(1),
        D=D,
        BLOCK_SIZE=256,
        BLOCK_N=block_n,
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        DTYPE=DTYPE,
        num_warps=num_warps,
    )
    return out



@triton.jit
def _flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    cu_seqlens_q_ptr,
    scale,
    stride_qn,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_on,
    stride_oh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # One program per (sequence, query head, query tile) on the non-prefix
    # (dense) prefill path: q/k/v are fresh [total_tokens, H|KVH, D] tensors.
    seq = tl.program_id(0)
    qh = tl.program_id(1)
    tile = tl.program_id(2)
    kvh = qh // (NUM_HEADS // NUM_KV_HEADS)

    q_start = tl.load(cu_seqlens_q_ptr + seq)
    seq_len = tl.load(cu_seqlens_q_ptr + seq + 1) - q_start

    q_local_start = tile * BLOCK_M
    if q_local_start >= seq_len:
        return
    q_local_end = tl.minimum(q_local_start + BLOCK_M, seq_len)

    q_idx = q_local_start + tl.arange(0, BLOCK_M)
    q_mask = q_idx < seq_len
    q = tl.load(
        q_ptr
        + (q_start + q_idx)[:, None] * stride_qn
        + qh * stride_qh
        + tl.arange(0, D)[None, :],
        mask=q_mask[:, None],
        other=0.0,
    )

    # Causal pruning: tile's last valid query position is q_local_end - 1, so
    # only K/V positions in [0, q_local_end) can be attended.
    num_k_tiles = (q_local_end + BLOCK_N - 1) // BLOCK_N
    m = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    for kt in range(num_k_tiles):
        k_local_start = kt * BLOCK_N
        k_idx = k_local_start + tl.arange(0, BLOCK_N)
        k_mask = k_idx < q_local_end
        k = tl.load(
            k_ptr
            + (q_start + k_idx)[:, None] * stride_kn
            + kvh * stride_kh
            + tl.arange(0, D)[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr
            + (q_start + k_idx)[:, None] * stride_vn
            + kvh * stride_vh
            + tl.arange(0, D)[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )
        # tl.dot inputs stay in storage dtype (fp16/bf16); fp32 accumulation.
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        # Causal: k_position <= q_position (local positions), plus k < q_local_end.
        causal = (k_idx[None, :] <= q_idx[:, None]) & k_mask[None, :]
        scores = tl.where(causal, scores, float("-inf"))
        # Invalid query rows (q == 0) yield scores == 0, never all -inf, so the
        # online-softmax state stays finite (no exp(-inf - -inf) NaN).
        tile_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m, tile_max)
        alpha = tl.exp(m - m_new)
        # m_new[:, None]: broadcast along the query axis. Without [:, None] the
        # 1D tensor aligns with the last dim (BLOCK_N) and corrupts every row.
        p = tl.exp(scores - m_new[:, None])
        # Rescale the running accumulator before adding the new tile, same as
        # the l update below; tl.dot(acc=...) only accumulates, it does not
        # apply the alpha rescale.
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(DTYPE), v, acc=acc, out_dtype=tl.float32)
        l = l * alpha + tl.sum(p, axis=1)
        m = m_new

    out = acc / l[:, None]
    tl.store(
        out_ptr
        + (q_start + q_idx)[:, None] * stride_on
        + qh * stride_oh
        + tl.arange(0, D)[None, :],
        out.to(DTYPE),
        mask=q_mask[:, None],
    )


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    block_m: int = 64,
    num_warps: int = 4,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    """Prefill FlashAttention, standalone (not yet wired into Attention.forward).

    Dense (non-prefix) prefill only: q is [total_q, num_heads, head_dim],
    k/v are [total_k, num_kv_heads, head_dim] with total_k == total_q, and
    cu_seqlens_q is [batch + 1] int32. Causal, forward-only, fp16/bf16.
    Matches flash_attn_varlen_func(...) semantics for the non-prefix case.

    max_seqlen is an optional host-side value; deriving it from the GPU tensor
    via .max() forces a device-to-host sync on every call, which dominates
    small-batch latency. Callers that know the batch's max sequence length
    (e.g. the scheduler metadata path) should pass it.
    """
    total_q, H, D = q.shape
    assert num_heads % num_kv_heads == 0
    assert D == 128
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.dtype == k.dtype == v.dtype
    assert k.shape[0] == total_q and v.shape[0] == total_q
    assert block_m in (32, 64)
    batch = cu_seqlens_q.numel() - 1
    if max_seqlen is None:
        max_seqlen = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max())
    t_max = (max_seqlen + block_m - 1) // block_m
    DTYPE = tl.float16 if q.dtype == torch.float16 else tl.bfloat16
    out = torch.empty_like(q)
    _flash_attention_kernel[(batch, H, t_max)](
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        scale,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=64,
        D=D,
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        DTYPE=DTYPE,
        num_warps=num_warps,
    )
    return out



class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
