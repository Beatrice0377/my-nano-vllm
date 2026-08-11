from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    # num_decode_tokens = number of decode rows at the head of the batch.
    # 0 -> pure prefill, == total rows -> pure decode, otherwise mixed.
    num_decode_tokens: int = 0
    # prefill-only fields (flash_attn_varlen_func)
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # decode-only fields (flash_attn_with_kvcache)
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # prefill block tables (prefix cache only); None for normal prefill
    prefill_block_tables: torch.Tensor | None = None
    # spans all rows (decode rows first, then prefill rows)
    slot_mapping: torch.Tensor | None = None
    # row indices to project at the LM head; None means all rows (pure decode)
    logits_indices: torch.Tensor | None = None


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    num_decode_tokens=0,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    prefill_block_tables=None,
    logits_indices=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        num_decode_tokens=num_decode_tokens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        prefill_block_tables=prefill_block_tables,
        logits_indices=logits_indices,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
