from collections import deque
from dataclasses import dataclass, field

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

# Prefix-cache affinity scheduling constants (Phase 6).
AFFINITY_WINDOW = 8  # bounded waiting-queue scan window
MIN_CACHED_BLOCKS = 2  # only reorder for a clearly reusable prefix (>= 2 blocks)
MAX_BYPASS = 8  # a FIFO head may be bypassed at most this many times


@dataclass(slots=True)
class ScheduleOutput:
    decode_seqs: list[Sequence] = field(default_factory=list)
    prefill_seqs: list[Sequence] = field(default_factory=list)

    @property
    def completed_prefill_seqs(self) -> list[Sequence]:
        return [
            seq
            for seq in self.prefill_seqs
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens
        ]


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # Scalar FIFO-head aging: track the current head and how many
        # consecutive scheduling opportunities cache affinity has bypassed it.
        self.affinity_head_seq_id: int | None = None
        self.affinity_head_bypasses = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _select_prefill(self) -> Sequence | None:
        """Pick the next prefill candidate, reordered by cache affinity.

        Returns None when the FIFO head cannot be allocated right now; the
        caller breaks, preserving Phase 5 memory-pressure semantics: affinity
        may reorder among feasible requests but never jump over an infeasible
        head. A request only becomes eligible once its 512-token shared prefix
        (2 full blocks) is already cached.

        Read-only with respect to ``waiting``: it returns the candidate, the
        caller decides whether the candidate is actually scheduled (budget
        check, queue mutation, aging update). Aging state is the only thing
        mutated here (head-change reset, MAX_BYPASS force), so a budget-limited
        round that never schedules anyone does not advance aging.
        """
        head = self.waiting[0]
        # In-progress chunked prefill always continues: never interleave a new
        # affinity candidate in front of it.
        if head.block_table:
            return head
        # Reset aging whenever the FIFO head changes.
        if head.seq_id != self.affinity_head_seq_id:
            self.affinity_head_seq_id = head.seq_id
            self.affinity_head_bypasses = 0
        head_cached, head_required = self.block_manager.probe_allocate(head)
        if head_required > len(self.block_manager.free_block_ids):
            return None
        # Aging guard: once the head has been bypassed MAX_BYPASS times,
        # force it regardless of cache affinity.
        if self.affinity_head_bypasses >= MAX_BYPASS:
            return head
        best, best_score = head, head_cached
        for i in range(1, min(AFFINITY_WINDOW, len(self.waiting))):
            cand = self.waiting[i]
            if cand.block_table:
                continue
            cached, required = self.block_manager.probe_allocate(cand)
            if required > len(self.block_manager.free_block_ids):
                continue
            if cached > best_score:
                best, best_score = cand, cached
        if best is not head and best_score >= MIN_CACHED_BLOCKS:
            return best
        return head

    def schedule(self) -> ScheduleOutput:
        decode_seqs = []
        prefill_seqs = []
        remaining_budget = self.max_num_batched_tokens

        # decode first: temporarily pop from running, run can_append/preemption,
        # then restore decode_seqs to running in their original order.
        while (
            self.running
            and len(decode_seqs) + len(prefill_seqs) < self.max_num_seqs
            and remaining_budget > 0
        ):
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                decode_seqs.append(seq)
                remaining_budget -= 1
        self.running.extendleft(reversed(decode_seqs))

        # prefill with the remaining shared token budget
        while (
            self.waiting
            and len(decode_seqs) + len(prefill_seqs) < self.max_num_seqs
            and remaining_budget > 0
        ):
            seq = self._select_prefill()
            if seq is None:
                break
            reordered = seq is not self.waiting[0]
            num_cached_blocks = 0
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                assert num_cached_blocks != -1  # probe said feasible
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if (
                remaining_budget < num_tokens and prefill_seqs
            ):  # only allow chunked prefill for the first seq
                # Nothing was scheduled this round: leave ``waiting`` and
                # aging untouched, otherwise the candidate would re-enter at
                # the head and look like a head change (resetting aging for a
                # bypass that never happened).
                break
            if reordered:
                self.waiting.remove(seq)
                self.affinity_head_bypasses += 1
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining_budget)
            remaining_budget -= seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                if not reordered:
                    self.waiting.popleft()
                self.running.append(seq)
            elif reordered:
                # partial chunk: move it back to the head so it continues
                # next round before any new affinity candidate.
                self.waiting.appendleft(seq)
            prefill_seqs.append(seq)

        return ScheduleOutput(decode_seqs, prefill_seqs)

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        token_idx = 0
        for seq in seqs:
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            token_id = token_ids[token_idx]
            token_idx += 1
            seq.append_token(token_id)
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        assert token_idx == len(token_ids)
