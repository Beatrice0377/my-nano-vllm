from collections import deque
from dataclasses import dataclass, field

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


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

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

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
            seq = self.waiting[0]
            num_cached_blocks = 0
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if (
                remaining_budget < num_tokens and prefill_seqs
            ):  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining_budget)
            remaining_budget -= seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
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
