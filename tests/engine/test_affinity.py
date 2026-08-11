"""Unit tests for Phase 6 prefix-cache affinity scheduling.

Covers block_manager.probe_allocate() (read-only probe) and the scheduler's
_select_prefill() cache-affinity reordering (bounded window, aging guard,
chunked-prefill priority, memory-pressure semantics).
"""

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import (
    Scheduler,
    AFFINITY_WINDOW,
    MAX_BYPASS,
    MIN_CACHED_BLOCKS,
)
from nanovllm.sampling_params import SamplingParams


def make_scheduler(num_blocks=4096, max_batched=16384):
    from types import SimpleNamespace

    config = SimpleNamespace(
        max_num_seqs=512,
        max_num_batched_tokens=max_batched,
        eos=-1,
        kvcache_block_size=256,
        num_kvcache_blocks=num_blocks,
    )
    return Scheduler(config)


def make_seq(num_tokens, token=0, max_tokens=64):
    return Sequence([token] * num_tokens, SamplingParams(max_tokens=max_tokens))


def cache_prefix(bm: BlockManager, num_full_blocks: int, token: int = 0):
    """Manually populate the prefix cache with num_full_blocks full blocks.

    Mirrors the end state of a request that prefilled, completed, and was
    kept running (blocks used + hashed). Avoids driving a seed request
    through schedule()/postprocess(), which would leave it in the running
    queue and let subsequent decode steps consume free blocks.
    """
    h = -1
    for _ in range(num_full_blocks):
        token_ids = [token] * bm.block_size
        h = bm.compute_hash(token_ids, h)
        block_id = bm._allocate_block()
        block = bm.blocks[block_id]
        block.update(h, token_ids)
        bm.hash_to_block_id[h] = block_id
    return h


# ---------------------------------------------------------------------------
# probe_allocate(): read-only prefix probe
# ---------------------------------------------------------------------------


class TestProbeAllocate:
    def test_probe_matches_can_allocate(self):
        sched = make_scheduler(num_blocks=16)
        bm = sched.block_manager
        cache_prefix(bm, 2, token=0)
        # 512 + 44 tokens = 3 blocks; only the first 2 (full) blocks
        # participate in prefix reuse, so cached == 2.
        hot = make_seq(512 + 44, token=0)
        cached, required = bm.probe_allocate(hot)
        assert cached == 2
        assert required == hot.num_blocks - 2  # 2 blocks already used
        assert bm.can_allocate(hot) == 2  # same prefix loop, same result

    def test_probe_matches_can_allocate_without_prefix(self):
        sched = make_scheduler(num_blocks=16)
        bm = sched.block_manager
        cold = make_seq(300, token=1)
        cached, required = bm.probe_allocate(cold)
        assert cached == 0
        assert required == cold.num_blocks
        assert bm.can_allocate(cold) == 0

    def test_probe_does_not_mutate_state(self):
        sched = make_scheduler(num_blocks=16)
        bm = sched.block_manager
        cache_prefix(bm, 2, token=0)
        before = (
            len(bm.free_block_ids),
            len(bm.used_block_ids),
            len(bm.hash_to_block_id),
            bm.cache_lookup_observer,
        )
        seen = []

        def observer(seq, cached, looked_up):
            seen.append((seq.seq_id, cached, looked_up))

        bm.cache_lookup_observer = observer
        hot = make_seq(512 + 44, token=0)
        cached, required = bm.probe_allocate(hot)
        after = (
            len(bm.free_block_ids),
            len(bm.used_block_ids),
            len(bm.hash_to_block_id),
        )
        assert cached == 2
        assert required == 1
        assert before[:3] == after  # no allocation / deallocation
        assert seen == []  # observer is not fired by the probe


# ---------------------------------------------------------------------------
# _select_prefill(): cache-affinity reordering
# ---------------------------------------------------------------------------


class TestAffinitySelect:
    def test_no_cache_hit_fifo_order(self):
        sched = make_scheduler()
        a = make_seq(100, token=0)
        b = make_seq(100, token=0)
        sched.add(a)
        sched.add(b)
        out = sched.schedule()
        # No prefix cache: both scheduled in FIFO order in one call.
        assert [s.seq_id for s in out.prefill_seqs] == [a.seq_id, b.seq_id]

    def test_highest_cached_first(self):
        sched = make_scheduler()
        cache_prefix(sched.block_manager, 2, token=0)
        cold = make_seq(300, token=1)
        hot = make_seq(512 + 44, token=0)
        sched.add(cold)
        sched.add(hot)
        out = sched.schedule()
        # hot (2 cached blocks) is scheduled before cold (0 cached).
        assert [s.seq_id for s in out.prefill_seqs][0] == hot.seq_id
        # Both eventually scheduled this round (budget is generous).
        assert set(s.seq_id for s in out.prefill_seqs) == {cold.seq_id, hot.seq_id}

    def test_score_tie_takes_fifo(self):
        sched = make_scheduler()
        cache_prefix(sched.block_manager, 2, token=0)
        cold1 = make_seq(300, token=1)
        cold2 = make_seq(300, token=2)
        hot1 = make_seq(512 + 44, token=0)
        hot2 = make_seq(512 + 44, token=0)
        sched.add(cold1)
        sched.add(cold2)
        sched.add(hot1)
        sched.add(hot2)
        out = sched.schedule()
        ids = [s.seq_id for s in out.prefill_seqs]
        # Both hot seqs tie at score 2; the older one (hot1) is chosen first,
        # then hot2; the cold seqs follow in FIFO order.
        assert ids.index(hot1.seq_id) < ids.index(hot2.seq_id)
        assert ids.index(hot1.seq_id) < ids.index(cold1.seq_id)
        assert ids.index(cold1.seq_id) < ids.index(cold2.seq_id)

    def test_candidate_outside_window_ignored(self):
        # One candidate per round: budget == 50 (each cold is 50 tokens).
        sched = make_scheduler(max_batched=50)
        cache_prefix(sched.block_manager, 2, token=0)
        colds = [make_seq(50, token=i + 1) for i in range(AFFINITY_WINDOW)]
        # hot sits beyond the window: index 8 == AFFINITY_WINDOW.
        hot = make_seq(512 + 44, token=0)
        for c in colds:
            sched.add(c)
        sched.add(hot)
        # Round 1: head only; hot is outside the scan window -> no reorder.
        out = sched.schedule()
        assert [s.seq_id for s in out.prefill_seqs] == [colds[0].seq_id]
        # Round 2: hot has slid to index 7 (inside the window) -> reordered.
        out = sched.schedule()
        assert [s.seq_id for s in out.prefill_seqs] == [hot.seq_id]

    def test_chunked_prefill_priority(self):
        sched = make_scheduler()
        # A partial-chunk seq (already owns blocks) sits at the head with a
        # high-affinity candidate behind it: the chunk must continue first.
        partial = make_seq(512 + 44, token=0)
        partial.block_table = [0, 1, 2]  # arbitrary; only truthiness matters
        partial.num_cached_tokens = 256
        hot = make_seq(512 + 44, token=0)
        sched.add(partial)
        sched.add(hot)
        out = sched.schedule()
        assert out.prefill_seqs[0] is partial

    def test_head_infeasible_breaks(self):
        sched = make_scheduler(num_blocks=5)
        cache_prefix(sched.block_manager, 2, token=0)  # free = 3
        big = make_seq(900, token=1)  # 4 blocks, 0 cached -> needs 4 free
        hot = make_seq(512 + 44, token=0)  # would be reordered, but...
        sched.add(big)
        sched.add(hot)
        out = sched.schedule()
        # Head cannot be allocated (needs 4 > 3 free): break, no reordering.
        assert out.prefill_seqs == []
        assert sched.waiting[0] is big  # order untouched

    def test_feasible_candidate_skipped_when_head_beats(self):
        sched = make_scheduler()
        # Two candidates both have some prefix; head has the higher score, so
        # no reorder happens (tie/less must not beat head).
        cache_prefix(sched.block_manager, 1, token=0)
        head = make_seq(256 + 44, token=0)  # 1 cached block
        cand = make_seq(256 + 44, token=0)  # 1 cached block (tie)
        sched.add(head)
        sched.add(cand)
        out = sched.schedule()
        assert out.prefill_seqs[0] is head
        assert [s.seq_id for s in out.prefill_seqs][1] == cand.seq_id

    def test_aging_forces_head_after_max_bypass(self):
        # Budget must fit one hot (44 tokens) plus MAX_BYPASS decode rows
        # (1 token each round) so each hot completes in a single round.
        sched = make_scheduler(max_batched=44 + MAX_BYPASS)
        cache_prefix(sched.block_manager, 2, token=0)
        cold = make_seq(300, token=1)
        sched.add(cold)
        hots = [make_seq(512 + 44, token=0) for _ in range(MAX_BYPASS)]
        for h in hots:
            sched.add(h)
        chosen_first = []
        for _ in range(MAX_BYPASS):
            out = sched.schedule()
            chosen_first.append(out.prefill_seqs[0].seq_id)
        # MAX_BYPASS rounds: head (cold) bypassed each time for a hot seq.
        assert all(c == cold.seq_id for c in chosen_first) is False
        assert sched.affinity_head_bypasses == MAX_BYPASS
        # Next schedule: aging guard forces the head.
        out = sched.schedule()
        assert out.prefill_seqs[0] is cold

    def test_aging_resets_on_head_change(self):
        sched = make_scheduler(max_batched=50)
        cache_prefix(sched.block_manager, 2, token=0)
        a = make_seq(300, token=1)
        b = make_seq(300, token=2)
        hot = make_seq(512 + 44, token=0)
        sched.add(a)
        sched.add(b)
        sched.add(hot)
        # a is bypassed once for hot.
        out = sched.schedule()
        assert out.prefill_seqs[0] is hot
        assert sched.affinity_head_bypasses == 1
        assert sched.affinity_head_seq_id == a.seq_id
        # Move b to the head; bypasses must reset on head change.
        sched.waiting.remove(a)
        sched.waiting.appendleft(a)  # keep a, but make b the new head
        sched.waiting.remove(b)
        sched.waiting.appendleft(b)
        assert sched.waiting[0] is b
        sched.schedule()
        assert sched.affinity_head_seq_id == b.seq_id
        assert sched.affinity_head_bypasses == 0

    def test_reordered_seq_removed_and_running(self):
        sched = make_scheduler(max_batched=50)
        cache_prefix(sched.block_manager, 2, token=0)
        cold = make_seq(300, token=1)
        hot = make_seq(512 + 44, token=0)
        sched.add(cold)
        sched.add(hot)
        out = sched.schedule()
        # hot completes this round: 512 cached + 44 scheduled == num_tokens.
        assert out.prefill_seqs[0] is hot
        assert hot.status == SequenceStatus.RUNNING
        assert hot not in sched.waiting
        assert hot in sched.running
        assert cold in sched.waiting

    def test_reordered_partial_chunk_returns_to_head(self):
        sched = make_scheduler(max_batched=100)
        cache_prefix(sched.block_manager, 2, token=0)
        cold = make_seq(200, token=1)
        hot = make_seq(512 + 188, token=0)  # 188 tokens after the 2-block prefix
        sched.add(cold)
        sched.add(hot)
        out = sched.schedule()
        # hot is reordered, then chunked (188 > 100 budget): it must be
        # placed back at the head of waiting so it continues next round.
        assert out.prefill_seqs[0] is hot
        assert hot.num_cached_tokens + hot.num_scheduled_tokens < hot.num_tokens
        # postprocess converts scheduled -> cached; the incomplete chunk
        # consumes no token (like the engine does every step).
        sched.postprocess(out.prefill_seqs, [])
        assert sched.waiting[0] is hot
        # Next round: the chunked seq continues (completing this round) and
        # the budget-limited reorder places it at the head only while
        # incomplete; once done it moves to running and cold is head again.
        out2 = sched.schedule()
        assert out2.prefill_seqs[0] is hot
        assert hot in sched.running
        assert sched.waiting[0] is cold

    def test_reorder_requires_min_cached_blocks(self):
        sched = make_scheduler()
        # Only 1 cached block: below MIN_CACHED_BLOCKS, no reorder.
        cache_prefix(sched.block_manager, 1, token=0)
        cold = make_seq(300, token=1)
        warm = make_seq(256 + 44, token=0)  # 1 cached block
        sched.add(cold)
        sched.add(warm)
        out = sched.schedule()
        assert out.prefill_seqs[0] is cold
        assert sched.affinity_head_bypasses == 0
