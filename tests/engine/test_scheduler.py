from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler, ScheduleOutput
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_scheduler(max_num_seqs=512, max_num_batched_tokens=16384, num_blocks=4096):
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=-1,
        kvcache_block_size=256,
        num_kvcache_blocks=num_blocks,
    )
    return Scheduler(config)


def make_seq(num_tokens, max_tokens=64):
    return Sequence([0] * num_tokens, SamplingParams(max_tokens=max_tokens))


def drain_prefill(scheduler):
    """Schedule + postprocess until no prefill work remains."""
    while True:
        out = scheduler.schedule()
        if not out.prefill_seqs:
            return out
        scheduler.postprocess(out.prefill_seqs, [0] * len(out.prefill_seqs))


def test_pure_decode_preserves_running_order_and_membership():
    scheduler = make_scheduler()
    seqs = [make_seq(100) for _ in range(3)]
    for s in seqs:
        scheduler.add(s)
    drain_prefill(scheduler)
    assert list(scheduler.running) == seqs

    out = scheduler.schedule()
    assert not out.prefill_seqs
    assert len(out.decode_seqs) == 3
    assert list(scheduler.running) == seqs


def test_decode_with_unscheduled_tail_keeps_original_order():
    scheduler = make_scheduler(max_num_seqs=2)
    seqs = [make_seq(100) for _ in range(4)]
    for s in seqs:
        scheduler.add(s)
    drain_prefill(scheduler)
    assert list(scheduler.running) == seqs[:2]

    out = scheduler.schedule()
    assert [s.seq_id for s in out.decode_seqs] == [s.seq_id for s in seqs[:2]]
    assert list(scheduler.running) == seqs[:2]


def test_decode_count_exceeding_token_budget_stays_within():
    scheduler = make_scheduler(max_num_seqs=64)
    seqs = [make_seq(100) for _ in range(16)]
    for s in seqs:
        scheduler.add(s)
    drain_prefill(scheduler)
    assert len(scheduler.running) == 16

    scheduler.max_num_batched_tokens = 8
    out = scheduler.schedule()
    assert len(out.decode_seqs) == 8
    assert not out.prefill_seqs
    assert sum(s.num_scheduled_tokens for s in out.decode_seqs) == 8
    assert len(scheduler.running) == 16


def test_mixed_decode_and_prefill_share_budget():
    scheduler = make_scheduler()
    dec = [make_seq(100) for _ in range(2)]
    for s in dec:
        scheduler.add(s)
    drain_prefill(scheduler)

    scheduler.max_num_batched_tokens = 16
    pref = make_seq(5)
    scheduler.add(pref)

    out = scheduler.schedule()
    assert [s.seq_id for s in out.decode_seqs] == [s.seq_id for s in dec]
    assert [s.seq_id for s in out.prefill_seqs] == [pref.seq_id]
    total = sum(s.num_scheduled_tokens for s in out.decode_seqs) + sum(
        s.num_scheduled_tokens for s in out.prefill_seqs
    )
    assert total == 2 + 5 <= 16
    assert pref.status == SequenceStatus.RUNNING
    assert pref not in scheduler.waiting


def test_chunked_prefill_stays_in_waiting():
    scheduler = make_scheduler(max_num_batched_tokens=4)
    seq = make_seq(10)
    scheduler.add(seq)

    out = scheduler.schedule()
    assert not out.decode_seqs
    assert [s.seq_id for s in out.prefill_seqs] == [seq.seq_id]
    assert seq.num_scheduled_tokens == 4
    assert seq.status == SequenceStatus.WAITING
    assert seq in scheduler.waiting
    assert seq.block_table  # allocated

    scheduler.postprocess(out.prefill_seqs, [0])
    assert seq.num_cached_tokens == 4
    assert seq.status == SequenceStatus.WAITING
    assert seq in scheduler.waiting

    out2 = scheduler.schedule()  # second chunk: 4 more
    assert seq.num_scheduled_tokens == 4
    scheduler.postprocess(out2.prefill_seqs, [0])
    assert seq.num_cached_tokens == 8
    assert seq.status == SequenceStatus.WAITING

    out3 = scheduler.schedule()  # final chunk: 2 tokens -> completes
    assert seq.num_scheduled_tokens == 2
    assert seq.status == SequenceStatus.RUNNING
    assert seq not in scheduler.waiting
    assert seq in scheduler.running


def test_prefill_skipped_when_decode_consumed_all_budget():
    scheduler = make_scheduler()
    dec = [make_seq(100) for _ in range(2)]
    for s in dec:
        scheduler.add(s)
    drain_prefill(scheduler)

    scheduler.max_num_batched_tokens = 2
    pref = make_seq(100)
    scheduler.add(pref)

    out = scheduler.schedule()
    assert len(out.decode_seqs) == 2
    assert not out.prefill_seqs
    assert pref.status == SequenceStatus.WAITING


def test_partial_block_not_added_to_prefix_hash():
    scheduler = make_scheduler()
    seq = make_seq(300)  # 1 full block (256) + 1 partial block (44)
    scheduler.add(seq)
    out = scheduler.schedule()
    scheduler.postprocess(out.prefill_seqs, [0])
    assert seq.status == SequenceStatus.RUNNING

    bm = scheduler.block_manager
    hashes = set(bm.hash_to_block_id)
    assert len(hashes) == 1  # only the full block entered the cache


def test_prefix_reuse_after_full_blocks():
    scheduler = make_scheduler()
    seq1 = make_seq(512)  # exactly 2 full blocks
    scheduler.add(seq1)
    out = scheduler.schedule()
    scheduler.postprocess(out.prefill_seqs, [0])
    assert seq1.status == SequenceStatus.RUNNING

    seq2 = make_seq(512)
    scheduler.add(seq2)
    out2 = scheduler.schedule()
    assert len(out2.prefill_seqs) == 1
    scheduler.postprocess(out2.prefill_seqs, [0])
    assert seq2.num_cached_tokens == 512  # both blocks reused
