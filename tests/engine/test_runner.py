from nanovllm.engine.scheduler import ScheduleOutput
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def make_seq(num_tokens, max_tokens=10):
    return Sequence([0] * num_tokens, SamplingParams(max_tokens=max_tokens))


def make_output(n_decode, prefill_specs):
    decode_seqs = [make_seq(100) for _ in range(n_decode)]
    prefill_seqs = []
    for cached, scheduled, total in prefill_specs:
        seq = make_seq(total)
        seq.num_cached_tokens = cached
        seq.num_scheduled_tokens = scheduled
        prefill_seqs.append(seq)
    return ScheduleOutput(decode_seqs=decode_seqs, prefill_seqs=prefill_seqs)


class TestCompletedPrefillSeqs:
    def test_mixed_all_completed(self):
        out = make_output(2, [(0, 10, 10), (0, 20, 20)])
        assert len(out.completed_prefill_seqs) == 2

    def test_mixed_some_chunked(self):
        out = make_output(2, [(50, 25, 100), (0, 20, 20)])
        completed = out.completed_prefill_seqs
        assert len(completed) == 1
        assert completed[0].num_tokens == 20

    def test_all_chunked_incomplete(self):
        out = make_output(0, [(40, 40, 100)])
        assert out.completed_prefill_seqs == []

    def test_chunk_boundary_exact(self):
        out = make_output(0, [(80, 20, 100)])
        assert len(out.completed_prefill_seqs) == 1
