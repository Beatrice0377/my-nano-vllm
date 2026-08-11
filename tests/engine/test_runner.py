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


def map_tokens(sampled, output):
    """Mirror of ModelRunner.run()'s token mapping: decode tokens first, then
    completed-prefill tokens; incomplete chunks get a placeholder."""
    nd = len(output.decode_seqs)
    completed = output.completed_prefill_seqs
    token_ids = list(sampled[:nd])
    for seq in output.prefill_seqs:
        if seq in completed:
            token_ids.append(sampled[nd + completed.index(seq)])
        else:
            token_ids.append(seq.last_token)
    return token_ids


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


class TestTokenMapping:
    def test_pure_decode(self):
        out = make_output(3, [])
        assert map_tokens([7, 8, 9], out) == [7, 8, 9]

    def test_pure_prefill_all_completed(self):
        out = make_output(0, [(0, 30, 30), (0, 50, 50)])
        assert map_tokens([111, 222], out) == [111, 222]

    def test_mixed_with_chunked_placeholder(self):
        out = make_output(1, [(0, 30, 30), (80, 15, 100)])
        # sampled = [decode, completed_prefill]; incomplete chunk -> its last_token
        assert map_tokens([11, 22], out) == [11, 22, out.prefill_seqs[1].last_token]

    def test_chunked_first_token_slot(self):
        # 2 decode + 1 complete prefill + 1 chunked; verify positional alignment
        out = make_output(2, [(0, 30, 30), (80, 15, 100)])
        sampled = [1, 2, 3]
        assert map_tokens(sampled, out) == [1, 2, 3, out.prefill_seqs[1].last_token]
