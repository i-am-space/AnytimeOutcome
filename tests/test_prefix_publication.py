from pathlib import Path
from types import SimpleNamespace
import multiprocessing as mp
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'third_party/ssd'))
from ssd.engine.prefix_publication import PrefixCheckpoint, PrefixSnapshot, record_stream_tree
from ssd.engine.draft_runner import DraftRunner


class Device:
    def __init__(self): self.enqueued = self.completed = 0; self.waited = []
    def event(self, **kw):
        device = self
        class Event:
            def record(self, stream): self.depth = device.enqueued
            def query(self): return self.depth <= device.completed
            def synchronize(self):
                device.waited.append(self.depth); device.completed = max(device.completed, self.depth)
        return Event()


def controller(submitted, completed, minimum=4):
    gpu = Device()
    ctl = PrefixCheckpoint(minimum, 10, 2, False, gpu.event)
    ctl.begin(None)
    for depth in range(1, submitted + 1):
        gpu.enqueued = depth; ctl.submitted(depth)
    gpu.completed = completed
    return gpu, ctl


def test_ready_prefix_does_not_wait_for_submitted_tail():
    gpu, ctl = controller(6, 4)
    assert ctl.before_step(6, True)
    assert ctl.final_depth(True) == 4
    assert gpu.waited == [] and not ctl.events[6].query()


def test_waits_for_minimum_only_when_no_valid_prefix_is_ready():
    gpu, ctl = controller(5, 3)
    assert ctl.before_step(5, True)
    assert gpu.waited == [4] and ctl.published == 4
    assert not ctl.events[5].query()


def test_late_request_after_last_launch_still_uses_completed_prefix():
    gpu, ctl = controller(10, 8)
    assert ctl.final_depth(True) == 8 and gpu.waited == []
    gpu, ctl = controller(10, 8)
    assert ctl.final_depth(False) == 10


def test_request_arriving_during_backpressure_is_seen_before_another_launch():
    gpu, ctl = controller(5, 3)
    assert ctl.before_step(5, lambda: bool(gpu.waited))
    assert ctl.published == 4 and gpu.enqueued == 5


def test_retired_source_storage_is_retained_until_tail_drain_finishes():
    r = DraftRunner.__new__(DraftRunner)
    storage = object(); r._pending_tree = storage
    r.publication_feedback = mp.get_context('spawn').Array('d', 12)
    class Event:
        def synchronize(self): assert r._pending_tree is storage
    r._tail_event = Event()
    r._drain_published_tail()
    assert r._pending_tree is None and r._tail_event is None
    assert r.publication_feedback[11] >= 0


def cache_runner(device):
    r = DraftRunner.__new__(DraftRunner)
    r.config = SimpleNamespace(speculate_k=7, MQ_LEN=8, use_eagle=False, early_publish_tokens=4, verbose=False)
    r.device = torch.device(device); r.cache_valid_length = 4
    r._fan_idx_hit = torch.arange(8, device=device); r._fan_idx_miss = r._fan_idx_hit
    r.tree_cache_keys = torch.empty((0, 3), device=device, dtype=torch.int64)
    r._handoff_stream = object()
    r._prefix_snapshot = PrefixSnapshot(7, [1]*8, [1]*8, 10, torch.float32, device)
    payload = dict(seq_ids_expanded=torch.full((8,), 42, device=device), rec_flat=torch.arange(8, device=device),
        cache_hits_list=[True], block_tables=torch.zeros(1, 1, device=device, dtype=torch.int32),
        base_num_tokens=torch.tensor([50], device=device), valid_length=4)
    return r, payload


def test_snapshot_has_no_storage_alias_or_unfinished_columns():
    r, payload = cache_runner('cpu')
    tokens = torch.full((8, 7), -99, dtype=torch.int64); tokens[:, :4] = 2
    logits = torch.full((8, 7, 10), float('nan')); logits[:, :4] = 3
    r._populate_tree_cache(payload, tokens, logits, None)
    assert r.tree_cache_tokens.shape == (5, 4)
    assert r.tree_cache_logits.shape == (5, 4, 10)
    tokens.fill_(99); logits.fill_(99)
    assert torch.all(r.tree_cache_tokens == 2) and torch.all(r.tree_cache_logits == 3)


@pytest.mark.parametrize('parent_length,expected_rows', [(1, 2), (4, 5), (7, 8)])
def test_short_snapshot_keeps_only_real_parent_branches_and_expected_context(parent_length, expected_rows):
    r, payload = cache_runner('cpu'); payload['valid_length'] = parent_length
    tokens = torch.full((8, 7), 2, dtype=torch.int64)
    logits = torch.full((8, 7, 10), 3.)
    r._populate_tree_cache(payload, tokens, logits, None)
    assert r.tree_cache_keys.shape == (expected_rows, 3)
    assert r.tree_cache_keys[:, 1].tolist() == list(range(expected_rows))
    assert r.tree_cache_expected_tokens.tolist() == list(range(51, 51 + expected_rows))
    assert r.tree_cache_tokens.shape == (expected_rows, 4)


def test_snapshot_hit_responds_without_draining_tail_but_stale_context_uses_jit():
    r, payload = cache_runner('cpu')
    r.config.jit_speculate = True; r.hf_config = SimpleNamespace(vocab_size=10, hidden_size=4, torch_dtype=torch.float32)
    tokens = torch.full((8, 7), 2, dtype=torch.int64); logits = torch.full((8, 7, 10), 3.)
    r._populate_tree_cache(payload, tokens, logits, None)
    drains = []
    r._tail_event = SimpleNamespace(synchronize=lambda: drains.append('tail'))
    r.publication_feedback = None
    r._pending_tree = object()
    def jit(*args):
        r._drain_published_tail()
        args[2].fill_(4); args[3].fill_(5)
    r.jit_speculate = jit
    req = torch.tensor([[42, 0, 0]])
    out, q, _, hits, _ = r.hit_cache_and_respond(req, 1, 7, torch.tensor([51]),
        torch.zeros(1), torch.zeros(1, 1, dtype=torch.int32))
    assert bool(hits[0]) and r.response_valid_length == 4 and drains == []
    assert torch.all(out[:, :4] == 2) and torch.all(q[:, :4] == 3)
    assert r._pending_tree is not None
    _, _, _, hits, _ = r.hit_cache_and_respond(req, 1, 7, torch.tensor([52]),
        torch.zeros(1), torch.zeros(1, 1, dtype=torch.int32))
    assert not bool(hits[0]) and drains == ['tail'] and r._pending_tree is None
    assert r.response_valid_length == 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA stream safety test')
def test_gpu_snapshot_completes_while_tail_is_still_writing():
    compute = torch.cuda.current_stream(); handoff = torch.cuda.Stream(priority=-1)
    r, payload = cache_runner('cuda'); r._handoff_stream = handoff
    tokens = torch.full((8, 7), 2, device='cuda', dtype=torch.int64)
    logits = torch.full((8, 7, 10), 3., device='cuda')
    tokens[:, 4:].fill_(99); logits[:, 4:].fill_(99)
    tokens[:, :4].fill_(2); logits[:, :4].fill_(3)
    # Warm allocation and indexing kernels before testing actual overlap.
    handoff.wait_stream(compute)
    with torch.cuda.stream(handoff): r._populate_tree_cache(payload, tokens, logits, None)
    torch.cuda.synchronize()
    r.tree_cache_keys = torch.empty((0, 3), device='cuda', dtype=torch.int64)
    ready, tail, sent = torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()
    ready.record(compute); tail.record(compute); sent.record(handoff)
    torch.cuda.synchronize()
    ready.record(compute)
    # A sequence of bounded kernels permits scheduling the handoff kernels
    # between them, just as a real multi-kernel draft step does. One monolithic
    # spin kernel may occupy the device until it exits: streams cannot preempt it.
    for _ in range(64): torch.cuda._sleep(10_000_000)
    # fill_ passes its scalar as a kernel argument; Python slice assignment
    # creates a host scalar tensor/copy that can itself wait on the tail.
    tokens[:, 4:].fill_(99); logits[:, 4:].fill_(99)
    tail.record(compute)
    with torch.cuda.stream(handoff):
        handoff.wait_event(ready)
        record_stream_tree((payload, tokens, logits), handoff)
        r._populate_tree_cache(payload, tokens, logits, None)
        sent.record(handoff)
    sent.synchronize()
    assert not tail.query(), 'Snapshot accidentally serialized behind unfinished tail'
    tail.synchronize()
    compute.wait_event(sent)
    record_stream_tree((r.tree_cache_tokens, r.tree_cache_logits), compute)
    assert torch.all(r.tree_cache_tokens == 2).item()
    assert torch.all(r.tree_cache_logits == 3).item()
    assert torch.all(tokens[:, 4:] == 99).item()
