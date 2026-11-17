"""Policy decisions and a simulated asynchronous device exercise safety invariants."""
import multiprocessing as mp
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party/ssd'))
from ssd.engine.publication_policy import PublicationCheckpoint, choose_depth, observe_verification


def test_policy_waits_only_when_extra_tokens_amortize_extra_drafting():
    assert choose_depth(4, 10, .030, .0001, .95) > 4
    assert choose_depth(4, 10, .030, .010, .95) == 4
    assert choose_depth(4, 10, .030, .0001, .10) == 4
    assert choose_depth(7, 7, .030, .001, .95) == 7
    for bad in (float('nan'), float('inf'), 0, -1):
        assert choose_depth(4, 10, bad, .001, .8) == 4


def test_acceptance_estimate_counts_only_the_first_rejection_not_padding():
    f = mp.get_context('spawn').Array('d', 8)
    observe_verification(f, .02, 3, 10)
    assert list(f[:4]) == [.02, 3, 1, 1]
    observe_verification(f, .04, 4, 4)
    assert list(f[:4]) == pytest.approx([.024, 3.2, .8, 2])
    assert f[4:] == [0] * 4


class Device:
    """Host can queue ahead; device progresses only on wait or explicit completion."""
    def __init__(self):
        self.enqueued = self.completed = 0
        self.waits = []
    def event(self, **kw):
        device = self
        class Event:
            def record(self, stream): self.depth = device.enqueued
            def query(self): return self.depth <= device.completed
            def synchronize(self):
                device.waits.append(self.depth)
                device.completed = max(device.completed, self.depth)
            def elapsed_time(self, end): return (end.depth - self.depth) * .1
        return Event()


@pytest.mark.parametrize('arrival', [0, 3, 4, 5, 9, 99])
@pytest.mark.parametrize('window', [1, 2, 3])
def test_queue_is_bounded_and_interruption_never_exposes_live_kv(arrival, window):
    gpu = Device()
    ctl = PublicationCheckpoint(4, 10, window, False, gpu.event)
    ctl.begin(None)
    stopped = False
    for depth in range(10):
        if ctl.before_step(depth, depth >= arrival):
            assert depth >= 4 and gpu.completed == depth
            stopped = True
            break
        gpu.enqueued += 1
        ctl.submitted(depth + 1)
        assert gpu.enqueued - gpu.completed <= window
    assert stopped == (arrival <= 9)
    if not stopped:
        # Cache population still has its own completion barrier in the engine.
        ctl.wait(10)
    assert gpu.completed == gpu.enqueued


def test_already_completed_steps_do_not_block_host_and_events_can_be_reused():
    gpu = Device()
    ctl = PublicationCheckpoint(4, 7, 2, False, gpu.event)
    for _ in range(2):
        gpu.enqueued = gpu.completed = 0
        ctl.begin(None)
        for depth in range(7):
            assert not ctl.before_step(depth, False)
            gpu.enqueued += 1
            ctl.submitted(depth + 1)
            gpu.completed = gpu.enqueued
    assert gpu.waits == []


def test_adaptive_rule_can_extend_prefix_then_stop_at_selected_depth():
    gpu = Device()
    feedback = mp.get_context('spawn').Array('d', [.03, 20, 0, 2, 0, 0, 0, 0])
    ctl = PublicationCheckpoint(2, 10, 2, True, gpu.event, feedback)
    ctl.begin(None)
    for depth in range(10):
        if ctl.before_step(depth, True): break
        gpu.enqueued += 1
        ctl.submitted(depth + 1)
    assert ctl.goal > 2
    assert gpu.enqueued == ctl.goal
    ctl.finish()
    assert feedback[6] == 1 and feedback[7] == 1


def test_cold_adaptive_rule_publishes_minimum_without_inventing_history():
    gpu = Device()
    feedback = mp.get_context('spawn').Array('d', 8)
    ctl = PublicationCheckpoint(2, 10, 2, True, gpu.event, feedback)
    ctl.begin(None)
    for depth in range(2):
        assert not ctl.before_step(depth, True)
        gpu.enqueued += 1
        ctl.submitted(depth + 1)
    assert ctl.before_step(2, True)
    assert ctl.goal == 2
