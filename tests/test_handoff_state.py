import importlib.util
from pathlib import Path
import threading
import time
import sys

import pytest

path = Path(__file__).resolve().parents[1]/'third_party/ssd/ssd/engine/handoff_state.py'
spec = importlib.util.spec_from_file_location('handoff_state_test', path)
state = importlib.util.module_from_spec(spec);sys.modules[spec.name] = state;spec.loader.exec_module(state)


class Event:
    def __init__(self, done=False):self.done=done;self.waited=False
    def query(self):return self.done
    def synchronize(self):self.waited=True;self.done=True


def tree():
    return state.LiveTree(5, 8, 7, {}, object(), object(), object(), [None]+[Event() for _ in range(7)])


def test_consumer_claims_only_ready_prefix_not_submitted_tail():
    t=tree()
    for d in range(1,7):t.submit(d)
    for d in range(1,5):t.events[d].done=True
    assert t.ready(4)==4
    assert not t.events[6].waited
    with pytest.raises(RuntimeError,match='consumed'):t.ready(4)
    with pytest.raises(RuntimeError,match='consumer'):t.retire(False)
    with pytest.raises(RuntimeError,match='submit'):t.retire(True)
    t.finish_submission();t.retire(True)
    assert t.events[6].waited and t.retired


def test_independent_service_waits_for_event_registration():
    t=tree();answer=[]
    consumer=threading.Thread(target=lambda:answer.append(t.ready(4)))
    consumer.start()
    for d in range(1,5):t.submit(d)
    consumer.join(2)
    assert answer==[4] and not consumer.is_alive()


def test_registry_does_not_serve_old_generation_while_next_is_building():
    r=state.Registry();old=tree();r.install(old);r.begin();seen=[]
    consumer=threading.Thread(target=lambda:seen.append(r.get(timeout=2)))
    consumer.start();new=tree();new.generation=6;r.install(new);consumer.join(2)
    assert seen==[new] and old not in seen


def test_failed_service_wakes_the_model_owner():
    w=state.Work(0,2,1,2);w.error=ValueError('broken');w.done.set()
    with pytest.raises(RuntimeError,match='service failed'):w.wait(w.done)


@pytest.mark.parametrize('posted,noticed,delivery,expected',[
    (100,110,150,'delivered_before_full'),
    (250,260,280,'request_after_full'),
    (100,250,280,'request_observed_after_full'),
    (100,110,250,'handoff_finished_after_full'),
])
def test_clock_correlated_diagnostic_classification(posted,noticed,delivery,expected):
    assert state.classify_trace(dict(prefix_ready_ns=80,request_posted_ns=posted,
        request_noticed_ns=noticed,delivery_ns=delivery,full_ready_ns=200))==expected


def test_unsubmitted_tail_is_never_extrapolated():
    assert state.classify_trace(dict(prefix_ready_ns=80,request_posted_ns=90,
        request_noticed_ns=100,delivery_ns=120,full_ready_ns=None))=='no_full_tail_measurement'


def test_complete_transport_requires_only_one_gpu_completion_event():
    last=Event()
    t=state.LiveTree(1,1,7,{},object(),object(),object(),[None]*7+[last])
    for depth in range(1,8):t.submit(depth)
    t.finish_submission()
    assert t.ready(7)==7 and last.waited
    t.retire(True)


def test_clock_uncertainty_does_not_create_a_false_late_delivery():
    assert state.classify_trace(dict(prefix_ready_ns=80,request_posted_ns=100,
        request_noticed_ns=110,delivery_ns=195,full_ready_ns=200,clock_uncertainty_ns=20))=='ambiguous_at_clock_resolution'
