"""Real CUDA ownership/late-consumer tests; executed by the dedicated queue."""
from pathlib import Path
import sys
import threading

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/ssd'))
from ssd.engine.handoff_state import LiveTree,Registry,Work
from ssd.engine.greedy_draft_runner import GreedyDraftRunner
from types import SimpleNamespace

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Dedicated CUDA run required')


def setup_tree():
    compute=torch.cuda.Stream();handoff=torch.cuda.Stream(priority=-1)
    with torch.cuda.stream(compute):
        tokens=torch.zeros((1,7),device='cuda',dtype=torch.int64)
        keys=torch.tensor([[42,0,8]],device='cuda');expected=torch.tensor([51],device='cuda')
        events=[None]+[torch.cuda.Event() for _ in range(7)]
        tree=LiveTree(1,2,7,{'valid_length':7},tokens,keys,expected,events)
        for i in range(4):tokens[:,i].fill_(i+1);events[i+1].record();tree.submit(i+1)
        for i in range(4,7):
            torch.cuda._sleep(12000000);tokens[:,i].fill_(i+1);events[i+1].record();tree.submit(i+1)
        tree.finish_submission()
    r=GreedyDraftRunner.__new__(GreedyDraftRunner);r.device=torch.device('cuda');r._handoff_stream=handoff
    r.config=SimpleNamespace(early_publish_tokens=4,speculate_k=7,handoff_bypass_ready=True,handoff_trace=False)
    r.publication_feedback=None
    r.registry=Registry();r.registry.install(tree)
    return r,tree,compute,handoff


def test_gpu_completed_prefix_is_copied_while_tail_remains_queued():
    r,t,compute,handoff=setup_tree()
    with torch.cuda.stream(handoff):
        packet=dict(keys=torch.tensor([[42,0,8]],device='cuda'),num_tokens=torch.tensor([51],device='cuda'))
        out,hits,n,owner=r._cached(Work(0,3,1,2,packet=packet))
        handoff.synchronize()
        assert 4<=n<=7 and out[0,:n].tolist()==list(range(1,n+1))
        assert out.data_ptr()!=t.tokens.data_ptr() or n==7
    t.retire(True);compute.synchronize()
    # Reusing producer storage cannot alter a copied shorter snapshot.
    if n<7:
        t.tokens.fill_(99);torch.cuda.synchronize()
        assert out[0,:n].tolist()==list(range(1,n+1))


def test_late_consumer_uses_complete_immutable_view():
    r,t,compute,handoff=setup_tree();compute.synchronize()
    with torch.cuda.stream(handoff):
        packet=dict(keys=torch.tensor([[42,0,8]],device='cuda'),num_tokens=torch.tensor([51],device='cuda'))
        out,hits,n,owner=r._cached(Work(0,3,1,2,packet=packet));handoff.synchronize()
    assert n==7 and out.data_ptr()==t.tokens.data_ptr()
    assert t.trace['full_ready_bypass'] and out[0].tolist()==list(range(1,8))
    t.retire(True)


def test_separate_host_consumer_can_wait_while_producer_submits():
    done=[];r,t,compute,handoff=setup_tree()
    def consume():
        torch.cuda.set_device(0)
        with torch.cuda.stream(handoff):
            packet=dict(keys=torch.tensor([[42,0,8]],device='cuda'),num_tokens=torch.tensor([51],device='cuda'))
            result=r._cached(Work(0,3,1,2,packet=packet));handoff.synchronize();done.append(result[2])
    consumer=threading.Thread(target=consume);consumer.start();consumer.join(10)
    assert not consumer.is_alive() and done and done[0]>=4
    t.retire(True)
