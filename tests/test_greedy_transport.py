from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/ssd'))
from ssd.utils.verify import verify
from ssd.engine.speculator_async import SpeculatorAsync
from ssd.engine.greedy_draft_runner import GreedyDraftRunner
from ssd.engine.handoff_state import LiveTree, Registry, Work


@pytest.mark.parametrize('k',[1,4,7,10])
def test_ids_only_acceptance_and_recovery_equal_original_for_every_rejection(k):
    pred=torch.arange(k+1).remainder(5).expand(k+1,-1)
    p=torch.full((k+1,k+1,6),-20.);p.scatter_(2,pred.unsqueeze(-1),20.)
    draft=pred[:,:k].clone()
    for r in range(k):draft[r,r]=(draft[r,r]+1)%6
    tokens=torch.cat((torch.full((k+1,1),5),draft),1);temp=torch.zeros(k+1)
    old=verify(p,torch.randn(k+1,k,6),tokens,temp,temp,jit_speculate=True)
    new=verify(p,None,tokens,temp,temp,jit_speculate=True)
    assert old==new


def test_stochastic_verification_fails_closed_without_draft_scores():
    with pytest.raises(ValueError,match='required'):
        verify(torch.ones(1,3,5),None,torch.tensor([[1,2,3]]),torch.ones(1),torch.ones(1),jit_speculate=True)


def test_stochastic_request_is_rejected_before_engine_enqueue():
    from ssd.engine.llm_engine import LLMEngine
    from ssd.sampling_params import SamplingParams
    engine=SimpleNamespace(config=SimpleNamespace(greedy_transport=True))
    with pytest.raises(ValueError,match='stochastic'):
        LLMEngine.add_request(engine,[1,2],SamplingParams(temperature=.7))


def test_ids_protocol_does_not_allocate_vocabulary_buffer():
    r=SpeculatorAsync.__new__(SpeculatorAsync)
    r.device=torch.device('cpu');r.K=7;r.async_fan_out=3;r.max_blocks=16
    r.publication_metadata=True;r.greedy_transport=True;r.vocab_size=151936;r.draft_dtype=torch.bfloat16
    r._alloc_handshake_bufs(1)
    assert r._logits_q is None
    assert r._fused_response.numel()==13


def test_live_full_response_bypasses_copy_and_partial_response_has_independent_storage(monkeypatch):
    import ssd.engine.greedy_draft_runner as module
    monkeypatch.setattr(module,'record_stream_tree',lambda *a:None)
    class E:
        def __init__(self,d):self.done=d
        def query(self):return self.done
        def synchronize(self):self.done=True
    r=GreedyDraftRunner.__new__(GreedyDraftRunner)
    r.device=torch.device('cpu');r._handoff_stream=None
    r.config=SimpleNamespace(early_publish_tokens=4,speculate_k=7,handoff_bypass_ready=True,handoff_trace=False)
    r.publication_feedback=None
    tokens=torch.arange(14).reshape(2,7)
    events=[None]+[E(True) for _ in range(7)]
    t=LiveTree(1,2,7,{'valid_length':7},tokens,torch.tensor([[42,0,8],[42,1,9]]),torch.tensor([51,52]),events)
    for n in range(1,8):t.submit(n)
    t.finish_submission();r.registry=Registry();r.registry.install(t)
    w=Work(0,3,1,2,packet=dict(keys=torch.tensor([[42,0,8]]),num_tokens=torch.tensor([51])))
    out,hits,n,_=r._cached(w)
    assert n==7 and out.data_ptr()==tokens.data_ptr() and t.trace['full_ready_bypass']
    t.claimed=False;events[5].done=events[6].done=events[7].done=False
    out,hits,n,_=r._cached(w)
    assert n==4 and out.data_ptr()!=tokens.data_ptr()
    tokens.fill_(-1)
    assert out[0,:4].tolist()==[0,1,2,3] and out[0,4:].tolist()==[0,0,0]


def test_stale_context_or_parent_outside_valid_prefix_never_hits(monkeypatch):
    import ssd.engine.greedy_draft_runner as module
    monkeypatch.setattr(module,'record_stream_tree',lambda *a:None)
    class E:
        def query(self):return True
        def synchronize(self):pass
    r=GreedyDraftRunner.__new__(GreedyDraftRunner);r.device=torch.device('cpu')
    r.config=SimpleNamespace(early_publish_tokens=4,speculate_k=7,handoff_bypass_ready=True,handoff_trace=False)
    r.publication_feedback=None
    t=LiveTree(1,2,7,{'valid_length':1},torch.zeros(1,7,dtype=torch.int64),torch.tensor([[42,2,8]]),torch.tensor([53]),[None]+[E() for _ in range(7)])
    for n in range(1,8):t.submit(n)
    r.registry=Registry();r.registry.install(t)
    assert r._cached(Work(0,3,1,2,packet=dict(keys=torch.tensor([[42,2,8]]),num_tokens=torch.tensor([53])))) is None
