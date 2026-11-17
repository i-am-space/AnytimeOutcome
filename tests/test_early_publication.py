"""Publication boundaries, context ownership, and acceptance safety."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/ssd'))
from ssd.engine.draft_runner import DraftRunner
from ssd.engine.verifier import Verifier
from ssd.engine.helpers.speculate_types import SpeculateResult
from ssd.utils.context import reset_context


def runner():
    r=DraftRunner.__new__(DraftRunner)
    r.config=SimpleNamespace(speculate_k=7,MQ_LEN=8,use_eagle=False,early_publish_tokens=4,
                             jit_speculate=True,verbose=False)
    r.hf_config=SimpleNamespace(vocab_size=10,torch_dtype=torch.float32,hidden_size=4)
    r.device=torch.device('cpu');r.cache_generation=0;r.served_epoch=2
    r.request_epoch=SimpleNamespace(value=2)
    r._fan_idx_hit=torch.arange(8);r._fan_idx_miss=torch.arange(8)
    r._reset_tree_cache_tensors()
    return r


@pytest.mark.parametrize('pending_at,expected',[(1,4),(5,5),(99,7)])
def test_interrupt_never_exposes_an_unfinished_step(monkeypatch,pending_at,expected):
    r=runner();completed=[];events=[]
    class Event:
        def record(self,*args):events.append(('record',len(completed)))
        def synchronize(self):events.append(('complete',len(completed)))
    r._publication_event=Event()
    monkeypatch.setattr(torch.cuda,'current_stream',lambda:None)
    r._compute_step_positions_and_slot_maps=lambda *args:(None,None,None,None)
    def step(depth,ids,*args):
        completed.append(depth)
        if len(completed)==pending_at:r.request_epoch.value=3
        return ids
    r._decode_tree_step=step
    payload=dict(metadata_ints=(1,7,1,8),positions=torch.arange(8),rope_positions=torch.arange(8),
        input_ids=torch.arange(8),block_tables=torch.zeros(1,1,dtype=torch.int32),temps=torch.zeros(8))
    tokens,logits,_=r._decode_tree(payload)
    assert len(completed)==expected and r.cache_valid_length==expected
    assert events[-1]==('complete',expected)
    assert r.cache_generation==1
    assert tokens.shape==(8,7) and logits.shape==(8,7,10)


def test_only_branches_with_real_parent_prefixes_are_published():
    r=runner()
    payload=dict(seq_ids_expanded=torch.full((8,),42),rec_flat=torch.arange(8),cache_hits_list=[True],
        block_tables=torch.zeros(1,1,dtype=torch.int32),base_num_tokens=torch.tensor([50]),valid_length=4)
    tokens=torch.ones(8,7,dtype=torch.int64);logits=torch.ones(8,7,10)
    r._populate_tree_cache(payload,tokens,logits,None)
    assert r.tree_cache_keys[:,1].tolist()==[0,1,2,3,4]
    assert r.tree_cache_expected_tokens.tolist()==[51,52,53,54,55]
    assert r.tree_cache_tokens.shape==(5,7)


@pytest.mark.parametrize('context_tokens,expected_hit',[(51,True),(52,False)])
def test_matching_key_with_stale_kv_context_is_a_miss(context_tokens,expected_hit):
    r=runner();r.cache_valid_length=4
    r.tree_cache_keys=torch.tensor([[42,0,3]])
    r.tree_cache_expected_tokens=torch.tensor([51])
    r.tree_cache_tokens=torch.full((1,7),2,dtype=torch.int64)
    r.tree_cache_logits=torch.ones(1,7,10)
    calls=[]
    def jit(*args):
        calls.append('jit')
        args[2].fill_(0);args[3].fill_(1)
    r.jit_speculate=jit
    _,_,_,hit,_=r.hit_cache_and_respond(torch.tensor([[42,0,3]]),1,7,
        torch.tensor([context_tokens]),torch.zeros(1),torch.zeros(1,1,dtype=torch.int32))
    assert bool(hit[0])==expected_hit
    assert bool(calls)==(not expected_hit)
    assert r.response_valid_length==(4 if expected_hit else 7)


@pytest.mark.parametrize('valid',[1,4,7])
def test_padding_that_matches_target_is_still_not_accepted(monkeypatch,valid):
    logits=torch.zeros(1,8,10);logits[:,:,2]=1
    target=SimpleNamespace(call=lambda *args:logits)
    metrics=dict(target_verify_times=[],accepted_suffix_lens_with_recovery=[],cache_hits=[],
        accepted_suffix_lens_on_hit=[],accepted_suffix_lens_on_miss=[],verified_lengths=[])
    verifier=Verifier(7,torch.device('cpu'),target,metrics=metrics)
    seq=SimpleNamespace(num_cached_tokens=100,temperature=0,draft_temperature=0)
    spec=SpeculateResult(torch.tensor([[3]+[2]*7]),torch.zeros(1,7,10),torch.ones(1),valid)
    result=verifier.verify([seq],spec)
    assert result.new_suffixes==[[3]+[2]*valid]
    assert metrics['verified_lengths']==[valid]
    reset_context()


def test_publication_notifier_is_monotonic_and_drain_ack_precedes_memory_reuse(monkeypatch):
    import multiprocessing as mp
    import ssd.engine.speculator_async as module
    s=module.SpeculatorAsync.__new__(module.SpeculatorAsync)
    s.device=torch.device('cpu');s.draft_runner_rank=1;s.async_pg=None
    s.request_epoch=mp.get_context('spawn').Value('q',3)
    s.metrics=dict(final_drain_times=[],draft_peak_memory=[])
    calls=[]
    monkeypatch.setattr(module.dist,'send',lambda payload,**kw:calls.append(('send',int(payload[0]))))
    def receive(payload,**kw):
        calls.append(('ack',s.request_epoch.value));payload.copy_(torch.tensor([4,100,200]))
    monkeypatch.setattr(module.dist,'recv',receive)
    s.drain()
    assert calls==[('send',3),('ack',4)] and s.request_epoch.value==4
    assert s.metrics['draft_peak_memory']==[dict(allocated_bytes=100,reserved_bytes=200)]
    monkeypatch.setattr(module.dist,'recv',lambda payload,**kw:payload.fill_(0))
    with pytest.raises(RuntimeError,match='Stale'):s.drain()


@pytest.mark.parametrize('overrides',[
    dict(speculate=False),dict(draft_async=False,num_gpus=1),dict(max_num_seqs=2),
    dict(use_eagle=True),dict(jit_speculate=False),dict(early_publish_tokens=8),
])
def test_unsupported_publication_configurations_fail_before_loading(monkeypatch,tmp_path,overrides):
    import ssd.config as module
    monkeypatch.setattr(module.AutoConfig,'from_pretrained',lambda *a,**kw:SimpleNamespace(
        max_position_embeddings=4096,model_type='qwen3'))
    kwargs=dict(model=str(tmp_path),draft=str(tmp_path),num_gpus=2,speculate=True,
        draft_async=True,jit_speculate=True,speculate_k=7,early_publish_tokens=4)
    kwargs.update(overrides)
    with pytest.raises(ValueError,match='Early publication'):
        module.Config(**kwargs)
