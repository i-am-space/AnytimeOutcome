from pathlib import Path
from types import SimpleNamespace
import multiprocessing as mp
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'third_party/ssd'))
from ssd.engine.outcome_depth import active_rows_by_depth, compact_mask, row_depths, validate_depths
from ssd.engine.draft_runner import DraftRunner
from ssd.engine.prefix_publication import PrefixSnapshot
from ssd.engine.llm_engine import LLMEngine
from ssd.utils.context import get_context


@pytest.mark.parametrize('k,depths,expected', [(7,[7,4,4],120),(10,[10,7,4],231)])
def test_real_rows_shrink_and_equal_work_controls_match(k,depths,expected):
    rows=active_rows_by_depth(k,3,depths)
    assert sum(map(len,rows))==expected < k*(k+1)*3
    assert sum(map(len,active_rows_by_depth(k,3,list(reversed(depths)))))==expected
    assert sum(map(len,active_rows_by_depth(k,3,[sum(depths)//3]*3)))==expected
    for previous,current in zip(rows,rows[1:]):assert set(current)<=set(previous)


@pytest.mark.parametrize('depths', [[],[7,4],[7,4,0],[8,4,4],[True,4,4],[7,4,4.0]])
def test_bad_depths_rejected(depths):
    with pytest.raises(ValueError):validate_depths(depths,7,3)


@pytest.mark.parametrize('k,depths', [(7,[7,4,4]),(10,[10,7,4]),(10,[4,7,10]),(10,[7,7,7])])
def test_masks_keep_canonical_ancestors_and_exclude_other_or_unwritten_rows(k,depths):
    f=3;width=(k+1)*f;prefix=17
    for step,active in enumerate(active_rows_by_depth(k,f,depths)):
        mask=compact_mask(k,f,step,prefix,active)
        for q,row in enumerate(active):
            expected=set(range(prefix+row//f+1))
            # All prefix tokens, then only glue tokens through this parent.
            expected=set(range(prefix))|set(range(prefix,prefix+row//f+1))
            expected|={prefix+k+1+d*width+row for d in range(step+1)}
            assert set(np.flatnonzero(mask[q]))==expected
            for d in range(step+1):
                written=active_rows_by_depth(k,f,depths)[d]
                allowed=mask[q,prefix+k+1+d*width:prefix+k+1+(d+1)*width]
                assert all(not allowed[r] for r in range(width) if r not in written)


def test_compact_step_executes_only_live_queries_and_keeps_original_kv_addresses():
    r=DraftRunner.__new__(DraftRunner);k=7;depths=[7,4,4];n=24;v=40
    r._outcome_indices=[torch.tensor(rows) for rows in active_rows_by_depth(k,3,depths)]
    r.config=SimpleNamespace(use_eagle=False)
    seen=[]
    def model(ids,positions,**kw):
        assert torch.equal(get_context().slot_mapping,100+positions)
        seen.append(len(ids));logits=torch.zeros(len(ids),v)
        logits.scatter_(1,((ids+1)%v)[:,None],1)
        return logits
    r.run_model=model
    ropes=torch.arange(n).repeat(k,1);slots=ropes+100;cls=torch.ones(k,1,dtype=torch.int32)
    payload={'_all_greedy':True,'_computed_query_rows':0,'cache_hits':torch.tensor([1])}
    tokens=torch.zeros(n,k,dtype=torch.int64);logits=torch.zeros(n,k,v)
    ids=torch.arange(n)
    for d in range(k):ids=r._decode_tree_step(d,ids,ropes,slots,cls,None,payload,tokens,logits,None)
    assert seen==[24]*4+[8]*3 and payload['_computed_query_rows']==120
    for row,depth in enumerate(row_depths(k,3,depths)):
        assert tokens[row,:depth].tolist()==[(row+i+1)%v for i in range(depth)]
        assert not tokens[row,depth:].any()


@pytest.mark.parametrize('overlap',[False,True])
def test_each_rank_offers_its_own_length_and_stale_context_falls_back(overlap):
    r=DraftRunner.__new__(DraftRunner);k=7;f=3;n=24
    r.config=SimpleNamespace(speculate_k=k,MQ_LEN=n,async_fan_out=f,use_eagle=False,
        early_publish_tokens=4 if overlap else 0,outcome_depths=[7,4,4],verbose=False,jit_speculate=True)
    r.device=torch.device('cpu');r.hf_config=SimpleNamespace(vocab_size=50,hidden_size=4,torch_dtype=torch.float32)
    r.outcome_depth_feedback=mp.get_context('spawn').Array('d',8)
    r._outcome_indices=[1];r._outcome_lengths=torch.tensor(row_depths(k,f,[7,4,4]))
    r.cache_valid_length=5 if overlap else 7
    r._fan_idx_hit=torch.arange(k+1).repeat_interleave(f);r._fan_idx_miss=r._fan_idx_hit
    r._handoff_stream=object() if overlap else None
    r._reset_tree_cache_tensors();r.cache_valid_length=5 if overlap else 7
    if overlap:r._prefix_snapshot=PrefixSnapshot(k,[f]*(k+1),[f]*(k+1),50,torch.float32,'cpu')
    payload=dict(seq_ids_expanded=torch.full((n,),42),rec_flat=torch.arange(n),cache_hits_list=[True],
        block_tables=torch.zeros(1,1,dtype=torch.int32),base_num_tokens=torch.tensor([50]),valid_length=4)
    tokens=torch.full((n,k),8,dtype=torch.int64);logits=torch.ones(n,k,50)
    r._populate_tree_cache(payload,tokens,logits,None)
    assert r.tree_cache_keys.shape[0]==15  # exclude unreachable parent positions >4
    jit=[]
    def fallback(*args):jit.append(True);args[2].fill_(1);args[3].fill_(9)
    r.jit_speculate=fallback
    for rank in range(3):
        _,_,_,hit,_=r.hit_cache_and_respond(torch.tensor([[42,0,rank]]),1,k,torch.tensor([51]),
            torch.zeros(1),torch.zeros(1,1,dtype=torch.int32))
        assert hit.item() and r.response_valid_length==min([7,4,4][rank],r.cache_valid_length)
    _,_,_,hit,_=r.hit_cache_and_respond(torch.tensor([[42,0,1]]),1,k,torch.tensor([52]),
        torch.zeros(1),torch.zeros(1,1,dtype=torch.int32))
    assert not hit.item() and jit and r.response_valid_length==7
    assert list(r.outcome_depth_feedback[3:6])==[1.,1.,1.]


def test_snapshot_lengths_and_storage_are_independent_from_producer():
    k=7;n=24;snapshot=PrefixSnapshot(k,[3]*8,[3]*8,4,torch.float32,'cpu')
    payload=dict(seq_ids_expanded=torch.ones(n,dtype=torch.int64),rec_flat=torch.arange(n),
        cache_hits_list=[True],base_num_tokens=torch.tensor([50]),valid_length=4)
    lengths=torch.tensor(row_depths(k,3,[7,4,4]));tokens=torch.ones(n,k,dtype=torch.int64)
    logits=torch.ones(n,k,4)
    result=snapshot.copy(payload,tokens,logits,5,torch.arange(8).repeat_interleave(3),lengths)
    lengths.fill_(1);tokens.fill_(9);logits.fill_(9)
    assert snapshot.lengths[:15].tolist()==[5,4,4]*5
    assert result[2].eq(1).all() and result[3].eq(1).all()


def test_outcome_feedback_resets_for_every_request():
    model=LLMEngine.__new__(LLMEngine);model.publication_feedback=None
    model.outcome_depth_feedback=mp.get_context('spawn').Array('d',[5.]*8)
    model.config=SimpleNamespace(max_steps=None,draft_async=False)
    model.create_inference_step=lambda cfg:None;model.is_finished=lambda:True
    model.generate([],[],use_tqdm=False,stream_callback=lambda *a:None)
    assert list(model.outcome_depth_feedback[:])==[0.]*8
