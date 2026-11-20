"""Small real CUDA-graph/FlashInfer check without loading either language model."""
from pathlib import Path
from types import SimpleNamespace
import math
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/ssd'))
from ssd.engine.model_runner import ModelRunner
from ssd.engine.helpers.cudagraph_helpers import capture_fi_tree_decode_cudagraph,run_fi_tree_decode_cudagraph
from ssd.engine.helpers import cudagraph_helpers
from ssd.engine.outcome_depth import active_rows_by_depth,compact_mask
from ssd.layers.attention import Attention
from ssd.utils.context import set_context,reset_context


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual compact CUDA graph and sparse KV test')
@pytest.mark.parametrize('k,depths',[(7,None),(10,None),(7,[7,4,4]),(10,[10,7,4]),(10,[4,7,10]),(7,[5,5,5]),(10,[7,7,7])])
def test_compact_graph_attention_matches_dense_masked_reference_and_leaves_holes_untouched(k,depths):
    torch.manual_seed(0);f=3;n=f*(k+1);prefix=13
    # Campaigns use a fresh process for every configuration. Match that cache scope.
    cudagraph_helpers.cache.clear()
    active_plan=(tuple(tuple(range(n)) for _ in range(k)) if depths is None else active_rows_by_depth(k,f,depths))
    cfg=SimpleNamespace(max_num_seqs=1,max_model_len=512,enforce_eager=False,async_fan_out=f,
        speculate_k=k,fan_out_list=[f]*(k+1),fan_out_list_miss=[f]*(k+1),outcome_depths=depths,use_eagle=False)
    hf=SimpleNamespace(num_attention_heads=4,num_key_value_heads=2,head_dim=128,hidden_size=512,
        vocab_size=64,torch_dtype=torch.bfloat16)
    cfg.hf_config=hf
    r=SimpleNamespace(config=cfg,hf_config=hf,rank=0,device=torch.device('cuda:0'),block_size=256,is_draft=True)
    ModelRunner._init_flashinfer_wrappers(r)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.emb=torch.nn.Embedding(64,512);self.head=torch.nn.Linear(512,64,bias=False)
            self.attn=Attention(4,128,1/math.sqrt(128),2,draft=True,speculate=True,draft_async=True,F=f,K=k)
            self.attn.only_prefill_wrapper=r.prefill_wrappers[1]
            self.attn.k_cache=torch.zeros(2,256,2,128,device='cuda',dtype=torch.bfloat16)
            self.attn.v_cache=torch.zeros_like(self.attn.k_cache)
        def forward(self,ids,positions):
            x=self.emb(ids)
            return self.attn(x,(x[:,:256]*.5).contiguous(),(x[:,:256]*.7).contiguous())
        def compute_logits(self,hidden,last_only):return self.head(hidden)
    r.model=Model().to(device='cuda',dtype=torch.bfloat16)
    old_dtype=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        gv,pool,graphs,sizes=capture_fi_tree_decode_cudagraph(r)
    finally:torch.set_default_dtype(old_dtype)
    r.graphs={'fi_tree_decode':graphs};r.graph_bs_list={'fi_tree_decode':sizes}
    assert set(graphs)==({1} if depths is None else {(1,len(rows)) for rows in active_plan})
    r.model.attn.k_cache.normal_(0,.1);r.model.attn.v_cache.normal_(0,.1)
    original_k=r.model.attn.k_cache.clone();original_v=r.model.attn.v_cache.clone()
    ids=torch.arange(n,device='cuda')%64;written=set()
    try:
        with torch.inference_mode():
            for step,active in enumerate(active_plan):
                rows=torch.tensor(active,device='cuda');inputs=ids[rows]
                slots=prefix+k+1+step*n+rows;written.update(slots.tolist())
                length=prefix+k+1+(step+1)*n
                set_context(is_prefill=False,slot_mapping=slots.to(torch.int32),
                    context_lens=torch.tensor([length],device='cuda',dtype=torch.int32),
                    block_tables=torch.tensor([[0,1]],device='cuda',dtype=torch.int32))
                actual=run_fi_tree_decode_cudagraph(r,inputs,rows,False,gv,step,torch.tensor([1],device='cuda'))
                query=r.model.emb(inputs).view(-1,4,128).float()
                keys=r.model.attn.k_cache.flatten(0,1)[:length].float().repeat_interleave(2,dim=1)
                values=r.model.attn.v_cache.flatten(0,1)[:length].float().repeat_interleave(2,dim=1)
                mask=torch.tensor(compact_mask(k,f,step,prefix,active),device='cuda',dtype=torch.bool)
                scores=torch.einsum('qhd,khd->hqk',query,keys)/math.sqrt(128)
                scores.masked_fill_(~mask[None],float('-inf'))
                hidden=torch.einsum('hqk,khd->qhd',scores.softmax(-1),values).reshape(-1,512).bfloat16()
                expected=r.model.compute_logits(hidden,False)
                torch.testing.assert_close(actual,expected,atol=.004,rtol=.02,check_dtype=False)
                ids=torch.zeros_like(ids);ids[rows]=actual.argmax(-1)
        untouched=[i for i in range(prefix+k+1,512) if i not in written]
        assert torch.equal(r.model.attn.k_cache.flatten(0,1)[untouched],original_k.flatten(0,1)[untouched])
        assert torch.equal(r.model.attn.v_cache.flatten(0,1)[untouched],original_v.flatten(0,1)[untouched])
    finally:reset_context()
