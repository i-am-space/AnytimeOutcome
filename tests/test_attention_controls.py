"""Check diagnostic backends on scattered pages and bottom-right causal masks."""
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'profiling'))
from attention_controls import (install_attention_control, install_rowwise_linear_control,
                                install_tokenwise_pointwise_control, paged_sdpa)
from test_paged_attention import make_cache, reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('backend', ['sdpa_flash', 'sdpa_math', 'portable_serial'])
@pytest.mark.parametrize('q_lens,kv_lens', [([1], [17]), ([7], [300]), ([8], [520]), ([3,7,1], [40,700,257])])
def test_control_matches_dense_reference(backend, q_lens, kv_lens):
    torch.manual_seed(19)
    hq, hkv, d = 32, 8, 128
    k,v,pt = make_cache(kv_lens,hkv,d,torch.bfloat16,'cuda')
    queries = [torch.randn(n,hq,d,device='cuda',dtype=torch.bfloat16) for n in q_lens]
    cu = torch.tensor([0]+list(torch.tensor(q_lens).cumsum(0)),device='cuda',dtype=torch.int32)
    q = torch.cat(queries)
    lengths = torch.tensor(kv_lens,device='cuda',dtype=torch.int32)
    got = paged_sdpa(q,k,v,lengths,pt,1/math.sqrt(d),cu_seqlens_q=cu,
                     max_seqlen_q=max(q_lens),backend=backend)
    ref = reference(queries,k,v,pt,kv_lens,1/math.sqrt(d))
    assert (got.float()-ref).abs().max() < 0.02
    # The first causal query must ignore future query tokens and their KV slots.
    q2=q.clone()
    q2[1:q_lens[0]] *= 10
    k2,v2=k.clone(),v.clone()
    first_visible=kv_lens[0]-q_lens[0]+1
    for pos in range(first_visible,kv_lens[0]):
        page=pt[0,pos//256]
        k2[page,pos%256] *= 10
        v2[page,pos%256] *= 10
    changed=paged_sdpa(q2,k2,v2,lengths,pt,1/math.sqrt(d),cu_seqlens_q=cu,
                       max_seqlen_q=max(q_lens),backend=backend)
    assert torch.equal(got[0],changed[0])
    if q_lens == [1]:
        single=paged_sdpa(q.unsqueeze(1),k,v,lengths,pt,1/math.sqrt(d),backend=backend)
        assert torch.equal(single.squeeze(1),got)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('backend', ['sdpa_flash', 'sdpa_math', 'portable_serial'])
def test_install_preserves_prefill_and_draft(monkeypatch, backend):
    from ssd.layers.attention import Attention
    from ssd.utils.context import get_context, set_context
    previous = get_context()
    sentinel = object()
    monkeypatch.setattr(Attention, 'forward', lambda *args: sentinel)
    install_attention_control(backend)
    target = Attention(32,128,1/math.sqrt(128),8)
    draft = Attention(32,128,1/math.sqrt(128),8,draft=True)
    k,v,pt = make_cache([17],8,128,torch.bfloat16,'cuda')
    target.k_cache,target.v_cache=k,v
    q=torch.randn(1,32*128,device='cuda',dtype=torch.bfloat16)
    key=torch.randn(1,8*128,device='cuda',dtype=torch.bfloat16)
    value=torch.randn_like(key)
    try:
        set_context(False,slot_mapping=(pt[0,:1]*256+16).int(),
                    context_lens=torch.tensor([17],device='cuda',dtype=torch.int32),block_tables=pt)
        got=target(q,key,value).view(1,32,128)
        ref=reference([q.view(1,32,128)],k,v,pt,[17],1/math.sqrt(128))
        assert (got.float()-ref).abs().max() < 0.02
        assert draft(q,key,value) is sentinel
        set_context(True)
        assert target(q,key,value) is sentinel
    finally:
        set_context(**vars(previous))


def test_rowwise_control_preserves_prefill_and_draft(monkeypatch):
    from types import SimpleNamespace
    import torch.nn.functional as functional
    from ssd.engine.model_runner import ModelRunner
    original_linear = functional.linear
    calls = []
    x, weight = torch.randn(7,17), torch.randn(19,17)
    def observed_linear(inputs, weights, bias=None):
        calls.append(inputs.shape[0])
        return original_linear(inputs,weights,bias)
    monkeypatch.setattr(functional,'linear',observed_linear)
    def fake_run(self, inputs, positions, is_prefill):
        return functional.linear(inputs,weight)
    monkeypatch.setattr(ModelRunner,'run_model',fake_run)
    install_rowwise_linear_control()
    runner=SimpleNamespace(is_draft=False)
    result=ModelRunner.run_model(runner,x,None,False)
    assert calls == [1]*7
    assert torch.equal(result,torch.cat([original_linear(row[None],weight) for row in x]))
    calls.clear()
    ModelRunner.run_model(runner,x,None,True)
    assert calls == [7]
    calls.clear()
    runner.is_draft=True
    ModelRunner.run_model(runner,x,None,False)
    assert calls == [7]


def test_compiled_controls_preserve_head_groups_and_residuals(monkeypatch):
    from ssd.layers.layernorm import RMSHeadNorm, RMSDNorm
    from ssd.models.qwen3 import Qwen3Attention
    calls=[]
    def head_forward(self,x):
        calls.append(x.shape[0])
        return x+1
    def norm_forward(self,x,residual=None):
        calls.append(x.shape[0])
        return (x+2,residual+3) if residual is not None else x+2
    monkeypatch.setattr(RMSHeadNorm,'forward',head_forward)
    monkeypatch.setattr(RMSDNorm,'forward',norm_forward)
    attention=Qwen3Attention.__new__(Qwen3Attention)
    torch.nn.Module.__init__(attention)
    attention.num_heads,attention.num_kv_heads=2,1
    attention.q_norm,attention.k_norm=RMSHeadNorm(4),RMSHeadNorm(4)
    norm=RMSDNorm(4)
    model=torch.nn.ModuleList([attention,norm])
    state={'active':True}
    install_tokenwise_pointwise_control(model,state)
    q=torch.zeros(6,4)
    assert torch.equal(attention.q_norm(q),q+1)
    assert calls == [2,2,2]
    calls.clear()
    x=torch.randn(3,4)
    a,b=norm(x,residual=x)
    assert torch.equal(a,x+2) and torch.equal(b,x+3)
    assert calls == [1,1,1]
    calls.clear()
    state['active']=False
    attention.q_norm(q)
    assert calls == [6]
