"""The shared prototype declares new AR arithmetic and preserves logical rows."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
from shared_numerics import WIDTH, fixed_linear, wrap_pointwise, install_shared
from ssd.layers.linear import QKVParallelLinear
from ssd.utils.context import get_context,set_context


@pytest.fixture(autouse=True)
def restore_context():
    old=get_context()
    yield
    set_context(**vars(old))


def test_shared_linear_uses_same_padded_shape_for_ar_and_verify(monkeypatch):
    import shared_numerics as backend
    calls=[]; original=backend.F.linear
    monkeypatch.setattr(backend.F,'linear',lambda x,w:calls.append(tuple(x.shape)) or original(x,w))
    x=torch.arange(24,dtype=torch.float64).reshape(3,8)
    weight=torch.ones(6,8,dtype=torch.float64)
    a=fixed_linear(x[:1],weight);b=fixed_linear(x,weight)
    assert calls==[(WIDTH,8),(WIDTH,8)]
    assert a.shape==(1,6) and b.shape==(3,6) and torch.equal(a[0],b[0])
    with pytest.raises(ValueError,match='1..16'):
        fixed_linear(x.repeat(6,1),weight)


def test_pointwise_padding_retains_head_groups_and_residual_tuple():
    calls=[]
    def original(x,residual):
        calls.append(tuple(x.shape))
        return x+residual,residual
    module=SimpleNamespace(forward=original)
    wrap_pointwise(module,rows_per_token=2)
    set_context(False)
    x=torch.arange(24,dtype=torch.float64).reshape(6,4)
    a,b=module.forward(x,x)
    assert calls==[(WIDTH*2,4)]
    assert torch.equal(a,2*x) and torch.equal(b,x)
    with pytest.raises(ValueError,match='token/head'):
        module.forward(x[:5],x[:5])


def test_prefill_uses_original_pointwise_shape():
    calls=[];module=SimpleNamespace(forward=lambda x:calls.append(len(x)) or x)
    wrap_pointwise(module)
    set_context(True)
    x=torch.ones(100,4)
    assert module.forward(x) is x and calls==[100]


def test_shared_installer_changes_decode_dispatch_and_preserves_prefill(monkeypatch):
    import shared_numerics as backend
    model=torch.nn.Module();model.tp_size=1
    model.qkv_proj=QKVParallelLinear(8,4,2,1)
    model.qkv_proj.weight.data.fill_(0.5)
    calls=[];original=backend.fixed_linear
    monkeypatch.setattr(backend,'fixed_linear',
                        lambda x,w:calls.append(len(x)) or original(x,w))
    install_shared(model)
    set_context(True);model.qkv_proj(torch.ones(3,8))
    assert not calls
    set_context(False);model.qkv_proj(torch.ones(1,8))
    set_context(False,cu_seqlens_q=torch.tensor([0,7]));model.qkv_proj(torch.ones(7,8))
    assert calls==[1,7]
    with pytest.raises(ValueError,match='already installed'):
        install_shared(model)


def test_parallel_target_prototype_rejects_tensor_parallelism():
    model=torch.nn.Module();model.tp_size=2
    with pytest.raises(ValueError,match='one target GPU'):
        install_shared(model)


def test_constructor_hook_installs_target_only(monkeypatch):
    import shared_numerics as backend
    calls=[]
    def init(self,*args,**kwargs):
        torch.nn.Module.__init__(self)
        self.draft=kwargs.get('draft',False)
    monkeypatch.setattr(backend.Qwen3ForCausalLM,'__init__',init)
    monkeypatch.setattr(backend,'install_shared',lambda model:calls.append(model.draft))
    backend.install_target_factory()
    backend.Qwen3ForCausalLM(draft=False)
    backend.Qwen3ForCausalLM(draft=True)
    assert calls==[False]


def test_parallel_attention_handles_extra_decode_capture_buckets(monkeypatch):
    import numerics_attention as attention
    calls=[]
    class Kernel:
        def __getitem__(self,grid):
            return lambda *args,**kwargs:calls.append((grid,args,kwargs))
    monkeypatch.setattr(attention,'_num_splits',lambda *args:32)
    monkeypatch.setattr(attention,'_paged_attn_split_kernel',Kernel())
    monkeypatch.setattr(attention,'_combine_kernel',Kernel())
    q=torch.ones(2,32,128)
    key=torch.ones(1,256,8,128)
    lengths=torch.tensor([50,70],dtype=torch.int32)
    tables=torch.zeros(2,1,dtype=torch.int32)
    attention.parallel_ar_attention(q,key,key,lengths,tables)
    grid,args,kwargs=calls[0]
    assert grid==(2,8,32) and args[3] is lengths and args[4] is tables
    assert kwargs['HAS_CU_Q'] is False and kwargs['NUM_SPLITS']==32
    calls.clear()
    attention.parallel_ar_attention(q,key,key,lengths[:1],tables[:1])
    assert torch.equal(calls[0][1][3],torch.tensor([49,50],dtype=torch.int32))
