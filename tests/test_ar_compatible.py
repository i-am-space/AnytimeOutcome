"""AR-shaped verification: shape fidelity, causal state, and isolation."""
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party/ssd'))
from ssd.engine.ar_compatible import (_wrap_attention, _wrap_head, _wrap_rows,
                                      install_ar_compatible, validate_ar_compatible)
from ssd.utils.context import get_context, reset_context, set_context


@pytest.fixture(autouse=True)
def context_cleanup():
    old = get_context()
    yield
    set_context(**vars(old))


def verify_context(n=3):
    set_context(False, cu_seqlens_q=torch.tensor([0, n], dtype=torch.int32),
                max_seqlen_q=n, slot_mapping=torch.arange(20, 20+n, dtype=torch.int32),
                context_lens=torch.tensor([20+n], dtype=torch.int32),
                block_tables=torch.tensor([[2, 0]], dtype=torch.int32))


def test_rows_keep_prefill_and_ar_decode_unchanged():
    calls = []
    def original(x, residual=None):
        calls.append(x.shape[0])
        return x+1 if residual is None else (x+residual, residual+2)
    module = SimpleNamespace(forward=original)
    _wrap_rows(module)
    x = torch.randn(3, 8)
    set_context(True)
    assert torch.equal(module.forward(x), x+1)
    assert calls == [3]
    calls.clear(); reset_context()
    module.forward(x)
    assert calls == [3]  # No verification metadata; original path.
    calls.clear(); verify_context()
    a,b = module.forward(x, residual=x)
    assert calls == [1,1,1]
    assert torch.equal(a, x+x) and torch.equal(b, x+2)


def test_head_groups_are_preserved():
    calls=[]
    module=SimpleNamespace(forward=lambda x: calls.append(len(x)) or x)
    _wrap_rows(module, rows=4)
    verify_context()
    x=torch.randn(12,8)
    assert torch.equal(module.forward(x),x)
    assert calls == [4,4,4]
    with pytest.raises(ValueError, match='incomplete'):
        module.forward(x[:11])


def test_attention_uses_ar_causal_lengths_and_restores_context():
    seen=[]
    def original(q,k,v):
        c=get_context()
        seen.append((c.context_lens.item(), c.slot_mapping.item(), c.cu_seqlens_q))
        return q+k+v
    module=SimpleNamespace(forward=original)
    _wrap_attention(module)
    verify_context(); before=get_context()
    q,k,v=[torch.randn(3,8) for _ in range(3)]
    assert torch.equal(module.forward(q,k,v),q+k+v)
    assert seen == [(21,20,None),(22,21,None),(23,22,None)]
    assert vars(get_context()) == vars(before)


def test_attention_restores_context_after_failure():
    def fail(*args): raise RuntimeError('kernel failure')
    module=SimpleNamespace(forward=fail); _wrap_attention(module)
    verify_context(); before=get_context()
    with pytest.raises(RuntimeError,match='kernel failure'):
        module.forward(*(torch.zeros(3,8) for _ in range(3)))
    assert vars(get_context()) == vars(before)


def test_vocabulary_projection_uses_single_rows_and_original_layout(monkeypatch):
    import torch.nn.functional as functional
    weight=torch.randn(19,8); x=torch.randn(3,8)
    original=functional.linear; calls=[]
    monkeypatch.setattr(functional,'linear',lambda a,b: calls.append(a.shape[0]) or original(a,b))
    module=SimpleNamespace(tp_size=1,weight=weight,forward=lambda x,last_only=True: 'unchanged')
    _wrap_head(module)
    set_context(True); assert module.forward(x) == 'unchanged'
    verify_context()
    result=module.forward(x,False)
    assert result.shape == (1,3,19) and calls == [1,1,1]
    assert torch.equal(result[0],torch.cat([original(a[None],weight) for a in x]))


@pytest.mark.parametrize('model,eagle,batch,tp',[
    ('llama',False,1,1),('qwen3',True,1,1),('qwen3',False,2,1),('qwen3',False,1,2)])
def test_unsupported_configuration_rejected(model,eagle,batch,tp):
    with pytest.raises(ValueError):
        validate_ar_compatible(SimpleNamespace(use_eagle=eagle,max_num_seqs=batch),model,tp)


def test_cached_rope_wrapper_does_not_change_drafter():
    from ssd.models.qwen3 import Qwen3Attention
    from ssd.layers.rotary_embedding import RotaryEmbedding
    # Build only the module structure; avoid allocating real model weights.
    target=Qwen3Attention.__new__(Qwen3Attention)
    torch.nn.Module.__init__(target)
    target.num_heads,target.num_kv_heads=2,1
    target.q_norm=torch.nn.Identity(); target.k_norm=torch.nn.Identity()
    rope=RotaryEmbedding(4,4,16,10000)
    target.rotary_emb=rope
    original_func=rope.forward.__func__
    install_ar_compatible(target)
    assert target.rotary_emb is not rope
    assert rope.forward.__func__ is original_func
    assert target.rotary_emb.cos_sin_cache.data_ptr() == rope.cos_sin_cache.data_ptr()
    with pytest.raises(ValueError,match='already installed'):
        install_ar_compatible(target)


@pytest.mark.parametrize('target_temp,draft_temp',[(1,None),(0,1)])
def test_non_greedy_requests_are_rejected_before_enqueue(target_temp,draft_temp):
    from ssd.engine.llm_engine import LLMEngine
    from ssd.sampling_params import SamplingParams
    engine=SimpleNamespace(config=SimpleNamespace(target_numerics='ar_compatible'))
    with pytest.raises(ValueError,match='greedy'):
        LLMEngine.add_request(engine,[1,2],SamplingParams(
            temperature=target_temp,draft_temperature=draft_temp))
