"""Diagnostic pairing, layout preservation, and causal isolation."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'profiling'))
from diagnose_fast_numerics import (OperationTrace, clone_tree, compare_outputs,
                                    difference, isolated_probe, row_linear, fixed_gemm)
from numerics_candidates import batch_gemv, install_hybrid_bmm
from ssd.layers.linear import QKVParallelLinear, RowParallelLinear
from ssd.layers.attention import Attention
from ssd.models.qwen3 import Qwen3Attention
from ssd.utils.context import get_context, set_context


def test_clone_preserves_split_view_layout_and_storage_independence():
    full = torch.arange(84).reshape(7, 12)
    view = full[:, 2:6]
    cloned = clone_tree((view, None))[0]
    assert cloned.stride() == view.stride()
    assert torch.equal(cloned, view)
    cloned.zero_()
    assert view.sum() > 0


def test_difference_rejects_broadcasting():
    with pytest.raises(ValueError, match='Unpaired tensor shapes'):
        difference(torch.ones(1, 4), torch.ones(4))


def test_output_comparison_preserves_head_groups_and_detects_second_head():
    single = torch.ones(2, 4)
    batch = torch.ones(6, 4)
    batch[1, 2] = 2
    result = compare_outputs(single, batch, rows=2)[0]
    assert not result['exact'] and result['unequal'] == 1
    assert result['elements'] == 8
    with pytest.raises(ValueError, match='structures'):
        compare_outputs((single, single), (batch,), 2)


def test_shared_rotary_instances_get_independent_trace_hooks():
    kwargs = dict(hidden_size=8, num_heads=2, num_kv_heads=1, head_dim=4, max_position=32)
    a, b = Qwen3Attention(**kwargs), Qwen3Attention(**kwargs)
    assert a.rotary_emb is b.rotary_emb
    shared = a.rotary_emb
    model = torch.nn.ModuleList([a, b])
    trace = OperationTrace(model)
    try:
        assert a.rotary_emb is not b.rotary_emb
        assert len(a.rotary_emb._forward_hooks) == len(b.rotary_emb._forward_hooks) == 1
        assert not shared._forward_hooks
    finally:
        trace.close()
    assert not a.rotary_emb._forward_hooks and not b.rotary_emb._forward_hooks


def test_attention_probe_uses_first_visible_prefix_and_restores_context():
    old = get_context()
    try:
        set_context(False, cu_seqlens_q=torch.tensor([0, 7]), max_seqlen_q=7,
                    slot_mapping=torch.arange(50, 57), context_lens=torch.tensor([57]),
                    block_tables=torch.tensor([[0]]))
        batch_context = get_context()
        module = Attention(2, 4, 0.5, 1)
        seen = []
        def forward(q, k, v):
            c = get_context()
            seen.append((c.context_lens.item(), c.slot_mapping.tolist(), c.cu_seqlens_q))
            return q * 2
        module.forward = forward
        x = torch.ones(7, 8)
        result = isolated_probe(dict(module=module, args=(x, x, x), rows=1,
                                     name='attention', output=x*2), batch_context)
        assert seen == [(51, [50], None)]
        assert result['same_input_row_vs_batch'][0]['exact']
        assert get_context().cu_seqlens_q is batch_context.cu_seqlens_q
        assert get_context().context_lens.item() == 57
    finally:
        set_context(**vars(old))


def test_batched_candidate_pairs_all_noncontiguous_input_rows():
    x = torch.arange(70, dtype=torch.float64).reshape(7,10)[:,1:9]
    weight = torch.arange(48, dtype=torch.float64).reshape(6,8)
    assert torch.equal(batch_gemv(x, weight), row_linear(x, weight))


def test_hybrid_keeps_ar_and_prefill_dispatch_and_uses_bmm_for_verify(monkeypatch):
    import numerics_candidates
    model = torch.nn.Module()
    model.qkv_proj = QKVParallelLinear(8,4,2,1)
    model.qkv_proj.weight.data.fill_(0.25)
    old = get_context()
    calls = []
    original = numerics_candidates.batch_gemv
    monkeypatch.setattr(numerics_candidates, 'batch_gemv',
                        lambda x,w: calls.append(tuple(x.shape)) or original(x,w))
    install_hybrid_bmm(model)
    try:
        x = torch.ones(7,8)
        set_context(True)
        model.qkv_proj(x)
        set_context(False)
        model.qkv_proj(x[:1])
        assert not calls
        set_context(False,cu_seqlens_q=torch.tensor([0,7]))
        model.qkv_proj(x)
        assert calls == [(7,8)]
        with pytest.raises(ValueError,match='already installed'):
            install_hybrid_bmm(model)
    finally:
        set_context(**vars(old))


def test_hybrid_output_projection_retains_individual_row_calls():
    model = torch.nn.Module()
    model.o_proj = RowParallelLinear(8,8)
    calls = []
    model.o_proj.forward = lambda x: calls.append(x.shape[0]) or x+1
    install_hybrid_bmm(model)
    old = get_context()
    try:
        set_context(False,cu_seqlens_q=torch.tensor([0,7]))
        assert torch.equal(model.o_proj(torch.ones(7,8)),torch.full((7,8),2.0))
        assert calls == [1]*7
    finally:
        set_context(**vars(old))


def test_fixed_gemm_prototype_preserves_logical_rows_and_rejects_overflow():
    x=torch.arange(70,dtype=torch.float64).reshape(7,10)[:,1:9]
    weight=torch.arange(48,dtype=torch.float64).reshape(6,8)
    assert torch.equal(fixed_gemm(x,weight),row_linear(x,weight))
    assert fixed_gemm(x[:1],weight).shape==(1,6)
    with pytest.raises(ValueError,match='declared tile'):
        fixed_gemm(x,weight,width=6)
