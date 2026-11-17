"""Shared fixed16-v1 target arithmetic for a NEW experimental AR reference.

This explicitly changes original AR decode arithmetic. Model weights and prefill
stay the same. No production defaults are changed. Install before graph capture
in the target constructor; draft models retain their original implementation.
"""
import copy

import torch
import torch.nn.functional as F

from ssd.layers.activation import SiluAndMul
from ssd.layers.attention import Attention, store_kvcache
from ssd.layers.embed_head import ParallelLMHead
from ssd.layers.layernorm import RMSDNorm
from ssd.layers.linear import LinearBase
from ssd.layers.rotary_embedding import RotaryEmbedding
from ssd.models.qwen3 import Qwen3Attention, Qwen3ForCausalLM
from ssd.utils.context import get_context
from numerics_attention import parallel_ar_attention

VERSION='shared-fixed16-v1'
WIDTH=16


def fixed_linear(x, weight):
    if not 1<=x.shape[0]<=WIDTH:
        raise ValueError('Shared fixed16 backend supports 1..16 target query rows')
    padded=x.new_zeros((WIDTH,x.shape[1]))
    padded[:x.shape[0]].copy_(x)
    return F.linear(padded,weight)[:x.shape[0]]


def wrap_pointwise(module, rows_per_token=1):
    original=module.forward
    def forward(*args,**kwargs):
        if get_context().is_prefill:
            return original(*args,**kwargs)
        n=args[0].shape[0]
        if n%rows_per_token or not 1<=n//rows_per_token<=WIDTH:
            raise ValueError('Invalid shared fixed16 pointwise token/head group')
        def padded(value):
            if not isinstance(value,torch.Tensor) or not value.ndim or value.shape[0]!=n:
                return value
            output=value.new_zeros((WIDTH*rows_per_token,*value.shape[1:]))
            output[:n].copy_(value)
            return output
        output=original(*(padded(x) for x in args),**{k:padded(v) for k,v in kwargs.items()})
        if isinstance(output,tuple):
            return tuple(x[:n] for x in output)
        return output[:n]
    module.forward=forward


def install_shared(model):
    if getattr(model,'_shared_numerics_version',None):
        raise ValueError('Shared backend already installed')
    if model.tp_size!=1:
        raise ValueError('Shared fixed16 prototype requires one target GPU')
    # Cached RoPE modules can belong to drafter and target simultaneously.
    for module in model.modules():
        if isinstance(module,Qwen3Attention):
            module.rotary_emb=copy.copy(module.rotary_emb)
            module.rotary_emb._forward_hooks=copy.copy(module.rotary_emb._forward_hooks)
            wrap_pointwise(module.q_norm,module.num_heads)
            wrap_pointwise(module.k_norm,module.num_kv_heads)
    for module in model.modules():
        original=module.forward
        if isinstance(module,LinearBase):
            if module.tp_size!=1 or getattr(module,'bias',None) is not None:
                raise ValueError('Shared fixed16 requires bias-free single-GPU projections')
            def linear(x,original=original,weight=module.weight):
                return original(x) if get_context().is_prefill else fixed_linear(x,weight)
            module.forward=linear
        elif isinstance(module,ParallelLMHead):
            def head(x,last_only=True,original=original,weight=module.weight):
                context=get_context()
                if context.is_prefill:
                    return original(x,last_only)
                output=fixed_linear(x,weight)
                return output.unsqueeze(0) if context.cu_seqlens_q is not None else output
            module.forward=head
        elif isinstance(module,(RMSDNorm,SiluAndMul,RotaryEmbedding)):
            wrap_pointwise(module)
        elif isinstance(module,Attention):
            def attention(q,k,v,original=original,module=module):
                context=get_context()
                if context.is_prefill:
                    return original(q,k,v)
                if context.context_lens.numel()!=1 or context.block_tables.shape[0]!=1:
                    raise ValueError('Shared fixed16 attention requires one logical request')
                q=q.view(-1,module.num_heads,module.head_dim)
                k=k.view(-1,module.num_kv_heads,module.head_dim)
                v=v.view(-1,module.num_kv_heads,module.head_dim)
                store_kvcache(k,v,module.k_cache,module.v_cache,context.slot_mapping)
                output=parallel_ar_attention(q,module.k_cache,module.v_cache,
                                             context.context_lens,context.block_tables)
                return output.view(-1,module.num_heads*module.head_dim)
            module.forward=attention
    model._shared_numerics_version=VERSION


def install_target_factory():
    """Diagnostic-process-local constructor hook; spawned drafters stay original."""
    original=Qwen3ForCausalLM.__init__
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        if not self.draft:
            install_shared(self)
    Qwen3ForCausalLM.__init__=init
