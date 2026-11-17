"""Experimental numerical candidates for diagnostics, never engine defaults.

Hybrid BMM retains batched attention and pointwise operations. QKV/MLP projections
use independent GEMVs in one batched call; output projections and the vocabulary
head use original single-row operations because their BMM reduction differs.
This has no speed or general exactness guarantee. Install before graph capture.
"""
import torch

from ssd.engine.ar_compatible import _verifying, _wrap_head, _wrap_rows
from ssd.layers.embed_head import ParallelLMHead
from ssd.layers.linear import LinearBase


def batch_gemv(x, weight):
    return torch.bmm(x.unsqueeze(1), weight.t().unsqueeze(0).expand(
        x.shape[0], -1, -1)).squeeze(1)


def install_hybrid_bmm(model):
    if getattr(model, '_hybrid_bmm_installed', False):
        raise ValueError('Hybrid BMM already installed')
    for name, module in model.named_modules():
        if isinstance(module, LinearBase):
            if module.tp_size != 1 or getattr(module, 'bias', None) is not None:
                raise ValueError('Hybrid BMM requires one target GPU and bias-free projections')
            if name.endswith('o_proj'):
                _wrap_rows(module)
            elif name.endswith(('qkv_proj', 'gate_up_proj', 'down_proj')):
                original = module.forward
                weight = module.weight
                def forward(x, original=original, weight=weight):
                    return batch_gemv(x, weight) if _verifying() else original(x)
                module.forward = forward
            else:
                raise ValueError(f'Unsupported hybrid projection: {name}')
        elif isinstance(module, ParallelLMHead):
            if module.tp_size != 1:
                raise ValueError('Hybrid BMM requires one target GPU')
            _wrap_head(module)
    model._hybrid_bmm_installed = True
