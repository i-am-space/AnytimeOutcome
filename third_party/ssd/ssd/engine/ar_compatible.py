"""Opt-in AR-shaped target verification, installed before CUDA graph capture.

Keep the original target prefill and single-token decode untouched. During
verification, each target operation uses the same per-token shapes as AR.
Operations are grouped by layer, so this is distinct from replaying a complete
target forward once per token. It sacrifices GEMM batching and must be measured;
this module does not promise a speedup or universal floating-point equivalence.
"""
import copy

import torch
import torch.nn.functional as F

from ssd.utils.context import get_context, set_context


def validate_ar_compatible(config, model_type, target_gpus):
    """Fail explicitly outside the first supported, testable configuration."""
    if model_type != 'qwen3' or config.use_eagle:
        raise ValueError('AR-compatible verification currently supports non-EAGLE Qwen3 only')
    if config.max_num_seqs != 1 or target_gpus != 1:
        raise ValueError('AR-compatible verification requires one request and one target GPU')


def _verifying():
    context = get_context()
    return not context.is_prefill and context.cu_seqlens_q is not None


def _wrap_rows(module, rows=1):
    original = module.forward

    def forward(*args, **kwargs):
        n = args[0].shape[0]
        if not _verifying() or n <= rows:
            return original(*args, **kwargs)
        if n % rows:
            raise ValueError('AR-compatible operation has an incomplete token/head group')

        def part(value, start):
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == n:
                return value[start:start + rows]
            return value

        outputs = [original(*(part(a, i) for a in args),
                            **{k: part(v, i) for k, v in kwargs.items()})
                   for i in range(0, n, rows)]
        if isinstance(outputs[0], tuple):
            return tuple(torch.cat([out[j] for out in outputs]) for j in range(len(outputs[0])))
        return torch.cat(outputs)

    module.forward = forward


def _wrap_attention(module):
    original = module.forward

    def forward(q, k, v):
        if not _verifying():
            return original(q, k, v)
        context = get_context()
        n = q.shape[0]
        if context.context_lens.numel() != 1 or context.block_tables.shape[0] != 1:
            raise ValueError('AR-compatible attention requires batch size one')
        outputs = []
        try:
            for j in range(n):
                # All metadata remains on device, including during graph capture.
                # Restrict the split-KV reduction to the exact AR visible length,
                # rather than masking future rows inside a wider reduction.
                set_context(False, slot_mapping=context.slot_mapping[j:j + 1],
                            context_lens=context.context_lens - n + j + 1,
                            block_tables=context.block_tables, is_jit=context.is_jit)
                outputs.append(original(q[j:j + 1], k[j:j + 1], v[j:j + 1]))
        finally:
            set_context(**vars(context))
        return torch.cat(outputs)

    module.forward = forward


def _wrap_head(module):
    original = module.forward

    def forward(x, last_only=True):
        if not _verifying():
            return original(x, last_only)
        if module.tp_size != 1:
            raise ValueError('AR-compatible vocabulary projection requires one target GPU')
        logits = torch.cat([F.linear(x[i:i + 1], module.weight) for i in range(x.shape[0])])
        return logits.unsqueeze(0)  # Original B=1 verification output layout.

    module.forward = forward


def install_ar_compatible(model):
    """Wrap target modules only; never monkey-patch global torch operations."""
    from ssd.layers.activation import SiluAndMul
    from ssd.layers.attention import Attention
    from ssd.layers.embed_head import ParallelLMHead
    from ssd.layers.layernorm import RMSDNorm
    from ssd.layers.linear import LinearBase
    from ssd.layers.rotary_embedding import RotaryEmbedding
    from ssd.models.qwen3 import Qwen3Attention

    if getattr(model, '_ar_compatible_installed', False):
        raise ValueError('AR-compatible operations are already installed')
    # get_rope() caches module instances. Detach the target wrapper from a RoPE
    # instance that may also belong to the synchronous drafter. Its read-only
    # cosine/sine buffers can remain shared.
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            module.rotary_emb = copy.copy(module.rotary_emb)

    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            _wrap_rows(module.q_norm, module.num_heads)
            _wrap_rows(module.k_norm, module.num_kv_heads)
        if isinstance(module, (LinearBase, RMSDNorm, SiluAndMul, RotaryEmbedding)):
            _wrap_rows(module)
        if isinstance(module, Attention):
            _wrap_attention(module)
        if isinstance(module, ParallelLMHead):
            _wrap_head(module)
    model._ar_compatible_installed = True
