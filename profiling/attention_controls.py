"""Diagnostic-only alternatives to the local paged-attention kernel.

KV is gathered through the real page table. Prefill and KV writes stay unchanged.
The flash control evaluates each causal query against its visible prefix so it
can use PyTorch's established FlashAttention backend without an arbitrary mask.
These eager controls prioritize attribution over performance.
"""
import importlib

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def paged_sdpa(q, k_cache, v_cache, cache_seqlens, page_table,
               softmax_scale=None, causal=True, cu_seqlens_q=None,
               max_seqlen_q=None, *, backend, **unused):
    single = cu_seqlens_q is None
    flat_q = q.reshape(-1, q.shape[-2], q.shape[-1])
    offsets = list(range(q.shape[0] + 1)) if single else cu_seqlens_q.tolist()
    outputs = []
    for b, length in enumerate(cache_seqlens.tolist()):
        page_size = k_cache.shape[1]
        pages = page_table[b, :(length + page_size - 1) // page_size].long()
        k = k_cache[pages].flatten(0, 1)[:length].transpose(0, 1).unsqueeze(0)
        v = v_cache[pages].flatten(0, 1)[:length].transpose(0, 1).unsqueeze(0)
        queries = flat_q[offsets[b]:offsets[b+1]].transpose(0, 1).unsqueeze(0)
        query_length = queries.shape[2]
        if backend == 'portable_serial':
            from ssd.layers.paged_attention_triton import flash_attn_with_kvcache
            rows = []
            for j in range(query_length):
                visible = length-query_length+j+1 if causal else length
                row = flash_attn_with_kvcache(
                    queries[0,:,j].unsqueeze(0).unsqueeze(0), k_cache, v_cache,
                    torch.tensor([visible],device=q.device,dtype=torch.int32),
                    page_table[b:b+1],softmax_scale=softmax_scale,causal=causal)
                rows.append(row.reshape(1,queries.shape[1],1,queries.shape[-1]))
            result = torch.cat(rows, dim=2)
        elif backend == 'sdpa_flash':
            rows = []
            # Causal query j can see keys 0 .. length-query_length+j.
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                for j in range(query_length):
                    visible = length-query_length+j+1 if causal else length
                    rows.append(F.scaled_dot_product_attention(
                        queries[:, :, j:j+1], k[:, :, :visible], v[:, :, :visible],
                        scale=softmax_scale, enable_gqa=True))
            result = torch.cat(rows, dim=2)
        elif backend == 'sdpa_math':
            mask = None
            if causal:
                mask = torch.ones(query_length, length, dtype=torch.bool, device=q.device).tril(length-query_length)
            with sdpa_kernel(SDPBackend.MATH):
                # Explicit FP32 inputs isolate the custom attention's reduced-
                # precision arithmetic while retaining BF16 model weights/KV.
                result = F.scaled_dot_product_attention(
                    queries.float(), k.float(), v.float(), attn_mask=mask,
                    scale=softmax_scale, enable_gqa=True).to(q.dtype)
        else:
            raise ValueError(backend)
        outputs.append(result.squeeze(0).transpose(0, 1))
    return torch.cat(outputs).reshape_as(q)


def install_attention_control(backend):
    if backend == 'portable':
        return
    if backend not in ('sdpa_flash', 'sdpa_math', 'portable_serial'):
        raise ValueError(backend)
    module = importlib.import_module('ssd.layers.attention')
    original_forward = module.Attention.forward
    def control(self, q, k, v):
        context = module.get_context()
        # Only target decode/verification changes. Draft execution and prefill
        # retain the existing implementation in every comparison.
        if self.draft or context.is_prefill:
            return original_forward(self, q, k, v)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        module.store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
        single = context.cu_seqlens_q is None
        result = paged_sdpa(q.unsqueeze(1) if single else q, self.k_cache, self.v_cache,
            context.context_lens, context.block_tables, self.scale,
            cu_seqlens_q=context.cu_seqlens_q, max_seqlen_q=context.max_seqlen_q,
            backend=backend)
        return result.reshape(-1, self.num_heads*self.head_dim)
    module.Attention.forward = control


def install_rowwise_linear_control():
    """Slow reference: target decode uses M=1 linear arithmetic for every row.

    Paired with portable_serial attention, this preserves the original AR
    numerical path. It is an attribution/correctness control, not a speed fix.
    """
    from ssd.engine.model_runner import ModelRunner
    original_linear = F.linear
    original_run_model = ModelRunner.run_model
    state = {'active': False}
    def linear(inputs, weight, bias=None):
        if not state['active']:
            return original_linear(inputs, weight, bias)
        flat = inputs.reshape(-1,inputs.shape[-1])
        if flat.shape[0] <= 1:
            return original_linear(inputs, weight, bias)
        outputs = [original_linear(flat[i:i+1],weight,bias) for i in range(flat.shape[0])]
        return torch.cat(outputs).reshape(*inputs.shape[:-1],weight.shape[0])
    @torch.inference_mode()
    def run_model(self, input_ids, positions, is_prefill, *args, **kwargs):
        previous = state['active']
        state['active'] = not self.is_draft and not is_prefill
        try:
            return original_run_model(self,input_ids,positions,is_prefill,*args,**kwargs)
        finally:
            state['active'] = previous
    F.linear = linear
    ModelRunner.run_model = run_model
    return state


def install_tokenwise_pointwise_control(model, state):
    """Match AR shapes for compiled norms, rotary embedding, and activation too."""
    from ssd.layers.activation import SiluAndMul
    from ssd.layers.layernorm import RMSDNorm
    from ssd.layers.rotary_embedding import RotaryEmbedding
    from ssd.models.qwen3 import Qwen3Attention
    seen = set()
    def wrap(module, rows):
        if id(module) in seen:
            return
        seen.add(id(module))
        original = module.forward
        def forward(*args, **kwargs):
            n = args[0].shape[0]
            if not state['active'] or n <= rows:
                return original(*args, **kwargs)
            def part(value, start):
                if isinstance(value,torch.Tensor) and value.ndim and value.shape[0] == n:
                    return value[start:start+rows]
                return value
            outputs = [original(*(part(a,i) for a in args),
                                **{k:part(v,i) for k,v in kwargs.items()}) for i in range(0,n,rows)]
            if isinstance(outputs[0],tuple):
                return tuple(torch.cat([out[j] for out in outputs]) for j in range(len(outputs[0])))
            return torch.cat(outputs)
        module.forward = forward
    for module in model.modules():
        if isinstance(module,Qwen3Attention):
            wrap(module.q_norm,module.num_heads)
            wrap(module.k_norm,module.num_kv_heads)
        if isinstance(module,(RMSDNorm,SiluAndMul,RotaryEmbedding)):
            wrap(module,1)
