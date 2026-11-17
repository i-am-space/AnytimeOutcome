"""Parallel query dispatch with the original AR attention reduction per query.

Diagnostic prototype: one kernel launch for all queries, fixed AR split count,
and each query's actual visible length. No target model numerical defaults change.
"""
import torch
import triton

from ssd.layers.paged_attention_triton import (_combine_kernel, _num_splits,
                                               _paged_attn_split_kernel)


def parallel_ar_attention(q, key, value, total_lens, table):
    if q.ndim!=3:
        raise ValueError('Parallel AR attention requires [query,head,dimension]')
    queries,heads,dim=q.shape
    _,page,kv_heads,_=key.shape
    group=heads//kv_heads
    splits=_num_splits(1,kv_heads,q.device)  # Original AR dispatch, independent of width.
    if total_lens.numel()==1 and table.shape[0]==1:
        lengths=total_lens-queries+1+torch.arange(queries,device=q.device,dtype=torch.int32)
        tables=table.expand(queries,-1)
    elif total_lens.numel()==queries and table.shape[0]==queries:
        # Extra single-query decode buckets captured by the upstream helper.
        # Each query is an independent sequence with the same per-query kernel.
        lengths=total_lens
        tables=table
    else:
        raise ValueError('Parallel AR attention requires one request or one query per request')
    partial=torch.empty(queries,splits,heads,dim,dtype=torch.float32,device=q.device)
    lse=torch.empty(queries,splits,heads,dtype=torch.float32,device=q.device)
    output=torch.empty_like(q)
    _paged_attn_split_kernel[(queries,kv_heads,splits)](
        q,key,value,lengths,tables,lengths,partial,lse,
        q.stride(0),q.stride(1),key.stride(0),key.stride(1),key.stride(2),
        tables.stride(0),partial.stride(0),partial.stride(1),partial.stride(2),
        lse.stride(0),lse.stride(1),dim**-0.5,
        G=group,HEAD_DIM=dim,PAGE=page,BLOCK_Q=1,
        BLOCK_M=max(16,triton.next_power_of_2(group)),BLOCK_N=64,
        NUM_SPLITS=splits,HAS_CU_Q=False,CAUSAL=True,num_warps=4,num_stages=2)
    _combine_kernel[(queries,heads)](
        partial,lse,output,partial.stride(0),partial.stride(1),partial.stride(2),
        lse.stride(0),lse.stride(1),output.stride(0),output.stride(1),
        HEAD_DIM=dim,NUM_SPLITS=splits,num_warps=4)
    return output
