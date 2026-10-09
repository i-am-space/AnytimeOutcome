"""Portable (sm80+) replacements for the sgl_kernel FA3 calls used by ssd/layers/attention.py.

sgl_kernel.flash_attn is FlashAttention-3, which only runs on Hopper (sm90). These kernels keep
the same call signatures for the subset SSD uses, so the engine runs on Ada (L40S, sm89) and Ampere:

  flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, page_table, softmax_scale, causal,
                          cu_seqlens_q=None, max_seqlen_q=None)
      Paged-KV attention for single-token decode (q: [B, 1, Hq, D]) and multi-token verify / glue
      decode (q: [total_q, Hq, D] + cu_seqlens_q). Split-KV ("flash-decoding") Triton kernel; all
      metadata stays on device, so it is safe inside CUDA-graph capture.

  flash_attn_varlen_func(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k,
                         softmax_scale, causal, block_table=None)
      Prefill. Uses torch SDPA (FlashAttention-2 backend) per sequence; prefill is never captured.

Causal masking is bottom-right aligned (FA convention): query j of a sequence with q_len queries and
kv_len keys may attend to keys [0, kv_len - q_len + j].
"""
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_NUM_SMS = {}


@triton.jit
def _paged_attn_split_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, seqlens_ptr, page_table_ptr, cu_q_ptr,
    out_part_ptr, lse_part_ptr,
    stride_qt, stride_qh,
    stride_kb, stride_kp, stride_kh,
    stride_pt_b,
    stride_ot, stride_os, stride_oh,
    stride_lt, stride_ls,
    scale,
    G: tl.constexpr, HEAD_DIM: tl.constexpr, PAGE: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr, HAS_CU_Q: tl.constexpr, CAUSAL: tl.constexpr,
):
    b = tl.program_id(0)
    h_kv = tl.program_id(1)
    split = tl.program_id(2)

    if HAS_CU_Q:
        q_start = tl.load(cu_q_ptr + b)
        q_len = tl.load(cu_q_ptr + b + 1) - q_start
    else:
        q_start = b
        q_len = 1
    kv_len = tl.load(seqlens_ptr + b)

    # Rows of the tile are (query token, head-in-group) pairs sharing one KV head (GQA packing).
    rows = tl.arange(0, BLOCK_M)
    qi = rows // G
    hq = h_kv * G + rows % G
    row_valid = (qi < q_len) & (qi < BLOCK_Q)
    d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_ptr + (q_start + qi)[:, None] * stride_qt + hq[:, None] * stride_qh + d[None, :]
    q = tl.load(q_ptrs, mask=row_valid[:, None], other=0.0)

    # This split's contiguous slice of the KV sequence, aligned to BLOCK_N.
    chunk = tl.cdiv(tl.cdiv(kv_len, NUM_SPLITS), BLOCK_N) * BLOCK_N
    kv_lo = split * chunk
    kv_hi = tl.minimum(kv_lo + chunk, kv_len)

    # Last key each row may see (inclusive bound is kv_len - q_len + qi).
    if CAUSAL:
        row_limit = kv_len - q_len + qi + 1
    else:
        row_limit = kv_len + qi * 0

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = scale * 1.4426950408889634  # fold log2(e) so we can use exp2

    for start in range(kv_lo, kv_hi, BLOCK_N):
        n = start + tl.arange(0, BLOCK_N)
        n_valid = n < kv_hi
        page = tl.load(page_table_ptr + b * stride_pt_b + start // PAGE)
        off = n % PAGE
        kv_off = page.to(tl.int64) * stride_kb + off[:, None] * stride_kp + h_kv * stride_kh + d[None, :]
        k = tl.load(k_cache_ptr + kv_off, mask=n_valid[:, None], other=0.0)
        v = tl.load(v_cache_ptr + kv_off, mask=n_valid[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)) * qk_scale
        allowed = (n[None, :] < row_limit[:, None]) & n_valid[None, :] & row_valid[:, None]
        s = tl.where(allowed, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    has = l_i > 0
    out = acc / tl.where(has, l_i, 1.0)[:, None]
    lse = tl.where(has, m_i + tl.log2(tl.where(has, l_i, 1.0)), float("-inf"))  # log2 domain

    t = q_start + qi
    o_ptrs = out_part_ptr + t[:, None] * stride_ot + split * stride_os + hq[:, None] * stride_oh + d[None, :]
    tl.store(o_ptrs, out, mask=row_valid[:, None])
    tl.store(lse_part_ptr + t * stride_lt + split * stride_ls + hq, lse, mask=row_valid)


@triton.jit
def _combine_kernel(
    out_part_ptr, lse_part_ptr, out_ptr,
    stride_ot, stride_os, stride_oh,
    stride_lt, stride_ls,
    stride_out_t, stride_out_h,
    HEAD_DIM: tl.constexpr, NUM_SPLITS: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.arange(0, NUM_SPLITS)
    d = tl.arange(0, HEAD_DIM)
    lse = tl.load(lse_part_ptr + t * stride_lt + s * stride_ls + h)
    m = tl.max(lse, 0)
    m = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp2(lse - m)
    o = tl.load(out_part_ptr + t * stride_ot + s[:, None] * stride_os + h * stride_oh + d[None, :])
    o = tl.sum(o * w[:, None], 0) / tl.maximum(tl.sum(w, 0), 1e-30)
    tl.store(out_ptr + t * stride_out_t + h * stride_out_h + d, o.to(out_ptr.dtype.element_ty))


def _num_splits(batch, num_kv_heads, device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _NUM_SMS:
        _NUM_SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    target = 2 * _NUM_SMS[idx]
    splits = max(1, min(32, target // max(1, batch * num_kv_heads)))
    return 1 << (splits.bit_length() - 1)  # power of two for tl.arange in the combine kernel


def flash_attn_with_kvcache(
    q, k_cache, v_cache, cache_seqlens, page_table, softmax_scale=None, causal=True,
    cu_seqlens_q=None, max_seqlen_q=None, **unused,
):
    single = cu_seqlens_q is None
    if single:  # q: [B, 1, Hq, D]
        assert q.dim() == 4 and q.shape[1] == 1, "decode path expects q of shape [B, 1, Hq, D]"
        q3 = q.reshape(q.shape[0], q.shape[2], q.shape[3])
        batch, block_q = q.shape[0], 1
    else:  # q: [total_q, Hq, D]
        q3 = q
        batch = cu_seqlens_q.numel() - 1
        block_q = triton.next_power_of_2(int(max_seqlen_q))

    total_q, num_heads, head_dim = q3.shape
    _, page, num_kv_heads, _ = k_cache.shape
    group = num_heads // num_kv_heads
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    block_m = max(16, triton.next_power_of_2(block_q * group))
    block_n = 64 if block_m <= 64 else 32
    assert page % block_n == 0, "page size must be a multiple of BLOCK_N"
    splits = _num_splits(batch, num_kv_heads, q.device)

    out_part = torch.empty(total_q, splits, num_heads, head_dim, dtype=torch.float32, device=q.device)
    lse_part = torch.empty(total_q, splits, num_heads, dtype=torch.float32, device=q.device)
    out = torch.empty_like(q3)

    _paged_attn_split_kernel[(batch, num_kv_heads, splits)](
        q3, k_cache, v_cache, cache_seqlens, page_table, cu_seqlens_q if not single else cache_seqlens,
        out_part, lse_part,
        q3.stride(0), q3.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        page_table.stride(0),
        out_part.stride(0), out_part.stride(1), out_part.stride(2),
        lse_part.stride(0), lse_part.stride(1),
        softmax_scale,
        G=group, HEAD_DIM=head_dim, PAGE=page,
        BLOCK_Q=block_q, BLOCK_M=block_m, BLOCK_N=block_n,
        NUM_SPLITS=splits, HAS_CU_Q=not single, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    _combine_kernel[(total_q, num_heads)](
        out_part, lse_part, out,
        out_part.stride(0), out_part.stride(1), out_part.stride(2),
        lse_part.stride(0), lse_part.stride(1),
        out.stride(0), out.stride(1),
        HEAD_DIM=head_dim, NUM_SPLITS=splits,
        num_warps=4,
    )
    return out.view_as(q) if single else out


def flash_attn_varlen_func(
    q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k,
    softmax_scale=None, causal=True, block_table=None, **unused,
):
    """Prefill attention. q: [total_q, Hq, D]; k/v: [total_k, Hkv, D] or paged caches with block_table."""
    num_heads, head_dim = q.shape[1], q.shape[2]
    out = torch.empty_like(q)
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    for i in range(len(cu_q) - 1):
        qs, qe = cu_q[i], cu_q[i + 1]
        kv_len = cu_k[i + 1] - cu_k[i]
        if block_table is not None:  # gather this sequence's keys from the paged cache
            page = k.shape[1]
            pages = block_table[i, : (kv_len + page - 1) // page].long()
            ki = k[pages].flatten(0, 1)[:kv_len]
            vi = v[pages].flatten(0, 1)[:kv_len]
        else:
            ki, vi = k[cu_k[i]:cu_k[i + 1]], v[cu_k[i]:cu_k[i + 1]]
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)
        ki = ki.transpose(0, 1).unsqueeze(0)
        vi = vi.transpose(0, 1).unsqueeze(0)
        q_len = qe - qs
        if causal and q_len != kv_len:  # bottom-right aligned causal mask
            mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device).tril(kv_len - q_len)
            o = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, scale=softmax_scale, enable_gqa=True)
        else:
            o = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal, scale=softmax_scale, enable_gqa=True)
        out[qs:qe] = o.squeeze(0).transpose(0, 1)
    return out
