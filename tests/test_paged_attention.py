"""Correctness tests for the portable paged-attention kernels against a dense PyTorch reference.

Run: python tests/test_paged_attention.py   (or pytest tests/)
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "ssd"))
from ssd.layers.paged_attention_triton import flash_attn_varlen_func, flash_attn_with_kvcache  # noqa: E402

PAGE = 256


def make_cache(kv_lens, hkv, d, dtype, dev):
    n_pages = sum((L + PAGE - 1) // PAGE for L in kv_lens) + 3
    k_cache = torch.randn(n_pages, PAGE, hkv, d, dtype=dtype, device=dev)
    v_cache = torch.randn(n_pages, PAGE, hkv, d, dtype=dtype, device=dev)
    max_pages = max((L + PAGE - 1) // PAGE for L in kv_lens)
    perm = torch.randperm(n_pages, device=dev)  # scattered, non-contiguous page assignment
    page_table = torch.full((len(kv_lens), max_pages + 2), -1, dtype=torch.int32, device=dev)
    nxt = 0
    for b, L in enumerate(kv_lens):
        np_ = (L + PAGE - 1) // PAGE
        page_table[b, :np_] = perm[nxt:nxt + np_].int()
        nxt += np_
    return k_cache, v_cache, page_table


def reference(q_list, k_cache, v_cache, page_table, kv_lens, scale):
    outs = []
    for b, (qb, L) in enumerate(zip(q_list, kv_lens)):
        pages = page_table[b, : (L + PAGE - 1) // PAGE].long()
        k = k_cache[pages].flatten(0, 1)[:L].float()  # [L, Hkv, D]
        v = v_cache[pages].flatten(0, 1)[:L].float()
        hq, hkv = qb.shape[1], k.shape[1]
        k = k.repeat_interleave(hq // hkv, dim=1)
        v = v.repeat_interleave(hq // hkv, dim=1)
        s = torch.einsum("qhd,khd->hqk", qb.float(), k) * scale
        ql = qb.shape[0]
        mask = torch.ones(ql, L, dtype=torch.bool, device=qb.device).tril(L - ql)
        s = s.masked_fill(~mask, float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", s.softmax(-1), v))
    return torch.cat(outs)


def check(name, got, ref, tol=2e-2):
    err = (got.float() - ref).abs().max().item()
    status = "ok" if err < tol else "FAIL"
    print(f"{status:4s} {name:55s} max_abs_err={err:.2e}")
    assert err < tol, name


def test_all(dev="cuda"):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    for hq, hkv in [(32, 8), (16, 8)]:  # Qwen3-8B, Qwen3-0.6B
        d = 128
        scale = 1 / math.sqrt(d)
        # single-token decode
        for kv_lens in [[1], [17], [300], [1024], [4097], [5, 700, 2048]]:
            k_cache, v_cache, pt = make_cache(kv_lens, hkv, d, dtype, dev)
            q = torch.randn(len(kv_lens), 1, hq, d, dtype=dtype, device=dev)
            got = flash_attn_with_kvcache(q, k_cache, v_cache, torch.tensor(kv_lens, dtype=torch.int32, device=dev), pt, scale, True)
            ref = reference([q[b] for b in range(len(kv_lens))], k_cache, v_cache, pt, kv_lens, scale)
            check(f"decode Hq={hq} kv={kv_lens}", got.view(-1, hq, d), ref)
        # multi-token verify / glue (variable q lens via cu_seqlens_q)
        for q_lens, kv_lens in [([8], [8]), ([8], [520]), ([8, 8], [300, 1500]), ([3, 15, 1], [40, 2000, 257])]:
            k_cache, v_cache, pt = make_cache(kv_lens, hkv, d, dtype, dev)
            q_list = [torch.randn(L, hq, d, dtype=dtype, device=dev) for L in q_lens]
            cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=dev)
            got = flash_attn_with_kvcache(torch.cat(q_list), k_cache, v_cache, torch.tensor(kv_lens, dtype=torch.int32, device=dev),
                                          pt, scale, True, cu_seqlens_q=cu, max_seqlen_q=max(q_lens))
            ref = reference(q_list, k_cache, v_cache, pt, kv_lens, scale)
            check(f"verify Hq={hq} q={q_lens} kv={kv_lens}", got, ref)
        # prefill varlen (dense k/v)
        q_lens = [37, 512]
        q = torch.randn(sum(q_lens), hq, d, dtype=dtype, device=dev)
        k = torch.randn(sum(q_lens), hkv, d, dtype=dtype, device=dev)
        v = torch.randn(sum(q_lens), hkv, d, dtype=dtype, device=dev)
        cu = torch.tensor([0, 37, 549], dtype=torch.int32, device=dev)
        got = flash_attn_varlen_func(q, k, v, 512, cu, 512, cu, scale, True)
        refs = []
        for s, e in [(0, 37), (37, 549)]:
            kk = k[s:e].float().repeat_interleave(hq // hkv, 1)
            vv = v[s:e].float().repeat_interleave(hq // hkv, 1)
            sc = torch.einsum("qhd,khd->hqk", q[s:e].float(), kk) * scale
            sc = sc.masked_fill(~torch.ones(e - s, e - s, dtype=torch.bool, device=dev).tril(), float("-inf"))
            refs.append(torch.einsum("hqk,khd->qhd", sc.softmax(-1), vv))
        check(f"prefill Hq={hq} q={q_lens}", got, torch.cat(refs))

    # CUDA-graph capture of the decode path (metadata changes between replays)
    hq, hkv, d = 32, 8, 128
    k_cache, v_cache, pt = make_cache([2048], hkv, d, dtype, dev)
    q = torch.randn(1, 1, hq, d, dtype=dtype, device=dev)
    lens = torch.tensor([100], dtype=torch.int32, device=dev)
    flash_attn_with_kvcache(q, k_cache, v_cache, lens, pt, 1 / math.sqrt(d), True)  # compile
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = flash_attn_with_kvcache(q, k_cache, v_cache, lens, pt, 1 / math.sqrt(d), True)
    for L in [100, 1000, 2048]:
        lens.fill_(L)
        g.replay()
        ref = reference([q[0]], k_cache, v_cache, pt, [L], 1 / math.sqrt(d))
        check(f"cuda-graph replay kv={L}", out.view(-1, hq, d), ref)
    print("all paged-attention tests passed")


if __name__ == "__main__":
    test_all()
