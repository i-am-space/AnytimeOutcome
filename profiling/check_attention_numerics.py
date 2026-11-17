"""Same-input causal attention at page and split-KV boundaries.

One GPU, no model weights or other worker required. Diagnostic exactness checks,
including CUDA graph replay. Does not measure speed or change engine defaults.
"""
import argparse
from pathlib import Path

import run_engine
import torch

from baseline_common import atomic_json, execution_fingerprint, file_digest, software_fingerprint
from diagnose_fast_numerics import difference
from numerics_attention import parallel_ar_attention
from ssd.layers.paged_attention_triton import flash_attn_with_kvcache, _num_splits


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new output filename')
    torch.manual_seed(0)
    device = torch.device('cuda:0')
    # Qwen3-8B GQA: 32 query heads, 8 KV heads, head dimension 128.
    key = torch.randn(17,256,8,128,device=device,dtype=torch.bfloat16)
    value = torch.randn_like(key)
    table = torch.arange(17,device=device,dtype=torch.int32)[None]
    result = dict(diagnostic_only=True, gpu=torch.cuda.get_device_name(), seed=0,
        execution=execution_fingerprint(), diagnostic_sha256=file_digest(__file__),
        parallel_ar_attention_sha256=file_digest(Path(__file__).with_name('numerics_attention.py')),
        software=software_fingerprint(), splits=_num_splits(1,8,device), cases=[])
    for width in [5,7,8,9]:
        for first_visible in [250,255,256,257,2042,2047,2048,2049,4080]:
            q = torch.randn(width,32,128,device=device,dtype=torch.bfloat16)
            total = torch.tensor([first_visible+width-1],device=device,dtype=torch.int32)
            cu = torch.tensor([0,width],device=device,dtype=torch.int32)
            lengths = [torch.tensor([first_visible+i],device=device,dtype=torch.int32)
                       for i in range(width)]
            def batch():
                return flash_attn_with_kvcache(q,key,value,total,table,
                    cu_seqlens_q=cu,max_seqlen_q=width)
            def serial():
                return torch.cat([flash_attn_with_kvcache(q[i:i+1,None],key,value,length,table)
                                  .reshape(1,32,128) for i,length in enumerate(lengths)])
            single = serial(); multi = batch()
            parallel = parallel_ar_attention(q,key,value,total,table)
            # Warm actual variants before graph capture; returned tensors stay alive.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = batch()
                captured_parallel = parallel_ar_attention(q,key,value,total,table)
            graph.replay(); torch.cuda.synchronize()
            entry = dict(width=width,first_visible=first_visible,
                single_vs_batch=difference(single,multi), graph_vs_eager=difference(multi,captured),
                single_vs_parallel=difference(single,parallel),
                parallel_graph_vs_eager=difference(parallel,captured_parallel),
                rows=[dict(row=i,unequal=int((single[i]!=multi[i]).sum()),
                           max_abs=float((single[i].float()-multi[i].float()).abs().max()))
                      for i in range(width)])
            result['cases'].append(entry)
            atomic_json(args.out,result)
            print(f'width={width} first_visible={first_visible} unequal={entry["single_vs_batch"]["unequal"]}',flush=True)
    result['status']='complete'
    atomic_json(args.out,result)


if __name__=='__main__':
    main()
