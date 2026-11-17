"""Locate shape-dependent target arithmetic using identical tokens and prefix KV.

Diagnostic only: eager model dispatch, existing compiled pointwise operations,
no drafter, no acceptance/rollback and no NCCL. Never a performance measurement.
Source profiling/env.sh and run with python -O. Production defaults are untouched.
"""
import argparse
import copy
import os
from pathlib import Path

import run_engine  # Existing optimized-Python/Dynamo compatibility workaround.
import torch
import torch.nn.functional as F

from baseline_common import (ROOT, atomic_json, compute_processes, digest,
                             execution_fingerprint, file_digest, read_json,
                             software_fingerprint, validate_models)
from ssd import LLM, SamplingParams
from ssd.engine.sequence import Sequence
from ssd.layers.activation import SiluAndMul
from ssd.layers.attention import Attention
from ssd.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from ssd.layers.layernorm import RMSDNorm, RMSHeadNorm
from ssd.layers.linear import LinearBase
from ssd.layers.rotary_embedding import RotaryEmbedding
from ssd.models.qwen3 import Qwen3Attention
from ssd.utils.context import get_context, set_context


def difference(a, b):
    if a.shape != b.shape:
        raise ValueError(f'Unpaired tensor shapes: {a.shape}, {b.shape}')
    delta = (a.double() - b.double()).abs()
    return dict(exact=torch.equal(a, b), unequal=int((a != b).sum()),
                elements=a.numel(), max_abs=float(delta.max()),
                mean_abs=float(delta.mean()))


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        # Preserve view strides: changing layout can itself select a new kernel.
        output = torch.empty_strided(value.shape, value.stride(),
                                     dtype=value.dtype, device=value.device)
        return output.copy_(value)
    if isinstance(value, tuple):
        return tuple(clone_tree(v) for v in value)
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    return value


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, tuple):
        return [t for v in value for t in tensors(v)]
    return []


def first_rows(value, rows):
    if isinstance(value, torch.Tensor) and value.ndim:
        return value[:rows]
    if isinstance(value, tuple):
        return tuple(first_rows(v, rows) for v in value)
    if isinstance(value, dict):
        return {k: first_rows(v, rows) for k, v in value.items()}
    return value


def compare_outputs(a, b, rows):
    aa, bb = tensors(a), tensors(b)
    if len(aa) != len(bb):
        raise ValueError('Unpaired output structures')
    return [difference(x.reshape(-1, x.shape[-1])[:rows],
                       y.reshape(-1, y.shape[-1])[:rows]) for x, y in zip(aa, bb)]


class OperationTrace:
    def __init__(self, model):
        self.active = False
        self.records = []
        self.handles = []
        groups = {}
        # The factory shares RoPE instances; give each layer its own hook site.
        for module in model.modules():
            if isinstance(module, Qwen3Attention):
                module.rotary_emb = copy.copy(module.rotary_emb)
                module.rotary_emb._forward_hooks = copy.copy(module.rotary_emb._forward_hooks)
                groups[id(module.q_norm)] = module.num_heads
                groups[id(module.k_norm)] = module.num_kv_heads
        kinds = (LinearBase, RMSDNorm, RMSHeadNorm, RotaryEmbedding,
                 SiluAndMul, Attention, VocabParallelEmbedding)
        for name, module in model.named_modules():
            if isinstance(module, kinds):
                rows = groups.get(id(module), 1)
                self.handles.append(module.register_forward_hook(self.hook(name, rows)))

    def hook(self, name, rows):
        def record(module, args, output):
            if self.active:
                self.records.append(dict(name=name, module=module, rows=rows,
                                         args=clone_tree(args), output=clone_tree(output)))
        return record

    def run(self, fn):
        self.records = []
        self.active = True
        try:
            output = fn().clone()
        finally:
            self.active = False
        return output, self.records

    def close(self):
        for handle in self.handles:
            handle.remove()


def linear_controls(module, args, batch_output, candidates=False):
    """Change reduction policy and evaluate selected dot products in FP64."""
    x = args[0]
    bias = getattr(module, 'bias', None)
    saved = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        full = F.linear(x, module.weight, bias)[0]
        single = F.linear(x[:1], module.weight, bias)[0]
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = saved
    original_single = F.linear(x[:1], module.weight, bias)[0]
    original_batch = batch_output.reshape(-1, batch_output.shape[-1])[0]
    changed = (original_single != original_batch).nonzero().flatten()[:8]
    selected = torch.unique(torch.cat((changed, original_single.topk(2).indices,
                                       original_batch.topk(2).indices)))
    # Only selected weight rows are converted; never allocate the whole head in FP64.
    exact = (module.weight[selected].double() * x[0].double()).sum(-1)
    if bias is not None:
        exact += bias[selected].double()
    rounded = exact.to(x.dtype)
    result = dict(reduced_precision_default=saved,
                no_reduced_precision_row_vs_batch=difference(single, full),
                no_reduced_precision_row_vs_original=difference(single, original_single),
                no_reduced_precision_batch_vs_original=difference(full, original_batch),
                selected_dot_products=[dict(column=int(j), fp64_dot=float(e),
                    rounded_bf16=float(r), row_bf16=float(s), batch_bf16=float(b))
                    for j,e,r,s,b in zip(selected, exact, rounded,
                                        original_single[selected], original_batch[selected])])
    if candidates:
        if bias is not None:
            raise ValueError('Candidate probe currently requires bias-free projections')
        try:
            # BF16 inputs / tensor cores, but keep FP32 output before the final cast.
            fp32 = torch.mm(x, module.weight.t(), out_dtype=torch.float32).to(x.dtype)[0]
            result['fp32_output_gemm_vs_original_row'] = difference(fp32, original_single)
            result['fp32_output_gemm_vs_original_batch'] = difference(fp32, original_batch)
        except TypeError as error:
            result['fp32_output_gemm_unavailable'] = str(error)
        # A single batched call with independent M=1 problems; shared weights
        # have batch stride zero. This is a candidate, not a speedup claim.
        independent = torch.bmm(x.unsqueeze(1), module.weight.t().unsqueeze(0).expand(
            x.shape[0], -1, -1)).squeeze(1)[0]
        result['batched_gemv_vs_original_row'] = difference(independent, original_single)
    return result


def isolated_probe(record, context, candidates=False):
    module, args, rows = record['module'], record['args'], record['rows']
    set_context(**vars(context))
    if isinstance(module, Attention):
        # The first row sees exactly the same prefix and freshly written current KV.
        set_context(False, slot_mapping=context.slot_mapping[:1],
                    context_lens=context.context_lens - args[0].shape[0] + 1,
                    block_tables=context.block_tables)
    try:
        if isinstance(module, ParallelLMHead):
            output = F.linear(args[0][:1], module.weight)
        else:
            output = module.forward(*first_rows(args, rows))
        result = dict(name=record['name'], operation=type(module).__name__,
                      same_input_row_vs_batch=compare_outputs(output, record['output'], rows))
        if isinstance(module, (LinearBase, ParallelLMHead)) and (
                record['name'].startswith('model.layers.0.') or isinstance(module, ParallelLMHead)):
            result['linear_controls'] = linear_controls(module, args, record['output'], candidates)
        return result
    finally:
        set_context(**vars(context))


def logit_summary(logits):
    row = logits.reshape(-1, logits.shape[-1])[0].float()
    values, ids = row.topk(5)
    return dict(argmax=int(row.argmax()), top_ids=ids.tolist(), top_logits=values.tolist(),
                top_two_margin=float(values[0]-values[1]))


def row_linear(x, weight):
    return torch.cat([F.linear(x[i:i+1], weight) for i in range(x.shape[0])])


def batch_gemv(x, weight):
    return torch.bmm(x.unsqueeze(1), weight.t().unsqueeze(0).expand(
        x.shape[0], -1, -1)).squeeze(1)


def linear_sweep(records):
    """Check ALL token rows at ALL linear operations on identical inputs."""
    results = []
    for record in records:
        module = record['module']
        if not isinstance(module, (LinearBase, ParallelLMHead)):
            continue
        if getattr(module, 'bias', None) is not None:
            raise ValueError('All-row candidate requires bias-free projections')
        x = record['args'][0]
        reference = row_linear(x, module.weight)
        batch = record['output'].reshape_as(reference)
        candidate = batch_gemv(x, module.weight)
        transposed = torch.bmm(module.weight.unsqueeze(0).expand(x.shape[0], -1, -1),
                               x.unsqueeze(-1)).squeeze(-1)
        results.append(dict(name=record['name'], input_shape=list(x.shape),
            input_stride=list(x.stride()), weight_shape=list(module.weight.shape),
            gemm=difference(reference, batch), bmm=difference(reference, candidate),
            bmm_transposed=difference(reference, transposed),
            rows=[dict(row=i, unequal=int((reference[i]!=candidate[i]).sum()))
                  for i in range(x.shape[0])]))
    return results


def kernel_profiles(records):
    from torch.profiler import profile, ProfilerActivity
    results = []
    for record in records:
        module = record['module']
        if not isinstance(module, (LinearBase, ParallelLMHead)) or not (
                record['name'].startswith('model.layers.0.') or isinstance(module, ParallelLMHead)):
            continue
        x, weight = record['args'][0], module.weight
        entry = dict(name=record['name'], kernels={})
        for name, fn in dict(ar=lambda: F.linear(x[:1], weight),
                             gemm=lambda: F.linear(x, weight),
                             bmm=lambda: batch_gemv(x, weight)).items():
            fn(); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                fn(); torch.cuda.synchronize()
            entry['kernels'][name] = [e.name for e in prof.events()
                if e.device_type == torch.autograd.DeviceType.CUDA]
        results.append(entry)
    return results


def linear_cost(records):
    """Captured sweep of actual layer weights; isolated cost, not request latency."""
    operations = [(r['args'][0], r['module'].weight) for r in records
                  if isinstance(r['module'], (LinearBase, ParallelLMHead))]
    results = {}
    # All 145 real weights are visited per sweep, exceeding the device L2 cache.
    for name, fn in [('gemm', F.linear), ('bmm', batch_gemv), ('serial', row_linear)]:
        for x, weight in operations:
            fn(x, weight)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        outputs = []
        with torch.cuda.graph(graph):
            for x, weight in operations:
                outputs.append(fn(x, weight))
        for _ in range(3):
            graph.replay()
        samples = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):
                graph.replay()
            end.record(); end.synchronize()
            samples.append(start.elapsed_time(end)/5)
        results[name] = dict(sweep_ms=samples, mean_ms=sum(samples)/len(samples),
                             operations=len(operations), graph_replays_per_sample=5)
        del graph, outputs
    return results


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--widths', type=int, nargs='+', default=[7, 8])
    parser.add_argument('--families', nargs='+', choices=['chat','code','doc'],
                        default=['chat','code','doc'])
    parser.add_argument('--linear-candidates', action='store_true')
    parser.add_argument('--all-linear-rows', action='store_true')
    parser.add_argument('--linear-cost', action='store_true',
                        help='Isolated graph sweep costs; not a full-request speedup claim')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Preserve old evidence; choose a new output path')
    if any(w < 2 or w > 16 for w in args.widths):
        parser.error('Widths must be between 2 and 16')
    definition_path = ROOT/'results/correctness/ar_compatible_2026-11-16_gpu0_v2/definition.json'
    ar_path = ROOT/'results/correctness/archived_68df65e_2026-11-16/ar.json'
    sd_path = ar_path.with_name('sd.json')
    definition = read_json(definition_path)
    if Path(os.environ['SSD_TARGET_MODEL']).resolve() != Path(definition['models']['target']['path']).resolve():
        raise ValueError('Configured target differs from the fingerprinted model')
    validate_models({'target': definition['models']['target']})
    uuids = [g['uuid'] for g in definition['hardware']['gpus']]
    from run_ar_compatible_gate import require_memory
    require_memory(uuids, 'ar')
    ar = {r['id']: r for r in read_json(ar_path)['requests']}
    sd = {r['id']: r for r in read_json(sd_path)['requests']}
    result = dict(diagnostic_only=True, status='running', cuda_graphs=False,
        compiled_pointwise=True, kv_fraction=0.10, capacity=4096,
        linear_candidates=args.linear_candidates,
        all_linear_rows=args.all_linear_rows, linear_cost=args.linear_cost,
        target=definition['models']['target'], execution=execution_fingerprint(),
        diagnostic_sha256=file_digest(__file__), software=software_fingerprint(),
        sources={str(p): file_digest(p) for p in [definition_path, ar_path, sd_path]},
        gpu_processes_start=compute_processes(uuids), requests=[])
    atomic_json(args.out, result)
    llm = LLM(os.environ['SSD_TARGET_MODEL'], num_gpus=1, max_num_seqs=1,
              max_model_len=4096, enforce_eager=True, gpu_memory_utilization=0.10,
              enable_prefix_cache=False)
    runner = llm.model_runner
    result['gpu'] = torch.cuda.get_device_name(0)
    trace = OperationTrace(runner.model)
    attention = [m for m in runner.model.modules() if isinstance(m, Attention)]
    try:
        for prompt in definition['prompts']:
            if prompt['id'] not in ar or prompt['family'] not in args.families:
                continue
            ids = ar[prompt['id']]['output_token_ids']
            other = sd[prompt['id']]['output_token_ids']
            d = next(i for i,(a,b) in enumerate(zip(ids, other)) if a != b)
            if d < 1 or d + max(args.widths) > len(ids):
                raise ValueError('Divergence lacks sufficient replay context')
            if digest(prompt['token_ids']) != ar[prompt['id']]['input_token_ids_sha256']:
                raise ValueError('Frozen input mismatch')
            seq = Sequence(prompt['token_ids'], SamplingParams(temperature=0))
            seq.block_table = list(range((len(seq)+d+max(args.widths)+255)//256))
            predicted = [runner.run([seq], True)[0]]
            seq.num_cached_tokens = len(seq)
            for i in range(d-1):
                seq.append_token(ids[i])
                predicted.append(runner.run([seq], False)[0])
                seq.num_cached_tokens += 1
            seq.append_token(ids[d-1])
            p = len(seq)-1
            bt = torch.tensor([seq.block_table], dtype=torch.int32, device=runner.device)
            positions = torch.arange(p, p+max(args.widths), device=runner.device)
            slots = bt[0,positions//256].long()*256+positions%256
            caches = []
            for module in attention:
                module.speculate = True
                for cache in (module.k_cache, module.v_cache):
                    flat = cache.view(-1,*cache.shape[2:])
                    caches.append((flat, flat[slots].clone()))

            def forward(width):
                for flat, saved in caches:
                    flat[slots] = saved
                set_context(False,
                    cu_seqlens_q=None if width==1 else torch.tensor([0,width],dtype=torch.int32,device=runner.device),
                    max_seqlen_q=width, slot_mapping=slots[:width].int(),
                    context_lens=torch.tensor([p+width],dtype=torch.int32,device=runner.device),
                    block_tables=bt)
                return runner.run_model(torch.tensor(ids[d-1:d-1+width],device=runner.device),
                                        positions[:width], False, width==1)

            plain = forward(1).clone()
            single, single_records = trace.run(lambda: forward(1))
            item = dict(id=prompt['id'], family=prompt['family'], divergence_zero_based=d,
                input_sha256=digest(prompt['token_ids']), prefix_output_ids=ids[:d],
                teacher_forced_first_difference=next((i for i,(a,b) in enumerate(zip(predicted,ids)) if a!=b), None),
                single=logit_summary(single), saved_ar_token=ids[d], saved_sd_token=other[d],
                instrumentation_matches_plain=difference(plain, single), widths={})
            for width in args.widths:
                plain_batch = forward(width).clone()
                batch, batch_records = trace.run(lambda: forward(width))
                context = get_context()
                operations = []
                if len(single_records) != len(batch_records):
                    raise ValueError('Execution traces differ in operation count')
                for single_record, record in zip(single_records, batch_records):
                    if single_record['name'] != record['name']:
                        raise ValueError('Execution traces differ in operation order')
                    probe = isolated_probe(record, context, args.linear_candidates)
                    probe['end_to_end_row_vs_batch'] = compare_outputs(
                        single_record['output'], record['output'], record['rows'])
                    operations.append(probe)
                item['widths'][str(width)] = dict(logits=logit_summary(batch),
                    instrumentation_matches_plain=difference(plain_batch, batch),
                    logit_delta=compare_outputs(single, batch, 1), operations=operations,
                    first_differing_operation=next((r['name'] for r in operations if any(
                        not v['exact'] for v in r['end_to_end_row_vs_batch'])), None),
                    single_after_probes=logit_summary(forward(1)))
                if args.all_linear_rows:
                    item['widths'][str(width)]['linear_all_rows'] = linear_sweep(batch_records)
                if args.linear_cost and not result['requests']:
                    item['widths'][str(width)]['kernel_profiles'] = kernel_profiles(batch_records)
                    item['widths'][str(width)]['isolated_linear_cost'] = linear_cost(batch_records)
                print(f"{prompt['id']} width={width}: {item['widths'][str(width)]['first_differing_operation']}", flush=True)
            result['requests'].append(item)
            atomic_json(args.out, result)
        result['status'] = 'complete'
        result['gpu_processes_end'] = compute_processes(uuids)
        atomic_json(args.out, result)
    finally:
        trace.close()
        llm.exit(hard=True)


if __name__ == '__main__':
    main()
