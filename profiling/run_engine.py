"""Shared benchmark driver for AR / synchronous SD / SSD on the SSD engine (one request at a time).

Measures full-request latency including prefills, first delivery, decode time, and committed tokens.
Profiler capture is opt-in. Frozen campaign manifests bypass prompt selection and tokenization.

Examples:
  python profiling/run_engine.py --mode ar  --n 4 --out results/engine/ar.json
  python profiling/run_engine.py --mode sd  --k 6 --n 4 --out results/engine/sd_k6.json
  python profiling/run_engine.py --mode ssd --k 7 --f 3 --n 4 --out results/engine/ssd_k7_f3.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import ssd.paths  # noqa: F401,E402  (sets TORCH_CUDA_ARCH_LIST before flashinfer import)
import torch  # noqa: E402

if sys.flags.optimize:
    # torch 2.10 + Python 3.13: under `python -O` Dynamo's assert detection compiles a probe `assert`
    # that -O strips, then raises StopIteration. There are no asserts to normalize under -O anyway.
    import torch._dynamo.symbolic_convert as _sc
    _sc._detect_and_normalize_assert_statement = lambda *a, **k: False
from ssd import LLM, SamplingParams  # noqa: E402
from ssd.engine.llm_engine import METRICS  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from workloads import get_prompts  # noqa: E402
from baseline_common import (allocation, atomic_json, digest, execution_fingerprint, ordered_prompts,
                             read_json, software_fingerprint, validate_manifest)
from baseline_common import worker_settings


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ar", "sd", "ssd"], required=True)
    p.add_argument("--target", default=os.environ.get("SSD_TARGET_MODEL"))
    p.add_argument("--draft", default=os.environ.get("SSD_DRAFT_MODEL"))
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--f", type=int, default=3)
    p.add_argument("--backup", choices=["jit", "fast"], default="jit")
    p.add_argument("--families", nargs="+", default=["chat", "code", "doc"])
    p.add_argument("--n", type=int, default=4, help="prompts per family")
    p.add_argument("--split", default="heldout")
    p.add_argument("--output_len", type=int, default=256)
    p.add_argument("--max_prompt_tokens", type=int, default=2048)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--gpu_mem_util", type=float, default=0.7,
                   help="fraction of free memory for KV cache; use ~0.08 under ncu (replay backs up written buffers)")
    p.add_argument("--draft_gpu_mem_util", type=float)
    p.add_argument('--target-numerics', choices=['batched', 'ar_compatible', 'shared_fixed16'], default='batched')
    p.add_argument("--outcome-depths", type=int, nargs="+", help="continuation depth per recovery-token rank")
    p.add_argument("--early-publish-tokens", type=int, default=0)
    p.add_argument("--publication-policy", choices=['uniform', 'bounded', 'adaptive', 'overlap'], default='uniform')
    p.add_argument("--publication-max-inflight", type=int, default=2)
    p.add_argument("--drain-draft-on-finish", action="store_true")
    p.add_argument("--ignore_eos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--protocol", choices=['fixed', 'natural'])
    p.add_argument("--profile-capture", action="store_true")
    p.add_argument("--disable-prefix-cache", action="store_true")
    p.add_argument("--manifest")
    p.add_argument("--job")
    p.add_argument("--order-seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def measure_request(llm, item, sampling, *, synchronize, clock=time.perf_counter, profile_capture=False):
    """Only generation and output readiness are timed; callback logging is disabled."""
    synchronize()
    if profile_capture:
        torch.cuda.nvtx.range_push(f"request:{item['id']}")
    first_delivery = None
    import builtins
    original_print = builtins.print
    builtins.print = lambda *a, **k: None
    t0 = clock()
    def delivered(seq_id, tokens):
        nonlocal first_delivery
        if tokens and first_delivery is None:
            first_delivery = clock() - t0
    try:
        outputs, metrics = llm.generate([item['token_ids']], [sampling], use_tqdm=False,
                                        stream_callback=delivered)
        synchronize()
        latency = clock() - t0
    finally:
        builtins.print = original_print
        if profile_capture:
            torch.cuda.nvtx.range_pop()
    from ssd.engine.outcome_depth import STAT_NAMES
    steps = metrics['target_step_times']
    accepted = metrics['accepted_suffix_lens_with_recovery']
    hits = metrics['cache_hits']
    return dict(id=item['id'], family=item['family'], category=item['category'],
        input_token_ids_sha256=digest(item['token_ids']), prompt_tokens=len(item['token_ids']),
        output_tokens=len(outputs[0]['token_ids']), latency_s=latency, first_delivery_s=first_delivery,
        prefill_s=metrics['prefill_total_time'], decode_s=metrics['decode_total_time'],
        num_steps=len(steps), mean_step_ms=1000*sum(steps[1:])/max(1,len(steps)-1),
        tokens_per_step=sum(accepted)/len(accepted) if accepted else 1.0,
        accepted_draft_tokens=sum(n-1 for n in accepted),
        verified_draft_tokens=sum(metrics.get('verified_lengths', [])) if metrics.get('verified_lengths') else
                              len(accepted)*llm.config.speculate_k if accepted else 0,
        outcome_depth_stats=(dict(zip(STAT_NAMES,
            list(llm.outcome_depth_feedback[:]))) if getattr(llm, 'outcome_depth_feedback', None) is not None else None),
        draft_wait_s=sum(metrics.get('draft_wait_times', [])),
        publication_records=list(metrics.get('publication_records', [])),
        publication_policy_stats=(list(llm.publication_feedback[:])
            if getattr(llm, 'publication_feedback', None) is not None else None),
        final_drain_s=sum(metrics.get('final_drain_times', [])),
        draft_peak_memory=list(metrics.get("draft_peak_memory", [])),
        accepted_suffix_lens_with_recovery=list(accepted),
        cache_hit_rate=sum(hits)/len(hits) if hits else None,
        output_token_ids=outputs[0]['token_ids'])


def execute_requests(llm, prompts, warm, sampling, *, synchronize, profile_capture=False,
                     clock=time.perf_counter):
    for item in warm:
        llm.generate([item['token_ids']],[sampling],use_tqdm=False,stream_callback=lambda *args:None)
    synchronize()
    if profile_capture:
        torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push('timed_region')
    try:
        return [measure_request(llm,item,sampling,synchronize=synchronize,clock=clock,
                                profile_capture=profile_capture) for item in prompts]
    finally:
        if profile_capture:
            torch.cuda.nvtx.range_pop()
            torch.cuda.profiler.stop()


def main():
    args = parse()
    if args.protocol:
        args.ignore_eos = args.protocol == 'fixed'
    manifest = read_json(args.manifest) if args.manifest else None
    job = read_json(args.job) if args.job else None
    if manifest:
        if args.target_numerics != manifest.get('target_numerics', 'batched'):
            raise ValueError('Numerical backend differs from the versioned campaign')
        if args.outcome_depths is not None:
            raise ValueError("Unequal outcome depths require a separate versioned study")
        if args.early_publish_tokens:
            raise ValueError('Early publication is a separate study, not baseline tuning')
        if args.drain_draft_on_finish != (manifest.get('drain_draft_on_finish', False) and args.mode=='ssd'):
            raise ValueError('Request-end draining differs from the frozen campaign')
        validate_manifest(manifest)
        args.target=args.target or manifest['models']['target']['path']
        args.draft=args.draft or manifest['models']['draft']['path']
        if execution_fingerprint() != manifest['execution'] or software_fingerprint() != manifest['software']:
            raise ValueError('Worker execution environment differs from frozen campaign')
        for role, path in [('target',args.target),('draft',args.draft)]:
            if os.path.realpath(path) != manifest['models'][role]['path']:
                raise ValueError(f'{role} model differs from frozen manifest')
    if job:
        if not manifest or digest({k:v for k,v in job.items() if k!='job_sha256'}) != job['job_sha256']:
            raise ValueError('Invalid campaign job')
        if job['manifest_sha256'] != manifest['manifest_sha256']:
            raise ValueError('Job manifest differs')
        for key,value in worker_settings(job).items():
            if vars(args).get(key)!=value:
                raise ValueError(f'Worker setting differs from declared campaign: {key}')
    if os.environ.get("AO_SPAWN_EXECUTABLE"):
        # Spawn the SSD drafter through a differently named interpreter so Nsight Compute can profile
        # only that process (--target-processes-filter); serializing both processes deadlocks NCCL.
        import multiprocessing
        multiprocessing.set_executable(os.environ["AO_SPAWN_EXECUTABLE"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if manifest:
        prompts = ordered_prompts(manifest,args.split,args.order_seed)
        warm = manifest['splits']['warmup']
        if job and [r['id'] for r in prompts] != job['prompt_ids']:
            raise ValueError('Worker prompt order differs from frozen job')
    else:
        tok = AutoTokenizer.from_pretrained(args.target)
        prompts = get_prompts(args.families, args.n, split=args.split, tokenizer=tok,
                              max_prompt_tokens=args.max_prompt_tokens)
        warm = get_prompts(["chat"], args.warmup, split="tune", tokenizer=tok)

    spec = args.mode in ("sd", "ssd")
    llm = LLM(
        args.target,
        num_gpus=2 if args.mode == "ssd" else 1,
        speculate=spec,
        draft=args.draft if spec else args.target,
        speculate_k=args.k,
        draft_async=args.mode == "ssd",
        async_fan_out=args.f,
        jit_speculate=args.backup == "jit",
        max_num_seqs=1,
        max_model_len=args.max_model_len,
        enforce_eager=args.eager,
        kvcache_block_size=256,
        gpu_memory_utilization=args.gpu_mem_util,
        draft_gpu_memory_utilization=(args.draft_gpu_mem_util if args.draft_gpu_mem_util is not None else
                                      args.gpu_mem_util if args.gpu_mem_util < 0.7 else None),
        enable_prefix_cache=not args.disable_prefix_cache,
        target_numerics=args.target_numerics,
        outcome_depths=args.outcome_depths,
        early_publish_tokens=args.early_publish_tokens,
        publication_policy=args.publication_policy,
        publication_max_inflight=args.publication_max_inflight,
        drain_draft_on_finish=args.drain_draft_on_finish,
    )
    sp = SamplingParams(temperature=0.0, max_new_tokens=args.output_len, ignore_eos=args.ignore_eos)

    results=execute_requests(llm,prompts,warm,sp,synchronize=torch.cuda.synchronize,
                             profile_capture=args.profile_capture)

    tot_tok = sum(r["output_tokens"] for r in results)
    tot_lat = sum(r["latency_s"] for r in results)
    summary = dict(
        mode=args.mode, k=args.k if spec else None, f=args.f if args.mode == "ssd" else None,
        backup=args.backup if args.mode == "ssd" else None, output_len=args.output_len, n_requests=len(results),
        mean_latency_s=tot_lat / len(results), committed_tok_per_s=tot_tok / tot_lat,
        decode_tok_per_s=tot_tok / sum(r["decode_s"] for r in results) if sum(r['decode_s'] for r in results)>0 else None,
        mean_tokens_per_step=sum(r["tokens_per_step"] for r in results) / len(results),
        mean_step_ms=sum(r["mean_step_ms"] for r in results) / len(results),
        gpu=torch.cuda.get_device_name(0),
    )
    print(json.dumps(summary, indent=2), flush=True)
    hardware = allocation(manifest,args.mode) if manifest else dict(
        gpu_count=2 if args.mode=='ssd' else 1, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        logical_devices=[torch.cuda.get_device_name(i) for i in range(2 if args.mode=='ssd' else 1)])
    atomic_json(args.out,dict(summary=summary,args=vars(args),requests=results,job=job,
        manifest_sha256=manifest['manifest_sha256'] if manifest else None,
        hardware_allocation=hardware,software=software_fingerprint(),
        hardware=manifest['hardware'] if manifest else None,
        actual_dtype=str(llm.config.hf_config.torch_dtype),torch_cuda_version=torch.version.cuda,
        prefix_cache_enabled=llm.config.enable_prefix_cache,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        correctness_status='Fast-mode exact equivalence is unresolved; compare saved output tokens.'))
    llm.exit(hard=True)


if __name__ == "__main__":
    main()
