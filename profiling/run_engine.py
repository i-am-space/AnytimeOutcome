"""Shared benchmark driver for AR / synchronous SD / SSD on the SSD engine (one request at a time).

Measures, per prompt: full-request latency (includes both prefills), time-to-first-token proxy
(prefill step), decode time, committed tokens, plus engine metrics (accepted tokens per step,
cache-hit rate). The timed region is wrapped in an NVTX range "timed_region" and
cudaProfilerStart/Stop so Nsight Systems / Compute can restrict capture to steady state.

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
    p.add_argument("--ignore_eos", action="store_true", default=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
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
    )
    sp = SamplingParams(temperature=0.0, max_new_tokens=args.output_len, ignore_eos=args.ignore_eos)

    for w in warm:
        llm.generate([w["token_ids"]], [sp], use_tqdm=False)

    results = []
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push("timed_region")
    for it in prompts:
        torch.cuda.nvtx.range_push(f"request:{it['id']}")
        t0 = time.perf_counter()
        outs, metrics = llm.generate([it["token_ids"]], [sp], use_tqdm=False)
        torch.cuda.synchronize()
        lat = time.perf_counter() - t0
        torch.cuda.nvtx.range_pop()
        steps = metrics["target_step_times"]
        acc = metrics["accepted_suffix_lens_with_recovery"]
        results.append(dict(
            id=it["id"], family=it["family"], category=it["category"],
            prompt_tokens=len(it["token_ids"]), output_tokens=len(outs[0]["token_ids"]),
            latency_s=lat, prefill_s=metrics["prefill_total_time"], decode_s=metrics["decode_total_time"],
            num_steps=len(steps), mean_step_ms=1e3 * sum(steps[1:]) / max(1, len(steps) - 1),
            tokens_per_step=(sum(acc) / len(acc)) if acc else 1.0,
            cache_hit_rate=(sum(metrics["cache_hits"]) / len(metrics["cache_hits"])) if metrics["cache_hits"] else None,
            output_token_ids=outs[0]["token_ids"],
        ))
        r = results[-1]
        print(f"[{args.mode}] {r['id']:22s} prompt={r['prompt_tokens']:5d} out={r['output_tokens']:4d} "
              f"lat={r['latency_s']*1e3:8.1f}ms prefill={r['prefill_s']*1e3:6.1f}ms "
              f"tok/step={r['tokens_per_step']:.2f} hit={r['cache_hit_rate']}", flush=True)
    torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()

    tot_tok = sum(r["output_tokens"] for r in results)
    tot_lat = sum(r["latency_s"] for r in results)
    summary = dict(
        mode=args.mode, k=args.k if spec else None, f=args.f if args.mode == "ssd" else None,
        backup=args.backup if args.mode == "ssd" else None, output_len=args.output_len, n_requests=len(results),
        mean_latency_s=tot_lat / len(results), committed_tok_per_s=tot_tok / tot_lat,
        decode_tok_per_s=tot_tok / sum(r["decode_s"] for r in results),
        mean_tokens_per_step=sum(r["tokens_per_step"] for r in results) / len(results),
        mean_step_ms=sum(r["mean_step_ms"] for r in results) / len(results),
        gpu=torch.cuda.get_device_name(0),
    )
    print(json.dumps(summary, indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=summary, args=vars(args), requests=results), f, indent=1)
    llm.exit(hard=True)


if __name__ == "__main__":
    main()
