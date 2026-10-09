"""Empirical roofline for the local GPU + GEMM-shape sweep for the Qwen3 target/draft pair.

Part 1 (machine roofline, slides "Constructing the Model for Any Machine"):
  - peak BF16 tensor-core throughput   (large square GEMM)
  - peak DRAM bandwidth                 (device-to-device copy + read-only reduction, >> L2)
  - peak L2 bandwidth                   (copy on a buffer that fits in the 96 MB L2)
  -> ridge point I* = pi / beta

Part 2 (kernel points): for every distinct linear-layer shape in Qwen3-8B and Qwen3-0.6B,
time Y[T, N] = X[T, K] @ W[K, N] for T = tokens per forward pass. Decode is T=1, sync-SD
verification is T=K+1, SSD tree decode on the drafter is T=B*(K+1)*F, prefill is T=prompt_len.
Each point gets an analytic arithmetic intensity (FLOPs / compulsory DRAM bytes) and a measured
FLOP/s, so it can be placed on the roofline.

Usage: python profiling/roofline_microbench.py --device 0 --out results/roofline
"""
import argparse
import json
import os
import time

import torch

# Linear-layer shapes (K_in, N_out) per transformer layer, plus lm_head, from the HF configs.
# Qwen3-8B: hidden 4096, 32 q heads / 8 kv heads x 128, intermediate 12288, 36 layers, vocab 151936.
# Qwen3-0.6B: hidden 1024, 16 q heads / 8 kv heads x 128, intermediate 3072, 28 layers, vocab 151936.
MODELS = {
    "qwen3-8b": dict(layers=36, shapes={
        "qkv_proj": (4096, (32 + 2 * 8) * 128),
        "o_proj": (32 * 128, 4096),
        "gate_up_proj": (4096, 2 * 12288),
        "down_proj": (12288, 4096),
    }, lm_head=(4096, 151936)),
    "qwen3-0.6b": dict(layers=28, shapes={
        "qkv_proj": (1024, (16 + 2 * 8) * 128),
        "o_proj": (16 * 128, 1024),
        "gate_up_proj": (1024, 2 * 3072),
        "down_proj": (3072, 1024),
    }, lm_head=(1024, 151936)),
}
L2_BYTES = 96 << 20  # L40S L2
TOKENS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096]


def time_fn(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    times.sort()
    return times[len(times) // 2], times[0]  # median, best (seconds)


def machine_roofline(dev):
    res = {}
    # Peak BF16 tensor throughput.
    n = 8192
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    med, best = time_fn(lambda: a @ b, iters=30)
    res["bf16_tensor_tflops_measured"] = 2 * n ** 3 / best / 1e12
    res["bf16_tensor_tflops_median"] = 2 * n ** 3 / med / 1e12
    del a, b

    # FP32 CUDA-core throughput (TF32 disabled).
    torch.backends.cuda.matmul.allow_tf32 = False
    n = 4096
    a = torch.randn(n, n, device=dev, dtype=torch.float32)
    med, best = time_fn(lambda: a @ a, iters=20)
    res["fp32_tflops_measured"] = 2 * n ** 3 / best / 1e12
    del a

    # DRAM bandwidth: 4 GiB buffers, far larger than the 96 MB L2.
    nbytes = 4 << 30
    src = torch.empty(nbytes // 2, device=dev, dtype=torch.bfloat16).normal_()
    dst = torch.empty_like(src)
    med, best = time_fn(lambda: dst.copy_(src), iters=20)
    res["dram_copy_GBps"] = 2 * nbytes / best / 1e9  # read + write
    med, best = time_fn(lambda: src.sum(dtype=torch.float32), iters=20)
    res["dram_read_GBps"] = nbytes / best / 1e9
    del src, dst

    # L2 bandwidth: 16 MiB src + 16 MiB dst stays resident in the 96 MB L2.
    nbytes = 16 << 20
    src = torch.empty(nbytes // 2, device=dev, dtype=torch.bfloat16).normal_()
    dst = torch.empty_like(src)
    med, best = time_fn(lambda: dst.copy_(src), iters=200, warmup=50)
    res["l2_copy_GBps"] = 2 * nbytes / best / 1e9
    del src, dst

    beta = max(res["dram_copy_GBps"], res["dram_read_GBps"]) * 1e9
    res["ridge_bf16_flop_per_byte_measured"] = res["bf16_tensor_tflops_measured"] * 1e12 / beta
    return res


def gemm_sweep(dev):
    rows = []
    for model, spec in MODELS.items():
        shapes = dict(spec["shapes"])
        shapes["lm_head"] = spec["lm_head"]
        for name, (k, n) in shapes.items():
            # Rotate through enough weight copies (>= 4x the 96 MB L2) that every timed GEMM streams
            # its weights from DRAM, as in a real forward pass where each layer is touched once.
            n_copies = max(1, -(-(4 * L2_BYTES) // (2 * k * n)))
            ws = [torch.randn(k, n, device=dev, dtype=torch.bfloat16) for _ in range(n_copies)]
            for t in TOKENS:
                x = torch.randn(t, k, device=dev, dtype=torch.bfloat16)
                it = iter(range(1 << 30))
                med, best = time_fn(lambda: x @ ws[next(it) % n_copies])
                flops = 2 * t * k * n
                dram_bytes = 2 * (k * n + t * k + t * n)  # compulsory traffic, bf16
                rows.append(dict(
                    model=model, layer=name, K=k, N=n, T=t,
                    time_us=med * 1e6, flops=flops, bytes=dram_bytes,
                    ai=flops / dram_bytes, tflops=flops / med / 1e12,
                    achieved_GBps=dram_bytes / med / 1e9,
                ))
            del ws
    return rows


def forward_estimates(rows, roof):
    """Sum per-layer GEMM times into a weights-only forward pass time per model and T."""
    out = []
    for model, spec in MODELS.items():
        for t in TOKENS:
            sel = {r["layer"]: r for r in rows if r["model"] == model and r["T"] == t}
            per_layer = sum(sel[name]["time_us"] for name in spec["shapes"])
            total = per_layer * spec["layers"] + sel["lm_head"]["time_us"]
            out.append(dict(model=model, T=t, linear_forward_us=total,
                            us_per_token=total / t))
    return out


def plot(roof, rows, fwd, out_dir, gpu_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pi = roof["bf16_tensor_tflops_measured"]
    beta = max(roof["dram_copy_GBps"], roof["dram_read_GBps"]) / 1e3  # TB/s
    pi_spec, beta_spec = 362.05, 0.864  # L40S datasheet: dense BF16 tensor, GDDR6

    fig, ax = plt.subplots(figsize=(9, 6))
    ai = [2 ** e for e in range(-2, 13)]
    ax.loglog(ai, [min(pi, x * beta) for x in ai], "k-", lw=2, label=f"measured roof ({pi:.0f} TFLOP/s, {beta*1e3:.0f} GB/s)")
    ax.loglog(ai, [min(pi_spec, x * beta_spec) for x in ai], "k--", lw=1, alpha=0.5, label="datasheet roof (362 TFLOP/s, 864 GB/s)")
    markers = {"qwen3-8b": "o", "qwen3-0.6b": "^"}
    colors = {"qkv_proj": "C0", "o_proj": "C1", "gate_up_proj": "C2", "down_proj": "C3", "lm_head": "C4"}
    for r in rows:
        ax.loglog(r["ai"], r["tflops"], markers[r["model"]], color=colors[r["layer"]], ms=4, alpha=0.8)
    for name, c in colors.items():
        ax.plot([], [], "s", color=c, label=name)
    for m, mk in markers.items():
        ax.plot([], [], mk, color="gray", label=m)
    ax.axvline(pi / beta, color="red", ls=":", lw=1)
    ax.text(pi / beta * 1.05, 0.3, f"ridge I*={pi/beta:.0f}", color="red")
    for t, lbl in [(1, "AR decode T=1"), (8, "SD verify T=K+1=8"), (24, "SSD tree T=(K+1)F=24")]:
        r = next(r for r in rows if r["model"] == "qwen3-8b" and r["layer"] == "gate_up_proj" and r["T"] == t)
        ax.annotate(lbl, (r["ai"], r["tflops"]), textcoords="offset points", xytext=(5, -12), fontsize=7)
    ax.set_xlabel("Arithmetic intensity (FLOP / DRAM byte)")
    ax.set_ylabel("Attained TFLOP/s (BF16)")
    ax.set_title(f"Roofline: Qwen3 linear layers on {gpu_name} (T = tokens per forward)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roofline_gemm.png"), dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for model in MODELS:
        pts = [f for f in fwd if f["model"] == model]
        ax[0].loglog([p["T"] for p in pts], [p["linear_forward_us"] / 1e3 for p in pts], "o-", label=model)
        ax[1].loglog([p["T"] for p in pts], [p["us_per_token"] for p in pts], "o-", label=model)
    ax[0].set_xlabel("tokens per forward pass (T)")
    ax[0].set_ylabel("linear-layer forward time (ms)")
    ax[0].set_title("Forward latency vs T: flat = memory-bound, linear = compute-bound")
    ax[1].set_xlabel("tokens per forward pass (T)")
    ax[1].set_ylabel("µs per token")
    ax[1].set_title("Amortized cost per token")
    for a in ax:
        a.grid(True, which="both", alpha=0.2)
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "forward_vs_tokens.png"), dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--out", default="results/roofline")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)

    t0 = time.time()
    roof = machine_roofline(dev)
    roof.update(gpu=props.name, sms=props.multi_processor_count, torch=torch.__version__,
                l2_bytes=getattr(props, "L2_cache_size", None))
    print(json.dumps(roof, indent=2))
    rows = gemm_sweep(dev)
    fwd = forward_estimates(rows, roof)
    with open(os.path.join(args.out, "machine_roofline.json"), "w") as f:
        json.dump(roof, f, indent=2)
    with open(os.path.join(args.out, "gemm_sweep.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(args.out, "forward_vs_tokens.json"), "w") as f:
        json.dump(fwd, f, indent=1)
    for f_ in fwd:
        print(f"{f_['model']:11s} T={f_['T']:5d}  linear fwd {f_['linear_forward_us']/1e3:8.3f} ms  {f_['us_per_token']:9.1f} us/token")
    plot(roof, rows, fwd, args.out, props.name)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
