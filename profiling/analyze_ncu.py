"""Summarize an Nsight Compute report (from profiling/run_ncu.sh) per kernel and per kernel family.

Per kernel (raw counters, exported with `ncu --import --page raw --csv`):
  duration, Speed-of-Light compute % and memory %, DRAM throughput %, DRAM bytes, BF16 tensor FLOPs and
  FP32 FLOPs (-> measured arithmetic intensity and FLOP/s for the roofline), achieved vs theoretical
  occupancy, waves/SM, L1/L2 hit rates, issue-slot utilization and the top warp-stall reasons.
Bound classification (slides' top-down workflow, step 2): memory-bound if memory SOL dominates and is
>= 60%, compute-bound if compute SOL dominates and is >= 60%, otherwise latency-bound (neither unit is
busy: too little parallelism, launch/tail effects, or stalls).

Usage: python profiling/analyze_ncu.py report.ncu-rep --name ar_decode --out results/ncu
"""
import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from analyze_nsys import categorize  # noqa: E402

NCU = os.environ.get("NCU", "/opt/nvidia/nsight-compute/2025.2.1/ncu")
COLS = {
    "duration_ns": "gpu__time_duration.sum",
    "sol_compute_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sol_memory_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_read_B": "dram__bytes_read.sum",
    "dram_write_B": "dram__bytes_write.sum",
    "tensor_bf16_flop": "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
    "ffma": "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "fadd": "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "fmul": "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "occ_achieved_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "occ_theoretical_pct": "sm__maximum_warps_per_active_cycle_pct",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "regs_per_thread": "launch__registers_per_thread",
    "grid_size": "launch__grid_size",
    "block_size": "launch__block_size",
    "l1_hit_pct": "l1tex__t_sector_hit_rate.pct",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm_mhz": "sm__cycles_elapsed.avg.per_second",
}
STALL_RX = re.compile(r"^smsp__average_warp_latency_issue_stalled_(\w+)\.ratio$|^smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$")


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return float("nan")


def load(rep):
    out = subprocess.run([NCU, "--import", rep, "--csv", "--page", "raw"], capture_output=True, text=True, check=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    header, units, data = rows[0], rows[1], rows[2:]
    return header, units, data


def classify(k):
    c, m = k["sol_compute_pct"], k["sol_memory_pct"]
    if max(c, m) < 60:
        return "latency-bound"
    return "memory-bound" if m >= c else "compute-bound"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/ncu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    header, units, data = load(args.rep)
    col = {h: i for i, h in enumerate(header)}
    name_col = col.get("Kernel Name")
    stall_cols = [(i, (STALL_RX.match(h).group(1) or STALL_RX.match(h).group(2))) for h, i in col.items() if STALL_RX.match(h)]

    kernels = []
    for r in data:
        k = dict(name=r[name_col])
        for key, metric in COLS.items():
            k[key] = num(r[col[metric]]) if metric in col else float("nan")
        # unit fix-ups: ncu may report durations in us/ms and bytes in KB/MB depending on magnitude
        u = units[col[COLS["duration_ns"]]] if COLS["duration_ns"] in col else "ns"
        k["duration_ns"] *= {"ns": 1, "us": 1e3, "usecond": 1e3, "ms": 1e6, "msecond": 1e6, "nsecond": 1}.get(u, 1)
        for b in ("dram_read_B", "dram_write_B"):
            if COLS[b] in col:
                ub = units[col[COLS[b]]]
                k[b] *= {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "KB": 1e3, "MB": 1e6, "GB": 1e9}.get(ub, 1)
        fp32 = 2 * k["ffma"] + k["fadd"] + k["fmul"]
        flops = (k["tensor_bf16_flop"] if k["tensor_bf16_flop"] == k["tensor_bf16_flop"] else 0) + (fp32 if fp32 == fp32 else 0)
        k["flop"] = flops
        dram = k["dram_read_B"] + k["dram_write_B"]
        k["dram_B"] = dram
        k["ai_flop_per_byte"] = flops / dram if dram > 0 else float("nan")
        k["tflops"] = flops / k["duration_ns"] / 1e3 if k["duration_ns"] > 0 else float("nan")
        k["dram_GBps"] = dram / k["duration_ns"] if k["duration_ns"] > 0 else float("nan")
        stalls = sorted(((num(r[i]), s) for i, s in stall_cols if num(r[i]) == num(r[i])), reverse=True)
        k["top_stalls"] = [(s, round(v, 2)) for v, s in stalls[:4]]
        k["bound"] = classify(k)
        k["category"] = categorize(k["name"])
        kernels.append(k)

    total = sum(k["duration_ns"] for k in kernels)
    fam = defaultdict(lambda: defaultdict(float))
    stall_by_fam = defaultdict(lambda: defaultdict(float))
    for k in kernels:
        f = fam[k["category"]]
        w = k["duration_ns"]
        f["time_ns"] += w
        f["n"] += 1
        f["flop"] += k["flop"]
        f["dram_B"] += k["dram_B"]
        for key in ("sol_compute_pct", "sol_memory_pct", "dram_pct", "occ_achieved_pct", "occ_theoretical_pct", "issue_active_pct", "l2_hit_pct"):
            if k[key] == k[key]:
                f[key + "_w"] += k[key] * w
        f["bound_" + k["bound"]] += w
        for s, v in k["top_stalls"]:
            stall_by_fam[k["category"]][s] += v * w
    families = {}
    for c, f in sorted(fam.items(), key=lambda x: -x[1]["time_ns"]):
        t = f["time_ns"]
        families[c] = dict(
            n_kernels=int(f["n"]), time_pct=100 * t / total, time_us=t / 1e3,
            sol_compute_pct=f["sol_compute_pct_w"] / t, sol_memory_pct=f["sol_memory_pct_w"] / t, dram_pct=f["dram_pct_w"] / t,
            occ_achieved_pct=f["occ_achieved_pct_w"] / t, occ_theoretical_pct=f["occ_theoretical_pct_w"] / t,
            issue_active_pct=f["issue_active_pct_w"] / t, l2_hit_pct=f["l2_hit_pct_w"] / t,
            ai_flop_per_byte=f["flop"] / f["dram_B"] if f["dram_B"] else None,
            achieved_tflops=f["flop"] / t / 1e3, achieved_dram_GBps=f["dram_B"] / t,
            bound_split_pct={b[6:]: 100 * v / t for b, v in f.items() if b.startswith("bound_")},
            top_stalls=[s for s, _ in sorted(stall_by_fam[c].items(), key=lambda x: -x[1])[:3]],
        )
    overall = dict(
        n_kernels=len(kernels), total_kernel_time_us=total / 1e3,
        flop=sum(k["flop"] for k in kernels), dram_B=sum(k["dram_B"] for k in kernels),
        bound_split_pct={b: 100 * sum(k["duration_ns"] for k in kernels if k["bound"] == b) / total
                         for b in ("memory-bound", "compute-bound", "latency-bound")},
    )
    overall["ai_flop_per_byte"] = overall["flop"] / overall["dram_B"]
    overall["achieved_dram_GBps"] = overall["dram_B"] / total
    res = dict(name=args.name, overall=overall, families=families,
               kernels=[{k2: v for k2, v in k.items()} for k in kernels])
    with open(os.path.join(args.out, f"{args.name}_ncu.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)

    print(f"\n===== ncu {args.name}: {len(kernels)} kernels, {total/1e3:.1f} us total =====")
    print(f"overall: AI={overall['ai_flop_per_byte']:.2f} FLOP/B, DRAM {overall['achieved_dram_GBps']:.0f} GB/s, "
          + ", ".join(f"{b} {v:.1f}%" for b, v in overall["bound_split_pct"].items()))
    print(f"{'family':34s} {'time%':>6s} {'SOLc%':>6s} {'SOLm%':>6s} {'DRAM%':>6s} {'occA%':>6s} {'issue%':>6s} {'AI':>7s} {'TFLOP/s':>8s}  bound split / top stalls")
    for c, f in families.items():
        ai = f"{f['ai_flop_per_byte']:.2f}" if f["ai_flop_per_byte"] else "-"
        bs = " ".join(f"{b[:3]}:{v:.0f}" for b, v in f["bound_split_pct"].items())
        print(f"{c:34s} {f['time_pct']:6.1f} {f['sol_compute_pct']:6.1f} {f['sol_memory_pct']:6.1f} {f['dram_pct']:6.1f} "
              f"{f['occ_achieved_pct']:6.1f} {f['issue_active_pct']:6.1f} {ai:>7s} {f['achieved_tflops']:8.2f}  {bs} | {','.join(f['top_stalls'])}")
    plot(res, args)


def plot(res, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roof_path = os.path.join(os.path.dirname(__file__), "..", "results", "roofline", "machine_roofline.json")
    roof = json.load(open(roof_path)) if os.path.exists(roof_path) else {"bf16_tensor_tflops_measured": 211, "dram_read_GBps": 747, "fp32_tflops_measured": 48}
    pi, beta = roof["bf16_tensor_tflops_measured"], roof["dram_read_GBps"] / 1e3
    pi32 = roof["fp32_tflops_measured"]
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [2 ** e for e in range(-6, 13)]
    ax.loglog(xs, [min(pi, x * beta) for x in xs], "k-", label=f"BF16 tensor roof ({pi:.0f} TFLOP/s)")
    ax.loglog(xs, [min(pi32, x * beta) for x in xs], "k--", lw=1, label=f"FP32 CUDA-core roof ({pi32:.0f} TFLOP/s)")
    cats = sorted({k["category"] for k in res["kernels"]})
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(cats):
        pts = [k for k in res["kernels"] if k["category"] == c and k["flop"] > 0 and k["dram_B"] > 0]
        if pts:
            ax.scatter([k["ai_flop_per_byte"] for k in pts], [k["tflops"] for k in pts],
                       s=[max(6, k["duration_ns"] / 2e3) for k in pts], color=cmap(i % 10), alpha=0.6, label=c)
    ax.axvline(pi / beta, color="red", ls=":", lw=1)
    ax.set_xlabel("measured arithmetic intensity (FLOP / DRAM byte)")
    ax.set_ylabel("measured TFLOP/s")
    ax.set_title(f"Nsight Compute roofline: {args.name} (marker area ∝ kernel time)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{args.name}_roofline.png"), dpi=140)


if __name__ == "__main__":
    main()
