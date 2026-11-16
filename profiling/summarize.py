"""Combine the Nsight Compute phase reports into markdown tables + one roofline figure for docs/perf_analysis.md.

Usage: python profiling/summarize.py  (reads results/ncu/*_ncu.json, results/roofline/machine_roofline.json)
"""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
PHASES = [
    ("ar_decode", "AR decode (target, T=1)"),
    ("sd_verify", "SD/SSD verify (target, T=k+1=7)"),
    ("sd_draft", "Sync-SD draft forward (drafter, T=1)"),
    ("ssd_tree", "SSD tree step (drafter, T=(k+1)f=24)"),
    ("ar_prefill", "Prefill (target, 631-token doc prompt)"),
]


def main():
    res = {k: json.load(open(os.path.join(ROOT, "results", "ncu", f"{k}_ncu.json"))) for k, _ in PHASES}
    lines = ["| phase | kernels | sum of kernel time | AI (FLOP/B) | DRAM GB/s | TFLOP/s | memory / compute / latency-bound (% of time) |",
             "|---|---|---|---|---|---|---|"]
    for k, label in PHASES:
        o = res[k]["overall"]
        b = o["bound_split_pct"]
        tf = o["flop"] / (o["total_kernel_time_us"] * 1e-6) / 1e12
        lines.append(f"| {label} | {o['n_kernels']} | {o['total_kernel_time_us']/1e3:.2f} ms | {o['ai_flop_per_byte']:.1f} | "
                     f"{o['achieved_dram_GBps']:.0f} | {tf:.1f} | {b['memory-bound']:.0f} / {b['compute-bound']:.0f} / {b['latency-bound']:.0f} |")
    out = ["\n".join(lines), ""]
    for k, label in PHASES:
        t = ["", f"**{label}** — kernel families (time-weighted)", "",
             "| family | time % | SOL compute % | SOL memory % | achieved occupancy % | issue active % | AI | top warp stalls |",
             "|---|---|---|---|---|---|---|---|"]
        for fam, f in res[k]["families"].items():
            if f["time_pct"] < 1.0:
                continue
            ai = f"{f['ai_flop_per_byte']:.1f}" if f["ai_flop_per_byte"] else "–"
            t.append(f"| {fam} | {f['time_pct']:.1f} | {f['sol_compute_pct']:.1f} | {f['sol_memory_pct']:.1f} | "
                     f"{f['occ_achieved_pct']:.1f} | {f['issue_active_pct']:.1f} | {ai} | {', '.join(f['top_stalls'][:3])} |")
        out.append("\n".join(t))
    md = "\n".join(out)
    with open(os.path.join(ROOT, "results", "ncu", "summary_tables.md"), "w") as fh:
        fh.write(md + "\n")
    print(md)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    roof = json.load(open(os.path.join(ROOT, "results", "roofline", "machine_roofline.json")))
    pi, beta = roof["bf16_tensor_tflops_measured"], roof["dram_read_GBps"] / 1e3
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [2 ** e for e in range(-3, 12)]
    ax.loglog(xs, [min(pi, x * beta) for x in xs], "k-", lw=2, label=f"measured L40S roof: {pi:.0f} TFLOP/s, {beta*1e3:.0f} GB/s")
    ax.axvline(pi / beta, color="red", ls=":", lw=1)
    ax.text(pi / beta * 1.08, 0.12, f"ridge {pi/beta:.0f} FLOP/B", color="red", fontsize=8)
    markers = ["o", "s", "^", "D", "*"]
    for (k, label), mk in zip(PHASES, markers):
        o = res[k]["overall"]
        tf = o["flop"] / (o["total_kernel_time_us"] * 1e-6) / 1e12
        ax.loglog(o["ai_flop_per_byte"], tf, mk, ms=11, label=f"{label}")
        gem = [f for n, f in res[k]["families"].items() if n.startswith(("gemm", "gemv"))]
        if gem:
            g = gem[0]
            ax.loglog(g["ai_flop_per_byte"], g["achieved_tflops"], mk, ms=7, mfc="none", color=ax.lines[-1].get_color())
    ax.set_xlabel("arithmetic intensity (FLOP / DRAM byte), measured by Nsight Compute")
    ax.set_ylabel("achieved TFLOP/s (BF16 tensor + FP32)")
    ax.set_title("Where each phase sits on the L40S roofline\n(filled: whole forward pass; hollow: its GEMM/GEMV kernels only)")
    ax.set_ylim(0.1, 400)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "phase_roofline.png"), dpi=150)


if __name__ == "__main__":
    main()
