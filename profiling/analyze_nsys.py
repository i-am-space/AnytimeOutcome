"""Analyze an Nsight Systems sqlite export of a run_engine.py capture.

Produces, restricted to the steady-state "timed_region" NVTX range of the main process:
  1. Per-GPU utilization: busy (union of kernel+memcpy+memset intervals) vs idle time.
  2. GPU time by kernel category (GEMM / GEMV / attention / norm+elementwise / sampling / NCCL / ...).
  3. Phase breakdown: for each NVTX range name, CPU wall time and GPU kernel time of kernels launched
     inside it (CUDA-graph nodes are attributed via the cudaGraphLaunch correlation id).
  4. Idle attribution: GPU idle gaps on each device, attributed to the innermost NVTX range open on the
     process driving that GPU (e.g. target idle during "SSD/wait_draft_tokens", drafter idle during
     "draft/idle_wait_cmd").
  5. Hardware counters over time (GPU-metrics sampling): mean DRAM read BW %, SM active %, SM issue %,
     tensor active % per phase -> the memory-bound vs compute-bound picture per phase.
  6. A timeline figure of a few decode steps.

Usage: python profiling/analyze_nsys.py trace.sqlite --name ssd_k7_f3 --out results/nsys
"""
import argparse
import bisect
import json
import os
import re
import sqlite3
from collections import defaultdict

CATEGORIES = [
    ("nccl", re.compile(r"nccl", re.I)),
    ("gemv (cuBLAS)", re.compile(r"gemv", re.I)),
    ("gemm (tensor core)", re.compile(r"gemm|cutlass|xmma|s16816|wmma", re.I)),
    ("attention (paged, Triton)", re.compile(r"_paged_attn|_combine_kernel", re.I)),
    ("attention (flashinfer tree)", re.compile(r"BatchPrefill|PrefillWithPaged|flashinfer", re.I)),
    ("attention (prefill SDPA)", re.compile(r"flash_fwd|fmha|efficient_attention", re.I)),
    ("kv-cache store", re.compile(r"store_kvcache", re.I)),
    ("rmsnorm/rope/silu (fused Triton)", re.compile(r"^triton_", re.I)),
    ("sampling/argmax", re.compile(r"Sampling|argmax|reduce_kernel|TopK|softmax", re.I)),
    ("memcpy/memset", re.compile(r"^\[memcpy|^\[memset", re.I)),
]
PCT_METRICS = {
    "DRAM Read Bandwidth [Throughput %]": "dram_read_pct",
    "DRAM Write Bandwidth [Throughput %]": "dram_write_pct",
    "SMs Active [Throughput %]": "sm_active_pct",
    "SM Issue [Throughput %]": "sm_issue_pct",
    "Tensor Active [Throughput %]": "tensor_active_pct",
    "PCIe RX Throughput [Throughput %]": "pcie_rx_pct",
    "PCIe TX Throughput [Throughput %]": "pcie_tx_pct",
    "GPC Clock Frequency [MHz]": "gpc_mhz",
}


def categorize(name):
    for cat, rx in CATEGORIES:
        if rx.search(name):
            return cat
    return "other elementwise"


def union_len(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def merged(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


class RangeIndex:
    """Innermost-NVTX-range lookup for one process: intervals sorted, nested ranges allowed."""

    def __init__(self, ranges):
        self.ranges = sorted(ranges, key=lambda r: (r[0], -r[1]))
        self.starts = [r[0] for r in self.ranges]

    def innermost(self, t):
        i = bisect.bisect_right(self.starts, t) - 1
        best = None
        # walk back over candidates that could contain t (ranges are short; bounded scan)
        j, scanned = i, 0
        while j >= 0 and scanned < 400:
            s, e, name = self.ranges[j]
            if s <= t < e and (best is None or (e - s) < (best[1] - best[0])):
                best = (s, e, name)
            j -= 1
            scanned += 1
        return best[2] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/nsys")
    ap.add_argument("--timeline_ms", type=float, default=120.0, help="window length for the timeline figure")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    db = sqlite3.connect(args.sqlite)
    q = lambda sql, *p: db.execute(sql, p).fetchall()  # noqa: E731

    strings = dict(q("select id, value from StringIds"))
    # NVTX push/pop ranges (eventType 59). Text may be inline or via textId.
    nvtx = [(s, e, txt if txt is not None else strings.get(tid), tid_g)
            for s, e, txt, tid, tid_g in q("select start, end, text, textId, globalTid from NVTX_EVENTS where eventType=59 and end is not null")]
    tr = [r for r in nvtx if r[2] == "timed_region"]
    assert tr, "no timed_region NVTX range found"
    W0, W1, main_tid = tr[0][0], tr[0][1], tr[0][3]
    window = W1 - W0
    pid_of = lambda gtid: gtid >> 24  # noqa: E731  (globalTid = pid << 24 | tid in nsys exports)

    # Kernels + memcpy/memset in window.
    acts = []
    for s, e, dev, corr, short, dem, gpid in q("select start, end, deviceId, correlationId, shortName, demangledName, globalPid from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ?", W0, W1):
        sname, dname = strings.get(short, "?"), strings.get(dem, "")
        # cuBLAS GEMV's short name is just "kernel": keep a readable label that still categorizes correctly
        label = sname if sname not in ("kernel", "Kernel2") else (("gemv:" if "gemv" in dname else "gemm:") + dname[dname.find("<") + 1:][:60])
        acts.append((max(s, W0), min(e, W1), dev, corr, label, gpid))
    for table, label in [("CUPTI_ACTIVITY_KIND_MEMCPY", "[memcpy]"), ("CUPTI_ACTIVITY_KIND_MEMSET", "[memset]")]:
        try:
            for s, e, dev, corr, gpid in q(f"select start, end, deviceId, correlationId, globalPid from {table} where end > ? and start < ?", W0, W1):
                acts.append((max(s, W0), min(e, W1), dev, corr, label, gpid))
        except sqlite3.OperationalError:
            pass
    devices = sorted({a[2] for a in acts})

    # Runtime API calls: correlationId -> (launch time, globalTid) for attributing GPU work to NVTX ranges.
    # CUPTI correlation ids are only unique per process, so key by (pid, correlationId).
    launch = {(gtid >> 24, corr): (s, gtid) for s, corr, gtid in q("select start, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where end > ? - 5e9 and start < ?", W0, W1)}

    # NVTX ranges per process (the drafter is a different process than the target).
    by_pid = defaultdict(list)
    for s, e, name, gtid in nvtx:
        if e > W0 - 1e9 and s < W1:
            by_pid[pid_of(gtid)].append((s, e, name))
    idx = {pid: RangeIndex(r) for pid, r in by_pid.items()}

    # Which process drives which GPU (most kernels).
    drv = defaultdict(lambda: defaultdict(int))
    for a in acts:
        drv[a[2]][a[5] >> 24 if a[5] > 1 << 24 else a[5]] += 1
    dev_pid = {d: max(c, key=c.get) for d, c in drv.items()}

    report = dict(name=args.name, window_ms=window / 1e6, devices={})
    for d in devices:
        da = [a for a in acts if a[2] == d]
        busy = union_len([(a[0], a[1]) for a in da])
        cat_t = defaultdict(float)
        for a in da:
            cat_t[categorize(a[4])] += (a[1] - a[0])
        top = defaultdict(lambda: [0, 0.0])
        for a in da:
            top[a[4]][0] += 1
            top[a[4]][1] += a[1] - a[0]

        # Phase attribution of GPU busy time (launch-site NVTX range) and of idle gaps (range open at gap).
        pid = dev_pid[d]
        ri = idx.get(pid)
        phase_gpu = defaultdict(float)
        for a in da:
            ls = launch.get((a[5] >> 24, a[3]))
            name = None
            if ls is not None and ri is not None:
                name = idx.get(pid_of(ls[1]), ri).innermost(ls[0])
            phase_gpu[name or "(unattributed)"] += a[1] - a[0]
        m = merged([(a[0], a[1]) for a in da])
        gaps = []
        prev = W0
        for s, e in m:
            if s > prev:
                gaps.append((prev, s))
            prev = max(prev, e)
        if prev < W1:
            gaps.append((prev, W1))
        phase_idle = defaultdict(float)
        gap_hist = defaultdict(lambda: [0, 0.0])
        for s, e in gaps:
            mid = (s + e) // 2
            name = ri.innermost(mid) if ri else None
            phase_idle[name or "(none)"] += e - s
            g = e - s
            b = "<5us" if g < 5e3 else "5-20us" if g < 2e4 else "20-100us" if g < 1e5 else "0.1-1ms" if g < 1e6 else ">1ms"
            gap_hist[b][0] += 1
            gap_hist[b][1] += g / 1e6

        report["devices"][str(d)] = dict(
            busy_ms=busy / 1e6, idle_ms=(window - busy) / 1e6, busy_pct=100 * busy / window,
            n_kernels=len(da), kernels_per_ms=len(da) / (window / 1e6),
            category_ms={k: v / 1e6 for k, v in sorted(cat_t.items(), key=lambda x: -x[1])},
            top_kernels=[dict(name=n, count=c, total_ms=t / 1e6, mean_us=t / c / 1e3)
                         for n, (c, t) in sorted(top.items(), key=lambda x: -x[1][1])[:15]],
            gpu_time_by_launch_phase_ms={k: v / 1e6 for k, v in sorted(phase_gpu.items(), key=lambda x: -x[1])},
            idle_by_phase_ms={k: v / 1e6 for k, v in sorted(phase_idle.items(), key=lambda x: -x[1])},
            idle_gap_histogram={k: dict(count=c, total_ms=t) for k, (c, t) in gap_hist.items()},
        )

    # CPU-side NVTX phase stats (all processes) within the window.
    phase_cpu = defaultdict(lambda: [0, 0.0])
    for s, e, name, gtid in nvtx:
        if s >= W0 and e <= W1:
            key = re.sub(r"_\d+$", "_*", name) if re.match(r"(SD/draft_fwd|draft/tree_step)_\d+", name or "") else name
            if key and not key.startswith("request:"):
                phase_cpu[key][0] += 1
                phase_cpu[key][1] += e - s
    report["nvtx_phases"] = {k: dict(count=c, total_ms=t / 1e6, mean_ms=t / c / 1e6)
                             for k, (c, t) in sorted(phase_cpu.items(), key=lambda x: -x[1][1])}

    # GPU-metrics sampling joined with phases.
    try:
        mnames = {(t, m): n for t, m, n in q("select typeId, metricId, metricName from TARGET_INFO_GPU_METRICS")}
        type_ids = sorted({t for t, m in mnames if any(n == "SMs Active [Throughput %]" for (tt, mm), n in mnames.items() if tt == t)})
        metric_by_dev = {}
        for i, t in enumerate(type_ids):  # typeIds enumerate GPUs in device order
            wanted = {m: PCT_METRICS[n] for (tt, m), n in mnames.items() if tt == t and n in PCT_METRICS}
            rows = q(f"select timestamp, metricId, value from GPU_METRICS where typeId=? and timestamp between ? and ? and metricId in ({','.join(map(str, wanted))})", t, W0, W1)
            metric_by_dev[str(i)] = (wanted, rows)
        report["gpu_metrics"] = {}
        for d, (wanted, rows) in metric_by_dev.items():
            overall = defaultdict(list)
            per_phase = defaultdict(lambda: defaultdict(list))
            ri = idx.get(dev_pid.get(int(d), -1))
            for ts, mid, val in rows:
                key = wanted[mid]
                if key == "gpc_mhz" and not (0 < val < 5000):  # clock samples can be garbage; drop them
                    continue
                overall[key].append(val)
                if ri is not None and key != "gpc_mhz":
                    per_phase[ri.innermost(ts) or "(none)"][key].append(val)
            report["gpu_metrics"][d] = dict(
                overall={k: sum(v) / len(v) for k, v in overall.items()},
                by_phase={ph: {k: sum(v) / len(v) for k, v in mv.items()} | {"samples": len(next(iter(mv.values())))}
                          for ph, mv in per_phase.items() if len(next(iter(mv.values()))) >= 20},
            )
        report["_metric_rows"] = None
    except sqlite3.OperationalError as e:
        report["gpu_metrics_error"] = str(e)
        metric_by_dev = {}

    with open(os.path.join(args.out, f"{args.name}_analysis.json"), "w") as f:
        json.dump({k: v for k, v in report.items() if not k.startswith("_")}, f, indent=1)
    print_report(report)
    try:
        plot_timeline(args, acts, devices, nvtx, metric_by_dev, W0, W1, dev_pid, pid_of)
    except Exception as e:  # plotting is best-effort
        print("timeline plot failed:", e)


def print_report(r):
    print(f"\n===== {r['name']}  timed window {r['window_ms']:.1f} ms =====")
    for d, x in r["devices"].items():
        print(f"\nGPU{d}: busy {x['busy_pct']:.1f}%  ({x['busy_ms']:.1f} ms busy / {x['idle_ms']:.1f} ms idle), "
              f"{x['n_kernels']} kernels ({x['kernels_per_ms']:.1f}/ms)")
        print("  GPU time by category:")
        for k, v in x["category_ms"].items():
            print(f"    {k:34s} {v:9.2f} ms  {100*v/max(x['busy_ms'],1e-9):5.1f}% of busy")
        print("  GPU time by launching phase:")
        for k, v in list(x["gpu_time_by_launch_phase_ms"].items())[:10]:
            print(f"    {str(k):34s} {v:9.2f} ms")
        print("  Idle time by open phase:")
        for k, v in list(x["idle_by_phase_ms"].items())[:10]:
            print(f"    {str(k):34s} {v:9.2f} ms")
        print("  Idle gap histogram:", {k: f"{v['count']}x/{v['total_ms']:.1f}ms" for k, v in x["idle_gap_histogram"].items()})
    print("\nNVTX phases (CPU wall):")
    for k, v in list(r["nvtx_phases"].items())[:25]:
        print(f"  {k:34s} n={v['count']:5d} total={v['total_ms']:9.2f} ms mean={v['mean_ms']:8.3f} ms")
    for d, g in r.get("gpu_metrics", {}).items():
        print(f"\nGPU{d} hardware counters (sampled 20 kHz): " + ", ".join(f"{k}={v:.1f}" for k, v in g["overall"].items()))
        for ph, mv in sorted(g["by_phase"].items(), key=lambda x: -x[1]["samples"])[:12]:
            print(f"  {ph:34s} " + "  ".join(f"{k}={v:5.1f}" for k, v in mv.items() if k != "samples") + f"  (n={mv['samples']})")


def plot_timeline(args, acts, devices, nvtx, metric_by_dev, W0, W1, dev_pid, pid_of):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Window: start 25% into the timed region, length timeline_ms.
    t0 = W0 + (W1 - W0) // 4
    t1 = min(W1, t0 + int(args.timeline_ms * 1e6))
    cats = [c for c, _ in CATEGORIES] + ["other elementwise"]
    cmap = plt.get_cmap("tab20")
    color = {c: cmap(i) for i, c in enumerate(cats)}
    n_dev = len(devices)
    fig, axes = plt.subplots(2 * n_dev, 1, figsize=(16, 2.6 * 2 * n_dev), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.3, 1] * n_dev))
    for i, d in enumerate(devices):
        ax = axes[2 * i]
        pid = dev_pid[d]
        # NVTX lanes for the process driving this GPU
        ranges = [(s, e, n) for s, e, n, g in nvtx if pid_of(g) == pid and e > t0 and s < t1 and n and
                  not n.startswith(("request:", "timed_region", "engine/step"))]
        names = sorted({re.sub(r"_\d+$", "_*", n) for _, _, n in ranges})
        lane = {n: j for j, n in enumerate(names)}
        for s, e, n in ranges:
            k = re.sub(r"_\d+$", "_*", n)
            ax.barh(lane[k] + 1, (e - s) / 1e6, left=(s - t0) / 1e6, height=0.7, color="0.6")
        for a in acts:
            if a[2] == d and a[1] > t0 and a[0] < t1:
                ax.barh(0, (a[1] - a[0]) / 1e6, left=(a[0] - t0) / 1e6, height=0.8, color=color[categorize(a[4])], lw=0)
        ax.set_yticks(range(len(names) + 1))
        ax.set_yticklabels(["GPU kernels"] + names, fontsize=7)
        ax.set_title(f"GPU{d} ({'target' if d == 0 else 'drafter'}) — kernels by category and NVTX phases", fontsize=9, loc="left")
        ax2 = axes[2 * i + 1]
        wanted, rows = metric_by_dev.get(str(d), ({}, []))
        series = defaultdict(lambda: ([], []))
        for ts, mid, val in rows:
            if t0 <= ts <= t1:
                series[wanted[mid]][0].append((ts - t0) / 1e6)
                series[wanted[mid]][1].append(val)
        for key, lbl in [("dram_read_pct", "DRAM read BW %"), ("tensor_active_pct", "Tensor active %"), ("sm_issue_pct", "SM issue %")]:
            if key in series:
                ax2.plot(*series[key], lw=0.8, label=lbl)
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("% of peak", fontsize=8)
        ax2.legend(fontsize=7, loc="upper right", ncol=3)
    axes[-1].set_xlabel("time (ms)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color[c]) for c in cats]
    fig.legend(handles, cats, loc="lower center", ncol=6, fontsize=7)
    fig.suptitle(f"{args.name}: steady-state timeline", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(os.path.join(args.out, f"{args.name}_timeline.png"), dpi=130)


if __name__ == "__main__":
    main()
