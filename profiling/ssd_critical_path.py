"""Per-round critical-path analysis of an SSD Nsight Systems capture (target on GPU0, drafter on GPU1).

For every decode round r (target process):
  t_req[r]      start of SSD/handshake_send (target asks the drafter for round r's speculation)
  t_resp[r]     end of SD/speculate (dist.recv only enqueues an NCCL kernel; the CPU actually blocks on
                the .tolist() of the received tokens inside SD/speculate, so its end = tokens arrived)
  wait[r]       t_resp - t_req: target stall at handoff
  verify[r]     duration of SD/verify
  miss[r]       a draft/jit_speculate(miss) range overlaps [t_req, t_resp]

For the drafter tree decode that prepares round r (started after serving round r-1), the GPU completion
time of each depth d is the end of the last GPU1 kernel launched inside draft/tree_step_d. The
"ready depth at handoff" is the number of depths already complete on the GPU at t_req[r] — i.e. the
prefix length an anytime (incremental-publication) cache could have handed over immediately, versus the
complete-tree policy that makes the target wait for all K depths.

Usage: python profiling/ssd_critical_path.py trace.sqlite --name ssd_k7_f3 --k 7 --out results/nsys
"""
import argparse
import bisect
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--out", default="results/nsys")
    args = ap.parse_args()
    db = sqlite3.connect(args.sqlite)
    q = lambda sql, *p: db.execute(sql, p).fetchall()  # noqa: E731
    strings = dict(q("select id, value from StringIds"))
    nv = [(s, e, t if t is not None else strings.get(tid), g >> 24)
          for s, e, t, tid, g in q("select start, end, text, textId, globalTid from NVTX_EVENTS where eventType=59 and end is not null")]
    W0, W1, main_pid = next((s, e, p) for s, e, n, p in nv if n == "timed_region")
    inwin = [r for r in nv if r[0] >= W0 and r[1] <= W1]
    byname = defaultdict(list)
    for s, e, n, p in inwin:
        byname[(p, n)].append((s, e))
    for v in byname.values():
        v.sort()
    draft_pid = next(p for (p, n) in byname if n == "draft/decode_tree")

    # GPU completion time of each tree depth: max end of GPU1 kernels launched inside draft/tree_step_d.
    rt = q("select start, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where (globalTid >> 24) = ? and start between ? and ?", draft_pid, W0, W1)
    rt.sort()
    rt_starts = [r[0] for r in rt]
    kend = dict(q("select correlationId, max(end) from CUPTI_ACTIVITY_KIND_KERNEL where (globalPid >> 24) = ? and end between ? and ? group by correlationId", draft_pid, W0, W1 + int(1e9)))

    def gpu_done(s, e):
        i, j = bisect.bisect_left(rt_starts, s), bisect.bisect_right(rt_starts, e)
        ends = [kend[c] for _, c in rt[i:j] if c in kend]
        return max(ends) if ends else None

    trees = byname[(draft_pid, "draft/decode_tree")]
    steps = {d: byname.get((draft_pid, f"draft/tree_step_{d}"), []) for d in range(args.k)}
    tree_ready = []  # per tree: list of GPU completion times per depth
    for s, e in trees:
        ready = []
        for d in range(args.k):
            rng = next(((a, b) for a, b in steps[d] if a >= s and b <= e), None)
            ready.append(gpu_done(*rng) if rng else None)
        tree_ready.append((s, ready))

    reqs = byname[(main_pid, "SSD/handshake_send")]
    resps = byname[(main_pid, "SD/speculate")]
    verifies = byname[(main_pid, "SD/verify")]
    jits = byname.get((draft_pid, "draft/jit_speculate(miss)"), [])
    rounds = []
    for i, (rs, _) in enumerate(reqs):
        resp_end = next((e for s, e in resps if s <= rs <= e), None)
        ver = next(((s, e) for s, e in verifies if s >= rs), None)
        if resp_end is None or ver is None:
            continue
        miss = any(js < resp_end and je > rs for js, je in jits)
        # the tree that prepares this round is the latest one started before the request
        prev = [t for t in tree_ready if t[0] < rs]
        ready_depth = None
        tree_done = None
        if prev and not miss:
            ready = prev[-1][1]
            ready_depth = sum(1 for t in ready if t is not None and t <= rs)
            tree_done = ready[-1]
        rounds.append(dict(t_req=rs, wait_ms=(resp_end - rs) / 1e6, verify_ms=(ver[1] - ver[0]) / 1e6, miss=miss,
                           ready_depth_at_request=ready_depth,
                           tree_done_after_request_ms=((tree_done - rs) / 1e6) if tree_done else None))

    hits = [r for r in rounds if not r["miss"]]
    misses = [r for r in rounds if r["miss"]]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    depth_hist = Counter(r["ready_depth_at_request"] for r in hits)
    summary = dict(
        name=args.name, rounds=len(rounds), miss_rate=len(misses) / len(rounds),
        mean_wait_ms_hit=mean([r["wait_ms"] for r in hits]), mean_wait_ms_miss=mean([r["wait_ms"] for r in misses]),
        mean_verify_ms=mean([r["verify_ms"] for r in rounds]),
        total_wait_ms=sum(r["wait_ms"] for r in rounds), total_verify_ms=sum(r["verify_ms"] for r in rounds),
        wait_share_of_hit_wait_from_unfinished_tree_ms=mean([max(0, r["tree_done_after_request_ms"]) for r in hits if r["tree_done_after_request_ms"] is not None]),
        ready_depth_at_request_hist={str(k): v for k, v in sorted(depth_hist.items(), key=lambda x: (x[0] is None, x[0]))},
    )
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.name}_critical_path.json"), "w") as f:
        json.dump(dict(summary=summary, rounds=rounds), f, indent=1)
    print(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist([r["wait_ms"] for r in hits], bins=40, alpha=0.7, label=f"cache hit (n={len(hits)})")
    ax[0].hist([r["wait_ms"] for r in misses], bins=40, alpha=0.7, label=f"cache miss → JIT (n={len(misses)})")
    ax[0].axvline(summary["mean_verify_ms"], color="k", ls=":", label="mean verify time")
    ax[0].set_xlim(0, 2 * max(summary["mean_wait_ms_miss"] if misses else 0, summary["mean_verify_ms"]))
    ax[0].set_xlabel("target wait at handoff (ms; long-tail outliers clipped)")
    ax[0].set_ylabel("rounds")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Target stall per round waiting for the drafter")
    ks = list(range(args.k + 1))
    ax[1].bar(ks, [depth_hist.get(k, 0) for k in ks])
    ax[1].set_xlabel(f"tree depths already complete on GPU1 when target asked (of K={args.k})")
    ax[1].set_ylabel("rounds (cache hits)")
    ax[1].set_title("Ready depth at handoff (headroom for anytime publication)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{args.name}_critical_path.png"), dpi=140)


if __name__ == "__main__":
    main()
