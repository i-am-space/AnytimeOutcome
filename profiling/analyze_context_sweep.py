"""CPU-only NVTX/CUDA reconstruction for the context-length diagnostic.

GPU spans include execution and intervening gaps, rather than CPU enqueue time.
Incomplete trees are kept separate from full K-step trees. Ready depth is
reported only for cache-hit requests with a completely correlated tree.
"""
import bisect
from collections import Counter, defaultdict
import sqlite3


def mean(values):
    return sum(values) / len(values) if values else None


def reconstruct(nvtx, launches, kernels, k):
    """Inputs: ranges (s,e,name,pid), launches (s,corr,pid), kernels (s,e,corr,pid)."""
    windows = [r for r in nvtx if r[2] == 'timed_region']
    if len(windows) != 1:
        raise ValueError('Expected exactly one timed_region')
    w0, w1, _, target_pid = windows[0]
    ranges = [r for r in nvtx if w0 <= r[0] < r[1] <= w1]
    draft_pids = {p for _, _, n, p in ranges if n == 'draft/decode_tree'}
    if len(draft_pids) != 1:
        raise ValueError('Expected one traced drafter with tree ranges')
    draft_pid = next(iter(draft_pids))
    by_pid = defaultdict(list)
    for start, corr, pid in launches:
        by_pid[pid].append((start, corr))
    starts = {}
    for pid in by_pid:
        by_pid[pid].sort()
        starts[pid] = [s for s, _ in by_pid[pid]]
    gpu = defaultdict(list)
    for start, end, corr, pid in kernels:
        gpu[pid, corr].append((start, end))

    def span(s, e, pid):
        times = starts.get(pid, [])
        lo, hi = bisect.bisect_left(times, s), bisect.bisect_left(times, e)
        work = [v for _, corr in by_pid.get(pid, [])[lo:hi] for v in gpu[pid, corr]]
        return (min(a for a, _ in work), max(b for _, b in work)) if work else None

    trees = []
    for s, e, name, pid in ranges:
        if name != 'draft/decode_tree' or pid != draft_pid:
            continue
        steps = {}
        for a, b, n, p in ranges:
            if p == pid and s <= a < b <= e and n.startswith('draft/tree_step_'):
                depth = int(n.rsplit('_', 1)[1])
                if depth in steps:
                    raise ValueError('Duplicate depth within one tree')
                steps[depth] = span(a, b, pid)
        if not steps:
            continue
        correlated = all(v is not None for v in steps.values())
        pieces = [v for v in steps.values() if v is not None]
        full = set(steps) == set(range(k)) and correlated
        trees.append(dict(start=s, end=e, steps=steps, correlated=correlated, full=full,
            gpu_span_ms=(max(b for _, b in pieces)-min(a for a, _ in pieces))/1e6 if pieces else None))
    trees.sort(key=lambda t: t['start'])
    tree_starts = [t['start'] for t in trees]
    request_ranges = sorted((s, e, n[8:]) for s, e, n, p in ranges
                            if p == target_pid and n.startswith('request:'))
    if not request_ranges:
        raise ValueError('Missing request ranges')
    request_starts = [s for s, _, _ in request_ranges]

    def request_at(t):
        i = bisect.bisect_right(request_starts, t)-1
        return request_ranges[i][2] if i >= 0 and t < request_ranges[i][1] else None

    verifies = sorted((s, e) for s, e, n, p in ranges if p == target_pid and n == 'SD/verify')
    forwards = [span(s, e, p) for s, e, n, p in ranges
                if p == target_pid and n == 'verify/target_fwd']
    if not verifies or not forwards or any(v is None for v in forwards):
        raise ValueError('Missing correlated verification GPU work')
    if not any(t['correlated'] for t in trees):
        raise ValueError('Missing correlated draft GPU work')
    responses = [(s, e) for s, e, n, p in ranges if p == target_pid and n == 'SD/speculate']
    misses = [(s, e) for s, e, n, p in ranges if p == draft_pid and n == 'draft/jit_speculate(miss)']
    handshakes = sorted(s for s, _, n, p in ranges if p == target_pid and n == 'SSD/handshake_send')
    rounds = []
    for start in handshakes:
        rid = request_at(start)
        response = next(((a, b) for a, b in responses if a <= start < b), None)
        if rid is None or response is None:
            continue
        end = response[1]
        verify = next(((a, b) for a, b in verifies if a >= end and request_at(a) == rid), None)
        if verify is None:
            continue
        miss = any(a < end and b > start for a, b in misses)
        i = bisect.bisect_right(tree_starts, start)-1
        tree = trees[i] if i >= 0 and request_at(trees[i]['start']) == rid else None
        ready = None
        if not miss and tree and tree['correlated']:
            ready = 0
            while ready in tree['steps'] and tree['steps'][ready][1] <= start:
                ready += 1
        rounds.append(dict(request_id=rid, target_wait_ms=(end-start)/1e6,
            verify_wall_ms=(verify[1]-verify[0])/1e6, miss=miss,
            ready_depth_at_request=ready,
            remaining_full_tree_ms=(max(0, tree['steps'][k-1][1]-start)/1e6
                if tree and tree['full'] and not miss else None)))
    if not rounds:
        raise ValueError('No matched target handoff/verification rounds')
    full = [t['gpu_span_ms'] for t in trees if t['full']]
    partial = [t['gpu_span_ms'] for t in trees if not t['full'] and t['correlated']]
    verify_wall = [(e-s)/1e6 for s, e in verifies]
    verify_gpu = [(e-s)/1e6 for s, e in forwards]
    return dict(
        k=k, full_trees=len(full), partial_trees=len(partial),
        uncorrelated_trees=sum(not t['correlated'] for t in trees),
        mean_full_tree_gpu_span_ms=mean(full), mean_partial_tree_gpu_span_ms=mean(partial),
        mean_verify_wall_ms=mean(verify_wall), mean_verify_gpu_span_ms=mean(verify_gpu),
        full_tree_over_verify_wall=(mean(full)/mean(verify_wall) if full else None),
        full_tree_over_verify_gpu=(mean(full)/mean(verify_gpu) if full else None),
        matched_rounds=len(rounds), mean_target_wait_ms=mean([r['target_wait_ms'] for r in rounds]),
        cache_hit_ready_depth_histogram=dict(sorted(Counter(str(r['ready_depth_at_request'])
            for r in rounds if not r['miss'] and r['ready_depth_at_request'] is not None).items())),
        unknown_ready_depth_rounds=sum(r['ready_depth_at_request'] is None for r in rounds),
        rounds=rounds,
        interpretation='Profiled mechanism evidence; no profiler-free latency claim. GPU spans include gaps. '
                       'Tree/verify ratios are descriptive phase means, not paired per-round overlap budgets.')


def analyze(path, k):
    with sqlite3.connect(path) as db:
        strings = dict(db.execute('select id,value from StringIds'))
        nvtx = [(s,e,text if text is not None else strings.get(tid,''),gtid >> 24)
            for s,e,text,tid,gtid in db.execute('select start,end,text,textId,globalTid from NVTX_EVENTS '
                                              'where eventType=59 and end is not null')]
        launches = [(s,c,g >> 24) for s,c,g in db.execute(
            'select start,correlationId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME')]
        kernels = [(s,e,c,g >> 24) for s,e,c,g in db.execute(
            'select start,end,correlationId,globalPid from CUPTI_ACTIVITY_KIND_KERNEL')]
    return reconstruct(nvtx, launches, kernels, k)
