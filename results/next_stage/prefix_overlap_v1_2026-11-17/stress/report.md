# Overnight stress results

Valid blocks: 4; requests: 12.
Exact outputs: 6/6 against shared AR.

| Protocol | Configuration | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Partial handoffs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | ar | 1 | 6.2650 | 6.2470 | 6.2998 | 40.86 | 0.0000 | 0 |
| fixed | overlap_k10 | 2 | 4.1682 | 3.8644 | 5.7606 | 61.42 | 2.0623 | 183 |
| natural | ar | 1 | 5.6311 | 6.2366 | 6.2391 | 40.90 | 0.0000 | 0 |
| natural | overlap_k10 | 2 | 4.2101 | 3.1014 | 6.2840 | 54.71 | 2.3074 | 160 |

Exactness is against shared-fixed16-v1 AR, not original historical AR.
Profiled and synthetic boundary timings are diagnostic only.
Target graph retains padded K+1 work; all final draft draining is included.
Publication wall time includes outstanding GPU completion in complete-cache SSD.

Correctness diagnostics only; shared GPU jobs are recorded in receipts.
Artificial tail delay is enabled only in the stress phase.

| Configuration | Partial prefixes | Pending tail at snapshot | Response before tail finished |
|---|---:|---:|---:|
| fixed_r0_overlap_k10 | 238 | 11 | 10 |
| natural_r0_overlap_k10 | 211 | 121 | 90 |
