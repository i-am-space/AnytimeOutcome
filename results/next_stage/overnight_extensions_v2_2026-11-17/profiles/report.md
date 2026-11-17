# Overnight profiles results

Valid blocks: 3; requests: 9.
Exact outputs: 6/6 against shared AR.

| Protocol | Configuration | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Partial handoffs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | ar | 1 | 6.8010 | 6.7807 | 6.8393 | 37.64 | 0.0000 | 0 |
| fixed | complete | 2 | 2.7356 | 2.6519 | 3.9043 | 93.58 | 0.7016 | 0 |
| fixed | early | 2 | 2.7939 | 2.7141 | 4.0004 | 91.63 | 0.7580 | 0 |

Exactness is against shared-fixed16-v1 AR, not original historical AR.
Profiled and synthetic boundary timings are diagnostic only.
Target graph retains padded K+1 work; all final draft draining is included.
Publication wall time includes outstanding GPU completion in complete-cache SSD.
