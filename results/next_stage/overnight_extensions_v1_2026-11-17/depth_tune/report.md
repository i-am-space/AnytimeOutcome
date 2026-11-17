# Overnight depth_tune results

Valid blocks: 10; requests: 240.
Exact outputs: 240/240 against shared AR.

| Protocol | Configuration | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Partial handoffs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | complete | 2 | 2.1212 | 2.1928 | 3.3103 | 120.69 | 0.4157 | 0 |
| fixed | p1 | 2 | 2.1900 | 2.2610 | 3.3997 | 116.90 | 0.4826 | 1 |
| fixed | p2 | 2 | 2.1800 | 2.2508 | 3.3950 | 117.43 | 0.4705 | 0 |
| fixed | p4 | 2 | 2.1585 | 2.2345 | 3.4871 | 118.60 | 0.4471 | 0 |
| fixed | p7 | 2 | 2.1393 | 2.2107 | 3.4803 | 119.67 | 0.4278 | 0 |

Exactness is against shared-fixed16-v1 AR, not original historical AR.
Profiled and synthetic boundary timings are diagnostic only.
Target graph retains padded K+1 work; all final draft draining is included.
Publication wall time includes outstanding GPU completion in complete-cache SSD.
