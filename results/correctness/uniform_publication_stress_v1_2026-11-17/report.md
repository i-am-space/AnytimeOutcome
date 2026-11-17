# Uniform early publication study

Artificial 4 ms/tree-step delay: safety stress test; no performance claim

Valid blocks: 4. Exact matches: 6/6.

## fixed

| Method | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Partial handoffs |
|---|---:|---:|---:|---:|---:|---:|---:|
| ar | 1 | 6.3002 | 6.2477 | 6.3935 | 40.63 | 0.0000 | 0 |
| early | 2 | 3.5046 | 3.3264 | 4.5488 | 73.05 | 1.3707 | 186 |

## natural

| Method | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Partial handoffs |
|---|---:|---:|---:|---:|---:|---:|---:|
| ar | 1 | 5.7481 | 6.2413 | 6.4018 | 40.07 | 0.0000 | 0 |
| early | 2 | 3.1900 | 2.4965 | 4.4013 | 72.21 | 1.2310 | 170 |

Artificial 4 ms/tree-step delay: safety stress test; no performance claim
Stress test only; fixed K7 F3 P4
Fixed target K+1 graph retained; padded tail is calculated but cannot be accepted.
New AR arithmetic; no original-AR exactness claim.
Uniform checkpoint waiting, NCCL handoff, and final obsolete-work draining are included.
