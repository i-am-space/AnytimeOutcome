# Uniform early publication study

Previously investigated engineering prompts; not held-out headline results

Valid blocks: 30. Exact matches: 180/180.

## fixed

| Method | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Partial handoffs |
|---|---:|---:|---:|---:|---:|---:|---:|
| ar | 1 | 6.2922 | 6.2568 | 6.4744 | 40.69 | 0.0000 | 0 |
| complete | 2 | 2.3201 | 2.5445 | 3.3637 | 110.34 | 0.4402 | 0 |
| early | 2 | 2.3596 | 2.5934 | 3.4527 | 108.49 | 0.4819 | 0 |

## natural

| Method | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Partial handoffs |
|---|---:|---:|---:|---:|---:|---:|---:|
| ar | 1 | 5.9951 | 6.2437 | 6.2983 | 40.72 | 0.0000 | 0 |
| complete | 2 | 2.2358 | 2.5412 | 3.3546 | 109.18 | 0.4258 | 0 |
| early | 2 | 2.2764 | 2.5930 | 3.4151 | 107.23 | 0.4576 | 2 |

Previously investigated engineering prompts; not held-out headline results
retuned shared baseline
Fixed target K+1 graph retained; padded tail is calculated but cannot be accepted.
New AR arithmetic; no original-AR exactness claim.
Uniform checkpoint waiting, NCCL handoff, and final obsolete-work draining are included.
