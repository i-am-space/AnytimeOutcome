# Uniform early publication study

Previously investigated engineering prompts; not held-out headline results

Valid blocks: 8. Exact matches: 54/54.

## fixed

| Method | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Partial handoffs |
|---|---:|---:|---:|---:|---:|---:|---:|
| ar | 1 | 6.3367 | 6.2739 | 6.5096 | 40.40 | 0.0000 | 0 |
| sd | 1 | 3.2797 | 3.6195 | 4.4030 | 78.06 | 0.0000 | 0 |
| complete | 2 | 2.2961 | 2.5479 | 3.3262 | 111.49 | 0.4412 | 0 |
| early | 2 | 2.3833 | 2.5928 | 3.3815 | 107.41 | 0.4792 | 0 |

## natural

| Method | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Partial handoffs |
|---|---:|---:|---:|---:|---:|---:|---:|
| ar | 1 | 6.0436 | 6.2399 | 6.4064 | 40.39 | 0.0000 | 0 |
| sd | 1 | 3.1466 | 3.6152 | 4.3094 | 77.58 | 0.0000 | 0 |
| complete | 2 | 2.2012 | 2.5415 | 3.2270 | 110.90 | 0.4154 | 0 |
| early | 2 | 2.2570 | 2.5888 | 3.2848 | 108.16 | 0.4555 | 0 |

Previously investigated engineering prompts; not held-out headline results
declared starting K/F; not retuned
Fixed target K+1 graph retained; padded tail is calculated but cannot be accepted.
New AR arithmetic; no original-AR exactness claim.
Uniform checkpoint waiting, NCCL handoff, and final obsolete-work draining are included.
