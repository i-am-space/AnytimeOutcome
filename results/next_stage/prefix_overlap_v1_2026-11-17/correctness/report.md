# Overnight correctness results

Valid blocks: 18; requests: 108.
Exact outputs: 96/96 against shared AR.

| Protocol | Configuration | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Partial handoffs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | ar | 1 | 12.7414 | 12.7275 | 12.9606 | 20.09 | 0.0000 | 0 |
| fixed | complete_k7 | 2 | 2.5461 | 2.7332 | 3.7002 | 100.54 | 0.5490 | 0 |
| fixed | uniform_k7 | 2 | 2.5664 | 2.6999 | 3.7723 | 99.75 | 0.6024 | 0 |
| fixed | bounded_k7 | 2 | 2.5490 | 2.6847 | 3.7467 | 100.43 | 0.5813 | 0 |
| fixed | overlap_k7 | 2 | 4.2779 | 4.3385 | 6.2624 | 59.84 | 0.4976 | 0 |
| fixed | complete_k10 | 2 | 3.7866 | 3.9840 | 5.6852 | 67.61 | 1.1041 | 0 |
| fixed | uniform_k10 | 2 | 4.6621 | 4.8460 | 6.9473 | 54.91 | 0.6953 | 11 |
| fixed | bounded_k10 | 2 | 3.0127 | 3.1339 | 4.4609 | 84.97 | 1.0209 | 300 |
| fixed | overlap_k10 | 2 | 2.9095 | 2.9743 | 4.2431 | 87.99 | 0.8299 | 323 |
| natural | ar | 1 | 6.0532 | 6.3562 | 6.4716 | 40.17 | 0.0000 | 0 |
| natural | complete_k7 | 2 | 2.4239 | 2.3655 | 3.7059 | 100.32 | 0.5386 | 0 |
| natural | uniform_k7 | 2 | 2.4661 | 2.4045 | 3.7673 | 98.60 | 0.5797 | 0 |
| natural | bounded_k7 | 2 | 2.4417 | 2.3821 | 3.7322 | 99.59 | 0.5548 | 0 |
| natural | overlap_k7 | 2 | 2.4447 | 2.3850 | 3.7361 | 99.47 | 0.5592 | 0 |
| natural | complete_k10 | 2 | 3.4418 | 3.3786 | 5.3863 | 70.65 | 1.6274 | 0 |
| natural | uniform_k10 | 2 | 2.6938 | 2.6004 | 4.0919 | 90.27 | 0.7499 | 296 |
| natural | bounded_k10 | 2 | 2.9006 | 2.8142 | 4.4575 | 83.83 | 0.9895 | 289 |
| natural | overlap_k10 | 2 | 2.8083 | 2.6766 | 4.2337 | 86.59 | 0.8034 | 310 |

Exactness is against shared-fixed16-v1 AR, not original historical AR.
Profiled and synthetic boundary timings are diagnostic only.
Target graph retains padded K+1 work; all final draft draining is included.
Publication wall time includes outstanding GPU completion in complete-cache SSD.

Correctness diagnostics only; shared GPU jobs are recorded in receipts.
Artificial tail delay is enabled only in the stress phase.

| Configuration | Partial prefixes | Pending tail at snapshot | Response before tail finished |
|---|---:|---:|---:|
| fixed_r0_overlap_k7 | 0 | 431 | 0 |
| fixed_r0_overlap_k10 | 453 | 453 | 323 |
| natural_r0_overlap_k7 | 0 | 413 | 0 |
| natural_r0_overlap_k10 | 436 | 436 | 310 |
