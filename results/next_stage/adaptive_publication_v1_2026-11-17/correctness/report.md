# Overnight correctness results

Valid blocks: 16; requests: 96.
Exact outputs: 84/84 against shared AR.

| Protocol | Configuration | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Partial handoffs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | ar | 1 | 6.3692 | 6.3854 | 6.4866 | 40.19 | 0.0000 | 0 |
| fixed | complete_k7 | 2 | 2.5125 | 2.6456 | 3.6945 | 101.89 | 0.5527 | 0 |
| fixed | bounded_k7 | 2 | 2.5401 | 2.6740 | 3.7351 | 100.78 | 0.5765 | 0 |
| fixed | adaptive_k7 | 2 | 2.5463 | 2.6812 | 3.7447 | 100.54 | 0.5822 | 0 |
| fixed | complete_k10 | 2 | 3.5454 | 3.7166 | 5.3632 | 72.21 | 1.6680 | 0 |
| fixed | uniform_k10 | 2 | 2.7983 | 2.9053 | 4.0932 | 91.48 | 0.7769 | 309 |
| fixed | bounded_k10 | 2 | 3.0398 | 3.2411 | 4.4492 | 84.22 | 1.0190 | 301 |
| fixed | adaptive_k10 | 2 | 3.0089 | 3.1314 | 4.4630 | 85.08 | 1.0236 | 301 |
| natural | ar | 1 | 6.0535 | 6.3564 | 6.4747 | 40.17 | 0.0000 | 0 |
| natural | complete_k7 | 2 | 2.4293 | 2.3695 | 3.7168 | 100.10 | 0.5440 | 0 |
| natural | bounded_k7 | 2 | 2.4379 | 2.3765 | 3.7285 | 99.75 | 0.5510 | 0 |
| natural | adaptive_k7 | 2 | 2.4424 | 2.3826 | 3.7346 | 99.56 | 0.5558 | 0 |
| natural | complete_k10 | 2 | 3.4383 | 3.3758 | 5.3779 | 70.72 | 1.6185 | 0 |
| natural | uniform_k10 | 2 | 2.6951 | 2.5993 | 4.0929 | 90.22 | 0.7499 | 296 |
| natural | bounded_k10 | 2 | 2.8963 | 2.8076 | 4.4465 | 83.96 | 0.9828 | 289 |
| natural | adaptive_k10 | 2 | 2.9033 | 2.8195 | 4.4612 | 83.76 | 0.9892 | 289 |

Exactness is against shared-fixed16-v1 AR, not original historical AR.
Profiled and synthetic boundary timings are diagnostic only.
Target graph retains padded K+1 work; all final draft draining is included.
Publication wall time includes outstanding GPU completion in complete-cache SSD.
