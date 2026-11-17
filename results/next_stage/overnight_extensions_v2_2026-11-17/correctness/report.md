# Overnight correctness results

Valid blocks: 20; requests: 240.
Exact outputs: 216/216 against shared AR.

| Protocol | Configuration | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Partial handoffs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | ar | 1 | 6.4182 | 6.4698 | 6.4713 | 39.89 | 0.0000 | 0 |
| fixed | sd_k4 | 1 | 3.5573 | 3.8624 | 4.6099 | 71.96 | 0.0000 | 0 |
| fixed | complete_k4 | 2 | 2.6359 | 2.9737 | 3.4868 | 97.12 | 0.3167 | 0 |
| fixed | early_k4 | 2 | 2.6450 | 2.9841 | 3.4992 | 96.79 | 0.3190 | 0 |
| fixed | sd_k7 | 1 | 3.7111 | 4.1742 | 5.1336 | 68.98 | 0.0000 | 0 |
| fixed | complete_k7 | 2 | 2.6145 | 3.1153 | 3.7629 | 97.92 | 0.5936 | 0 |
| fixed | early_k7 | 2 | 2.6384 | 3.1425 | 3.7942 | 97.03 | 0.6194 | 0 |
| fixed | sd_k10 | 1 | 4.1042 | 4.7639 | 5.9385 | 62.38 | 0.0000 | 0 |
| fixed | complete_k10 | 2 | 3.6901 | 4.4911 | 5.5012 | 69.38 | 1.7426 | 0 |
| fixed | early_k10 | 2 | 3.1065 | 3.7147 | 4.4930 | 82.41 | 1.0445 | 604 |
| natural | ar | 1 | 6.2636 | 6.4706 | 6.4825 | 39.85 | 0.0000 | 0 |
| natural | sd_k4 | 1 | 3.4914 | 3.8653 | 4.6116 | 71.49 | 0.0000 | 0 |
| natural | complete_k4 | 2 | 2.5912 | 2.9789 | 3.4927 | 96.32 | 0.3115 | 0 |
| natural | early_k4 | 2 | 2.5937 | 2.9837 | 3.4973 | 96.23 | 0.3156 | 0 |
| natural | sd_k7 | 1 | 3.6430 | 4.1782 | 5.1362 | 68.51 | 0.0000 | 0 |
| natural | complete_k7 | 2 | 2.5698 | 3.1188 | 3.7684 | 97.12 | 0.5894 | 0 |
| natural | early_k7 | 2 | 2.5903 | 3.1407 | 3.7982 | 96.35 | 0.6075 | 0 |
| natural | sd_k10 | 1 | 4.0463 | 4.7772 | 5.9471 | 61.68 | 0.0000 | 0 |
| natural | complete_k10 | 2 | 3.6351 | 4.4947 | 5.5014 | 68.66 | 1.7186 | 0 |
| natural | early_k10 | 2 | 3.0315 | 3.7175 | 4.4941 | 82.33 | 1.0278 | 592 |

Exactness is against shared-fixed16-v1 AR, not original historical AR.
Profiled and synthetic boundary timings are diagnostic only.
Target graph retains padded K+1 work; all final draft draining is included.
Publication wall time includes outstanding GPU completion in complete-cache SSD.
