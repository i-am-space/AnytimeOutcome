# Controlled context-length diagnostic

Controlled synthetic length diagnostic; three task templates, not held-out generalization.
Fixed 256 output tokens, EOS ignored; three repeated rounds; BF16, batch one, graphs, full prefills and final draining.
AR uses one GPU; SSD uses two. All accepted blocks passed dedicated-GPU monitoring.
Exact speculative outputs: 252/252 against fresh shared AR. Original historical AR is separate.

## Unprofiled request measurements

| Input tokens | Configuration | GPUs | Mean request s | Decode s | Wait s | Tokens/verify | Actual overlap handoffs |
|---:|---|---:|---:|---:|---:|---:|---:|
| 256 | ar | 1 | 6.2545 | 6.2158 | 0.0000 | 0.000 | 0 |
| 512 | ar | 1 | 6.2768 | 6.2292 | 0.0000 | 0.000 | 0 |
| 1024 | ar | 1 | 6.3440 | 6.2582 | 0.0000 | 0.000 | 0 |
| 2048 | ar | 1 | 6.4687 | 6.3086 | 0.0000 | 0.000 | 0 |
| 256 | complete_k7 | 2 | 1.7907 | 1.7521 | 0.2824 | 4.488 | 0 |
| 512 | complete_k7 | 2 | 2.0493 | 2.0011 | 0.3564 | 4.047 | 0 |
| 1024 | complete_k7 | 2 | 2.5670 | 2.4807 | 0.4754 | 3.318 | 0 |
| 2048 | complete_k7 | 2 | 2.6677 | 2.5065 | 0.5575 | 3.460 | 0 |
| 256 | uniform_k7_p4 | 2 | 1.8197 | 1.7808 | 0.3121 | 4.488 | 0 |
| 512 | uniform_k7_p4 | 2 | 2.0807 | 2.0324 | 0.3879 | 4.047 | 0 |
| 1024 | uniform_k7_p4 | 2 | 2.6057 | 2.5188 | 0.5135 | 3.318 | 0 |
| 2048 | uniform_k7_p4 | 2 | 2.7045 | 2.5434 | 0.5942 | 3.460 | 0 |
| 256 | overlap_k7_p4_w2 | 2 | 1.8088 | 1.7700 | 0.3012 | 4.488 | 0 |
| 512 | overlap_k7_p4_w2 | 2 | 2.0684 | 2.0201 | 0.3750 | 4.047 | 0 |
| 1024 | overlap_k7_p4_w2 | 2 | 2.5912 | 2.5041 | 0.4981 | 3.318 | 0 |
| 2048 | overlap_k7_p4_w2 | 2 | 2.6900 | 2.5291 | 0.5784 | 3.460 | 0 |
| 256 | complete_k10 | 2 | 2.2929 | 2.2395 | 1.0081 | 5.420 | 0 |
| 512 | complete_k10 | 2 | 2.8919 | 2.8297 | 1.2912 | 4.410 | 0 |
| 1024 | complete_k10 | 2 | 3.6014 | 3.5008 | 1.6138 | 3.562 | 0 |
| 2048 | complete_k10 | 2 | 3.6550 | 3.4807 | 1.6652 | 3.767 | 0 |
| 256 | uniform_k10_p4 | 2 | 1.8852 | 1.8460 | 0.3827 | 4.529 | 0 |
| 512 | uniform_k10_p4 | 2 | 2.3043 | 2.2560 | 0.5024 | 3.813 | 0 |
| 1024 | uniform_k10_p4 | 2 | 2.7442 | 2.6582 | 0.6151 | 3.289 | 0 |
| 2048 | uniform_k10_p4 | 2 | 2.8528 | 2.6916 | 0.7180 | 3.455 | 0 |
| 256 | overlap_k10_p4_w2 | 2 | 2.0233 | 1.9790 | 0.5139 | 4.529 | 414 |
| 512 | overlap_k10_p4_w2 | 2 | 2.3713 | 2.3203 | 0.4909 | 3.671 | 524 |
| 1024 | overlap_k10_p4_w2 | 2 | 2.7573 | 2.6677 | 0.5900 | 3.234 | 579 |
| 2048 | overlap_k10_p4_w2 | 2 | 2.8956 | 2.7329 | 0.7237 | 3.390 | 516 |

## Separate profiled phase measurements

| Input tokens | Configuration | Full trees | Mean full-tree GPU span ms | Verify GPU span ms | Verify wall ms | Tree/verify GPU ratio | Hit ready-depth histogram |
|---:|---|---:|---:|---:|---:|---:|---|
| 256 | complete_k10 | 143 | 44.046 | 28.176 | 29.084 | 1.563 | {'4': 1, '5': 106, '6': 1, '7': 3} |
| 256 | overlap_k10_p4_w2 | 1 | 43.305 | 27.254 | 28.049 | 1.589 | {'4': 118, '5': 22, '6': 1, '7': 3, '8': 1} |
| 256 | complete_k7 | 172 | 27.187 | 27.111 | 27.886 | 1.003 | {'5': 10, '6': 124, '7': 5} |
| 512 | complete_k7 | 192 | 27.234 | 27.224 | 28.012 | 1.000 | {'3': 1, '5': 29, '6': 119, '7': 5} |
| 512 | complete_k10 | 178 | 45.510 | 27.782 | 28.565 | 1.638 | {'10': 2, '4': 2, '5': 133, '7': 3} |
| 512 | overlap_k10_p4_w2 | 0 | — | 27.312 | 28.107 | — | {'4': 146, '5': 25, '6': 1, '7': 2} |
| 1024 | complete_k7 | 233 | 27.441 | 27.256 | 28.067 | 1.007 | {'5': 154, '6': 28, '7': 4} |
| 1024 | complete_k10 | 217 | 44.637 | 27.403 | 28.192 | 1.629 | {'4': 10, '5': 157, '7': 3, '8': 1} |
| 1024 | overlap_k10_p4_w2 | 2 | 44.417 | 27.872 | 28.697 | 1.594 | {'0': 1, '10': 1, '4': 157, '5': 30, '6': 2, '7': 2, '8': 1} |
| 2048 | complete_k10 | 206 | 45.604 | 28.002 | 28.803 | 1.629 | {'0': 1, '4': 5, '5': 141, '7': 2, '8': 3} |
| 2048 | overlap_k10_p4_w2 | 2 | 44.582 | 28.742 | 29.532 | 1.551 | {'2': 1, '3': 1, '4': 129, '5': 37, '6': 1, '7': 4} |
| 2048 | complete_k7 | 224 | 28.005 | 28.155 | 28.941 | 0.995 | {'4': 1, '5': 152, '6': 10, '7': 7} |

## Interpretation boundaries

Use matched K controls to assess publication; do not interpret a K10 improvement over K10 as a win over a faster K7 baseline. Examine the tree/verify ratio, ready depth, wait and accepted work together. More overlap alone is not a speedup.
Profiled GPU spans include gaps; CPU launch-range duration is not GPU execution time. Incomplete or uncorrelated trees are not extrapolated to a full K-step cost. Initial context grows by up to 256 generated tokens during each request.
Paired 95% intervals in summary.json use 10,000 seeded draws over rounds and templates within families. With one template per family these intervals do not estimate workload generalization.

Synthetic repeated filler; only one scored task template per family.
Three repeated rounds measure repeatability; no claim about unseen prompts.
Measured context includes input plus up to 256 generated tokens.
Partial trees are not reported as full K-step drafting.
Profiled wall/GPU spans may be perturbed; performance comparisons use separate unprofiled runs.
