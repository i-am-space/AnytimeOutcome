| phase | kernels | sum of kernel time | AI (FLOP/B) | DRAM GB/s | TFLOP/s | memory / compute / latency-bound (% of time) |
|---|---|---|---|---|---|---|
| AR decode (target, T=1) | 540 | 23.06 ms | 0.9 | 780 | 0.7 | 95 / 0 / 5 |
| SD/SSD verify (target, T=k+1=7) | 545 | 22.21 ms | 13.5 | 748 | 10.1 | 95 / 0 / 5 |
| Sync-SD draft forward (drafter, T=1) | 470 | 3.10 ms | 1.0 | 499 | 0.5 | 71 / 0 / 29 |
| SSD tree step (drafter, T=(k+1)f=24) | 320 | 2.24 ms | 54.5 | 460 | 25.1 | 63 / 0 / 37 |
| Prefill (target, 631-token doc prompt) | 557 | 51.92 ms | 365.2 | 476 | 174.0 | 11 / 85 / 4 |


**AR decode (target, T=1)** — kernel families (time-weighted)

| family | time % | SOL compute % | SOL memory % | achieved occupancy % | issue active % | AI | top warp stalls |
|---|---|---|---|---|---|---|---|
| gemv (cuBLAS) | 95.3 | 13.6 | 94.6 | 24.0 | 7.1 | 0.9 | long_scoreboard, lg_throttle, barrier |
| rmsnorm/rope/silu (fused Triton) | 2.7 | 0.1 | 1.3 | 15.2 | 10.3 | 1.1 | long_scoreboard, imc_miss, selected |
| attention (paged, Triton) | 1.3 | 5.5 | 14.3 | 11.2 | 15.3 | 11.7 | long_scoreboard, imc_miss, barrier |

**SD/SSD verify (target, T=k+1=7)** — kernel families (time-weighted)

| family | time % | SOL compute % | SOL memory % | achieved occupancy % | issue active % | AI | top warp stalls |
|---|---|---|---|---|---|---|---|
| gemm (tensor core) | 93.9 | 9.1 | 91.2 | 7.4 | 5.3 | 13.6 | long_scoreboard, wait, selected |
| rmsnorm/rope/silu (fused Triton) | 2.7 | 1.0 | 5.6 | 16.5 | 11.2 | 1.5 | long_scoreboard, imc_miss, wait |
| attention (paged, Triton) | 1.9 | 6.6 | 48.5 | 13.0 | 10.6 | 5.2 | long_scoreboard, barrier, imc_miss |
| other elementwise | 1.2 | 0.4 | 2.1 | 8.3 | 3.1 | – | long_scoreboard, imc_miss, no_instruction |

**Sync-SD draft forward (drafter, T=1)** — kernel families (time-weighted)

| family | time % | SOL compute % | SOL memory % | achieved occupancy % | issue active % | AI | top warp stalls |
|---|---|---|---|---|---|---|---|
| gemv (cuBLAS) | 70.7 | 11.5 | 80.5 | 18.1 | 7.1 | 0.8 | long_scoreboard, lg_throttle, misc |
| rmsnorm/rope/silu (fused Triton) | 17.3 | 0.1 | 0.7 | 9.5 | 6.9 | 0.7 | imc_miss, long_scoreboard, wait |
| attention (paged, Triton) | 7.2 | 5.7 | 10.4 | 11.2 | 16.5 | 16.6 | long_scoreboard, imc_miss, short_scoreboard |
| kv-cache store | 2.2 | 0.0 | 0.4 | 8.2 | 1.4 | – | long_scoreboard, imc_miss, no_instruction |
| sampling/argmax | 1.6 | 0.7 | 3.5 | 61.2 | 47.3 | 0.6 | barrier, not_selected, wait |
| other elementwise | 1.0 | 1.6 | 6.0 | 11.9 | 5.6 | 1.0 | imc_miss, long_scoreboard, no_instruction |

**SSD tree step (drafter, T=(k+1)f=24)** — kernel families (time-weighted)

| family | time % | SOL compute % | SOL memory % | achieved occupancy % | issue active % | AI | top warp stalls |
|---|---|---|---|---|---|---|---|
| gemm (tensor core) | 61.3 | 16.5 | 81.7 | 11.9 | 6.4 | 57.3 | long_scoreboard, barrier, wait |
| rmsnorm/rope/silu (fused Triton) | 12.4 | 0.8 | 6.3 | 9.1 | 9.9 | 1.6 | long_scoreboard, imc_miss, selected |
| other elementwise | 10.9 | 3.0 | 14.0 | 19.0 | 5.5 | – | long_scoreboard, imc_miss, drain |
| attention (flashinfer tree) | 9.7 | 1.0 | 7.9 | 11.8 | 8.4 | 54.3 | imc_miss, long_scoreboard, wait |
| reductions (argmax etc.) | 3.4 | 2.8 | 11.0 | 25.0 | 14.5 | 0.5 | long_scoreboard, imc_miss, no_instruction |
| kv-cache store | 1.6 | 0.1 | 5.5 | 8.2 | 1.4 | – | long_scoreboard, imc_miss, no_instruction |

**Prefill (target, 631-token doc prompt)** — kernel families (time-weighted)

| family | time % | SOL compute % | SOL memory % | achieved occupancy % | issue active % | AI | top warp stalls |
|---|---|---|---|---|---|---|---|
| gemm (tensor core) | 87.7 | 77.1 | 52.6 | 16.4 | 12.9 | 444.6 | math_pipe_throttle, wait, selected |
| rmsnorm/rope/silu (fused Triton) | 7.4 | 15.0 | 78.9 | 80.0 | 16.2 | 1.6 | long_scoreboard, barrier, short_scoreboard |
| gemv (cuBLAS) | 3.3 | 13.1 | 97.3 | 24.6 | 6.6 | 0.9 | long_scoreboard, lg_throttle, barrier |
| other elementwise | 1.1 | 17.0 | 57.3 | 71.2 | 21.5 | 0.0 | long_scoreboard, imc_miss, wait |
