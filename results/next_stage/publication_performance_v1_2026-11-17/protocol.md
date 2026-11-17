# Declared publication performance protocol

Only dedicated GPU measurements can count. Both GPUs are checked even during AR. One worker at a time; contamination invalidates the entire block and stops scheduling until GPUs return.

| Method | Tuning grid | Requests |
|---|---|---:|
| AR control | Fresh shared AR | 48 |
| Complete SSD | K=4/7/10, F=3 | 144 |
| Original uniform publication | K=7/10, P=2/4/7, F=3 | 288 |
| Overlap publication | K=7/10, P=2/4/7, queue=1/2, F=3 | 576 |

Total tuning: 1,056 scored requests in 44 fresh worker blocks, two repetitions with reversed seeded configuration order. Three separate family-balanced warmups per block. Tune fixed 256-token output with EOS ignored; select lowest mean full-request latency. Exact ties choose smaller K, then P, then queue bound.

After selection is frozen, test AR and the three winners on 75 prompts × five rounds × two output protocols: 3,000 scored requests / 40 blocks. If complete SSD selects K other than seven, add a declared K7/F3 anchor: 750 further scored requests / ten blocks. Each round uses a seeded method-order permutation drawn without replacement; protocol order alternates and prompt order matches across methods.

Primary comparisons: overlap vs tuned complete SSD, and overlap vs tuned original publication. AR is a fresh output reference and latency context. Natural EOS reuses fixed-length-selected settings. BF16, batch one, CUDA graphs, JIT misses, 4096-token model capacity, input ≤2048 tokens, full prefills, 0.70 KV fractions, UUID-pinned GPUs, no profiler, and final drain accounting are fixed.

Reused frozen test prompts; not a newly unseen workload. Test results are never used to select configurations. All timing results are fresh. Original historical AR is a separate reference; exactness is checked against shared-fixed16-v1 AR.

The campaign is complete only when all valid blocks, selection.json, and the final report exist. A GPU-wait status is pending work, not completed measurement.
