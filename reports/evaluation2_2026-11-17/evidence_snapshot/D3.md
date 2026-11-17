# Shared fast target arithmetic: experimental reference v1

The project now has a fast prototype that uses the same numerical target
implementation for AR, SD and SSD. The user explicitly approved prototyping this
new reference after the original-AR-compatible candidates proved too slow.
It is a separate, versioned experiment. Original AR remains an output and
performance control; the default engine and frozen campaign are unchanged.

Batch-independent arithmetic is existing prior work, not this project's research
contribution. See [Thinking Machines Lab's explanation](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
This prototype supplies numerical infrastructure for testing the proposed early
publication and depth-allocation policies on a consistent fast baseline.

## Implementation

`profiling/shared_numerics.py` installs target-only wrappers before CUDA graph
capture. Draft model construction and operations remain original.

- Target decode/verify linear projections pad to 16 rows and use the same
  tensor-core GEMM shape in all three methods, discarding unused rows.
- Normalization, activation and rotary operations use fixed padded shapes,
  preserving query/key head groups and residuals.
- Attention evaluates queries in parallel using the same single-query kernel,
  split count and visible length for each query. This avoids changing the
  reduction when a verify block crosses a split boundary.
- Target prefill retains the original implementation. All methods receive the
  same complete frozen inputs and perform full prefills.

Weights remain Qwen3-8B / Qwen3-0.6B in BF16. The supported prototype scope is
greedy, batch size one, one target GPU, CUDA graphs, and query blocks of at most
16 tokens. AR and SD use GPU0; SSD uses GPU0 plus GPU1 for the drafter. Engine
defaults are not modified: the experimental gate installs the prototype inside
its own worker process.

The 16 rows are an internal matrix shape, not 16 simultaneous requests. Request
batch size remains one throughout these experiments.

The first attempt failed before scoring because the engine captures extra
single-query decode buckets even with `max_num_seqs=1`. The corrected attention
dispatch supports these setup buckets. The failed attempt is retained, excluded
from evidence, and covered by a regression test.

## Completed numerical and correctness evidence

The projection test covers all 145 projections, every query row, three current
failure prefixes and query widths seven/eight: 870 operation cases, 6,525 rows.
The fixed-shape GEMM agrees exactly with its own one-query reference in 870/870
cases. It differs from original AR in every projection case at the tensor level;
this does not mean every generated token must differ.

Isolated CUDA-graph sweeps over real layer weights measured approximately
22.6 ms for fixed-shape GEMM versus 22.1 ms for ordinary fast GEMM. The BMM
candidate took 122–140 ms and was rejected for speed. These are contextual
operation costs, not full-request speedups.

Attention boundary tests cover widths 5, 7, 8 and 9 at nine prefix lengths,
including page edges and the 2,048-token split boundary. Original batched
attention matched individual queries in 26/36 cases. The parallel-query
prototype matched in 36/36 cases and matched CUDA graph replay in all cases.

The complete shared-backend gate passed **36/36 paired comparisons**:
nine prompts, SD and SSD, fixed 256 outputs and natural EOS capped at 256.
It includes rejection rounds and actual EOS stops. Each configuration/protocol
uses a fresh process and three warmups. The gate uses 0.10 KV fractions and
records shared GPU activity; its request times are diagnostic.

Fixed-length diagnostic means were 6.330 s for shared AR, 3.298 s for shared SD
and 2.327 s for shared SSD. Natural-EOS means were 6.055 s, 3.172 s and 2.187 s.
These observations motivate a controlled benchmark, not a headline speedup claim.

Agreement is **against `shared-fixed16-v1` AR**. Original AR is retained and its
differences are explicitly recorded: the shared AR output differs from original
AR on **7/9 prompts under each protocol**. This prototype does not repair exact
agreement with original AR, and answer quality has not been separately evaluated.
Its older historical reproduction gate
still matches only one of three saved outputs. The failed historical gate has
not been relabeled a pass. The summary exposes paired correctness and historical
reproduction as separate fields.

## Dedicated prototype benchmark

The separate prototype benchmark declares three already-investigated prompts,
one per family, fixed 256 outputs, five rounds, and four methods: original AR,
shared AR, shared SD K6 and shared SSD K7/F3. It requires 20 complete valid worker
blocks and 60 scored requests, with three warmups per worker. It uses 0.70 KV
fractions, full prefills, BF16 and CUDA graphs, with no profiler capture.

Five distinct seeded method-order permutations are used. Prompt order is paired
within each round. The supervisor requires both GPU UUIDs to be free before each
worker, monitors for unrelated compute, invalidates contaminated blocks, and
stops scheduling. Resume retains valid blocks and creates a new attempt for a
failed block; unrelated jobs are never terminated.

Outputs include request latency, first delivery, committed tokens/s, acceptance,
cache hits, token IDs, per-round exactness and within-method repeatability. Paired
confidence intervals resample rounds 10,000 times with seed 0. There is only one
fixed prompt per family, so these intervals do not estimate prompt generalization.

This is an engineering benchmark, not the 75-prompt held-out Evaluation 1
campaign. K/F configurations have not been retuned for the new backend.

### Completed dedicated results

All **20/20 blocks** completed with valid receipts, 60 scored requests and no
observed unrelated GPU compute. Each method has five fresh-process repetitions
on all three prompts. Shared SD/SSD matched the paired shared AR output in
**30/30 comparisons**. All four methods reproduced their own outputs across all
five rounds. These are additional repeatability checks, not additional unseen
prompts.

| Method | GPUs | Mean latency | p50 | p95 | Committed tokens/s | Mean first delivery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original AR control | 1 | 5.619 s | 5.549 s | 5.858 s | 45.56 | 94.5 ms |
| Shared AR v1 | 1 | 6.281 s | 6.244 s | 6.427 s | 40.76 | 62.5 ms |
| Shared SD K6 | 1 | 3.491 s | 3.747 s | 4.492 s | 73.33 | 137.6 ms |
| Shared SSD K7/F3 | 2 | 2.500 s | 2.836 s | 3.343 s | 102.40 | 181.6 ms |

SSD is **2.51 times faster than shared AR** on this declared prompt mix, with a
paired round-bootstrap 95% interval of **2.46–2.58**. SD is **1.80 times faster**,
interval **1.78–1.81**. These intervals concern repeated rounds, not new prompts.

SSD is also 2.25 times faster than original AR (interval 2.19–2.31), but this is
only a performance control: the two use different target arithmetic and can
produce different tokens. The new shared AR is approximately 11.8% slower than
original AR; retaining the original control makes that reference cost visible.

| Family / fixed prompt | Original AR | Shared AR | Shared SD | Shared SSD |
| --- | ---: | ---: | ---: | ---: |
| Chat / specbench-148 | 5.545 s | 6.240 s | 4.487 s | 3.341 s |
| Code / HumanEval/36 | 5.657 s | 6.253 s | 2.184 s | 1.394 s |
| Document / specbench-553 | 5.656 s | 6.351 s | 3.801 s | 2.765 s |

First-delivery outliers are retained rather than trimmed. A separate compiler
audit checks scored requests with `fail_on_recompile` and records Dynamo graph
counters. All four method audits passed on all three prompts (12 scored
requests), with unchanged counters and no recompilation errors. This is a
separate audit of the warmup strategy, not compiler instrumentation retroactively
added to the original 20 benchmark blocks. The initial audit's final
success flag was not reached because the engine hard-exits its worker. A fixed
audit persists its conclusion before the last request returns and has a
regression test; the initial attempt remains preserved separately.

The raw summary is
`results/correctness/shared_fixed16_prototype_benchmark_2026-11-17/summary.json`.

## Reproduction and remaining work

After sourcing `profiling/env.sh`:

```bash
python -O profiling/run_ar_compatible_gate.py \
  --candidate shared_fixed16 --allow-shared-gpus \
  --outdir results/correctness/shared_fixed16_new_attempt

python -O profiling/benchmark_shared_numerics.py \
  --gate results/correctness/shared_fixed16_v1_2026-11-17_v2 \
  --outdir results/correctness/shared_fixed16_prototype_benchmark_2026-11-17
```

The benchmark command also resumes a compatible pending/interrupted run. A code
or environment change is rejected; create a new experiment version in that case.

Before adopting the prototype: expand exactness across K values, longer prefixes,
more prompts and repetitions; then retune and freeze a new performance campaign.
The original manifest must not be
reused for this changed numerical backend. Repeat bottleneck profiling as well:
the old Nsight traces describe the original backend, not this version. Uniform early publication and adaptive
depth allocation remain subsequent project work.

Raw evidence is under `results/correctness/`, including
`fast_numerics_all_layers_2026-11-17.json`,
`attention_boundaries_2026-11-17.json`,
`fixed_gemm_controls_2026-11-17.json`,
`shared_fixed16_v1_2026-11-17_v2/`, and
`shared_fixed16_prototype_benchmark_2026-11-17/`.
Exact executed source snapshots accompany the experiment records.

The final CPU regression command reports **99 passed, 15 skipped**:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests -q --ignore=tests/test_paged_attention.py
```

The existing CUDA-only paged-attention suite is excluded from that CPU command.
GPU evidence is recorded separately in the full-generation gates, attention
boundary checks, dedicated benchmark and compiler-guard audits. The complete
machine-readable evidence summary is
`results/correctness/shared_fast_backend_progress_2026-11-17.json`.
