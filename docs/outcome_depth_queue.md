# Evaluation 2: unequal-depth successor queue

**Current amendment:** full AR timing repeats have been removed. The active
version-2 plan is [AR reference reuse](ar_reference_reuse.md). The original
version-1 plan below is retained as historical context; its watcher was replaced.

The new implementation is isolated in
`.worktrees/outcome-depth-v1/`, on branch `codex/outcome-depth-v1`. This preserves
the frozen execution code used by both existing experiment queues. The exact
new engine and driver sources are also archived in the successor's `source.zip`.

Implementation details and ownership rules:
`.worktrees/outcome-depth-v1/docs/outcome_depth.md`.

## Status of the five proposed improvements

| Improvement | Current state |
|---|---|
| Replace unconditional publication-checkpoint waits | Implemented in bounded/overlap policies. Queue backpressure and necessary safety barriers remain; negligible overhead has not been demonstrated. |
| Dynamically publish or continue drafting | A causal local heuristic exists in the separate adaptive policy. It did not choose extra depth in the completed GPU diagnostic and has no demonstrated performance benefit. It is not combined with overlap or unequal-depth allocation here. |
| Stable prefix handoff before the submitted tail finishes | Implemented and validated on the declared shared-AR workload. Actual response-before-tail events were observed. The mechanism overlaps already-submitted work, then drains before KV/graph reuse. Dedicated performance comparison remains pending. |
| Shorter verification graphs | Not implemented or queued. Target verification still runs K+1 rows. Shared arithmetic pads some operations to 16 rows, so savings must be measured. |
| Different continuation depth per recovery outcome | New opt-in static rank-based prototype and successor campaign. Kernel/CPU validation is separate from the queued full-model correctness gate and performance study. |

## Overnight order

1. Existing publication performance campaign.
2. Existing controlled context-length sweep and profiles.
3. New unequal-depth correctness gate, tuning, frozen selection and comparisons.

The third watcher is `ao-outcome-depth`. Its results and live state are in
`results/next_stage/outcome_depth_v1_2026-11-18/`.

The new allocation uses K7 depths `[7,4,4]` and K10 depths `[10,7,4]` for the
three recovery-token ranks, at each possible acceptance position. Fan-out stays
three. Deeper steps use smaller actual CUDA-graph batches. Every outcome carries
its own valid length; the target can never accept its uncomputed tail.

Controls include complete SSD K4/7/10, overlap with P4/window2 at K7/10, uniform
depths with equal full-tree query-row budgets, and reversed rank priorities.
Overlap K is retuned with P/window fixed; this is not exhaustive publication
tuning. Core winners are selected on tuning prompts alone. Ablation K follows
the selected unequal-depth K by a predeclared rule.

The successor contains 168 correctness requests, 672 tuning requests and 5250
held-out requests, plus 378 unscored warmups. The detailed immutable protocol,
counts, allocation, source/model hashes, and limitations are in its `protocol.md`
and `queue.json`. The 75 held-out prompts are disjoint from tuning but have been
used in prior studies. Natural EOS uses the fixed-output-selected configurations.

Both GPUs must be dedicated for any accepted campaign block. All three watchers
share the same main-checkout lock and start one worker at a time. A foreign job
invalidates its current block; already completed valid blocks survive. Foreign
processes are never terminated. The successor waits outside the lock until both
predecessor reports and valid results exist, and stops on correctness/code drift.

Existing baseline timings suggest roughly five to six additional dedicated GPU
hours for the successor and around ten to twelve hours for all three queues.
GPU waiting and invalidated retries are extra; the new policy's speed is unknown.
The calculation and historical sources are saved in the successor's
`runtime_estimate.json`.

For Evaluation 2, completing these comparisons is more useful than adding another
large sweep now. A subsequent short-prefix-verification prototype would answer a
separate question and require a fresh correctness gate and versioned measurements.
