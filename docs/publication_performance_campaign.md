# Dedicated-GPU publication performance campaign

This is the performance follow-up to the completed stable-prefix correctness
gate. It compares complete SSD, original uniform early publication, and the
new stable-prefix overlap path. Fresh shared AR supplies token references and
one-GPU latency context. All SSD methods allocate two GPUs.

## Declared experiment

| Method | Tuning settings, F=3 | Configurations | Scored requests |
|---|---|---:|---:|
| AR | Fresh shared reference | 1 | 48 |
| Complete SSD | K=4,7,10 | 3 | 144 |
| Original uniform publication | K=7,10; P=2,4,7 | 6 | 288 |
| Stable-prefix overlap | K=7,10; P=2,4,7; queue bound=1,2 | 12 | 576 |

Tuning uses the same 24 frozen, balanced prompts (eight per chat/code/document
family), two repetitions, fixed 256-token output with EOS ignored. Each worker
performs three separate warmup requests, one per family. After the fresh AR
control, speculative configurations use seed-0 shuffled order in repetition
one and its reverse in repetition two. Prompt order matches across methods
within each repetition. F=3 is held constant from the prior baseline tuning;
this campaign searches K, minimum publication depth, and queue size.

One winner per SSD method minimizes mean full-request tuning latency. Exact
ties favor smaller K, then P, then queue bound. Selection reads tuning files
only, verifies exact outputs against fresh AR, and freezes `selection.json`
before creating or reading any held-out result. There is no EOS retuning.

The frozen winners and fresh AR run on the same 75 test prompts (25 per family),
five times each, under fixed-256 and natural-EOS/cap-256 protocols: 3,000 scored
test requests in 40 worker blocks. If complete SSD selects another K, an explicit
complete K7/F3 anchor adds 750 requests in ten blocks. If K7 is selected, its
existing winner results supply the anchor without duplicate measurements.

Method orders use five seed-0 permutations sampled without replacement. Each
round alternates protocol order and uses one matching seeded prompt order
across all methods. The two primary comparisons are overlap versus tuned
complete SSD and overlap versus tuned original uniform publication.

These 75 prompts were used in previous project evaluations. They remain separate
from configuration tuning, but are not a newly unseen workload. The campaign
records that limitation explicitly. No historical timing or output result is
substituted for a fresh measurement.

## Fixed execution conditions

- Target on physical GPU0, drafter on physical GPU1; UUID assignments and
  topology are frozen. AR uses only GPU0 but still requires both GPUs dedicated.
- BF16, batch one, CUDA graphs, JIT cache-miss fallback, and the approved
  shared-fixed16-v1 target numerical backend.
- 4096-token model capacity, complete input sequences at most 2048 tokens,
  identical saved token IDs, and target/draft KV fractions both 0.70.
- Weights and kernels warm; prefix caching disabled so every scored request
  performs its full prefill. No profiler or artificial delay hook.
- Full-request timing includes both prefills, handoff, verification, acceptance,
  and final obsolete-work draining. Loading, compilation, warmups, logging, and
  raw-result serialization stay outside the scored request timer.

The current engine must match the completed correctness gate's execution
fingerprint. All 102 speculative checks from that gate must pass, including
actual response-before-tail evidence. Tuning and testing also check outputs
against their fresh AR references. Exactness is scoped to shared AR and the
tested inputs; original historical AR is a separate numerical reference.

## Dedicated access and unattended execution

The driver waits until both GPUs have no unrelated compute jobs and low GPU
utilization. One benchmark worker runs at a time. During each worker it polls
both GPUs once per second and identifies only its own process group or exact
inherited run token as owned. If an unrelated job appears, the whole block is
invalidated, only our worker is stopped, and scheduling waits for dedicated
access to return. Valid earlier blocks remain resumable. No shared-GPU option
is available in this performance driver.

Execution code, software/environment, model/tokenizer files, source workload,
GPU identities, prompts, selections, raw results, and receipts are pinned or
validated. Interrupted/contaminated attempts cannot count toward five valid
repetitions. Resume rejects incompatible inputs or changed execution code.
Monitoring samples GPU process state; it cannot exclude jobs shorter than the
one-second sampling interval. Every accepted block retains its monitoring receipt.

The driver is `profiling/run_publication_campaign.py` and the campaign directory
is `results/next_stage/publication_performance_v1_2026-11-17/`.

```
source profiling/env.sh
python -O profiling/run_publication_campaign.py --watch
```

The unattended tmux session is `ao-publication-performance`. Attach with
`tmux attach -t ao-publication-performance`. `status.json` distinguishes running,
pending dedicated access, stopped on a validation error, and complete. The
watcher automatically resumes valid progress when a dedicated window returns.
It does not retry execution or correctness failures indefinitely.

The CLI also supports `--phase prepare`, `tune`, `select`, `heldout`, and
`report`, plus `--dry-run`. Intermediate phases enforce their dependencies.
Preparation freezes inputs without GPU computation. A new execution version
requires a new output directory.

## Analysis and completion

The tuning report lists every setting and the selection rationale. The final
report and figures show mean/p50/p95 request latency, committed tokens/s,
first-delivery latency, target waiting, acceptance, cache hits, tokens per
verification, output lengths, memory, and draining. Raw overlap telemetry
distinguishes partial snapshots from actual response-before-tail events.

Paired 95% confidence intervals use 10,000 seed-0 bootstrap samples of rounds
and prompts within families. Mean latency, speed ratios, per-family regressions,
exact-output agreement, first divergence positions, and within-method
repeatability are retained. Intervals are conditional on frozen configurations
and are per-comparison, without a familywise multiple-comparison adjustment.
Fixed-length and natural-EOS results remain separate.

The campaign requires 1,056 tuning plus 3,000–3,750 test requests. Warmups are
additional: 252 requests without the extra anchor, or 282 with it. Expect roughly
4–6 hours of dedicated GPU time; unavailable GPU time adds unpredictable waiting.
Completion requires every valid block, frozen selection, and the final report.
A prepared or waiting campaign is not a completed experiment.
