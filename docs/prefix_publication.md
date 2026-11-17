# Stable-prefix handoff with deferred tail draining

This opt-in prototype addresses the drain-before-handoff cost observed in the
bounded/adaptive publication diagnostic. Set `publication_policy='overlap'`,
`early_publish_tokens=4`, and `publication_max_inflight=2`. Target arithmetic,
the shared AR numerical reference, and the full K+1 verification graph remain
the same. The ordinary uniform-publication policy remains the default.

## Ownership and execution

The draft compute stream records a separate completion event after each tree
step's KV writes, logits, and tokens. It queues at most the configured number
of unfinished steps. When the target requests a continuation, the host selects
the deepest completed prefix of at least P tokens and stops submitting further
steps. If P is not yet complete, it waits for P only, rather than the submitted
tail. A late request after the final launch may also select a completed prefix.

A separate high-priority CUDA handoff stream waits on the selected prefix
event, not the tail event. It copies the eligible branches and completed
token/logit columns into independent, preallocated snapshot buffers. Parent
positions beyond the previous valid prefix are excluded. The snapshot retains
sequence/recovery keys, expected committed context lengths, and existing
generation/epoch checks. No dynamic boolean indexing or device allocation is
required to construct this snapshot. The snapshot storage is reused only after
its response consumer has finished.

Cache hits read that snapshot on the handoff stream and send the existing
NCCL response. The target may then verify while obsolete work finishes on the
draft compute stream. This is overlap of handoff/verification with work already
submitted; it is not a kernel-preemption mechanism and does not submit an
unbounded deeper continuation after publication.

The engine retains producer tensors and records their cross-stream allocator
use until all outstanding reads/writes finish. The following barriers remain:

- A cache miss drains the old tail before JIT drafting uses model/graph buffers
  or KV pages.
- After sending a hit response, the drafter joins handoff completion and drains
  the old tail before rebuilding the next canonical KV/glue state or launching
  another tree.
- Prefill, request-end acknowledgment, and exit drain outstanding work before
  clearing caches or reusing memory. Full-request latency includes final drain.

These rules prevent incomplete logits, stale contexts, allocator reuse, and
concurrent writes through reused CUDA-graph buffers. The target can accept only
the declared real prefix, even though its captured verification graph computes
padded tail rows.

High stream priority is a scheduling hint. Kernel overlap is not guaranteed:
long device kernels, lazy kernel loading, communication dependencies, or device
resource limits may still serialize a handoff. Therefore actual response-before-
tail evidence is required, rather than inferring overlap from stream creation.

This implementation chooses a short prefix when it observes a pending request
between bounded draft submissions or at the final submission check. It has no
independent host service thread. A request arriving after the host has already
chosen a full-depth snapshot can still wait for that snapshot. This limits the
overlap opportunities; the gate measures actual responses rather than assuming
every request can interrupt immediately.

## Validation campaign

`profiling/run_prefix_publication.py` freezes a new source/model/software/GPU
manifest and never resumes the predecessor with changed execution code.

The normal correctness phase has nine configurations: fresh shared AR and
complete, original uniform, bounded, and overlap SSD at K=7 and K=10. SSD uses
F=3, publication P=4, and bounded/overlap queue bounds of two. Six inputs cover
chat, code, document, and one synthetic 2048-token boundary per family. Fixed
256-token and natural-EOS/cap-256 protocols produce 18 blocks / 108 measured
requests. Every block has three unscored warmup requests.

A separate stress phase compares fresh AR and overlap K=10 on the three real
prompts under both protocols: four blocks / 12 measured requests. Its isolated
worker hook adds 32 short CUDA sleeps of two million cycles after steps at or
beyond depth four. This leaves submitted tail work observable during handoff.
The hook is never installed in ordinary engine or performance workers. Delay
is in device clock cycles, not a declared millisecond duration.

Both phases use BF16, batch one, CUDA graphs, JIT cache misses, full prefills,
4096-token model capacity, and 0.10 KV-memory fractions. These are engineering
diagnostics, not headline timing experiments. Every speculative block must
match fresh AR token-for-token before further blocks are scheduled. The stress
gate also requires actual responses delivered before tail completion in both
output protocols. This reference is shared-fixed16-v1, not original historical
AR. Broader correctness is always limited to the declared tested workload.

By default the gate waits for both GPUs to be dedicated and invalidates
contaminated attempts. The explicit `--allow-shared-correctness` option permits
the user's previously authorized correctness checks while unrelated jobs run;
receipts preserve those jobs and mark `performance_evidence=false`. It cannot
produce dedicated-access headline timing results. The monitor never terminates
unrelated processes. Interrupted/failed blocks cannot count as completed.

```
source profiling/env.sh
python -O profiling/run_prefix_publication.py \
  --outdir results/next_stage/prefix_overlap_v1_2026-11-17 \
  --allow-shared-correctness --watch
```

The `ao-prefix-overlap` tmux session runs this gate. Source changes reject
resume; a new versioned output directory is required after an execution change.

## Telemetry and remaining evaluation

The first eight entries of each request's `publication_policy_stats` retain
the existing verification/checkpoint counters. Four added entries are:

| Index | Meaning |
|---|---|
| 8 | Trees published with a valid continuation shorter than K |
| 9 | Snapshots started while the submitted tail event was incomplete |
| 10 | Responses completed on the handoff stream while the tail was incomplete |
| 11 | Wall seconds spent in deferred tail completion barriers |

The counters distinguish merely creating a partial snapshot from actually
delivering a response before tail completion. A pending-tail counter alone is
not proof of useful overlap. Existing request wait, prefix lengths, acceptance,
cache hits, output lengths, and target/draft memory accounting remain recorded.

After correctness and actual overlap coverage pass, a fresh dedicated-GPU
campaign must compare tuned complete SSD and both publication paths using
0.70 KV fractions, balanced tuning prompts, frozen winners, held-out repetitions,
and full-request latency including drain. A smaller handoff wait does not by
itself establish lower response latency: shorter prefixes can require more
verification rounds, and extra snapshot buffers add memory/copy costs.

Variable-length verification graphs and unequal continuation-depth allocation
remain separate changes. This prototype specifically removes the requirement
to finish already-submitted tail work before handing over a completed prefix.

## Completed validation, 17 November 2026 at 19:39 IST

All 22 blocks completed with valid monitored receipts, covering 120 measured
requests. All 102 speculative comparisons matched fresh shared AR exactly:
96/96 in the normal phase and 6/6 in the artificial-delay stress phase. The 12
normal-phase AR prompt/protocol outputs also matched the preceding shared AR
reference. This validates the declared inputs and shared numerical reference,
not original historical AR or all possible inputs.

Actual responses completed before their submitted tail event in normal K=10
runs: 323 fixed-output and 310 natural-EOS responses, 633 total. The stress
phase recorded 10 fixed-output and 90 natural-EOS responses before tail
completion. Both phases included rejection, 2048-token input boundaries (normal
phase), natural EOS at 179 tokens, and end-of-request draining. K=7 recorded no
response-before-tail events in these inputs; the feature has no demonstrated
overlap opportunity there.

The CPU suite passed 183 tests with 16 CUDA-specific skips, excluding the
standalone paged-attention GPU test from that CPU invocation. A separate GPU
invocation passed all 16 prefix and campaign tests, including the explicit
snapshot-before-tail and post-tail snapshot-integrity check. Syntax and
whitespace checks passed. Logs, source snapshots, receipts, and output tokens
are preserved under `results/next_stage/prefix_overlap_v1_2026-11-17/`.

This establishes the implemented overlap mechanism. It does not establish a
speedup. These were single-repetition, 0.10-KV correctness diagnostics with
changing unrelated GPU jobs recorded in their receipts. Their latency tables
must not be used to rank methods or replace dedicated held-out measurements.
The default policy is unchanged. A new dedicated-GPU tuning/held-out comparison
against complete K=7 SSD and original uniform publication remains necessary.
