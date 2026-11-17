# Bounded and adaptive early publication, version 1

This implements the first two improvements after the uniform-publication study:
bounded asynchronous checkpointing and a request-local adaptive publication rule.
It does not change target numerical operations or replace complete-cache SSD.
All modes remain opt-in; `publication_policy='uniform'` preserves the first prototype.

## Completed predecessor experiments

The corrected overnight queue finished on 17 November 2026 at 15:26 IST.
Its 1,500 held-out SSD requests all matched the shared AR reference. Selected
P=7 with K=7 never published a shorter prefix. Fixed-output mean request latency
was 2.2006 s for complete SSD versus 2.2230 s with publication enabled; natural
EOS means were 1.9045 s and 1.9203 s. Neither protocol showed an improvement.

The separate K=10 correctness diagnostic did exercise partial handoffs: 604
fixed-output and 592 natural-EOS handoffs at P=7. Fixed-output mean latency was
3.6901 s complete versus 3.1065 s early, but complete K=7 was still faster at
2.6145 s. These are diagnostic timings, including synthetic boundary prompts,
not a performance win over the strongest baseline.

Existing results and source snapshots remain under
`results/next_stage/overnight_extensions_v2_2026-11-17/` and
`results/next_stage/shared_uniform_v1_2026-11-17/`.
The new source fingerprint cannot resume these completed campaigns.

## Bounded checkpointing

Set `early_publish_tokens=4`, `publication_policy='bounded'`, and
`publication_max_inflight=2`. Every draft depth gets its own preallocated CUDA
event. The CPU queries completion without blocking and may enqueue at most two
unfinished steps. It waits only when it reaches this queue bound or must hand
over a ready prefix to a pending target request.

A request drains the bounded submitted tail before exposing its valid length.
All submitted tokens are then safe, and their KV writes have finished. No later
step is submitted after stopping. Existing generation/epoch/context checks,
acceptance slicing, cache-population completion, and final request draining
remain active. The algorithm does not preempt an executing CUDA kernel.

This still can block under backpressure. It removes the unconditional
one-step barrier, not every CPU/GPU synchronization. Additional events and
queries have costs, so lower latency must be measured rather than assumed.

## Adaptive rule

Use `publication_policy='adaptive'` with the same minimum depth and queue bound.
The target shares exponentially smoothed verification time, accepted draft
tokens, and whether a rejection occurred (alpha 0.2). Statistics reset at each
request, including after warmup. Untested tokens after the first rejection do
not count as failures. A weak Beta(0.5, 0.5) prior stabilizes the estimated
per-token survival probability.

When a request is pending and the minimum prefix is ready, the drafter compares
candidate depths d from the submitted depth r to K using

`estimated_cost(d) = [verify_seconds + (d-r)*draft_step_seconds] / [1 + sum(p**i, i=1..d)]`.

Draft step duration comes from completed CUDA event timestamps for the current
tree, including intervening idle gaps. The target's time includes its forward
and acceptance processing. The rule waits for additional tokens only if its
estimated cost falls by more than 5%; otherwise it publishes the current
completed prefix. With fewer than two verification observations, it publishes
without extrapolating acceptance history. Ties favor shorter depth.

This is a causal, local heuristic. The geometric acceptance approximation is
imperfect; it does not model cache misses, future-round effects, variable
verification lengths, or transfer time separately. It may choose poorly and
does not promise a speedup. Target verification still computes K+1 rows.

## Validation and invocation

The new driver is `profiling/run_adaptive_publication.py`. Its frozen gate uses
eight configurations: fresh shared AR; complete, bounded, and adaptive SSD at
K=7; complete, original uniform, bounded, and adaptive SSD at K=10. All SSD arms
use F=3; publication arms use P=4 and bounded/adaptive arms allow two unfinished
steps. Six prompts include three real prompts and three synthetic 2048-token
boundary prompts, with fixed and natural EOS output protocols. This is 16
blocks and 96 measured requests, plus three warmups per block.

The gate uses BF16, batch one, CUDA graphs, JIT misses, a 4096-token capacity,
prefix caching disabled, and 0.10 KV fractions. Both GPUs must be dedicated.
Contaminated blocks do not count. The driver checks each completed speculative
block against fresh AR before scheduling more, and requires actual partial
handoffs in both new policies before marking the gate complete. All timing in
this gate is diagnostic; a new 0.70-KV tuning and held-out campaign is needed
before making performance claims.

```
source profiling/env.sh
python -O profiling/run_adaptive_publication.py \
  --outdir results/next_stage/adaptive_publication_v1_2026-11-17 --watch
```

Use `--dry-run` to freeze and inspect the workload without GPU computation.
Code, software, model, prompt, and GPU identities are recorded. Resume rejects
changed execution code. The successor retains a source ZIP and a pre-change
source archive. Live status is in its `status.json`; accepted results require
a valid block receipt, not just a worker output file.

Raw request field `publication_policy_stats` stores eight entries: verification
time EWMA, accepted-count EWMA, rejection-count EWMA, observation count, event
queries, explicit checkpoint waits, publication decisions, and decisions that
continued beyond the ready prefix. Ready-length histograms and ordinary wait,
acceptance, memory, and end-drain accounting remain available.

## Remaining improvements

Transferring a published prefix while deeper work continues, variable-length
verification graphs, and unequal continuation-depth allocation are not part of
this change. They require separate buffer/stream ownership or target execution
changes and independent correctness and performance validation.

## Completed validation, 17 November 2026 at 17:46 IST

The CPU suite passed 168 tests, with 15 CUDA-specific skips; the standalone
paged-attention GPU test was excluded from that CPU invocation. The actual GPU
gate completed all 16 valid blocks / 96 requests. All 84 speculative outputs
matched fresh shared AR. The 12 fresh AR prompt/protocol outputs also matched
the predecessor's shared AR exactly. Both bounded and adaptive modes exercised
590 partial handoffs, including rejection and natural EOS stopping at 179 tokens.

Mean full-request latency in this six-prompt, single-repetition diagnostic:

| Configuration | Fixed 256 tokens | Natural EOS, cap 256 |
|---|---:|---:|
| Complete K=7 | 2.5125 s | 2.4293 s |
| Bounded K=7 | 2.5401 s | 2.4379 s |
| Adaptive K=7 | 2.5463 s | 2.4424 s |
| Complete K=10 | 3.5454 s | 3.4383 s |
| Original uniform P=4, K=10 | 2.7983 s | 2.6951 s |
| Bounded P=4, K=10 | 3.0398 s | 2.8963 s |
| Adaptive P=4, K=10 | 3.0089 s | 2.9033 s |

The new variants did not improve on original uniform publication or the
stronger complete K=7 control. These are 0.10-KV diagnostic results, including
synthetic contexts, not a repeated held-out performance campaign. No default
was changed and no acceleration claim is made for the new variants.

The fixed K=10 records explain the trade-off: original uniform publication
mostly handed over six tokens (284 rounds), whereas bounded mostly handed over
seven (272 rounds). Mean target waiting rose from 0.7769 s to 1.0190 s per
request. Adaptive similarly handed over seven tokens in 274 rounds, with
1.0236 s waiting. The adaptive decision rule never elected to extend the ready
prefix in this gate; extra depth came from work already submitted, which the
handoff had to drain. Unit tests separately exercise the decision-to-extend
branch. A future GPU workload should exercise that branch before claiming
broad behavioral coverage of the adaptive policy.

At K=7 all seven tokens were ready at handoff. Checkpoint/event management
therefore added costs without saving draft waiting on this workload. Eliminating
the drain-before-handoff dependency using an immutable prefix and explicit
buffer ownership is the next concrete engineering question. It must still
include eventual draining and KV reuse costs in full-request latency.

Raw data and the generated report are under
`results/next_stage/adaptive_publication_v1_2026-11-17/correctness/`.
The CPU log and AR continuity checks are stored in the parent directory.
