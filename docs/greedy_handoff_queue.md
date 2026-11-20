# Greedy handoff successor — 19 November 2026

The existing complete SSD baseline is preserved in the main checkout at
`/home/vishalrao/AnytimeOutcome`. Its engine files are unchanged. The candidate
lives in `/home/vishalrao/AnytimeOutcome/.worktrees/greedy-handoff-v1`, on
`codex/greedy-handoff-v1`, based on commit `1084d91`.

Here, original SSD means our current shared-fixed16 complete SSD. It is not a
claim that its numerical implementation reproduces the older upstream AR.

## Implemented candidate

- `greedy_transport=True`: cache and send draft token IDs, without keeping the
  tree's full vocabulary-score history or transferring draft score arrays.
  The draft model still computes its vocabulary head to select each token.
  The target retains its existing shared arithmetic and K+1 verification graph.
- `handoff_service='threaded'`: a dedicated CPU thread observes requests and
  serves completed token prefixes on its own CUDA stream. The producer remains
  the only thread allowed to run models or write KV/graph buffers. Cache misses
  delegate JIT drafting to that producer.
- `handoff_bypass_ready=True`: a fully completed continuation can be delivered
  through an immutable view, avoiding a snapshot copy. This removes that copy;
  it does not promise that every publication checkpoint becomes free.
- `handoff_trace=True`: diagnostic-only GPU events aligned to a bracketed host
  clock, including its uncertainty. Logs distinguish prefix readiness, request
  posting, host observation, snapshot selection/copy, target receipt, and full
  continuation readiness. A tail that was never submitted has no invented
  completion timestamp.

Token-only transport explicitly rejects stochastic decoding. Original defaults
remain unchanged in the candidate checkout as well. Performance workers never
enable trace calibration or artificial delays.

## Ownership rules

Each live tree has a generation, producer epoch, immutable recovery keys,
expected committed context lengths, separate completion events, and a single
consumer claim. The consumer reads only completed columns and valid parent
positions. Partial replies use independent token storage. A complete-view
bypass retains its producer storage until response completion. Before another
model call or KV reuse, the producer joins the response and every submitted
tail write. Prefill, request end, and shutdown follow the same retirement rule.

Model execution and the process-global attention context remain on one thread.
The independent service thread performs communication and completed-buffer
reads only. This is handoff overlap with queued work, not GPU preemption or
unbounded speculative work after publication.

## Frozen comparisons

All timing arms use fresh workers and dedicated UUID-pinned GPUs. The original
arms import the actual main-checkout engine, not an optimization-disabled
approximation of the candidate. Each result records its imported module path.

| Arm | Implementation | Purpose |
|---|---|---|
| `legacy_complete` | Unchanged original checkout | Preserve the existing baseline |
| `legacy_overlap` | Unchanged original checkout | Existing publication comparator |
| `ids_complete` | New version; complete continuation, IDs only | Isolate transport/cache savings |
| `ids_inline` | New version; publication on producer CPU thread | Isolate token-only publication |
| `ids_threaded` | New version; independent CPU service | Isolate host service separation |
| `ids_bypass` | New version; independent service plus ready bypass | Measure the combined candidate |

Complete arms tune K=4/7/10. Publication arms tune K=7/10 with F=3, P=4 and
queue bound 2 fixed. This study retunes K, not an exhaustive P/window grid.
Two repetitions on 24 balanced tuning prompts select each arm by mean full
request latency, with smaller K as the tie break. Winners are frozen before
reading held-out results. The 75 frozen test prompts are reused, not newly unseen.

The selected six arms run five repetitions under fixed256 and natural-EOS/cap256
protocols. Prompt order is shared within rounds, method orders are seeded
permutations without replacement, and protocol order alternates. Natural EOS
does not retune the fixed-output-selected settings.

Primary interpretation requires both comparisons: new versus original complete
SSD, and publication versus new IDs-only complete SSD. A transport improvement
alone is reported as a transport optimization, not proof of publication benefit.

## Queue order and size

The queue waits for all valid blocks and the final report of
`results/next_stage/outcome_depth_v2_2026-11-18`. It verifies predecessor receipts
and exact outputs before acquiring the shared campaign lock.

| Phase | Blocks | Scored requests |
|---|---:|---:|
| CUDA buffer/stream ownership microtests | 1 | 0 |
| Fresh small AR plus 14 SSD correctness configurations | 30 | 180 |
| Artificial-tail stress with a small fresh AR control | 4 | 12 |
| Original/new timing diagnostics, excluded from performance | 6 | 18 |
| K tuning, two repetitions | 28 | 672 |
| Six selected SSD arms, five rounds, two protocols | 60 | 4,500 |
| **Total** | **129** | **5,382** |

Every model worker first runs three unscored warmup requests: 384 warmups in
total. Correctness uses six real/boundary inputs and both output protocols.
Stress must show a real response before submitted tail completion. Performance
cannot proceed after an exact-output failure or absent stress coverage.

Approximate budget: **4–6 dedicated GPU hours**, after the predecessor completes.
This is a proxy based on earlier worker timings; candidate timings are unknown.
GPU waiting, retries, compilation changes and failures can extend elapsed time.

## Operation and evidence

- Controller: `.worktrees/greedy-handoff-v1/profiling/run_greedy_handoff_campaign.py`.
- Results: `results/next_stage/greedy_handoff_v1_2026-11-19`.
- tmux session: `ao-greedy-handoff`.
- AR uses physical GPU0; SSD target GPU0 and draft GPU1. Both GPUs must be free.
- GPU UUIDs, topology, software, model/tokenizer fingerprints, exact input IDs,
  original/candidate source hashes and a source archive are frozen before work.
- BF16, batch one, CUDA graphs, JIT fallback, capacity4096, full prefills, disabled
  prefix lookup and .70 target/draft KV fractions remain fixed for performance.
- Only one worker runs at a time. External jobs invalidate only the current
  block. Completed valid blocks survive interruptions. No unrelated job is killed.
- CUDA initialization failures are saved as invalid attempts and retried after
  backoff. Other worker errors or correctness failures stop the queue for review.
- Changed code, workload, GPU identities or software reject resume.
- No full AR performance campaign is repeated. Saved shared AR validates tuning
  and held-out outputs; small fresh AR controls validate gate/stress inputs.
- Reports include paired 10,000-sample confidence intervals, per-family results,
  repeatability, acceptance, waiting, memory and final drain, plus separate
  diagnostic classifications and payload counters.

CPU tests and protocol dry runs validate the harness before launch. Full-model
CUDA correctness and performance remain pending until the dedicated queue runs.
