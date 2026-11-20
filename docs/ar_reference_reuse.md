# Overnight queue amendment: reuse completed AR

The user approved removing redundant full AR timing runs on 18 November 2026
(IST). All previously completed baseline AR results remain intact. The three
waiting version-1 queues had no valid completed blocks when their controllers
were stopped. Their manifests, source archives, attempts and logs are preserved;
replacement queues use new directories and fingerprints.

## What changes

- Reuse the completed shared-fixed16-v1 baseline's two AR tuning blocks and ten
  AR held-out blocks for exact token references. This is the shared numerical
  reference, not the original pre-fix AR implementation.
- Validate their original monitored receipts, job/output hashes, frozen inputs,
  model/tokenizer, software and GPU identities. Compare target numerical source
  hashes and the AR CUDA-graph functions with the archived numerical contract.
- Run one short fresh AR continuity block on three tuning prompts (one per
  family) before publication tuning. All 256 output tokens must match saved AR.
- Keep the unequal-depth engineering gate's two small AR blocks: six coverage
  inputs under each output protocol. Full AR tuning/held-out timing arms are
  removed from both performance campaigns.
- Keep the context-length sweep's fresh AR blocks: those synthetic inputs are
  new and require their own references.
- Rerun every SSD configuration and comparison as declared. SSD-versus-SSD
  latency remains a fresh paired comparison, with five held-out repetitions.

No historical AR receipt is converted into a new receipt or counted as a newly
completed worker block. Reports show historical AR timing in a separate labeled
table. Ratios to historical AR are cross-campaign context; their bootstrap pairs
prompt IDs but resamples old and new rounds independently. Primary SSD intervals
still resample paired current rounds and prompts within families. Both use
10,000 seeded samples. Temporal GPU-state drift is not controlled for historical
AR ratios.

Selection loads tuning outputs only. The reference bank pins held-out files by
hash at preparation but does not parse their outputs/timing for tuning. Held-out
AR rows are loaded only for the held-out phase and final report.

## Replacement order and counts

| Campaign | Directory under results/next_stage | New blocks | Scored requests |
|---|---|---:|---:|
| Publication | publication_performance_v2_2026-11-18 | 73–83 | 3261–4011 |
| Context sweep | context_length_v2_2026-11-18 | 33 | 288 |
| Unequal depths | outcome_depth_v2_2026-11-18 | 114 | 5292 |
| Total | | 220–230 | 8841–9591 |

The optional ten blocks are the complete-K7 anchor if publication tuning selects
another complete-SSD K. Publication work is one continuity block, 42 SSD tuning
blocks, and 30–40 SSD held-out blocks. Unequal-depth work is 28 correctness-gate
blocks, 26 SSD tuning blocks, and 60 SSD held-out blocks. Warmups are unscored.

Only 51 scored AR requests remain: three continuity, twelve unequal-depth gate,
and thirty-six context-sweep requests. The amendment removes 1593 redundant
scored AR requests and 23 worker blocks net.

Historical worker timings predict around 2.8 dedicated GPU hours saved. Expected
total work is approximately 7.5–8 hours without waiting/retries; allow roughly
8–10 hours because new SSD policy costs are not established. The exact estimate
and assumptions are in `results/next_stage/queue_ar_reuse_2026-11-18.json`.

## Controllers and validation

The tmux session names remain `ao-publication-performance`,
`ao-context-length-sweep`, and `ao-outcome-depth`, now launching their version-2
drivers. Context waits for validated publication completion; unequal depths
waits for both validated predecessor campaigns. Both dependencies include the
continuity evidence.

Publication driver: `profiling/run_publication_reuse_ar.py`.
Context driver: `profiling/run_context_length_sweep_v2.py`.
Unequal-depth driver:
`.worktrees/outcome-depth-v1/profiling/run_outcome_depth_reuse_ar.py`.
Reference validation/reporting: `profiling/saved_ar_reference.py` (also copied
and fingerprinted in the isolated worktree).

Engine execution code is unchanged by this amendment. Both-GPU dedicated access,
UUID allocation, shared global locking, one worker at a time, contamination
monitoring and block-level resume remain in force. Correctness or source drift
stops a queue; GPU contention waits and retries only the interrupted block.
