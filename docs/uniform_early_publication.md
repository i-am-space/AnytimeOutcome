# Uniform early publication: implementation and experiment protocol

**Status update, 17 November 2026:** The baseline, publication pilot, and
corrected overnight extension queue have finished. The earlier pending-status
notes below describe the initial launch. The extension held-out comparison
completed 1,500 requests, all matching shared AR, with no latency improvement
from the selected P=7/K=7 publication setting. See
`results/next_stage/overnight_extensions_v2_2026-11-17/heldout/report.md` and
[the bounded/adaptive successor](adaptive_publication.md) for the next implementation.

This is the first core project feature: SSD may return a completed continuation
prefix before its maximum drafting depth finishes. It is opt-in. Ordinary SSD
still builds its complete continuation by default.

## Numerical reference

Use `target_numerics='shared_fixed16'` consistently for AR, SD and SSD. This
integrates the approved `shared-fixed16-v1` prototype into the engine before
CUDA graph capture. Weights and prefill remain unchanged. The new AR arithmetic
does not promise original AR's outputs. Prior prototype evidence and the
original AR control remain in `docs/shared_fast_backend.md`.

The engine copies the validated prototype's numerical operations exactly;
a regression test checks that they have not drifted. A fresh correctness run
is required after integration because the execution code has changed.

## Policy and safety

Set `early_publish_tokens=4` for uniform minimum depth P=4, with maximum K=7
as the initial engineering configuration. This is a request-driven policy:
if the target requests a continuation while the tree is still building, the
drafter stops at the first completed step at or beyond P. If the request arrives
later, the response can contain five, six, or all seven ready tokens. P is a
minimum ready depth, not a forced truncation to four tokens on every hit.

The target increments a CPU shared-memory request epoch before entering the
NCCL handshake. The draft process checks it between GPU tree steps. This signal
is independent of the CUDA/NCCL stream, which can otherwise wait for the draft
stream to finish before the drafter learns that a request is pending.

After P, a reused CUDA event follows each step's KV writes, logits copy, and
sampled token writes. The worker waits for that event before exposing its valid
length. Only one step remains queued at those checkpoints; this deliberately
simple implementation includes its synchronization overhead in the experiment.

The cache and response carry the sequence/recovery key, valid continuation
length, cache generation, and expected committed-context length. Only fork positions
whose parent prefix is real are eligible for publication. A matching recovery
key with a different expected context is a miss and uses JIT drafting.

The response includes the current request epoch. The target rejects an invalid
length or stale epoch. Padded tail slots are initialized, but acceptance is
sliced to the actual published length. They cannot become accepted output.
The following glue pass rebuilds the canonical draft KV state for the committed
prefix and recovery token. All prior tree writes finish before these pages can
be overwritten or reused. Early publication currently supports greedy,
batch-one, non-EAGLE SSD with JIT cache-miss fallback.

## What this prototype still calculates

The target retains its captured K+1 verification graph. It computes padded tail
rows even when fewer than K draft tokens are available. Only the real prefix
participates in acceptance. The drafter also retains its existing branch layout
and filters invalid parent branches from its published cache. This keeps the
first feature compatible with existing graphs and KV allocation; reducing those
extra calculations is a subsequent optimization, not a claimed benefit here.

Early publication does not yet implement adaptive depth allocation, a new
fanout algorithm, or an early-publication speedup guarantee.

## Accounting

Both arms of the new SSD comparison enable `drain_draft_on_finish=True`.
Request latency includes prefills, communication, verification, acceptance,
and a final acknowledgment that obsolete work is finished and pages are safe
to reuse. The same rule is used by the new baseline SSD campaign. Older timing
results that omitted this final drain are a separate protocol.

Each request saves output IDs, accepted/verified counts, target wait across the
complete handshake, ready lengths, generation/epoch metadata, publication wall
time (including any outstanding GPU completion in the complete-cache arm), interrupt/drain wall time, final request drain, and target/draft peak
allocated and reserved CUDA memory. Target wait includes handoff and receive;
it is not interpreted as pure draft computation. Allocator memory includes the
configured KV allocation; unchanged capacity need not reduce memory usage.

## New baseline campaign

The original campaign cannot resume under this execution code. Prepare a new
versioned directory with `run_baselines.py --target-numerics shared_fixed16`.
The declared workload and tuning grid remain unchanged: 480 tuning requests,
then 2,250 held-out requests, 24/75 prompts balanced across families, two output
protocols, five held-out rounds, prefix reuse disabled, and three unscored
warmups per worker. Selection uses tuning only. Numerical contract and drain
policy are hashed into the new manifest and jobs. Existing original-AR controls
are retained separately; old winners and measurements are not imported.

## Publication study

`profiling/run_publication_study.py` has two modes:

- `correctness`: integrated AR, SD, complete-cache SSD and early-publication SSD
  on nine previously investigated prompts, both EOS protocols, one repetition,
  .10 KV fraction. These timings are diagnostic.
- `compare`: AR plus the two SSD arms, five paired rounds, both protocols,
  .70 KV fraction, 30 valid blocks / 270 measured requests. With
  `--baseline-dir`, K/F are read from the validated new tuning selection. No
  tuning of P on these results is performed. The pipeline declares P=min(4,K-1)
  before tuning so K=4 still has an early checkpoint.

Both modes require dedicated GPUs. UUID pinning, source/software/model hashes,
block receipts, and process monitoring are recorded. Interrupted or contaminated
blocks cannot count; only our own worker process group is stopped. `--watch`
waits for the dedicated window to return. Execution changes reject resume.

The comparison reports full-request mean/p50/p95 latency, throughput, first
delivery, target waiting, tokens per verification, accepted/verified draft
counts, partial handoffs, publication/drain overhead, memory, output lengths,
per-family means, exact agreement, and within-method repeatability. Paired 95%
intervals use 10,000 seeded draws of rounds and prompts within families.
These nine seen prompts are engineering evidence. They do not replace the
75-prompt held-out baseline or establish unseen-workload generalization.

## Completion criteria

The feature requires unit tests and actual GPU rejection/EOS validation, with
observed partial handoffs. The baseline is complete only after all 50 valid
blocks and its final report exist. The publication comparison is complete only
after its 30 valid blocks and report exist. Pending or running jobs are not
completed experiments. Numerical correctness is scoped to the declared shared
reference and tested inputs; original historical AR drift remains unresolved.

`check_publication_interrupt.py` adds a separate artificial 4 ms delay before
each draft tree step. It runs shared AR and early SSD on the three historical
coverage prompts (chat `specbench-148`, code `HumanEval/36`, and the
EOS-producing document `specbench-496`), under both protocols, to exercise
incomplete-tree handoffs end to end. This source-hashed
process-local hook is never installed in baseline or performance workers. A
stress-test speedup is not a performance result. If natural runs have zero
partial handoffs, report that directly: the feature has no demonstrated waiting
benefit on that workload.

The unattended sequence is `profiling/run_next_stage.py`: normal correctness,
artificial interruption safety, a new baseline campaign, and then the paired
publication study. It stops on output mismatch, validation failure, or code
change. Dedicated GPU interruptions wait and resume invalid blocks. Expected
wall time is several hours of dedicated access; waiting for shared GPUs is
additional and cannot be predicted.

## Completed initial checks (17 November 2026 IST)

The normal integrated-engine gate completed 8 valid blocks / 72 requests.
SD, complete-cache SSD, and early SSD matched the new AR in all 54 paired
comparisons across fixed and natural EOS protocols. Integrated AR also matched
the earlier shared prototype on all nine inputs under both protocols.

These normal runs observed zero partial handoffs: all seven tokens were ready
by handoff. Single-repetition diagnostic means were 2.296 s complete-cache
versus 2.383 s early for fixed length, and 2.201 s versus 2.257 s for natural
EOS, at .10 KV fractions. These are not five-repeat headline comparisons and
do not demonstrate an early-publication speedup.

The CPU suite passed 122 tests; 15 CUDA-specific tests were skipped in that
CPU invocation, and the existing standalone paged-attention GPU test was
excluded from it. The separate real-GPU gate above did run CUDA graphs.

The separate 4 ms/tree-step safety stress check completed 4 valid blocks /
12 requests. All six early-SSD outputs matched shared AR. It observed 186
partial handoffs under fixed length and 170 under natural EOS (356 total),
including the document stopping at 179 tokens. None of these artificial-delay
latencies are performance evidence.

The unattended session is `ao-next-stage`. Results and status are under
`results/next_stage/shared_uniform_v1_2026-11-17/`. Attach with
`tmux attach -t ao-next-stage`. The baseline and publication reports are
pending until their required valid blocks have actually completed.
