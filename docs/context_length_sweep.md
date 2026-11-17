# Queued context-length diagnostic

This successor starts only after the dedicated publication performance campaign
finishes successfully. It leaves that campaign and engine source unchanged.
The queue is under `results/next_stage/context_length_v1_2026-11-17/`; its
predecessor is `results/next_stage/publication_performance_v1_2026-11-17/`.

The question is whether increased cached context makes a full draft tree take
longer relative to target verification, and whether ready-prefix publication
then reduces total request latency. No benefit is presumed.

## Workload and comparisons

- Exactly 256, 512, 1,024, and 2,048 complete input tokens.
- One fixed synthetic task per chat/code/document family, extended with
  explicitly irrelevant repeated notes. Tasks and chat formatting are not
  truncated. Exact token IDs, text, tokenizer/template and model fingerprints
  are frozen. This is a controlled synthetic sensitivity experiment, not a
  held-out workload or a measure of answer quality.
- Fixed 256 output tokens, EOS ignored; current context grows by up to another
  256 tokens during a request. A longer-output study is not in this queue.
- Fresh shared AR; complete SSD, original uniform publication and overlap at
  K=7 and K=10, F=3. Publication uses P=4 and queue bound two. K stays matched
  within policy comparisons. These are declared diagnostic configurations,
  rather than configurations selected using this new workload.
- BF16, batch one, graphs, JIT cache misses, 4,096-token capacity, 0.70 target
  and draft KV fractions, prompt-prefix caching disabled, complete prefills
  and final draining included. AR allocates GPU0; SSD uses GPU0 and GPU1.

Unprofiled comparisons have three rounds with seeded method order and matching
prompt order across methods in each round. AR runs first to establish a fresh
reference. Seven configurations × 12 inputs × three rounds = **252 scored
requests in 21 fresh-worker blocks**. Each worker first runs 12 separate warmup
inputs, covering every length and family; these 252 warmup requests are unscored.

Separate Nsight Systems captures run complete K7, complete K10 and overlap K10
at each length, with three family inputs per capture: **36 scored diagnostic
requests in 12 blocks**, plus 36 length-matched warmups. Profiling uses existing
NVTX ranges and CUDA graph-node tracing, without adding per-step synchronization.
Profiled request times never enter the unprofiled performance comparison.

## Queue order and dedicated access

The new watcher waits without taking the global campaign lock or launching GPU
workers. Only a predecessor `complete` status plus validated tuning/testing
receipts, exactness, frozen selection and final report permit progression. It
then acquires the same global campaign lock. If the predecessor stops or is
still waiting, this sweep remains waiting.

Both UUID-pinned GPUs must be dedicated, even for AR. A one-second monitor
identifies worker-owned processes using process groups and inherited run tokens.
Contamination invalidates the entire block, stops only the owned worker, and
waits before resuming. Unrelated jobs are never terminated. Valid completed
blocks survive interruption. Exactness, provenance, execution or trace-analysis
failures stop the sweep rather than retrying indefinitely.

Execution code, software/environment, profiler version, model files, workload,
GPU identities and predecessor identity are pinned. Changed execution requires
a new versioned output directory. No old timings or outputs are substituted for
fresh measurements on these synthetic inputs.

## Analysis

`report.md`, `summary.json`, `summary.csv`, `length_latency.png`, and
`draft_verify_scaling.png` are produced only after every required valid block.
They report request/decoding latency, first delivery, target waiting, acceptance,
tokens per verification, cache hits, memory, draining, partial publications and
actual response-before-tail counts.

The separate trace analysis reconstructs draft-step GPU completion through
process-scoped CUDA correlations. It reports full-tree GPU spans, verification
GPU spans and wall durations, their descriptive ratios, and cache-hit ready
depth at target requests. CPU enqueue duration is not treated as GPU duration.
Incomplete trees are kept separate; missing correlations are not extrapolated.
GPU spans include intervening gaps, so they are not a kernel-busy percentage.
Per-handoff raw evidence is retained in each `phase_analysis.json`.

Paired intervals use 10,000 seeded bootstrap samples over rounds and templates
within families. There is only one scored template per family; these intervals
describe repeat variability on these inputs and do not establish generalization.
An improvement over complete K10 must also be judged against a stronger K7
baseline. More overlap alone does not establish faster request completion.

## Operation

Source the environment exactly once per shell:

```bash
source profiling/env.sh
python -O profiling/run_context_length_sweep.py --watch
```

The detached session is `ao-context-length-sweep`. Its `status.json` distinguishes
waiting for the predecessor, waiting for GPUs, running, stopped, and complete.
Use `--prepare-only` or `--dry-run` to freeze/inspect without GPU work. To resume,
run the same command with the same environment and unchanged code. The CLI also
supports `--phase measure`, `profiles`, and `report`; later phases still require
their prerequisite references and valid outputs.

Validation covers deterministic exact input sizes, complete task retention,
pairing by length/template/round, dependency gating, contamination/interruption,
profile separation, exact outputs, asynchronous GPU completion, partial trees,
and process-scoped correlation IDs. The trace parser was also checked on a saved
complete-SSD capture, reconstructing 210 complete trees and matched rounds with
no missing tree correlations. That smoke check is parser validation, not a new
context-length measurement.
