# AR-compatible verification: implementation and evidence

The follow-up now includes a separately versioned fast prototype with a new,
explicitly approved shared AR reference. See [shared fast backend](shared_fast_backend.md).
The historical-reference path described here remains a diagnostic control.

This is the first implementation step toward the proposal's exact greedy-output
requirement. It is an opt-in numerical path, not a completed fast correctness fix.
The default batched path and all historical campaign artifacts remain unchanged.

## What changed

`target_numerics='ar_compatible'` installs target-only operation wrappers before
CUDA graph capture. Verification still processes a block of proposed tokens, but
linear projections, normalization, rotary embedding, activation and the vocabulary
projection use the same per-token shapes as AR. Query/key head groups are preserved.
Each attention query uses its actual visible prefix length, including the same
single-query split-KV reduction used by AR. Context metadata is restored after
each attention operation, including on errors.

Target prefill and AR single-token operations call the original implementations.
The drafter keeps its original operations. Because rotary modules may be shared by
the cached factory, target wrappers receive separate module objects sharing only
read-only cosine/sine buffers. No global torch numerical function is patched.

The supported initial scope is greedy Qwen3, batch size one, one target GPU, and
either synchronous or two-GPU asynchronous drafting. Unsupported model, batching,
tensor-parallel and non-greedy configurations are rejected explicitly.

The path sacrifices batched matrix multiplication. Retaining CUDA graphs reduces
some overhead compared with the older eager diagnostic reference, but it does not
make this an acceleration result. The measured diagnostic requests are slower
than the original fast implementation.

## Correctness runs

The gate uses nine prompts: the three historical failures and two existing tuning
prompts per family. It checks fixed 256-token generation and normal EOS with a
256-token cap. Frozen definitions include exact inputs, warmups, model hashes,
execution hashes, software, GPU identities and explicit 0.10 KV fractions.

Synchronous SD matches fresh, unchanged-code AR in **18/18 comparisons** across
the two protocols. The natural-EOS test includes two actual early stops, at 179
and 226 output tokens. CUDA graphs are enabled. This is small-workload correctness
evidence, not five-repetition performance evidence or a universal proof.

SSD also matches fresh AR in **18/18 comparisons**. Combined SD/SSD agreement is
**36/36 comparisons** against the 18 corresponding AR generations, across nine
prompts and two stopping protocols. SSD validation is recorded separately in
`results/correctness/ar_compatible_2026-11-16_shared_v4/summary.json`.
Only completed worker receipts count. Reused AR/SD records retain their original
paths and hashes; no output is rewritten to look like a new measurement.

## Historical AR reproduction remains unresolved

The fresh AR output reproduces only one of the three saved historical profiling
outputs. Chat first differs at zero-based index 12 and document at index 170; code
matches. The fresh AR fixed/natural workers agree on their common prefixes for
all nine prompts. Matching the original chat warmup workload did not restore the
two older outputs, so warmup selection alone does not explain the discrepancy.

Model files and package/environment fingerprints match the prior campaign.
Restoring the original 0.70 KV-memory fraction also leaves the same two
differences, ruling out the small correctness-run allocation as an explanation
for these cases.

An independent control extracts commit `68df65e` into a temporary directory. Its
execution fingerprint exactly matches the frozen original:
`3f5ca1112ffdd8258141e1092ef3d156ff4a82d79b727b626a5b09542b29fad6`.
That pre-change engine still shows AR/SD discrepancies on all three historical
prompts, at indices 12, 181 and 194 in this new run. Its AR outputs exactly match
the fresh reference used by the new path, including the same two historical
reproduction failures. Today's AR-compatible implementation therefore did not
introduce either observed discrepancy. This control uses our prior adapted L40S
engine; it is not a test of the untouched upstream Hopper backend.

The archived source, worker commands, process records and receipts are recorded
in `results/correctness/archived_68df65e_2026-11-16/`. Reproduce that control with
`python profiling/check_archived_engine.py --outdir <new-directory>` after sourcing
the environment. The combined evidence is summarized in
`results/correctness/ar_compatible_progress_2026-11-16.json`.

Further controls are needed for initialization and compiled numerical paths.
The current gate intentionally remains failed if historical AR outputs
do not reproduce, even when every speculative candidate matches fresh AR.
Neither tolerance-aware diagnostics nor changing the reference silently counts
as a pass.

## Shared GPU policy

Correctness does not require exclusive GPUs. `--allow-shared-gpus` admits a worker
when enough memory is free, records other jobs, and never terminates them.
Its timings cannot be used as dedicated-access performance measurements. The
original performance campaign's dedicated-window and contamination rules remain
unchanged.

An initial dedicated attempt stopped when another GPU1 job appeared. A later SSD
attempt failed during drafter startup because the runner's Python 3.13 / torch 2.10
optimized-mode workaround was not applied in spawned children. The workaround is
now installed during module import, and an optimized spawn-import regression test
checks it. Failed attempts remain saved and are excluded.

## Validation and next decision

The CPU suite reports **75 passed, 15 CUDA-dependent tests skipped**; the existing
CUDA-only paged-attention suite is excluded from that command. Tests cover operation
shapes, causal metadata, residuals, drafter isolation, greedy-only admission,
reference pairing, length mismatches, contamination, resource admission,
incompatible reuse, output tampering, and the child-process workaround.

Before claiming the proposal's correctness-preserving acceleration, reproduce
the historical target under declared settings, validate a wider prompt set, and
recover batching efficiency without changing the chosen reference semantics.
Changed inference code requires a new performance campaign version and affected
retuning. Uniform early publication is the next implementation milestone after
these correctness and baseline decisions; it is not implemented by this change.

## Fast implementation investigation, 17 November 2026

The slow implementation remains a diagnostic reference. The project's performance
work should use batched implementations; serial verification is not the proposed
acceleration. No production numerical defaults were changed by this investigation.

`profiling/diagnose_fast_numerics.py` replays the three current archived AR/SD
divergences using only the target model. It teacher-forces the same AR token prefix,
restores identical suffix KV slots before each forward, and compares one query with
seven/eight queries. The first query has the same visible history. There is no
drafter, acceptance/rollback machinery, or cross-GPU communication in this probe.
Each module is then tested on identical inputs to separate local arithmetic from
differences propagated by earlier layers. Tensor strides and Q/K head groups are
preserved. Instrumented and uninstrumented forwards agree exactly in every case.

Results are in `results/correctness/fast_numerics_2026-11-17.json` and
`results/correctness/fast_numerics_candidates_2026-11-17.json`. These are eager
diagnostics with the existing compiled pointwise operations, not full-generation
correctness gates or performance benchmarks. Model contents are rehashed; source,
software and input/result provenance are recorded.

| Prompt | Current first differing output index (zero based) | First differing operation | Unequal first-row elements |
| --- | ---: | --- | ---: |
| Chat: specbench-148 | 12 | Layer 0 Q/K/V projection | 15 / 6,144 |
| Code: HumanEval/36 | 181 | Layer 0 Q/K/V projection | 8 / 6,144 |
| Document: specbench-553 | 194 | Layer 0 Q/K/V projection | 18 / 6,144 |

For both query widths, this target-only experiment reproduces the current AR and
fast SD token choices for all three failures. Teacher-forced single-token
predictions match the saved current AR history up to each tested position. Thus
shape-dependent target arithmetic alone is sufficient to reproduce these failures;
an SSD cache or communication bug is not necessary to produce them. This does not
prove the absence of other bugs or resolve the separate historical AR drift.

The first projection differs by at most 0.00048828125 in chat/document and
0.00006103515625 in code. Selected FP64 dot products show values close enough to
BF16 rounding boundaries that different accumulation paths produce adjacent BF16
values. Differences are also found independently in all 144 transformer linear
projections and the vocabulary projection for each tested width/prefix. Under the
same-input isolation, attention, normalization, RoPE and activation match at these
particular positions. That is not coverage of every position, compilation state,
or attention split boundary.

Final score changes explain the token selection:

- Chat: AR ties token 7460 and 17601 at 32.5; the batch lowers token 7460 to 32.25.
- Code: AR scores token 476 at 37.0 and token 334 at 36.75; the batch ties both at
  37.0, so argmax picks the smaller token index, 334.
- Document: AR scores token 3070 at 28.875 and token 1496 at 28.75; the batch ties
  both at 28.875, so argmax picks token 1496.

These are BF16 score ties/near-ties, not evidence that a larger token index should
be preferred or that changing tie-breaking would repair the upstream arithmetic.
AR is the chosen reference, not a claim of universally more accurate arithmetic.

### Candidate controls and next experiments

Three controls were evaluated on the first layer's four projections and the
vocabulary head, for three prefixes and two widths (30 operation cases):

1. Disabling `allow_bf16_reduced_precision_reduction` changed neither the single-row
   nor batched outputs in these cases. It did not remove the discrepancies.
2. BF16 matrix multiplication with FP32 output, followed by a BF16 cast, reproduced
   the existing batched result and retained its disagreements with AR.
3. One `torch.bmm` call containing independent matrix-vector products, with shared
   weights, exactly matched AR for the tested Q/K/V and both MLP projections
   (18/30 cases). The output projection still differed in 1–3 elements and the
   vocabulary head in 12–19 elements (the remaining 12/30 cases). This is a
   promising operation-level control, not a validated or measured fast fix.

Next, test the matrix-vector alternative across all layers and rows, and measure
representative kernel costs against both ordinary GEMM and serial reference calls.
Inspect the remaining dispatch/reduction differences in square output projections
and the large vocabulary projection before integrating any candidate. Retain
batched work wherever identical arithmetic can be established; replacing a Python
loop with a batched API does not itself prove a speedup.

Also test attention at causal/page/split boundaries, particularly when a verify
block crosses the portable kernel's 2,048-token split-chunk threshold on this GPU,
and test pointwise compilation state. Then validate complete SD and SSD generation
with CUDA graphs, multiple K values, EOS and rejection/rollback cases. Only after
exact-output validation should a new dedicated performance campaign measure cost.

If efficient exact matching to the current AR kernels proves infeasible, a distinct
option is a common batch-independent numerical backend for all three methods. That
changes the reference contract and requires an explicit project decision, new
references and revalidation; the existing historical gate must not be relabeled a
pass. Keep original AR as an additional performance control so changing its backend
does not manufacture an apparent speedup. A heuristic small-logit-margin fallback
alone is not sufficient to prove the existing token-for-token requirement.

Five focused CPU tests pass for tensor layout preservation, strict shape pairing,
head grouping, isolation of shared RoPE hook sites and causal-context restoration.

## All-layer candidate follow-up

The follow-up checks every query row at all 145 projections, for the three current
failure prefixes and widths seven/eight: 870 operation cases and 6,525 operation
rows. BMM and its transposed formulation agree with original AR in all rows of
648/870 cases (108 projections per forward). The remaining 222 cases are all 36
attention-output projections and the vocabulary head in each forward. Therefore
the first-layer result was representative, but plain BMM is not an exact fix.

CUDA profiler kernel names explain the remaining dispatch distinction. Original
AR's QKV and MLP projections and BMM select the same cuBLAS `internal::gemvx`
variant 7. AR's attention-output and vocabulary projections select variant 6,
while BMM selects variant 7. The ordinary fast verifier uses a CUTLASS BF16
tensor-core GEMM. This identifies the different implementations selected; it does
not claim to reconstruct every instruction inside proprietary cuBLAS kernels.

The following isolated CUDA-graph sweeps visit all 145 real weight tensors, whose
combined size exceeds L2. Each entry averages five samples of five graph replays,
after three warm replays. Inputs are captured activations, not a connected model
forward. These are contextual kernel costs, not request speedups: GPU0 was used
for the probe while unrelated work occupied GPU1, and configuration order was not
randomized.

| Queries | Ordinary fast GEMM | Batched GEMV | Individual row calls |
| ---: | ---: | ---: | ---: |
| 7 | 22.09 ms | 122.37 ms | 129.70 ms |
| 8 | 22.11 ms | 139.77 ms | 152.51 ms |

BMM costs approximately 5.5–6.3 times the fast GEMM sweep. Removing Python row
loops does not recover the weight reuse and throughput of batched tensor-core
GEMM. This candidate fails the performance screen even before adding attention,
normalization and drafting costs.

Artifacts: `results/correctness/fast_numerics_all_layers_2026-11-17.json` and the
matching `.source.py` snapshot. The full-generation hybrid is confined to
`profiling/numerics_candidates.py`; it is not exposed as an engine default.
