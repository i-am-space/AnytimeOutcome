# Unequal continuation depths, version 1

This opt-in prototype changes how much continuation is prepared for each
recovery-token alternative. It keeps F=3 recovery guesses at every possible
acceptance position. It does not implement Saguaro's geometric fan-out schedule.

## Allocation

`outcome_depths` gives one continuation budget per recovery-token rank. The
existing top-k recovery guesses are ordered by their drafter logits after
excluding the previously proposed token. Rank zero is the highest-ranked
remaining guess at that acceptance position; it is not a calibrated probability
of the complete outcome. The same budget rule applies at every acceptance
position. There is no learned allocation across acceptance positions in v1.

| Maximum K | Ranked depths | Reversed priorities | Uniform equal-work depths |
|---|---|---|---|
| 7 | 7, 4, 4 | 4, 4, 7 | 5, 5, 5 |
| 10 | 10, 7, 4 | 4, 7, 10 | 7, 7, 7 |

With K10/F3 there are 33 possible recovery outcomes. All 33 run the first four
steps, 22 run the next three, and 11 run the final three. A complete tree uses
231 query rows rather than 330: 30% fewer model rows. K7 uses 120 rather than
168 rows, a 28.6% reduction. Those are work counts, not latency improvements.
More launches with smaller batches can be slower on a latency-bound drafter.

## Fast implementation and ownership

- Capture CUDA graphs at the actual live query counts, including the reduced
  counts at deeper steps. Retired rows are removed from model forwards, rather
  than executing them and masking their results afterward.
- Keep canonical KV addresses. The compact attention mask maps each live query
  to its own original branch ancestors and valid glue prefix. Retired KV holes
  remain unwritten and are never visible through an attention mask.
- Store a valid length for each outcome. Cache-hit responses use that outcome's
  length, clipped to the globally completed publication prefix. Parent positions
  beyond the previous real continuation are excluded. Expected committed context,
  request epoch and cache generation checks remain active.
- Stable-prefix overlap snapshots copy per-outcome lengths and independent
  token/logit buffers. The tail still drains before graph buffers or KV state are
  reused. A miss uses the full existing JIT fallback after draining old work.
- Target verification retains its K+1 graph and shared-fixed16-v1 arithmetic.
  Full-request timing includes prefills and eventual draining. Reserved KV
  capacity and rectangular logits/snapshot storage are not reduced by this v1.

The default `outcome_depths=None` retains complete equal-depth behavior. The new
path requires plain Qwen-style asynchronous SSD, batch one, CUDA graphs, JIT
misses, and uniform recovery fan-out. With early publication enabled every
assigned outcome depth must reach P. This prototype supports original uniform
handoff or the stable-prefix overlap policy; bounded/adaptive combinations are
deliberately rejected until separately validated.

## Overnight study

Driver: `profiling/run_outcome_depth_campaign.py` in this isolated worktree.
Results: the main checkout's
`results/next_stage/outcome_depth_v1_2026-11-18/`.

The queue waits outside the global campaign lock until the existing publication
performance campaign and context-length sweep both complete successfully. It
then validates their receipts, outputs and final reports before acquiring GPU
work. The engine changes here cannot invalidate their source fingerprints.

1. Correctness: all 14 configurations, six inputs (three real tasks and three
   2048-token boundaries), fixed and natural EOS protocols. 28 blocks / 168
   requests at 0.10 KV. Stop on any token mismatch. Require actual reduced-row
   execution and short cached outcomes before proceeding to performance.
2. Tuning: AR; complete SSD K4/7/10; overlap, ranked, ranked+overlap, reversed,
   and equal-work uniform depths at K7/10. F3, P4, queue bound two. Twenty-four
   balanced tuning prompts, two repetitions, fixed 256 output tokens: 28 blocks
   / 672 requests. Shuffle with seed zero and reverse the second repetition.
3. Freeze complete, overlap, ranked and ranked+overlap winners on tuning only.
   Tie-break by smaller K. Choose reversed/equal-work ablation K to match the
   selected ranked configuration by a declared rule.
4. Evaluate fresh AR and the six SSD arms on all 75 frozen held-out prompts,
   five repetitions, fixed256 and natural EOS capped at256: 70 blocks / 5250
   requests. Same prompt order per round, five method-order permutations without
   replacement, alternating protocol order. The test prompts have been used in
   prior studies, so they are not newly unseen.

Performance uses 0.70 KV fractions for both models, BF16, request batch one,
graphs, full prefills, no prefix-cache lookup, no profiler, and three separate
warmup requests per fresh worker. AR uses physical GPU0. SSD uses GPU0 for the
target and GPU1 for the drafter. UUID pins, dedicated access, global locking,
owned-process monitoring, and invalidation of contaminated blocks are retained.
Completed valid blocks survive interruptions. Foreign jobs are never killed.

The report includes latency, first delivery, throughput, wait, acceptance, cache
hits, output lengths, memory, final drain, actual compacted row counts, exact
outputs, repeatability, per-family regressions, and 10,000 seeded paired bootstrap
samples over rounds and prompts within families. It compares unequal depth with
both tuned complete SSD and overlap SSD, as well as reversed and equal-work
controls. Ranked+overlap may select a different K, so the latter two controls are
matched-K ablations for ranked only; that limitation is disclosed.

Total: 6090 scored requests plus 378 warmup requests. Model loading, compilation,
warmup and serialization are outside scored latency. Full language-model
correctness and performance remain pending until this queue completes. Small
kernel tests alone do not establish token-for-token generation correctness or
an acceleration claim.
