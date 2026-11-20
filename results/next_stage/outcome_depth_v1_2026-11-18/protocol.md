# Unequal outcome-depth queue



Runs after both existing publication and context-length queues complete successfully. Separate worktree keeps their execution fingerprints unchanged. Same global campaign lock; one worker at a time.

Gate: 14 configurations × six real/boundary inputs × two protocols = 168 scored requests, 28 blocks; 0.10 KV. Must match fresh shared AR, exercise short cache hits and reduce actual draft query rows before performance starts.

Tuning: 24 balanced prompts, two fixed-256 repetitions, three separate warmups per fresh worker. 14 configurations = 28 blocks / 672 requests. Seed 0, reversed configuration order for second repetition. F=3; complete K4/7/10; K7/10 for overlap, unequal, unequal+overlap, equal-work uniform and reversed-rank controls.

Unequal rank depths: K7 [7,4,4], K10 [10,7,4]. Equal-work depths: [5,5,5] or [7,7,7]. Reversed controls reverse only recovery-rank priority. Overlap uses P4 and queue bound two.

Select complete, overlap, unequal and unequal+overlap winners on tuning only, ties smaller K. Equal-work and reverse controls use the unequal winner’s K by predeclared rule.

Held-out: fresh AR + four selected methods + equal-work/reverse controls; 75 prompts, five rounds, fixed256 and natural EOS cap256 = 70 blocks / 5250 requests. Same frozen prompt order within a round; five method-order permutations without replacement; alternate protocol order.

Performance KV fractions 0.70; BF16, batch one, graphs, full prefills, capacity4096, no prefix caching, no profiler. AR GPU0; SSD target GPU0 / draft GPU1; UUID pinned. Both GPUs must be dedicated. Foreign jobs invalidate only their current block; completed valid blocks survive. Never terminate foreign jobs.

Report full-request latency, first delivery, throughput, waits, acceptance, cache hits, memory, final draining, query-row counts, per-family behavior, paired 10,000-sample intervals, exact outputs and repeatability.

This is a static recovery-rank allocation prototype. It does not learn outcome probabilities, change fan-out, shrink target verification graphs, or reduce reserved KV capacity. Performance is unestablished until completed.

Totals excluding warmups: 168 gate + 672 tuning + 5250 held-out = 6090 scored requests.
