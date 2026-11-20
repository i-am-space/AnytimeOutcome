# Controlled context-length diagnostic

Runs only after the publication performance campaign completes with valid receipts, exact outputs, frozen selection and a final report. Shares the global campaign lock; one worker at a time.

- Inputs: exactly 256, 512, 1024, 2048 tokens. Three synthetic chat/code/document tasks, extended using repeated irrelevant notes; complete tasks and chat formatting remain intact.
- Outputs: fixed 256, EOS ignored. BF16, batch one, CUDA graphs, JIT misses, shared AR arithmetic, 4096 capacity, 0.70 target/draft KV fractions, prefix caching disabled, final drain included.
- Unprofiled: fresh AR plus complete/uniform/overlap at K7 and K10; F3, P4, queue bound two. Three seeded rounds, same prompt order within a round, fresh worker per configuration/round: 21 blocks / 252 scored requests. Twelve separate warmup requests per worker, covering every length/family.
- Profiles: complete K7, complete K10, overlap K10 at each length; three family requests per block: 12 blocks / 36 scored requests, three length-matched separate warmups per block. CUDA graph nodes and NVTX captured with Nsight Systems, without intrusive per-step synchronization.
- Both GPUs must remain dedicated, including AR. UUID-pinned allocation, one-second foreign-process monitoring, invalidation and wait/resume; never terminate unrelated jobs.
- Every speculative result must match fresh same-round AR; profiles match round-zero AR. Stop on exactness, execution, provenance, or trace-correlation errors.
- Report full-request/decoding time, wait, acceptance, tokens per verification, partial/actual handoffs, draining, memory; separately report tree/verify GPU spans and CPU wall durations, and ready-depth histograms. Raw traces, outputs, receipts and plots are retained.

This is a synthetic length-sensitivity diagnostic, not a newly unseen benchmark or an automatic speedup claim. Paired bootstrap intervals describe only these three task templates and repeated rounds. No extra output-length sweep or retuning is part of this queue.
