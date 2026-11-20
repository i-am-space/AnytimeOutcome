# Unequal-depth successor with saved AR v2



User-authorized amendment removes full AR timing repetitions. Saved shared-fixed16-v1 AR outputs are validated by receipts, input IDs, model/software/GPU identity and target numerical code. The small fresh AR correctness gate remains; its timings are diagnostic.

After the publication v2 and context v2 queues complete, run the full six-input gate: fresh AR plus13 SSD settings, both protocols =28 blocks /168 requests at0.10 KV. All speculative outputs must match fresh gate AR, with actual reduced rows and short cached outcomes.

Tuning retains all13 SSD settings: complete K4/7/10, overlap, ranked, ranked+overlap, reverse and uniform equal-work at K7/10; F3, P4, queue2. Two repetitions ×24 balanced prompts =26 blocks /624 requests. All tuning outputs must also match saved baseline AR. Selection reads tuning only.

Freeze complete/overlap/ranked/ranked+overlap winners; budget/reverse K follows the ranked winner. Six SSD arms ×75 held-out prompts ×five repetitions ×two protocols =60 blocks /4500 requests. Saved AR supplies exact output references; no new full AR timing arm is executed.

Primary SSD comparisons are fresh and paired by round/prompt. Any ratio to AR uses explicitly historical timing with independent historical/current round resampling. No contemporaneous AR claim.

One worker at a time, shared main lock, dedicated UUID-pinned GPUs, contamination invalidates current block only. BF16,batch1,graphs,JIT,full prefills,capacity4096,prefix cache off,no profiler,final drain included. Performance KV0.70. AR gate GPU0; SSD targetGPU0/draftGPU1.

Totals:114 blocks /5292 scored requests, plus342 warmups. This replaces v1 without deleting its artifacts. Static rank budgets are K7:[7,4,4], K10:[10,7,4]. Unequal depths do not reduce target verification graph size.
