# Publication comparison, reused AR v2



User-authorized amendment: remove repeated full AR timing arms. Historical AR outputs and receipts are hash-pinned. Target arithmetic and AR graph functions, weights/tokenizer, software and GPU identities must match.

First run one short AR continuity block on three tuning prompts, one per family, with three warmups and fixed256 output. It must match saved AR exactly before tuning starts. Its timing is diagnostic and not an AR performance arm.

All 21 SSD configurations remain: complete K4/7/10; uniform K7/10 × P2/4/7; overlap same K/P × queue bounds1/2. F3. Two repetitions on 24 balanced tuning prompts = 42 blocks / 1008 requests.

Freeze complete/uniform/overlap winners using tuning alone. Held-out results are not used for selection. Evaluate the three SSD winners on75 prompts × five rounds × two protocols =30 blocks /2250 requests. Add completeK7 if needed:10 blocks /750 requests. Natural EOS reuses fixed-length-selected winners.

Primary latency comparisons are fresh SSD versus SSD, paired by round/prompt. Saved AR timing appears only as historical context; historical ratios use independent old/new round resampling with paired prompts, not contemporaneous-run claims.

AR continuity uses GPU0; SSD target GPU0 / draft GPU1. Both GPUs must be dedicated. Same global lock, UUIDs, contamination handling and block resume. BF16, batch1, graphs, JIT misses, capacity4096, full prefills,0.70 KV fractions, no prefix lookup, no profiler, final drain included.

Total new work:73–83 blocks /3261–4011 scored requests including three continuity checks. Previously completed AR blocks are references only and cannot count as newly executed blocks.
