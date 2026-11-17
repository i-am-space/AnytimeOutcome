# Overnight additions

Wait for the existing baseline/publication pipeline to complete successfully.

1. Tune P on 24 tuning prompts, two fixed-256 repetitions: unique P=1,2,4,K,
   plus complete-cache P=0. Freeze the best early P; exact ties choose smaller P.
2. Capture fresh shared AR / complete SSD / selected early SSD Nsight Systems
   traces on three seen prompts. Profiled timings are diagnostic only.
3. Check K=4,7,10 with AR/SD/complete SSD/early SSD, both EOS protocols, on three
   seen coverage prompts and nine synthetic inputs of 2040/2047/2048 tokens.
4. Evaluate selected P on all 75 held-out prompts: fresh paired complete/early
   SSD, both protocols, five rounds (1500 requests). Baseline AR supplies exact
   token references only. Use 10,000 paired hierarchical bootstrap draws.

All workers are serial, UUID pinned, prefix caching disabled, graphs enabled,
greedy BF16, capacity4096, with three separate warmups. Both GPUs must remain
dedicated. Contaminated attempts are invalidated; unrelated jobs are untouched.
Code/environment changes and correctness failures stop dependent measurements.
