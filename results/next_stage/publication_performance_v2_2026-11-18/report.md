# Fresh SSD comparisons with saved shared-AR references

AR is not rerun as a full performance arm. Saved AR outputs come from validated dedicated baseline receipts. Short fresh AR checks must pass before this campaign proceeds. Target numerical files, AR graph functions, model/tokenizer, software and GPU identities are checked against the saved reference.
Reused frozen test prompts; not a newly unseen workload. Selection uses tuning only; natural EOS is secondary.
Exact speculative outputs: 2250/2250 against shared-fixed16-v1. Original historical AR is a separate reference.

## Historical AR timing context — measured earlier, not rerun

| Protocol | GPUs | Earlier mean s | Earlier p50 s | Earlier p95 s |
|---|---:|---:|---:|---:|
| fixed | 1 | 6.2711 | 6.2447 | 6.3593 |
| natural | 1 | 5.4191 | 6.2362 | 6.3180 |

Historical-AR ratios in summary.json are cross-campaign context. Their bootstrap pairs prompts but resamples historical/current rounds independently. It does not control temporal GPU-state differences. Primary SSD-versus-SSD comparisons use fresh paired rounds on dedicated GPUs.

## fixed: fresh SSD measurements

| Method | GPUs | Mean s | p50 s | p95 s | tok/s | First delivery ms | Wait s | Tokens/verify |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| complete_k7 | 2 | 2.2039 | 1.9993 | 3.6817 | 116.16 | 95.06 | 0.4269 | 3.854 |
| uniform_k7_p7 | 2 | 2.2245 | 2.0169 | 3.7143 | 115.08 | 94.88 | 0.4486 | 3.854 |
| overlap_k7_p4_w2 | 2 | 2.2229 | 2.0146 | 3.7147 | 115.17 | 94.58 | 0.4454 | 3.854 |

| Method | Family | Fresh reference | Speed ratio | Paired 95% CI |
|---|---|---|---:|---|
| complete | all | uniform | 1.009 | 1.007–1.012 |
| uniform | all | complete | 0.991 | 0.988–0.993 |
| overlap | all | complete | 0.991 | 0.989–0.994 |
| overlap | all | uniform | 1.001 | 0.999–1.003 |
| complete | chat | uniform | 1.009 | 1.006–1.011 |
| uniform | chat | complete | 0.991 | 0.989–0.994 |
| overlap | chat | complete | 0.992 | 0.989–0.995 |
| overlap | chat | uniform | 1.001 | 1.000–1.001 |
| complete | code | uniform | 1.011 | 1.007–1.016 |
| uniform | code | complete | 0.989 | 0.984–0.993 |
| overlap | code | complete | 0.991 | 0.986–0.998 |
| overlap | code | uniform | 1.002 | 0.998–1.009 |
| complete | doc | uniform | 1.009 | 1.005–1.012 |
| uniform | doc | complete | 0.991 | 0.988–0.995 |
| overlap | doc | complete | 0.991 | 0.989–0.994 |
| overlap | doc | uniform | 1.000 | 0.998–1.003 |

## natural: fresh SSD measurements

| Method | GPUs | Mean s | p50 s | p95 s | tok/s | First delivery ms | Wait s | Tokens/verify |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| complete_k7 | 2 | 1.9031 | 1.6122 | 3.2613 | 116.22 | 94.53 | 0.3581 | 3.830 |
| uniform_k7_p7 | 2 | 1.9178 | 1.6254 | 3.2886 | 115.33 | 93.37 | 0.3739 | 3.830 |
| overlap_k7_p4_w2 | 2 | 1.9187 | 1.6258 | 3.2855 | 115.27 | 94.15 | 0.3745 | 3.830 |

| Method | Family | Fresh reference | Speed ratio | Paired 95% CI |
|---|---|---|---:|---|
| complete | all | uniform | 1.008 | 1.006–1.009 |
| uniform | all | complete | 0.992 | 0.991–0.994 |
| overlap | all | complete | 0.992 | 0.990–0.994 |
| overlap | all | uniform | 1.000 | 0.997–1.001 |
| complete | chat | uniform | 1.008 | 1.006–1.010 |
| uniform | chat | complete | 0.992 | 0.990–0.994 |
| overlap | chat | complete | 0.992 | 0.990–0.993 |
| overlap | chat | uniform | 1.000 | 0.996–1.001 |
| complete | code | uniform | 1.009 | 1.007–1.011 |
| uniform | code | complete | 0.991 | 0.990–0.994 |
| overlap | code | complete | 0.990 | 0.985–0.993 |
| overlap | code | uniform | 0.999 | 0.994–1.002 |
| complete | doc | uniform | 1.007 | 1.003–1.009 |
| uniform | doc | complete | 0.993 | 0.991–0.997 |
| overlap | doc | complete | 0.993 | 0.989–0.997 |
| overlap | doc | uniform | 1.000 | 0.996–1.002 |

## Interpretation

Lower latency must be established against the strongest fresh matched SSD comparator. Overlap counts and fewer query rows alone do not establish a speedup. All request timing includes prefills and final draining, excludes loading/warmup/serialization, and uses no profiler. Target verification retains K+1 rows. Detailed acceptance, cache hits, memory, output lengths, first divergences, repeatability and per-family regressions are in summary.json.
