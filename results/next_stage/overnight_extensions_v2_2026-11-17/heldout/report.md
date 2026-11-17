# Overnight heldout results

Valid blocks: 20; requests: 1500.
Exact outputs: 1500/1500 against shared AR.

| Protocol | Configuration | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Partial handoffs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | complete | 2 | 2.2006 | 1.9971 | 3.6824 | 116.33 | 0.4250 | 0 |
| fixed | early | 2 | 2.2230 | 2.0170 | 3.7118 | 115.16 | 0.4460 | 0 |
| natural | complete | 2 | 1.9045 | 1.6130 | 3.2616 | 116.13 | 0.3617 | 0 |
| natural | early | 2 | 1.9203 | 1.6266 | 3.2888 | 115.18 | 0.3755 | 0 |

| Protocol / family | Complete / early latency | Paired 95% CI |
|---|---:|---|
| fixed/all | 0.990× | 0.987–0.993 |
| fixed/chat | 0.991× | 0.989–0.993 |
| fixed/code | 0.986× | 0.978–0.992 |
| fixed/doc | 0.991× | 0.986–0.995 |
| natural/all | 0.992× | 0.988–0.995 |
| natural/chat | 0.992× | 0.990–0.994 |
| natural/code | 0.993× | 0.985–1.001 |
| natural/doc | 0.991× | 0.984–0.995 |

Exactness is against shared-fixed16-v1 AR, not original historical AR.
Profiled and synthetic boundary timings are diagnostic only.
Target graph retains padded K+1 work; all final draft draining is included.
Publication wall time includes outstanding GPU completion in complete-cache SSD.
