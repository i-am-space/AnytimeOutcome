# Publication tuning

Dedicated GPUs; 24 balanced tuning prompts × two repetitions per setting.
Fixed 256 output tokens, EOS ignored. Selection uses tuning only. F=3 throughout.

| Configuration | Mean request latency (s) | Winner |
|---|---:|---|
| complete_k4 | 2.288818 |  |
| complete_k7 | 2.116604 | yes |
| complete_k10 | 2.969658 |  |
| uniform_k7_p2 | 2.178487 |  |
| uniform_k7_p4 | 2.157913 |  |
| uniform_k7_p7 | 2.148126 | yes |
| uniform_k10_p2 | 2.357538 |  |
| uniform_k10_p4 | 2.353612 |  |
| uniform_k10_p7 | 2.483834 |  |
| overlap_k7_p2_w1 | 2.173833 |  |
| overlap_k7_p2_w2 | 2.152640 |  |
| overlap_k7_p4_w1 | 2.180338 |  |
| overlap_k7_p4_w2 | 2.137226 | yes |
| overlap_k7_p7_w1 | 2.190568 |  |
| overlap_k7_p7_w2 | 2.138318 |  |
| overlap_k10_p2_w1 | 2.367905 |  |
| overlap_k10_p2_w2 | 2.462715 |  |
| overlap_k10_p4_w1 | 2.364070 |  |
| overlap_k10_p4_w2 | 2.459858 |  |
| overlap_k10_p7_w1 | 2.538804 |  |
| overlap_k10_p7_w2 | 2.498215 |  |
