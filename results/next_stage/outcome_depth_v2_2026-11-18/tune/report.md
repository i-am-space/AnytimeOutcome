# Publication tuning

Dedicated GPUs; 24 balanced tuning prompts × two repetitions per setting.
Fixed 256 output tokens, EOS ignored. Selection uses tuning only. F=3 throughout.

| Configuration | Mean request latency (s) | Winner |
|---|---:|---|
| complete_k4 | 2.285560 |  |
| complete_k7 | 2.121355 | yes |
| complete_k10 | 2.972764 |  |
| overlap_k7 | 2.151039 | yes |
| overlap_k10 | 2.458474 |  |
| ranked_k7 | 2.204087 | yes |
| ranked_k10 | 3.019356 |  |
| ranked_overlap_k7 | 2.215944 | yes |
| ranked_overlap_k10 | 2.472936 |  |
| reverse_k7 | 2.467168 |  |
| reverse_k10 | 3.523359 |  |
| budget_k7 | 2.254661 |  |
| budget_k10 | 2.498694 |  |
