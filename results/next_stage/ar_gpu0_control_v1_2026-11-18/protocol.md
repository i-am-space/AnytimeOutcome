# GPU0-only contextual AR control

Two repetitions of the exact 24 tuning prompts and seeded orders from the publication campaign; 48 scored requests and six separate warmups. No held-out inputs are accessed. BF16, greedy batch one, shared-fixed16-v1, CUDA graphs, 4096 capacity, fixed 256 output tokens ignoring EOS, 0.70 KV fraction, no prefix cache or profiler. Full-request timing includes prefill.

Only physical GPU0 must be free; GPU1 activity is recorded. GPU0 contamination invalidates a block. One worker at a time under the same global campaign lock. These results are contextual and cannot count as either of the frozen both-GPUs-dedicated AR blocks. The main campaign resumes afterward and retains its original protocol.
