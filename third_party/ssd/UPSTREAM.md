# Vendored SSD engine

Upstream: https://github.com/tanishqkumar/ssd @ `d7eb8fa0edb77a6d0876af1903367b9bb82f54e7` (MIT, see LICENSE).
Vendored (not a submodule) because this project modifies the engine directly.
`LOCAL_CHANGES.patch` is the full diff against upstream at the time of vendoring.

Local changes:
- **Ada/Ampere support** (`ssd/layers/attention.py`, new `ssd/layers/paged_attention_triton.py`): upstream calls
  `sgl_kernel` FlashAttention-3, which only runs on Hopper (sm90). On other GPUs we use a split-KV Triton
  paged-attention kernel (decode + multi-token verify/glue; CUDA-graph safe) and torch SDPA for prefill.
  Prefill with a prefix-cache hit now passes the block table (upstream dropped it). Tests: `tests/test_paged_attention.py`.
- **flashinfer 0.6 compatibility**: `ssd/layers/sampler.py` (`get_seed_and_offset` gained a `device` arg) and
  `ssd/engine/helpers/cudagraph_helpers.py` (fa2 `plan()` gained `num_colocated_ctas`).
- **NVTX instrumentation** (`ssd/utils/nvtx.py`; ranges in `step.py`, `verifier.py`, `speculator_{sync,async}.py`,
  `draft_runner.py`, `llm_engine.py`) for Nsight Systems/Compute phase attribution. No behavioral change.
- **Configurable drafter KV fraction** (`ssd/config.py`, `draft_runner.py`): optional
  `draft_gpu_memory_utilization` (default `None` = upstream's hardcoded 0.75 sync / 0.8 async). Needed to
  keep KV small under Nsight Compute, whose kernel replay backs up every writable buffer.
- **Optional prefix-cache control** (`ssd/config.py`, `engine/block_manager.py`, `engine/scheduler.py`):
  `enable_prefix_cache=True` preserves normal behavior. The Evaluation 1 campaign disables target
  and draft prompt-cache lookup so every timed request includes full prefill with warm model weights.
- **Experimental AR-compatible verification** (`ssd/engine/ar_compatible.py`, `model_runner.py`):
  `target_numerics='ar_compatible'` uses per-token target operation shapes, installed before CUDA
  graph capture. Default `'batched'` behavior and AR operations are preserved. This first version
  supports greedy, batch-one non-EAGLE Qwen3 on one target GPU, including two-GPU asynchronous
  speculation. It prioritizes numerical consistency and sacrifices batched matrix multiplication;
  it is not a demonstrated acceleration. Tests: `tests/test_ar_compatible*.py`.
