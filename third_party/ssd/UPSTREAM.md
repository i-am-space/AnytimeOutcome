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
