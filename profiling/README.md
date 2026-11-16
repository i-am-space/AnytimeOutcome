# Profiling: AR vs synchronous SD vs SSD on L40S

Everything here runs on the vendored SSD engine (`third_party/ssd`, see its `UPSTREAM.md`) with
Qwen3-8B (target) and Qwen3-0.6B (drafter), greedy, batch size 1. Results and figures land in `results/`;
raw traces (`.nsys-rep`, `.sqlite`, `.ncu-rep`) go to `/scratch/$USER/ao_traces` (too large for git).
The write-up is `docs/perf_analysis.md`.

## Setup (this machine)
```bash
source profiling/env.sh          # model paths, CUDA_HOME=/usr/local/cuda-12.9, LD_PRELOAD for flashinfer JIT
python tests/test_paged_attention.py   # portable attention kernels vs dense reference (25 cases)
```
Python env: miniconda base (torch 2.10+cu128, flashinfer 0.6.4, triton 3.6, transformers 4.57). Run the
engine with `python -O` (strips SSD's `__debug__` prints/syncs; `run_engine.py` patches a torch-2.10/py3.13
Dynamo bug that `-O` triggers).

Models: `Qwen3-8B` (sha256-verified against HF) and `Qwen3-0.6B` under `$SSD_HF_CACHE`.
Workloads (`profiling/workloads.py`): chat = Spec-Bench MT-bench categories, code = HumanEval,
doc = Spec-Bench summarization + RAG; Qwen3 chat template, thinking disabled; fixed tune/held-out split.

## Scripts
| Script | What it measures |
|---|---|
| `roofline_microbench.py` | Measured BF16 tensor peak, DRAM/L2 bandwidth, ridge point; every Qwen3-8B/0.6B linear-layer GEMM swept over tokens-per-forward T (weights rotated to defeat the 96 MB L2). `results/roofline/` |
| `run_engine.py` | Shared driver for `--mode ar|sd|ssd`: full-request latency (incl. both prefills), tokens/step, cache hits, committed tok/s, output token ids. Wraps the timed region in NVTX `timed_region` + cudaProfilerStart/Stop. |
| `run_nsys.sh` + `analyze_nsys.py` | Nsight Systems: per-GPU busy/idle, GPU time by kernel category and NVTX phase, idle gaps attributed to phases, GPU-metrics sampling (DRAM BW %, SM issue %, tensor active %) per phase, timeline figure. |
| `ssd_critical_path.py` | Per SSD round: target stall at handoff (hit vs miss), and how many tree depths were already complete on the drafter GPU when the target asked (headroom for anytime publication). |
| `run_ncu.sh` + `analyze_ncu.py` | Nsight Compute on one steady-state forward selected by NVTX range: SOL compute/memory, DRAM bytes, tensor FLOPs, occupancy, issue utilization, warp stalls; memory/compute/latency-bound classification; measured kernel roofline. |

## Reproduce the report
```bash
python profiling/roofline_microbench.py --device 0 --out results/roofline
profiling/run_nsys.sh ar        --mode ar            --n 1 --warmup 3 --output_len 256
profiling/run_nsys.sh sd_k6     --mode sd  --k 6     --n 1 --warmup 3 --output_len 256
profiling/run_nsys.sh ssd_k7_f3 --mode ssd --k 7 --f 3 --n 1 --warmup 3 --output_len 256
python profiling/ssd_critical_path.py /scratch/$USER/ao_traces/ssd_k7_f3.sqlite --name ssd_k7_f3 --k 7
profiling/run_ncu.sh ar_decode  "AR\\/decode/"          5400 540 --mode ar
profiling/run_ncu.sh ar_prefill "AR\\/prefill/"            0 600 --mode ar --families doc --warmup 0
profiling/run_ncu.sh sd_verify  "verify\\/target_fwd/"  2700 545 --mode sd --k 6
profiling/run_ncu.sh sd_draft   "SD\\/speculate/"       9900 470 --mode sd --k 6
profiling/run_ncu.sh ssd_tree   "draft\\/decode_tree/" 15000 320 --mode ssd --k 7 --f 3
```

## Gotchas found while setting this up
- `ncu --nvtx-include` treats `/` as a range-nesting separator: escape it (`AR\/decode/`).
- ncu kernel replay backs up every buffer a kernel can write; with SSD's default KV cache (70% of free
  memory) that is ~20 GB per pass (2 kernels/min). `run_ncu.sh` passes `--gpu_mem_util 0.08` (~1 kernel/s).
- In SSD, `dist.recv` only enqueues an NCCL kernel; the target actually blocks at the `.tolist()` inside
  `SD/speculate`, and the waiting shows up on GPU0 as long-running NCCL kernels, not as idle time.
- NVTX ranges are CPU-side; with CUDA graphs the GPU work of `verify/target_fwd` runs after the range closes,
  inside `SD/verify` (where the CPU blocks). Per-phase hardware counters therefore use the enclosing blocking range.
- `pkill -f <pattern>` matches the invoking shell if the pattern appears in its command line; kill by PID.

## Greedy correctness investigation

`results/correctness/` contains diagnostic logit captures, matching-history summaries, and test output.
These runs copy logits to the CPU and replay forwards; **do not use their timings as benchmarks**.
The production inference implementation is unchanged by these diagnostics.

```bash
source profiling/env.sh
python -O profiling/investigate_correctness.py --mode ar --out results/correctness/ar.json
python -O profiling/investigate_correctness.py --mode sd --out results/correctness/sd.json
python -O profiling/investigate_correctness.py --mode ssd --warmup 0 --out results/correctness/ssd.json
python -O profiling/investigate_correctness.py --mode sd --eager --warmup 0 --output-len 192 --out results/correctness/sd_eager.json
python -O profiling/replay_correctness_prefix.py
python -O profiling/replay_correctness_prefix.py --schedule results/correctness/ssd.json --family doc
python profiling/summarize_correctness.py
python -m pytest tests/test_greedy_verify.py -q
python tests/test_paged_attention.py
```

The first script targets the three saved baseline prompts, records logits around their first AR/SD
divergences, and replays relevant verification blocks one token at a time from the same starting KV.
It restores overwritten suffix KV after replay. The summary excludes rows conditioned on rejected
draft proposals: only rows whose input history matches the common committed prefix are witnesses.
The prefix-control script builds histories by teacher-forcing the saved AR tokens, varies query width
from the identical prefix, and separately varies the final vocabulary-projection shape. Its optional
schedule replay executes captured target inputs at explicit logical positions without SSD's draft,
branch manager, verifier, or communication. Existing saved baseline results are never overwritten.

### Attention attribution and numerical-reference controls

`attention_controls.py` provides diagnostic target-only substitutions. Prefill and draft execution
stay unchanged. `sdpa_flash` gathers paged KV and forces PyTorch's FlashAttention backend for each
causal query. `sdpa_math` uses PyTorch math attention with explicit FP32 inputs and a bottom-right
causal mask. `portable_serial` applies our original single-query attention kernel to each query.
These paths use host metadata and require eager execution; they are not production replacements.

```bash
# Compare identical token histories and the captured SSD target schedule.
python -O profiling/replay_correctness_prefix.py --attention sdpa_math --schedule results/correctness/ssd.json --out results/correctness/attention_math_controls.json
python -O profiling/replay_correctness_prefix.py --attention sdpa_flash --schedule results/correctness/ssd.json --out results/correctness/attention_flash_controls.json

# Actual generation: compare AR and SD against each other under the same backend.
python -O profiling/investigate_correctness.py --mode ar --attention sdpa_flash --eager --warmup 0 --out results/correctness/attention_backends/flash_ar.json
python -O profiling/investigate_correctness.py --mode sd --attention sdpa_flash --eager --warmup 0 --out results/correctness/attention_backends/flash_sd.json

# Preserve the ORIGINAL AR numerical path, without changing the reference outputs.
# Includes one-token shapes for projections, attention, compiled norms, RoPE, and SiLU.
python -O profiling/investigate_correctness.py --mode sd --attention portable_serial --rowwise-linear --tokenwise-pointwise --eager --warmup 0 --progress --out results/correctness/attention_backends/serial_reference_sd.json
python profiling/summarize_attention_controls.py
python -m pytest tests/test_attention_controls.py tests/test_greedy_verify.py -q
```

The math free-generation control uses `--attention sdpa_math --families chat --output-len 64` for
both AR and SD; its evidence covers only that shorter sample. The failed reference limited to
attention/projection shapes is preserved as `linear_attention_reference_sd.json`. The expanded
reference also matches per-token shapes for compiled normalization, rotary embedding, and activation.
Reference controls deliberately sacrifice batching and must not be presented as acceleration results.
The backend comparison summary reports both agreement within each alternative backend and agreement
with the original saved AR, so a changed reference cannot silently be counted as a fix.

## Evaluation 1 baseline campaign

The campaign freezes 24 tuning prompts (8 per family), 75 held-out prompts (25 per family), and
three separate warmup prompts. The three investigated held-out prompts are excluded. Complete input
token IDs, source IDs, tokenizer/model SHA256 fingerprints, execution code, software settings, and
GPU identities are saved in `results/baselines/eval1/manifest.json` before measurement.

SD tunes K=4/6/8; SSD tunes K=4/7/10 × F=1/3. AR is the tuning control. Each setting runs twice on
24 tuning prompts (480 measurements total). The lowest balanced mean full-request latency selects
one SD and one SSD configuration, with smaller K/F breaking exact ties. Selection never reads
held-out results and is frozen with hashes of all validated tuning evidence.

The winners and AR then run five repetitions on all 75 held-out prompts under fixed-256 and
natural-EOS/cap-256 protocols (2,250 measurements). Natural EOS uses the fixed-length-tuned winners;
it does not retune on test data. Every block starts a fresh worker and performs the three warmups.
Model/compilation warmup, console logging, and file writing are outside request timing. The delivery
callback records actual first committed output. Prompt-prefix lookup is disabled for target and
drafter, so repeated requests include full prefill; normal engine defaults retain prefix caching.

```bash
source profiling/env.sh
python profiling/run_baselines.py --phase prepare
python profiling/run_baselines.py --phase dry-run
python profiling/run_baselines.py --phase all --resume

# Individual phases are also available:
python profiling/run_baselines.py --phase tune --resume
python profiling/run_baselines.py --phase select
python profiling/run_baselines.py --phase heldout --resume
python profiling/run_baselines.py --phase report
```

Both physical L40S GPUs must be dedicated before any block starts. GPU UUIDs pin target to physical
GPU0, with SD's drafter also on GPU0 and SSD's drafter on GPU1. Both KV fractions are explicitly
0.70; BF16, batch one, CUDA graphs, JIT misses, and 4,096 model capacity remain fixed. No profiler is
attached. GPU processes are monitored every second; unrelated jobs invalidate the block and stop
further scheduling. Unrelated jobs are never terminated. A busy machine produces
`pending_dedicated_gpu_window` in `status.json`, **not shared-GPU benchmark results**. Resume after
the window becomes available; only evidence with complete requests, matching hashes, successful
exit, and clean monitoring can be skipped. Invalid attempts are archived, not counted. Changed
models, software, or execution code require a new campaign directory (`--outdir`).

Only all 50 valid blocks permit the final report. Outputs include raw request token IDs/timings,
receipts/logs, frozen winners, `summary.json`, `summary.csv`, tuning and latency figures, and
`report.md`. Statistics include per-family mean/p50/p95 request latency, tokens/s, first delivery,
output length, acceptance, cache hits, exact output comparisons, repeatability, and regressions.
Paired 95% intervals use 10,000 seeded bootstrap draws over repetition rounds and prompt IDs,
stratified by family. AR/SD use one GPU; SSD uses two, explicitly labeled. Fast-mode exactness remains
unresolved and is disclosed beside performance results. The slow correctness reference is separate;
natural-EOS latency ratios may also reflect different output lengths.

```bash
python -m pytest tests/test_baseline_campaign.py tests/test_baseline_engine.py -q
```

These tests use synthetic evidence in temporary directories to validate orchestration and reporting.
Synthetic results are never stored as real campaign measurements.

### Automatically wait and resume in tmux

The watcher checks both frozen GPU UUIDs every 30 seconds. Once no unrelated compute jobs remain,
both GPUs must pass the existing three consecutive idle checks. It then starts the campaign with
resume enabled. During each block, the campaign checks for other GPU processes every second. On
contention or a monitoring failure, it terminates only its own worker group, invalidates the entire
block, and returns to waiting. After dedicated access returns, the block runs again with a fresh
process and warmups. Completed valid blocks remain saved. This is a restart of an interrupted
block; interrupted timing samples are never combined with later samples.

Launch a new tmux session, then run these commands inside it (source the environment once):

```bash
tmux new -s eval1
cd /home/vishalrao/AnytimeOutcome
source profiling/env.sh
python /home/vishalrao/AnytimeOutcome/profiling/watch_baselines.py
```

Detach with Ctrl+B then D. Reattach with `tmux attach -t eval1`. Ctrl+C stops the watcher and its
campaign workers safely; launch the watcher again to resume. Use `--poll-seconds 10` for more
frequent idle checks, or `--check` to validate the campaign and check availability without starting
it. The watcher stops on execution/validation errors rather than retrying those errors indefinitely.
Only one watcher may own a campaign at a time. Progress is recorded in `watcher_status.json` and
`watcher.log` in the campaign directory, alongside the existing campaign status and per-block logs.

Budget approximately **3–6 hours of uninterrupted dedicated access**, plus any waiting and discarded
blocks. This is a planning estimate, not a benchmark result: prior profiled three-prompt runs averaged
5.82 s/request for AR, 4.09 s for SD K6, and 3.18 s for SSD K7/F3. Applying those rates to the declared
method counts gives about 3.2 hours for scored generation if natural-EOS outputs also reach 256
tokens. Allow additional time for 50 model loads/compilations, 150 warmups, and slower grid settings.
Natural EOS may finish sooner. Dedicated timings and other K/F settings may differ from those
earlier runs; wall-clock completion cannot be bounded while other users occupy the GPUs.
