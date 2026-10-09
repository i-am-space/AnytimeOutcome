#!/bin/bash
# Nsight Compute: per-kernel Speed-of-Light, occupancy, memory workload, warp stalls and roofline for the
# kernels of one steady-state forward pass, selected by NVTX range.
#   profiling/run_ncu.sh <name> <nvtx-include> <launch-skip> <launch-count> <run_engine.py args...>
# Examples (see profiling/README.md for the set used in the report):
#   profiling/run_ncu.sh ar_decode   "AR\\/decode/"         1200 460 --mode ar
#   profiling/run_ncu.sh sd_verify   "verify\\/target_fwd/" 1200 460 --mode sd --k 6
#   profiling/run_ncu.sh sd_draft    "SD\\/speculate/"      1500 300 --mode sd --k 6
#   profiling/run_ncu.sh ssd_tree    "draft\\/decode_tree/" 1500 400 --mode ssd --k 7 --f 3
# Notes: ncu serializes kernels and replays each one, so cache state differs from the real pipelined run;
# clocks are left unlocked (--clock-control none) so durations/TFLOP/s are comparable to Nsight Systems.
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME=$1; NVTX=$2; SKIP=$3; COUNT=$4; shift 4
TRACE_DIR=${TRACE_DIR:-/scratch/$USER/ao_traces}
mkdir -p "$TRACE_DIR" "$AO_ROOT/results/ncu"
NCU=${NCU:-/opt/nvidia/nsight-compute/2025.2.1/ncu}
$NCU --target-processes all \
  --nvtx --nvtx-include "$NVTX" \
  --graph-profiling node \
  --launch-skip "$SKIP" --launch-count "$COUNT" \
  --section SpeedOfLight --section SpeedOfLight_RooflineChart \
  --section Occupancy --section LaunchStats \
  --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
  --section WarpStateStats --section SchedulerStats \
  --metrics sm__ops_path_tensor_src_bf16_dst_fp32.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum \
  --clock-control none \
  --force-overwrite -o "$TRACE_DIR/ncu_$NAME" \
  python -O "$AO_ROOT/profiling/run_engine.py" --n 1 --families chat --warmup 2 --output_len 64 --gpu_mem_util 0.08 \
    --out "$AO_ROOT/results/ncu/${NAME}_run.json" "$@"
python "$AO_ROOT/profiling/analyze_ncu.py" "$TRACE_DIR/ncu_$NAME.ncu-rep" --name "$NAME" --out "$AO_ROOT/results/ncu"
