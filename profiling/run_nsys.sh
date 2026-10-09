#!/bin/bash
# Nsight Systems capture of the steady-state timed region for one decoding mode.
#   profiling/run_nsys.sh <name> <run_engine.py args...>
# e.g. profiling/run_nsys.sh ssd_k7_f3 --mode ssd --k 7 --f 3 --n 2 --output_len 128
# Trace: CUDA API + kernels (CUDA-graph nodes expanded), NVTX, NCCL via CUDA, and GPU-metrics sampling
# (SM active / tensor active / DRAM bandwidth timelines) on every GPU.
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME=$1; shift
TRACE_DIR=${TRACE_DIR:-/scratch/$USER/ao_traces}
mkdir -p "$TRACE_DIR" "$AO_ROOT/results/nsys"
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --gpu-metrics-devices=all --gpu-metrics-frequency=20000 \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --sample=none --cpuctxsw=none \
  --force-overwrite=true -o "$TRACE_DIR/$NAME" \
  python -O "$AO_ROOT/profiling/run_engine.py" --out "$AO_ROOT/results/nsys/${NAME}_run.json" "$@"
nsys export --type sqlite --force-overwrite=true -o "$TRACE_DIR/$NAME.sqlite" "$TRACE_DIR/$NAME.nsys-rep"
python "$AO_ROOT/profiling/analyze_nsys.py" "$TRACE_DIR/$NAME.sqlite" --name "$NAME" --out "$AO_ROOT/results/nsys"
