# Source before running anything in this repo:  source profiling/env.sh
export AO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SSD_HF_CACHE=${SSD_HF_CACHE:-/scratch/$USER/hf/hub}
export SSD_DATASET_DIR=${SSD_DATASET_DIR:-/scratch/$USER/datasets/processed_datasets}
export SSD_CUDA_ARCH=${SSD_CUDA_ARCH:-8.9}          # L40S / RTX 4090 = 8.9, RTX 3090 = 8.6, RTX 2080 = 7.5
export SSD_TARGET_MODEL=${SSD_TARGET_MODEL:-$SSD_HF_CACHE/models--Qwen--Qwen3-8B/snapshots/local}
export SSD_DRAFT_MODEL=${SSD_DRAFT_MODEL:-$(ls -d $SSD_HF_CACHE/models--Qwen--Qwen3-0.6B/snapshots/* 2>/dev/null | head -1)}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}  # flashinfer JIT needs nvcc >= 12.4 for sm89
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$AO_ROOT/third_party/ssd${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/scratch/$USER}
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}  # flashinfer JIT .so need the system libstdc++ (miniconda ships an older one)
