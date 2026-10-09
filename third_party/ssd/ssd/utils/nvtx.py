"""NVTX ranges for Nsight Systems. Near-zero cost when no profiler is attached."""
from contextlib import contextmanager

import torch


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
