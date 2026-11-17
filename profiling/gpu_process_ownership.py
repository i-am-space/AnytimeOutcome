"""Identify run-owned CUDA processes even when a profiler changes their group."""
import os
from pathlib import Path
import signal

TOKEN_ENV='AO_GPU_RUN_TOKEN'


def has_token(pid,token,proc_root=Path('/proc')):
    if not token:return False
    try:
        entries=(proc_root/str(pid)/'environ').read_bytes().split(b'\0')
    except (FileNotFoundError,ProcessLookupError,PermissionError):
        return False
    return (TOKEN_ENV+'='+token).encode() in entries


def partition(processes,launcher_group,token):
    owned=[];others=[]
    for row in processes:
        try:
            same_group=os.getpgid(row['pid'])==launcher_group
        except ProcessLookupError:
            continue
        if same_group or has_token(row['pid'],token):owned.append(row)
        else:others.append(row)
    return owned,others


def cleanup_tagged(processes,token):
    """Never signal a PID without rechecking its inherited run token."""
    for pid in {r['pid'] for r in processes}:
        if has_token(pid,token):
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
