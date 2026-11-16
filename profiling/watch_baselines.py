"""Wait for dedicated GPUs and resume the frozen campaign; suitable for tmux.

Source profiling/env.sh once before launching. Ctrl-C stops our workers safely.
Only dedicated-window exits are retried; execution/validation errors stop the watcher.
"""
import argparse
import fcntl
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

from baseline_common import ROOT, atomic_json
from run_baselines import PendingWindow, interrupted, load_campaign, now, require_idle


def run_attempt(folder):
    worker=subprocess.Popen([sys.executable,str(ROOT/'profiling/run_baselines.py'),
        '--phase','all','--resume','--outdir',str(folder)],cwd=ROOT,start_new_session=True)
    try:
        return worker.wait()
    except KeyboardInterrupt:
        # The campaign handles SIGINT/SIGTERM by cleaning up its own worker group.
        try:
            worker.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            worker.wait(timeout=25)
        except subprocess.TimeoutExpired:
            worker.terminate()
            worker.wait(timeout=25)
        raise


def watch(folder, uuids, poll_seconds=30, check=False):
    def status(state, detail):
        entry=dict(status=state,at=now(),detail=detail)
        if not check:
            atomic_json(folder/'watcher_status.json',entry)
            with (folder/'watcher.log').open('a') as stream:
                stream.write(f'{entry["at"]} {state}: {detail}\n')
        print(f'{state}: {detail}',flush=True)
    waiting=False
    while True:
        try:
            require_idle(uuids)
        except PendingWindow as exc:
            if not waiting:
                status('waiting_for_gpus',str(exc))
            if check:
                return 2
            waiting=True
            time.sleep(poll_seconds)
            continue
        if check:
            status('ready','Both GPUs are idle; no campaign was started by --check.')
            return 0
        status('running','Both GPUs idle; starting/resuming validated blocks.')
        code=run_attempt(folder)
        if code==0:
            status('complete','All valid blocks and the final report are complete.')
            return 0
        if code==2:
            status('waiting_for_gpus','Dedicated window lost; incomplete/contaminated block will be rerun.')
            waiting=True
            time.sleep(poll_seconds)
            continue
        status('stopped',f'Campaign exited with code {code}; inspect status.json and worker logs. No automatic error retry.')
        return code if code>0 else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=ROOT/'results/baselines/eval1')
    parser.add_argument('--poll-seconds',type=float,default=30)
    parser.add_argument('--check',action='store_true',help='Validate inputs/check GPU availability without starting jobs')
    args=parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds<1:
        parser.error('--poll-seconds must be a finite number of at least 1')
    folder=args.outdir.resolve()
    signal.signal(signal.SIGTERM,interrupted)
    try:
        with (folder/'.watcher.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            manifest=load_campaign(folder)
            return watch(folder,[g['uuid'] for g in manifest['hardware']['gpus']],args.poll_seconds,args.check)
    except KeyboardInterrupt:
        if not args.check:
            atomic_json(folder/'watcher_status.json',dict(status='stopped',at=now(),detail='Stopped by user; valid blocks retained.'))
        print('Stopped; resume by launching this watcher again.',flush=True)
        return 130
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
        print(f'Watcher stopped: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
