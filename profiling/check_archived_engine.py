"""Check the pre-change engine on today's hardware, without changing the checkout.

The git archive must match the original frozen execution fingerprint. Correctness
may share GPUs; timings are not performance evidence. Only our worker is stopped
on interruption or a bounded timeout. Source artifacts and old results stay intact.
"""
import argparse
import fcntl
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

from baseline_common import ROOT, atomic_json, compute_processes, file_digest, read_json
from run_ar_compatible_gate import first_difference, require_memory
from run_baselines import interrupted, now, stop_worker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args(); folder=args.outdir.resolve()
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'definition.json').exists(): raise ValueError('Preserve the existing experiment; use a new directory')
    signal.signal(signal.SIGTERM,interrupted)
    original=read_json(ROOT/'results/baselines/eval1/manifest.json')
    commit=subprocess.check_output(['git','rev-parse','68df65e'],cwd=ROOT,text=True).strip()
    archive=Path(tempfile.mkdtemp(prefix='anytime-archived-engine-'))
    data=subprocess.check_output(['git','archive',commit],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(data)) as source:
        source.extractall(archive,filter='data')
    env=dict(os.environ)
    env['AO_ROOT']=str(archive)
    env['PYTHONPATH']=str(archive/'third_party/ssd')
    env.pop('AO_SPAWN_EXECUTABLE',None)
    probe=('import sys,json; sys.path.insert(0,"profiling"); '
           'from baseline_common import execution_fingerprint; print(json.dumps(execution_fingerprint()))')
    execution=__import__('json').loads(subprocess.check_output(
        [sys.executable,'-c',probe],cwd=archive,env=env,text=True))
    if execution!=original['execution']:
        raise ValueError('Archived source does not match the old frozen execution')
    uuids=[g['uuid'] for g in original['hardware']['gpus']]
    env['CUDA_VISIBLE_DEVICES']=uuids[0]
    definition=dict(at=now(),commit=commit,archive=str(archive),execution=execution,
        supervisor_sha256=file_digest(__file__),models=original['models'],
        shared_correctness_only=True,warmup_requests=3,scored_prompts_per_mode=3,
        output_cap=256,ignore_eos=True,prefix_cache=True,target_kv_fraction=.70,
        draft_kv_fraction='upstream default',modes=['ar','sd'])
    atomic_json(folder/'definition.json',definition)
    with (ROOT/'results/baselines/eval1/.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for mode in definition['modes']:
            admission=require_memory(uuids,mode)
            command=[sys.executable,'-O',str(archive/'profiling/run_engine.py'),
                '--mode',mode,'--n','1','--warmup','3','--output_len','256',
                '--gpu_mem_util','0.70','--target',original['models']['target']['path'],
                '--draft',original['models']['draft']['path'],'--out',str(folder/f'{mode}.json')]
            process=None; samples=[]; error=None; started=time.monotonic()
            print('Checking archived '+mode,flush=True)
            try:
                with (folder/f'{mode}.log').open('w') as log:
                    process=subprocess.Popen(command,cwd=archive,env=env,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while process.poll() is None:
                        samples.append(dict(at=now(),processes=compute_processes(uuids)))
                        if time.monotonic()-started>300: raise TimeoutError('Archived worker exceeded 300 seconds')
                        time.sleep(1)
            except BaseException as exc:
                error=str(exc) or type(exc).__name__
                raise
            finally:
                if process is not None: stop_worker(process)
                valid=process is not None and process.returncode==0 and error is None
                if valid:
                    valid=len(read_json(folder/f'{mode}.json')['requests'])==3
                atomic_json(folder/f'{mode}.receipt.json',dict(valid=valid,error=error,
                    command=command,admission_memory_free_mib=admission,
                    samples=samples,shared_correctness_only=True))
            if not valid: raise RuntimeError('Archived worker failed: '+mode)
    ar=read_json(folder/'ar.json')['requests']; sd=read_json(folder/'sd.json')['requests']
    old={r['id']:r['output_token_ids'] for r in read_json(ROOT/'results/nsys/ar_run.json')['requests']}
    rows=[]
    for a,s in zip(ar,sd):
        if a['id']!=s['id']: raise ValueError('Unpaired archived outputs')
        rows.append(dict(id=a['id'],ar_matches_historical=a['output_token_ids']==old[a['id']],
            ar_first_difference=first_difference(a['output_token_ids'],old[a['id']]),
            sd_matches_ar=s['output_token_ids']==a['output_token_ids'],
            sd_first_difference=first_difference(s['output_token_ids'],a['output_token_ids'])))
    atomic_json(folder/'summary.json',dict(at=now(),commit=commit,
        exact_frozen_execution_match=True,shared_correctness_only=True,comparisons=rows))
    print(rows,flush=True)


if __name__=='__main__': main()
