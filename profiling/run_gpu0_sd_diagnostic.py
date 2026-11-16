"""GPU0-only SD run authorized while GPU1 is occupied; contextual evidence only."""
from pathlib import Path
import fcntl
import os
import signal
import subprocess
import time
from baseline_common import (ROOT,atomic_json,compute_processes,external_processes,file_digest,
    freeze_json,heldout_jobs,read_json,validate_block)
from run_baselines import (PendingWindow,interrupted,load_campaign,now,require_idle,stop_worker,
    validated_selection,worker_command)
from run_fixed_window import BudgetExpired,expired


def main():
    folder=ROOT/'results/baselines/eval1'
    dest=folder/'gpu0_only_sd_r0';dest.mkdir(exist_ok=True)
    request=read_json(folder/'fixed_window_request.json')
    left=request['deadline_unix']-time.time()
    if left<=0:raise ValueError('Existing two-hour window expired; no GPU run started')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,left)
    with (folder/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=load_campaign(folder);selection=validated_selection(folder,manifest)
        job=next(j for j in heldout_jobs(manifest,selection) if j['protocol']=='fixed' and
            j['repeat']==0 and j['config']['mode']=='sd')
        uuids=[g['uuid'] for g in manifest['hardware']['gpus']]
        output=dest/'run.json';receipt=dest/'receipt.json';jobpath=dest/'job.json'
        if output.exists() or receipt.exists():raise ValueError('Diagnostic attempt exists; preserve it before another')
        freeze_json(jobpath,job)
        definition=dict(scope='GPU0 exclusive; unrelated GPU1 compute explicitly allowed by user',
            valid_for_original_dedicated_campaign=False,manifest_sha256=manifest['manifest_sha256'],
            job_sha256=job['job_sha256'],supervisor_sha256=file_digest(__file__),
            requested_outputs=256,requests=75,repetitions=1,at=now())
        freeze_json(dest/'definition.json',definition)
        atomic_json(dest/'status.json',dict(status='waiting_for_gpu0',at=now()))
        while True:
            try:require_idle(uuids[:1]);break
            except PendingWindow as exc:
                print('Waiting for GPU0: '+str(exc),flush=True);time.sleep(10)
        command=worker_command(folder,manifest,job,jobpath,output)
        env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=uuids[0]
        env['PYTHONPATH']=str(ROOT/'third_party/ssd')+os.pathsep+env.get('PYTHONPATH','')
        for key in list(env):
            if key.startswith('SSD_PROFILE'):env[key]='0'
        env.pop('AO_SPAWN_EXECUTABLE',None)
        if env.get('CUDA_INJECTION64_PATH') or env.get('NVTX_INJECTION64_PATH'):
            raise ValueError('Profiler injection is not allowed')
        monitor=dict(started_at=now(),samples=0,gpu0_external=[],gpu1_process_changes=[],error=None)
        worker=None;last_gpu1=None;failure=None
        def sample():
            nonlocal last_gpu1
            rows=compute_processes(uuids)
            other=external_processes([r for r in rows if r['uuid']==uuids[0]],worker.pid)
            gpu1=[r for r in rows if r['uuid']==uuids[1]]
            monitor['samples']+=1
            if other:monitor['gpu0_external'].append(dict(at=now(),processes=other))
            if gpu1!=last_gpu1:
                monitor['gpu1_process_changes'].append(dict(at=now(),processes=gpu1));last_gpu1=gpu1
        try:
            atomic_json(dest/'status.json',dict(status='running',at=now(),scope=definition['scope']))
            print('SD K6 on GPU0: 75 frozen prompts, fixed 256 outputs; GPU1 activity disclosed.',flush=True)
            with (dest/'worker.log').open('w') as log:
                worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT,start_new_session=True)
                while worker.poll() is None:
                    try:sample()
                    except (OSError,ValueError,subprocess.SubprocessError) as exc:monitor['error']=str(exc)
                    if monitor['gpu0_external'] or monitor['error']:break
                    time.sleep(1)
                try:sample()
                except (OSError,ValueError,subprocess.SubprocessError) as exc:monitor['error']=str(exc)
        except (KeyboardInterrupt,BudgetExpired) as exc:
            failure=str(exc) or 'Interrupted'
        finally:
            if worker is not None:stop_worker(worker)
            signal.setitimer(signal.ITIMER_REAL,0)
        valid=(worker is not None and worker.returncode==0 and monitor['samples']>0 and
            not monitor['gpu0_external'] and monitor['error'] is None and failure is None)
        if valid:
            try:validate_block(read_json(output),job,manifest)
            except (ValueError,KeyError,OSError) as exc:valid=False;failure=str(exc)
        monitor['finished_at']=now()
        atomic_json(receipt,dict(valid=False,diagnostic_valid=valid,valid_for_dedicated_campaign=False,
            scope=definition['scope'],job_sha256=job['job_sha256'],manifest_sha256=manifest['manifest_sha256'],
            execution_sha256=manifest['execution']['sha256'],
            output_sha256=file_digest(output) if output.exists() else None,
            returncode=worker.returncode if worker else None,monitoring=monitor,error=failure))
        atomic_json(dest/'status.json',dict(status='complete_contextual_diagnostic' if valid else 'invalid',
            at=now(),requests=75 if valid else 0,valid_for_dedicated_campaign=False,error=failure))
        print('Diagnostic validity: '+str(valid)+'; dedicated-campaign eligibility: false',flush=True)
        return 0 if valid else 1


if __name__=='__main__':
    try:raise SystemExit(main())
    finally:signal.setitimer(signal.ITIMER_REAL,0)
