"""Reproducible Evaluation 1 campaign. Source profiling/env.sh before invoking."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from baseline_common import (ROOT, SPEC, allocation, atomic_json, compute_processes, config_id,
    configurations, digest, execution_fingerprint, external_processes, file_digest, freeze_json,
    gpu_inventory, heldout_jobs, make_manifest, model_fingerprint, read_json, software_fingerprint,
    tuning_jobs, validate_block, validate_manifest, validate_models)


class PendingWindow(RuntimeError):
    pass


def stop_worker(worker):
    """Release only our worker session, including its async drafter."""
    try:
        os.killpg(worker.pid,signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        worker.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(worker.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        worker.wait()


def interrupted(signum, frame):
    raise KeyboardInterrupt


def now():
    return datetime.now(timezone.utc).isoformat()


def paths(folder, job):
    base = folder / 'raw' / job['id']
    return (base.with_suffix('.json'), base.with_suffix('.receipt.json'),
            folder / 'jobs' / (job['id'] + '.json'), folder / 'logs' / (job['id'] + '.log'))


def accepted_result(folder, job, manifest):
    output, receipt_path, _, _ = paths(folder,job)
    if not output.exists() or not receipt_path.exists():
        raise ValueError(f'Missing completed block: {job["id"]}')
    receipt = read_json(receipt_path)
    monitor=receipt.get('monitoring',{})
    if (receipt.get('valid') is not True or receipt.get('job_sha256') != job['job_sha256'] or
            receipt.get('output_sha256') != file_digest(output) or
            receipt.get('execution_sha256') != manifest['execution']['sha256'] or
            receipt.get('returncode')!=0 or monitor.get('samples',0)<1 or
            monitor.get('external_processes') or monitor.get('error') is not None):
        raise ValueError(f'Invalid/interrupted/contaminated block: {job["id"]}')
    result = read_json(output)
    validate_block(result,job,manifest)
    return result


def select_winners(folder, manifest):
    jobs = tuning_jobs(manifest)  # Never inspect held-out results for selection.
    scores=[]
    evidence={}
    for config in configurations():
        rows=[]
        for job in jobs:
            if job['config'] != config:
                continue
            result=accepted_result(folder,job,manifest)
            rows.extend(result['requests'])
            evidence[job['id']]=file_digest(paths(folder,job)[1])
        if len(rows) != 48:
            raise ValueError('Selection requires both complete tuning repetitions')
        scores.append(dict(config=config,mean_latency_s=sum(r['latency_s'] for r in rows)/len(rows),
                           requests=len(rows)))
    winners={}
    for mode in ('ar','sd','ssd'):
        eligible=[r for r in scores if r['config']['mode']==mode]
        winners[mode]=min(eligible,key=lambda r:(r['mean_latency_s'],r['config']['k'] or 0,
                                               r['config']['f'] or 0))['config']
    body=dict(manifest_sha256=manifest['manifest_sha256'],execution_sha256=manifest['execution']['sha256'],
              objective='mean fixed-256 full-request latency on 24 balanced tuning prompts, two repetitions',
              winners=winners,scores=scores,tuning_receipt_sha256=evidence)
    return dict(body,selection_sha256=digest(body))


def validated_selection(folder, manifest):
    selection=read_json(folder/'selection.json')
    if selection != select_winners(folder,manifest):
        raise ValueError('Frozen selection differs from validated tuning evidence')
    return selection


def prepare(folder, target, draft, target_numerics='batched'):
    from transformers import AutoTokenizer
    from workloads import get_prompts
    if not target or not draft:
        raise ValueError('Set SSD_TARGET_MODEL and SSD_DRAFT_MODEL, or supply --target/--draft')
    print('Fingerprinting model/tokenizer files and freezing the workload...',flush=True)
    models=dict(target=model_fingerprint(target),draft=model_fingerprint(draft))
    tokenizer=AutoTokenizer.from_pretrained(models['target']['path'],local_files_only=True)
    manifest=make_manifest(tokenizer,models,execution_fingerprint(),software_fingerprint(),
                           gpu_inventory(),get_prompts)
    if target_numerics != 'batched':
        from baseline_common import seal_manifest
        manifest = seal_manifest({k:v for k,v in manifest.items() if k != 'manifest_sha256'} |
            dict(schema_version=2, target_numerics=target_numerics, drain_draft_on_finish=True,
                 reference_contract='shared-fixed16-v1; new AR arithmetic; original AR is a separate control'))
        validate_manifest(manifest)
    freeze_json(folder/'manifest.json',manifest)
    freeze_json(folder/'tuning_jobs.json',tuning_jobs(manifest))
    atomic_json(folder/'status.json',dict(status='prepared',at=now(),tuning_blocks=20,
        heldout_blocks=30,tuning_measurements=480,heldout_measurements=2250,
        correctness='Fast-mode exact equivalence is unresolved; all output tokens will be retained.'))
    return manifest


def load_campaign(folder):
    manifest=read_json(folder/'manifest.json')
    validate_manifest(manifest)
    if execution_fingerprint()!=manifest['execution'] or software_fingerprint()!=manifest['software']:
        raise ValueError('Execution code/software changed; use a new campaign directory')
    if gpu_inventory()['gpus']!=manifest['hardware']['gpus']:
        raise ValueError('GPU identities or driver changed; use a new campaign directory')
    validate_models(manifest['models'])
    return manifest


def require_idle(uuids):
    idle=0
    for _ in range(30):
        other=external_processes(compute_processes(uuids))
        if other:
            raise PendingWindow(f'Dedicated window unavailable: unrelated GPU processes {other}')
        text=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,utilization.gpu',
                                      '--format=csv,noheader,nounits'],text=True,timeout=10)
        values={line.split(',')[0].strip():float(line.split(',')[1]) for line in text.strip().splitlines()}
        idle=idle+1 if all(values[u] <= 5 for u in uuids) else 0
        if idle>=3:
            return
        time.sleep(1)
    raise PendingWindow('Both GPUs must remain idle for three consecutive checks')


def worker_command(folder, manifest, job, job_path, output):
    cfg=job['config']
    cmd=[sys.executable,'-O',str(ROOT/'profiling/run_engine.py'), '--mode',cfg['mode'],
         '--target',manifest['models']['target']['path'],'--draft',manifest['models']['draft']['path'],
         '--manifest',str(folder/'manifest.json'),
         '--job',str(job_path),'--split','tune' if job['phase']=='tune' else 'heldout',
         '--protocol',job['protocol'],'--order-seed',str(job['order_seed']),
         '--output_len','256','--max_prompt_tokens','2048','--max_model_len','4096',
         '--warmup','3','--gpu_mem_util','0.70','--draft_gpu_mem_util','0.70',
         '--backup','jit','--disable-prefix-cache','--out',str(output)]
    if 'target_numerics' in manifest:
        cmd += ['--target-numerics', manifest['target_numerics']]
        if manifest.get('drain_draft_on_finish') and cfg['mode']=='ssd':
            cmd += ['--drain-draft-on-finish']
    if cfg['k'] is not None:
        cmd+=['--k',str(cfg['k'])]
    if cfg['f'] is not None:
        cmd+=['--f',str(cfg['f'])]
    return cmd


def run_block(folder, manifest, job, resume):
    output, receipt_path, job_path, log_path=paths(folder,job)
    if output.exists() or receipt_path.exists():
        try:
            accepted_result(folder,job,manifest)
        except (ValueError,KeyError,OSError):
            if not resume:
                raise ValueError(f'Existing incomplete block; use --resume: {job["id"]}')
            archive=folder/'invalid_attempts'/job['id']/uuid.uuid4().hex
            archive.mkdir(parents=True)
            for path in (output,receipt_path,log_path):
                if path.exists():
                    path.rename(archive/path.name)
        else:
            if not resume:
                raise ValueError(f'Completed block exists; use --resume: {job["id"]}')
            print('Resume: validated '+job['id'],flush=True)
            return
    uuids=[g['uuid'] for g in manifest['hardware']['gpus']]
    require_idle(uuids)
    freeze_json(job_path,job)
    log_path.parent.mkdir(parents=True,exist_ok=True)
    command=worker_command(folder,manifest,job,job_path,output)
    env=dict(os.environ)
    env['PYTHONPATH']=str(ROOT/'third_party/ssd')+os.pathsep+env.get('PYTHONPATH','')
    env['CUDA_VISIBLE_DEVICES']=','.join(uuids if job['config']['mode']=='ssd' else uuids[:1])
    for key in list(env):
        if key.startswith('SSD_PROFILE'):
            env[key]='0'
    env.pop('AO_SPAWN_EXECUTABLE',None)
    if env.get('CUDA_INJECTION64_PATH') or env.get('NVTX_INJECTION64_PATH'):
        raise ValueError('Profiler injection is incompatible with headline timing')
    print('Running '+job['id'],flush=True)
    monitoring=dict(started_at=now(),samples=0,external_processes=[],error=None)
    worker=None
    try:
        with log_path.open('w') as log:
            worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,
                                    cwd=ROOT,start_new_session=True)
            while worker.poll() is None:
                try:
                    other=external_processes(compute_processes(uuids),worker.pid)
                    monitoring['samples']+=1
                    if other:
                        monitoring['external_processes'].append(dict(at=now(),processes=other))
                except (OSError,ValueError,subprocess.SubprocessError) as exc:
                    monitoring['error']=str(exc)
                if monitoring['external_processes'] or monitoring['error']:
                    break  # Discard this block and release our GPUs immediately.
                time.sleep(1)
            # Include the end boundary, excluding only this worker's descendants.
            try:
                other=external_processes(compute_processes(uuids),worker.pid)
                if other:
                    monitoring['external_processes'].append(dict(at=now(),processes=other))
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                monitoring['error']=str(exc)
    finally:
        if worker is not None:
            # Only the process group we created; never touch unrelated GPU jobs.
            stop_worker(worker)
    monitoring['finished_at']=now()
    valid=(worker.returncode==0 and monitoring['samples']>0 and
           not monitoring['external_processes'] and monitoring['error'] is None)
    error=None
    if valid:
        try:
            validate_block(read_json(output),job,manifest)
        except (ValueError,KeyError,OSError) as exc:
            valid=False
            error=str(exc)
    receipt=dict(valid=valid,job_sha256=job['job_sha256'],execution_sha256=manifest['execution']['sha256'],
        output_sha256=file_digest(output) if output.exists() else None,
        returncode=worker.returncode,error=error,monitoring=monitoring,
        hardware_allocation=allocation(manifest,job['config']['mode']))
    atomic_json(receipt_path,receipt)
    if monitoring['external_processes'] or monitoring['error']:
        raise PendingWindow(f'Block invalidated by contention/monitoring failure: {job["id"]}')
    if not valid:
        raise ValueError(f'Worker failed validation: {job["id"]}; inspect {log_path}')


def main():
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['prepare','tune','select','heldout','held-out','report','all','dry-run'],default='all')
    parser.add_argument('--outdir',type=Path,default=ROOT/'results/baselines/eval1')
    parser.add_argument('--target',default=os.environ.get('SSD_TARGET_MODEL'))
    parser.add_argument('--draft',default=os.environ.get('SSD_DRAFT_MODEL'))
    parser.add_argument('--target-numerics',choices=['batched','shared_fixed16'],default='batched')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.phase=='held-out':
        args.phase='heldout'
    folder=args.outdir.resolve()
    if args.phase=='dry-run':
        manifest=load_campaign(folder)
        print('20 tuning blocks: 480 requests; 30 held-out blocks: 2,250 requests; 150 warmups.')
        print('Tuning configuration order:',[config_id(j['config']) for j in tuning_jobs(manifest)])
        print('Current external GPU processes:',compute_processes([g['uuid'] for g in manifest['hardware']['gpus']]))
        return 0
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if not (folder/'manifest.json').exists():
                if args.phase not in ('prepare','all'):
                    raise ValueError('Prepare the campaign first')
                manifest=prepare(folder,args.target,args.draft,args.target_numerics)
            else:
                manifest=load_campaign(folder)
            freeze_json(folder/'tuning_jobs.json',tuning_jobs(manifest))
            if args.phase=='prepare':
                print(f'Frozen campaign: {folder}',flush=True)
                return 0
            if args.phase in ('tune','all'):
                for job in tuning_jobs(manifest):
                    run_block(folder,manifest,job,args.resume)
            if args.phase in ('select','all'):
                freeze_json(folder/'selection.json',select_winners(folder,manifest))
            if args.phase in ('heldout','all'):
                selection=validated_selection(folder,manifest)
                jobs=heldout_jobs(manifest,selection)
                freeze_json(folder/'heldout_jobs.json',jobs)
                for job in jobs:
                    run_block(folder,manifest,job,args.resume)
            if args.phase in ('report','all'):
                from analyze_baselines import write_report
                write_report(folder,manifest,validated_selection(folder,manifest))
                atomic_json(folder/'status.json',dict(status='complete',at=now(),valid_blocks=50,
                                                     scored_requests=2730,report=str(folder/'report.md')))
            elif args.phase in ('tune','select','heldout'):
                atomic_json(folder/'status.json',dict(status=args.phase+'_complete',at=now()))
        except KeyboardInterrupt:
            atomic_json(folder/'status.json',dict(status='interrupted',phase=args.phase,at=now()))
            return 130
        except PendingWindow as exc:
            atomic_json(folder/'status.json',dict(status='pending_dedicated_gpu_window',phase=args.phase,
                                                 at=now(),reason=str(exc)))
            print(str(exc),file=sys.stderr)
            return 2
        except (ValueError,KeyError,OSError,subprocess.SubprocessError) as exc:
            atomic_json(folder/'status.json',dict(status='incomplete',phase=args.phase,at=now(),reason=str(exc)))
            print(str(exc),file=sys.stderr)
            return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
