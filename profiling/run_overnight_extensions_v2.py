"""Ownership-monitor correction; engine and inference worker remain unchanged.

Use a new queue directory. Explicitly reuse only the valid, pinned v1 tuning
calibration. Invalid profiling attempts are preserved and never imported.
"""
import argparse
import importlib.util
import inspect
from pathlib import Path
import tempfile
import uuid
import zipfile

import run_overnight_extensions as base
from baseline_common import ROOT,atomic_json,digest,file_digest,freeze_json,read_json
from gpu_process_ownership import TOKEN_ENV,cleanup_tagged,partition

ORIGINAL_SCRIPT=Path(base.__file__).resolve()
DEFAULT_SOURCE=ROOT/'results/next_stage/overnight_extensions_v1_2026-11-17'
_original_code_files=base.code_files
_original_enqueue=base.enqueue
_original_block_result=base.block_result
_original_run_block=base.run_block
_original_worker_command=base.worker_command
_contaminations={}


def code_files():
    return dict(_original_code_files(),**{
        'profiling/run_overnight_extensions_v2.py':file_digest(__file__),
        'profiling/gpu_process_ownership.py':file_digest(ROOT/'profiling/gpu_process_ownership.py')})


def source_reader():
    spec=importlib.util.spec_from_file_location('overnight_v1_evidence_reader',ORIGINAL_SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def enqueue(args):
    reader=source_reader();old=reader.load_queue(DEFAULT_SOURCE)
    old_phase=reader.load_phase(DEFAULT_SOURCE/'depth_tune',old)
    # Revalidate all ten receipts, result hashes, paired AR outputs and selection.
    reader.select_depth(DEFAULT_SOURCE/'depth_tune',old_phase,old)
    paths=[DEFAULT_SOURCE/'queue.json',DEFAULT_SOURCE/'depth_tune/definition.json',
           DEFAULT_SOURCE/'depth_tune/selection.json']
    paths+=sorted((DEFAULT_SOURCE/'depth_tune').rglob('receipt.json'))
    paths+=sorted((DEFAULT_SOURCE/'depth_tune').rglob('result.json'))
    with tempfile.TemporaryDirectory(prefix='.prepare-',dir=args.outdir) as tmp:
        staged=argparse.Namespace(**vars(args));staged.outdir=Path(tmp)
        q=_original_enqueue(staged)
        for key in ('execution','software','models','hardware','warmup','tuning_prompts',
                    'heldout_prompts','depth_tuning','reference_contract'):
            if q[key]!=old[key]:raise ValueError('Cannot reuse calibration: '+key+' differs')
        if inspect.getsource(base.worker)!=inspect.getsource(reader.worker):
            raise ValueError('Inference worker changed; rerun calibration')
        body={k:v for k,v in q.items() if k!='sha256'}
        body['controller_version']='v2: inherited run-token ownership; no inference arithmetic change'
        body['reused_depth']=dict(folder=str(DEFAULT_SOURCE),queue_sha256=old['sha256'],
            artifact_sha256={str(p):file_digest(p) for p in paths},
            reason='Only process ownership monitoring changed; source worker and engine are identical')
        q=dict(body,sha256=digest(body));freeze_json(args.outdir/'queue.json',q)
        plan=(Path(tmp)/'plan.md').read_text()
    with zipfile.ZipFile(args.outdir/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in {**q['execution']['files'],**q['queue_code']}:archive.write(ROOT/name,name)
        archive.write(args.outdir/'queue.json','queue.json')
    (args.outdir/'plan.md').write_text(plan+'\nVersion 2 explicitly reuses the completed 240-request depth calibration. '
        'Its artifacts and unchanged inference worker are verified; failed v1 traces remain invalid.\n')
    atomic_json(args.outdir/'status.json',dict(status='prepared',at=base.now(),reused_depth_requests=240))
    return q


def block_result(folder,j,d,q):
    reuse=q.get('reused_depth')
    if d['phase']!='depth_tune' or not reuse:
        return _original_block_result(folder,j,d,q)
    for path,sha in reuse['artifact_sha256'].items():
        if file_digest(path)!=sha:raise ValueError('Reused calibration evidence changed')
    reader=source_reader();source=Path(reuse['folder'])
    old=reader.load_queue(source);old_d=reader.load_phase(source/'depth_tune',old)
    if old['sha256']!=reuse['queue_sha256']:raise ValueError('Reused source queue changed')
    if inspect.getsource(base.worker)!=inspect.getsource(reader.worker):
        raise ValueError('Inference worker changed since calibration')
    for key,value in d.items():
        if key not in ('queue_sha256','sha256') and old_d[key]!=value:
            raise ValueError('Reused calibration definition differs: '+key)
    # Return the original evidence, with its original provenance. Do not rewrite
    # raw results to pretend these were new worker runs.
    return reader.block_result(source/'depth_tune',j,old_d,old)


def run_block(args,d,j,q):
    token=uuid.uuid4().hex;owned=[]
    original_popen=base.subprocess.Popen
    original_external=base.external_processes
    parent=args.outdir/d['phase']/j['id']
    before=set(parent.glob('attempt_*'))
    def launch(*a,**kw):
        if kw.get('start_new_session') and 'env' in kw:
            kw['env']=dict(kw['env'],**{TOKEN_ENV:token})
        return original_popen(*a,**kw)
    def unrelated(rows,group):
        ours,others=partition(rows,group,token);owned.extend(ours)
        return others
    base.subprocess.Popen=launch;base.external_processes=unrelated
    try:
        return _original_run_block(args,d,j,q)
    except base.PendingWindow:
        # A continuously occupied GPU simply waits without creating attempts.
        # Repeated invalidated launches need inspection rather than infinite retries.
        if set(parent.glob('attempt_*'))-before:
            key=(d['phase'],j['id'])
            _contaminations[key]=_contaminations.get(key,0)+1
            if _contaminations[key]>=5:
                raise RuntimeError('Five launches were contaminated; stopping for ownership/window inspection')
        raise
    finally:
        base.subprocess.Popen=original_popen;base.external_processes=original_external
        cleanup_tagged(owned,token)
        for attempt in set(parent.glob('attempt_*'))-before:
            path=attempt/'receipt.json'
            if path.exists():
                receipt=read_json(path)
                receipt['ownership_monitor']=dict(strategy='launcher group or inherited per-run token',
                    run_token=token,owned_gpu_pids=sorted({r['pid'] for r in owned}))
                atomic_json(path,receipt)


def worker_command(args,j,d,attempt):
    command=_original_worker_command(args,j,d,attempt)
    if d['profile_capture']:
        # Device-side event tracing can introduce false stream dependencies.
        command.insert(2,'--cuda-event-trace=false')
    return command


def install_controller():
    base.code_files=code_files
    base.enqueue=enqueue
    base.block_result=block_result
    base.run_block=run_block
    base.worker_command=worker_command
    base.__file__=__file__


# Spawned drafters import this entry point with __mp_main__ as their name.
if __name__ in ('__main__','__mp_main__'):
    install_controller()
    if __name__=='__main__':raise SystemExit(base.main())
