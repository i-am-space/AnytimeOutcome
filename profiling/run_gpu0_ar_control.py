"""Contextual AR control on dedicated GPU0 while GPU1 may run unrelated jobs.

This separate experiment cannot satisfy the frozen both-GPUs-dedicated campaign.
Reuses its AR worker and owned-process monitor; never edits its engine or inputs.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import time
import zipfile

import run_publication_campaign as campaign
from baseline_common import (ROOT, atomic_json, compute_processes, digest,
    execution_fingerprint, file_digest, freeze_json, gpu_inventory, read_json,
    software_fingerprint, validate_models)
from run_baselines import PendingWindow, interrupted, now, require_idle

DEFAULT_OUT=ROOT/'results/next_stage/ar_gpu0_control_v1_2026-11-18'
SOURCE=campaign.DEFAULT_OUT
ORIGINAL_CHECK=campaign.check_runtime
ORIGINAL_VALIDATE_RECEIPT=campaign.validate_receipt
ORIGINAL_VALIDATE_RESULT=campaign.validate_result
ORIGINAL_RUN_BLOCK=campaign.run_block


def code_files():
    return {**campaign.code_files(),'profiling/run_gpu0_ar_control.py':file_digest(__file__)}


def prepare(folder):
    source=read_json(SOURCE/'queue.json');ORIGINAL_CHECK(source)
    old=read_json(SOURCE/'tune/definition.json')
    q=dict(created_at=now(),execution=execution_fingerprint(),software=software_fingerprint(),
        queue_code=code_files(),hardware=gpu_inventory(),models=source['models'],
        source=dict(path=str(SOURCE/'queue.json'),sha256=file_digest(SOURCE/'queue.json')),
        reference_contract=source['reference_contract'],scope='GPU0-only contextual AR; GPU1 may be busy',
        headline_performance_evidence=False,monitor_scope='Physical GPU0 only; record GPU1 activity separately')
    q['sha256']=digest(q);freeze_json(folder/'queue.json',q)
    d=dict(queue_sha256=q['sha256'],phase='tune',prompts=old['prompts'],warmup=old['warmup'],
        configurations=[campaign.config('ar')],jobs=[j for j in old['jobs'] if j['config']['method']=='ar'],
        repetitions=2,protocols=['fixed'],output_cap=256,capacity=4096,kv_fraction=.70,profile_capture=False)
    d['sha256']=digest(d);freeze_json(folder/'tune/definition.json',d)
    with zipfile.ZipFile(folder/'source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(q['execution']['files'])|set(q['queue_code'])):archive.write(ROOT/name,name)
    atomic_json(folder/'status.json',dict(status='prepared',at=now(),requests=48,contextual_only=True))
    (folder/'protocol.md').write_text(
        '# GPU0-only contextual AR control\n\n'
        'Two repetitions of the exact 24 tuning prompts and seeded orders from the publication campaign; '
        '48 scored requests and six separate warmups. No held-out inputs are accessed. '
        'BF16, greedy batch one, shared-fixed16-v1, CUDA graphs, 4096 capacity, fixed 256 output tokens '
        'ignoring EOS, 0.70 KV fraction, no prefix cache or profiler. Full-request timing includes prefill.\n\n'
        'Only physical GPU0 must be free; GPU1 activity is recorded. GPU0 contamination invalidates a block. '
        'One worker at a time under the same global campaign lock. '
        'These results are contextual and cannot count as either of the frozen both-GPUs-dedicated AR blocks. '
        'The main campaign resumes afterward and retains its original protocol.\n')
    return q


def check_runtime(q):
    if (digest({k:v for k,v in q.items() if k!='sha256'})!=q['sha256'] or
        q['execution']!=execution_fingerprint() or q['queue_code']!=code_files() or
        q['software']!=software_fingerprint() or q['hardware']['gpus']!=gpu_inventory()['gpus'] or
        file_digest(q['source']['path'])!=q['source']['sha256']):
        raise ValueError('Contextual AR execution, software, source or GPU identities changed')


def validate_receipt(r,d,output):
    if (r.get('dedicated_gpus') is not False or r.get('allocated_gpu_dedicated') is not True or
        r.get('contextual_only') is not True or r.get('headline_performance_evidence') is not False or
        r.get('monitor_scope')!='physical GPU0 only'):
        raise ValueError('Incorrect resource scope for contextual AR')
    # Apply the same interruption/output-integrity checks, after explicitly
    # checking the stricter resource label above. The stored receipt stays false.
    ORIGINAL_VALIDATE_RECEIPT(dict(r,dedicated_gpus=True),d,output)


def run_block(args,d,j,q):
    all_uuids=[g['uuid'] for g in q['hardware']['gpus']];observed={}
    def monitored(_):
        rows=compute_processes(all_uuids)
        for row in rows:
            if row['uuid']!=all_uuids[0]:observed[row['uuid'],row['pid']]=row
        return [r for r in rows if r['uuid']==all_uuids[0]]
    campaign.check_runtime=check_runtime
    campaign.require_idle=lambda _:require_idle(all_uuids[:1])
    campaign.compute_processes=monitored
    campaign.__file__=__file__
    parent=args.outdir/'tune'/j['id']
    before=set(parent.glob('attempt_*'))
    try:
        ORIGINAL_RUN_BLOCK(args,d,j,q)
    finally:
        for attempt in set(parent.glob('attempt_*'))-before:
            path=attempt/'receipt.json'
            if path.exists():
                r=read_json(path)
                r.update(dedicated_gpus=False,allocated_gpu_dedicated=True,contextual_only=True,
                    headline_performance_evidence=False,monitor_scope='physical GPU0 only',
                    nonallocated_gpu_processes=list(observed.values()))
                atomic_json(path,r)


def install_validators():
    campaign.check_runtime=check_runtime
    campaign.validate_receipt=validate_receipt
    campaign.validate_result=ORIGINAL_VALIDATE_RESULT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--job')
    parser.add_argument('--phase',default='tune');parser.add_argument('--attempt',type=Path)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();args.outdir=args.outdir.resolve()
    if args.worker:
        q=read_json(args.outdir/'queue.json');check_runtime(q)
        campaign.base.check_runtime=check_runtime
        campaign.engine_worker.worker(args);return 0
    signal.signal(signal.SIGTERM,interrupted)
    args.outdir.mkdir(parents=True,exist_ok=True)
    q=read_json(args.outdir/'queue.json') if (args.outdir/'queue.json').exists() else prepare(args.outdir)
    check_runtime(q)
    if args.prepare_only:print('Prepared contextual GPU0-only AR: 48 scored requests, two fresh workers.');return 0
    install_validators()
    try:
        with (ROOT/'results/baselines/eval1/.campaign.lock').open('a') as lock:
            while True:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    atomic_json(args.outdir/'status.json',dict(status='waiting_for_campaign_lock',at=now()))
                    if not args.watch:return 2
                    time.sleep(2)
            validate_models(q['models'])
            d=read_json(args.outdir/'tune/definition.json')
            if digest({k:v for k,v in d.items() if k!='sha256'})!=d['sha256'] or d['queue_sha256']!=q['sha256']:
                raise ValueError('Contextual AR definition changed')
            blocks=[]
            for j in d['jobs']:
                while campaign.block_result(args.outdir/'tune',j,d,q) is None:
                    try:
                        atomic_json(args.outdir/'status.json',dict(status='running',job=j['id'],at=now(),gpu_count=1,contextual_only=True))
                        run_block(args,d,j,q)
                    except PendingWindow as exc:
                        atomic_json(args.outdir/'status.json',dict(status='pending_gpu0',reason=str(exc),at=now()))
                        if not args.watch:return 2
                        time.sleep(15)
                blocks.append(campaign.block_result(args.outdir/'tune',j,d,q))
            first={r['id']:r['output_token_ids'] for r in blocks[0]['requests']}
            repeatable=all(first[r['id']]==r['output_token_ids'] for r in blocks[1]['requests'])
            rows=[r for b in blocks for r in b['requests']]
            summary=dict(status='complete',requests=len(rows),valid_blocks=len(blocks),gpu_count=1,
                mean_latency_s=sum(r['latency_s'] for r in rows)/len(rows),
                committed_tok_s=sum(r['output_tokens'] for r in rows)/sum(r['latency_s'] for r in rows),
                repeatable=repeatable,contextual_only=True,headline_performance_evidence=False,
                numerics=q['reference_contract'],queue_sha256=q['sha256'])
            atomic_json(args.outdir/'summary.json',summary)
            (args.outdir/'report.md').write_text(
                '# GPU0-only contextual AR control\n\n'
                f'Completed {len(rows)} tuning requests in two repetitions. Mean request latency: '
                f'{summary["mean_latency_s"]:.4f} s; committed throughput: {summary["committed_tok_s"]:.2f} tokens/s. '
                f'Within-method output repeatability: {repeatable}.\n\n'
                'GPU0 was monitored for unrelated jobs; GPU1 jobs are recorded in the block receipts. '
                'Contextual timings only: these do not replace the dedicated AR control in the main campaign.\n')
            atomic_json(args.outdir/'status.json',dict(summary,at=now()))
    except BaseException as exc:
        atomic_json(args.outdir/'status.json',dict(status='stopped',reason=str(exc),at=now()));raise
    return 0


if __name__=='__main__':raise SystemExit(main())
