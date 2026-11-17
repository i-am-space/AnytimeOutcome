"""Stable-prefix overlap gate. Shared GPU access is correctness-only and explicit.

Normal and artificial-tail-delay phases use fresh AR references, CUDA graphs,
rejection/EOS inputs, immutable source fingerprints, and monitored block receipts.
No result from this driver is a headline performance measurement.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import zipfile

import run_adaptive_publication as previous
import run_overnight_extensions as base
from baseline_common import (ROOT, atomic_json, compute_processes, digest,
    execution_fingerprint, file_digest, freeze_json, gpu_inventory, read_json,
    software_fingerprint, validate_models)
from gpu_process_ownership import TOKEN_ENV, partition, cleanup_tagged
from run_baselines import PendingWindow, interrupted, now, require_idle, stop_worker

PHASES = ('correctness', 'stress')


def code_files():
    names = ('run_prefix_publication.py', 'run_adaptive_publication.py',
             'run_overnight_extensions.py', 'run_publication_study.py',
             'benchmark_shared_numerics.py', 'gpu_process_ownership.py')
    return {f'profiling/{name}': file_digest(ROOT/'profiling'/name) for name in names}


def configurations(stress=False):
    cfgs = [dict(label='ar', mode='ar', k=7, f=3, p=0, policy='uniform', window=2)]
    for k in ((10,) if stress else (7, 10)):
        for label in (('overlap',) if stress else ('complete', 'uniform', 'bounded', 'overlap')):
            cfgs.append(dict(label=f'{label}_k{k}', mode='ssd', k=k, f=3,
                p=0 if label == 'complete' else 4,
                policy='uniform' if label == 'complete' else label, window=2))
    return cfgs


def prepare(folder, shared):
    source = ROOT/'results/next_stage/overnight_extensions_v2_2026-11-17/queue.json'
    old = read_json(source)
    validate_models(old['models'])
    q = dict(created_at=now(), execution=execution_fingerprint(), software=software_fingerprint(),
        queue_code=code_files(), hardware=gpu_inventory(), models=old['models'],
        reference_contract='shared-fixed16-v1; unchanged target arithmetic; original AR separate',
        source=dict(path=str(source), sha256=file_digest(source)),
        allow_shared_correctness=shared, performance_evidence=False,
        scope='Single-repetition correctness and overlap coverage; timings diagnostic only',
        policy_contract='prefix-overlap-v1; P4; at most two unfinished steps; one preallocated snapshot',
        stress_contract='After depth>=4, 32 CUDA sleeps of 2M cycles; test hook only, never performance')
    q['sha256'] = digest(q); freeze_json(folder/'queue.json', q)
    real_ids = {p['id'] for p in old['profile_prompts']}
    for phase in PHASES:
        stress = phase == 'stress'
        prompts = [p for p in old['correctness_prompts']
                   if p['id'] in real_ids or (not stress and p['id'].endswith('-2048'))]
        cfgs = configurations(stress)
        d = dict(queue_sha256=q['sha256'], phase=phase, configurations=cfgs, prompts=prompts,
            warmup=old['warmup'], repetitions=1, protocols=['fixed', 'natural'], output_cap=256,
            capacity=4096, kv_fraction=.10, profile_capture=False,
            artificial_delay=stress, performance_evidence=False)
        d['jobs'] = base.make_jobs(cfgs, 1, d['protocols'], len(prompts))
        d['jobs'].sort(key=lambda j: (d['protocols'].index(j['protocol']), j['config']['mode'] != 'ar'))
        d['sha256'] = digest(d); freeze_json(folder/phase/'definition.json', d)
    with zipfile.ZipFile(folder/'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(q['execution']['files']) | set(q['queue_code'])):
            archive.write(ROOT/name, name)
    return q


def install_stress_hook():
    # Spawn imports this entry point too; this remains outside production code.
    from ssd.engine.draft_runner import DraftRunner
    import torch
    original = DraftRunner._decode_tree_step
    def delayed(self, depth, *args, **kwargs):
        result = original(self, depth, *args, **kwargs)
        if self.config.publication_policy == 'overlap' and depth >= self.config.early_publish_tokens:
            for _ in range(32): torch.cuda._sleep(2_000_000)
        return result
    DraftRunner._decode_tree_step = delayed


def validate_receipt(receipt, q, d, output):
    if (receipt.get('definition_sha256') != d['sha256'] or receipt.get('returncode') != 0 or
            receipt.get('samples', 0) < 1 or receipt.get('error') or
            receipt.get('output_sha256') != file_digest(output) or
            receipt.get('shared_correctness') != q['allow_shared_correctness'] or
            receipt.get('performance_evidence') is not False or
            (receipt.get('external_processes') and not q['allow_shared_correctness'])):
        raise ValueError('Invalid or contaminated prefix-publication receipt')


def block_result(folder, job, d, q):
    for attempt in sorted((folder/job['id']).glob('attempt_*')):
        path = attempt/'receipt.json'
        if not path.exists(): continue
        receipt = read_json(path)
        if receipt.get('valid') is not True: continue
        validate_receipt(receipt, q, d, attempt/'result.json')
        result = read_json(attempt/'result.json')
        base.validate_result(result, job, d, q)
        return result
    return None


def run_block(args, d, j, q):
    base.check_runtime(q)
    uuids = [g['uuid'] for g in q['hardware']['gpus']]
    if not q['allow_shared_correctness']: require_idle(uuids)
    parent = args.outdir/d['phase']/j['id']; parent.mkdir(parents=True, exist_ok=True)
    attempt = parent/f'attempt_{len(list(parent.glob("attempt_*")))}'; attempt.mkdir()
    token = uuid.uuid4().hex
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(uuids if j['config']['mode'] == 'ssd' else uuids[:1]))
    env[TOKEN_ENV] = token
    env.pop('AO_SPAWN_EXECUTABLE', None)
    for key in list(env):
        if key.startswith('SSD_PROFILE'): env[key] = '0'
    if env.get('CUDA_INJECTION64_PATH') or env.get('NVTX_INJECTION64_PATH'):
        raise ValueError('Profiler injection is unsupported in this correctness gate')
    command = [sys.executable, '-O', str(Path(__file__).resolve()), '--worker',
        '--outdir', str(args.outdir), '--phase', d['phase'], '--job', j['id'], '--attempt', str(attempt)]
    if d['artificial_delay']: command.append('--stress-worker')
    process = None; error = None; samples = 0; foreign = {}; owned = []
    started = time.monotonic()
    try:
        with (attempt/'worker.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            while True:
                owned, others = partition(compute_processes(uuids), process.pid, token)
                for row in others: foreign[(row['pid'], row['uuid'])] = row
                samples += 1
                if others and not q['allow_shared_correctness']: break
                if process.poll() is not None: break
                if time.monotonic() - started > 1200:
                    error = 'Correctness worker exceeded 1200 seconds'; break
                time.sleep(1)
    except BaseException as exc:
        error = str(exc) or type(exc).__name__; raise
    finally:
        if process is not None: stop_worker(process)
        cleanup_tagged(owned, token)
        output = attempt/'result.json'
        valid = (process is not None and process.returncode == 0 and samples > 0 and not error and
                 (not foreign or q['allow_shared_correctness']))
        if valid:
            try: base.validate_result(read_json(output), j, d, q)
            except (ValueError, KeyError, OSError) as exc: valid = False; error = str(exc)
        atomic_json(attempt/'receipt.json', dict(valid=bool(valid), definition_sha256=d['sha256'],
            command=command, returncode=process.returncode if process else None, samples=samples,
            external_processes=list(foreign.values()), shared_correctness=q['allow_shared_correctness'],
            performance_evidence=False, error=error, run_token=token, at=now(),
            output_sha256=file_digest(output) if output.exists() else None))
    if not valid:
        if foreign and not q['allow_shared_correctness']:
            raise PendingWindow('Contaminated attempt invalidated; waiting for dedicated GPUs')
        raise RuntimeError(f'Prefix correctness worker failed: {attempt}/worker.log; {error}')


def verify_block(folder, j, d, q):
    if j['config']['mode'] == 'ar': return
    ar = next(b for b in d['jobs'] if b['protocol'] == j['protocol'] and b['config']['mode'] == 'ar')
    ref = block_result(folder, ar, d, q)
    refs = {(ar['protocol'], ar['round'], row['id']): row for row in ref['requests']}
    checks = base.exact_checks([block_result(folder, j, d, q)], refs)
    atomic_json(folder/f'{j["id"]}_agreement.json', checks)
    if not all(c['exact'] for c in checks): raise ValueError('Exact output mismatch; further blocks stopped')


def report(folder, d, q):
    original = base.block_result
    try:
        base.block_result = block_result
        summary = base.report_phase(folder, d, q)
    finally: base.block_result = original
    stats = {}
    for j in d['jobs']:
        if j['config']['policy'] != 'overlap': continue
        rows = block_result(folder, j, d, q)['requests']
        stats[j['id']] = dict(partial_prefixes=sum(r['publication_policy_stats'][8] for r in rows),
            snapshots_with_pending_tail=sum(r['publication_policy_stats'][9] for r in rows),
            responses_before_tail_done=sum(r['publication_policy_stats'][10] for r in rows),
            deferred_drain_s=sum(r['publication_policy_stats'][11] for r in rows))
    summary['overlap_coverage'] = stats
    summary['shared_correctness'] = q['allow_shared_correctness']
    summary['performance_evidence'] = False
    atomic_json(folder/'summary.json', summary)
    with (folder/'report.md').open('a') as stream:
        stream.write('\nCorrectness diagnostics only; shared GPU jobs are recorded in receipts.\n')
        stream.write('Artificial tail delay is enabled only in the stress phase.\n\n')
        stream.write('| Configuration | Partial prefixes | Pending tail at snapshot | Response before tail finished |\n')
        stream.write('|---|---:|---:|---:|\n')
        for label, row in stats.items():
            stream.write(f'| {label} | {row["partial_prefixes"]:.0f} | {row["snapshots_with_pending_tail"]:.0f} | {row["responses_before_tail_done"]:.0f} |\n')
    if d['phase'] == 'stress' and not all(stats[f'{p}_r0_overlap_k10']['responses_before_tail_done'] > 0
                                         for p in d['protocols']):
        raise ValueError('Stress outputs match, but actual response-before-tail coverage is missing')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--allow-shared-correctness', action='store_true')
    parser.add_argument('--watch', action='store_true'); parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--stress-worker', action='store_true')
    parser.add_argument('--phase', choices=PHASES); parser.add_argument('--job'); parser.add_argument('--attempt', type=Path)
    args = parser.parse_args(); args.outdir = args.outdir.resolve()
    base.code_files = code_files
    if args.worker:
        if (args.phase == 'stress') != args.stress_worker: raise ValueError('Stress hook setting differs')
        previous.worker(args); return 0
    signal.signal(signal.SIGTERM, interrupted)
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (ROOT/'results/baselines/eval1/.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q = read_json(args.outdir/'queue.json') if (args.outdir/'queue.json').exists() else prepare(args.outdir, args.allow_shared_correctness)
        if q['allow_shared_correctness'] != args.allow_shared_correctness: raise ValueError('GPU sharing policy changed')
        base.check_runtime(q)
        definitions = [base.load_phase(args.outdir/phase, q) for phase in PHASES]
        if args.dry_run:
            print([(d['phase'], len(d['jobs']), len(d['jobs'])*len(d['prompts'])) for d in definitions]); return 0
        validate_models(q['models'])
        try:
            for d in definitions:
                folder = args.outdir/d['phase']
                for j in d['jobs']:
                    while block_result(folder, j, d, q) is None:
                        try:
                            atomic_json(args.outdir/'status.json', dict(status='running', phase=d['phase'], job=j['id'], at=now()))
                            run_block(args, d, j, q)
                        except PendingWindow as exc:
                            atomic_json(args.outdir/'status.json', dict(status='pending_dedicated_gpus', reason=str(exc), at=now()))
                            if not args.watch: return 2
                            time.sleep(30)
                    verify_block(folder, j, d, q)
                report(folder, d, q)
            atomic_json(args.outdir/'status.json', dict(status='complete', performance_evidence=False, at=now()))
        except BaseException as exc:
            atomic_json(args.outdir/'status.json', dict(status='stopped', reason=str(exc), at=now())); raise
    return 0


# multiprocessing spawn executes this entrypoint before entering DraftRunner.
if __name__ in ('__main__', '__mp_main__') and '--stress-worker' in sys.argv:
    install_stress_hook()
if __name__ == '__main__': raise SystemExit(main())
