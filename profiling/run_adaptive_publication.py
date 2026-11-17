"""Versioned, dedicated-GPU validation of bounded/adaptive publication.

Reuses the receipt, process monitoring, pairing, and reporting machinery from
the completed overnight campaign. Never resumes or edits that campaign.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import zipfile

import run_overnight_extensions as base
from baseline_common import (ROOT, allocation, atomic_json, digest, execution_fingerprint,
    file_digest, freeze_json, gpu_inventory, read_json, software_fingerprint, validate_models)
from run_baselines import interrupted, now


def code_files():
    names = ('run_adaptive_publication.py', 'run_overnight_extensions.py',
             'run_publication_study.py', 'benchmark_shared_numerics.py')
    return {f'profiling/{name}': file_digest(ROOT/'profiling'/name) for name in names}


def configurations():
    values = [dict(label='ar', mode='ar', k=7, f=3, p=0, policy='uniform', window=2)]
    for k in (7, 10):
        policies = ('complete', 'bounded', 'adaptive') if k == 7 else ('complete', 'uniform', 'bounded', 'adaptive')
        for policy in policies:
            values.append(dict(label=f'{policy}_k{k}', mode='ssd', k=k, f=3,
                p=0 if policy == 'complete' else 4,
                policy='uniform' if policy == 'complete' else policy, window=2))
    return values


def prepare(folder):
    source = ROOT/'results/next_stage/overnight_extensions_v2_2026-11-17'
    if read_json(source/'status.json')['status'] != 'complete':
        raise ValueError('Prior queue must complete before creating this successor')
    old = read_json(source/'queue.json')
    validate_models(old['models'])
    q = dict(created_at=now(), execution=execution_fingerprint(), software=software_fingerprint(),
        queue_code=code_files(), hardware=gpu_inventory(), models=old['models'],
        reference_contract='shared-fixed16-v1; same target arithmetic; original AR separate',
        source=dict(path=str(source/'queue.json'), sha256=file_digest(source/'queue.json')),
        policy_contract='bounded-adaptive-v1; P4, queue<=2, EWMA alpha=.2, beta prior=.5/.5, gain margin=5%',
        scope='Engineering correctness and timing diagnostics; not held-out headline performance')
    q['sha256'] = digest(q)
    freeze_json(folder/'queue.json', q)
    # Three real prompts exercise rejection and natural EOS; one 2048-token
    # boundary per family exercises the longest declared input. No new selection.
    prompts = [p for p in old['correctness_prompts']
               if p['id'] in {b['id'] for b in old['profile_prompts']} or p['id'].endswith('-2048')]
    if len(prompts) != 6: raise ValueError('Expected six declared coverage prompts')
    cfgs = configurations()
    d = dict(queue_sha256=q['sha256'], phase='correctness', configurations=cfgs,
        prompts=prompts, warmup=old['warmup'], repetitions=1, protocols=['fixed', 'natural'],
        output_cap=256, capacity=4096, kv_fraction=.10, profile_capture=False)
    d['jobs'] = base.make_jobs(cfgs, 1, d['protocols'], len(prompts))
    # Run fresh AR first in each protocol so each following block can fail fast.
    d['jobs'].sort(key=lambda j: (d['protocols'].index(j['protocol']), j['config']['mode'] != 'ar'))
    d['sha256'] = digest(d)
    freeze_json(folder/'correctness/definition.json', d)
    with zipfile.ZipFile(folder/'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(q['execution']['files']) | set(q['queue_code'])):
            archive.write(ROOT/name, name)
    return q


def worker(args):
    import torch
    from ssd import LLM, SamplingParams
    q = read_json(args.outdir/'queue.json')
    d = base.load_phase(args.outdir/args.phase, q)
    base.check_runtime(q)
    j = next(j for j in d['jobs'] if j['id'] == args.job)
    c = j['config']; mode = c['mode']
    model = LLM(q['models']['target']['path'], draft=q['models']['draft']['path'],
        target_numerics='shared_fixed16', num_gpus=2 if mode == 'ssd' else 1,
        speculate=mode != 'ar', draft_async=mode == 'ssd', speculate_k=c['k'], async_fan_out=c['f'],
        jit_speculate=True, max_num_seqs=1, max_model_len=4096, enable_prefix_cache=False,
        gpu_memory_utilization=d['kv_fraction'], draft_gpu_memory_utilization=d['kv_fraction'],
        early_publish_tokens=c['p'], publication_policy=c['policy'], publication_max_inflight=c['window'],
        drain_draft_on_finish=mode == 'ssd')
    sampling = SamplingParams(temperature=0, max_new_tokens=256, ignore_eos=j['protocol'] == 'fixed')
    for p in d['warmup']:
        model.generate([p['token_ids']], [sampling], use_tqdm=False, stream_callback=lambda *a: None)
    result = dict(queue_sha256=q['sha256'], definition_sha256=d['sha256'], job=j, requests=[],
        hardware_allocation=allocation(q, mode), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        actual_dtype=str(model.config.hf_config.torch_dtype), cuda_graphs=True, prefix_cache=False,
        kv_fraction=d['kv_fraction'], profile_capture=False, software=software_fingerprint())
    for i in j['prompt_order']:
        torch.cuda.reset_peak_memory_stats()
        row = base.run_engine.measure_request(model, d['prompts'][i], sampling, synchronize=torch.cuda.synchronize)
        row['target_peak_memory'] = dict(allocated_bytes=torch.cuda.max_memory_allocated(),
                                        reserved_bytes=torch.cuda.max_memory_reserved())
        result['requests'].append(row)
        atomic_json(args.attempt/'result.json', result)
        print(f'{j["id"]}: {row["id"]} done', flush=True)
    model.exit(hard=True)


def verify_completed_block(folder, j, d, q):
    result = base.block_result(folder, j, d, q)
    if j['config']['mode'] == 'ar': return
    ar = next(b for b in d['jobs'] if b['protocol'] == j['protocol'] and b['config']['mode'] == 'ar')
    ref = base.block_result(folder, ar, d, q)
    refs = {(ar['protocol'], ar['round'], row['id']): row for row in ref['requests']}
    checks = base.exact_checks([result], refs)
    if not all(c['exact'] for c in checks):
        atomic_json(folder/f'{j["id"]}_failures.json', checks)
        raise ValueError('Token mismatch; stopped before scheduling more blocks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--phase', default='correctness', choices=['correctness'])
    parser.add_argument('--job'); parser.add_argument('--attempt', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args(); args.outdir = args.outdir.resolve()
    # Mutation is confined to this entrypoint process; importing this module is safe.
    base.__file__ = __file__; base.code_files = code_files
    if args.worker: worker(args); return 0
    signal.signal(signal.SIGTERM, interrupted)
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (ROOT/'results/baselines/eval1/.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q = read_json(args.outdir/'queue.json') if (args.outdir/'queue.json').exists() else prepare(args.outdir)
        base.check_runtime(q)
        d = base.load_phase(args.outdir/'correctness', q)
        if args.dry_run:
            print(f'{len(d["jobs"])} blocks / {len(d["jobs"])*len(d["prompts"])} scored requests; three warmups/block')
            return 0
        validate_models(q['models'])
        try:
            for j in d['jobs']:
                while base.block_result(args.outdir/'correctness', j, d, q) is None:
                    try:
                        atomic_json(args.outdir/'status.json', dict(status='running', job=j['id'], at=now()))
                        base.run_block(args, d, j, q)
                    except base.PendingWindow as exc:
                        atomic_json(args.outdir/'status.json', dict(status='pending_dedicated_gpus', reason=str(exc), at=now()))
                        if not args.watch: return 2
                        base.time.sleep(30)
                verify_completed_block(args.outdir/'correctness', j, d, q)
            summary = base.report_phase(args.outdir/'correctness', d, q)
            partial = {policy: sum(r['partial_handoffs'] for r in summary['statistics']
                       if r['config']['policy'] == policy) for policy in ('bounded', 'adaptive')}
            if not all(partial.values()):
                raise ValueError(f'Actual partial handoff coverage missing: {partial}')
            atomic_json(args.outdir/'status.json', dict(status='complete', at=now(), partial_handoffs=partial))
        except BaseException as exc:
            atomic_json(args.outdir/'status.json', dict(status='stopped', reason=str(exc), at=now()))
            raise
    return 0


if __name__ == '__main__': raise SystemExit(main())
