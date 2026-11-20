"""Dedicated-GPU publication tuning, frozen selection, and paired evaluation.

Source profiling/env.sh once. SSD timings are fresh; saved AR timing is explicitly historical.
Every measurement requires both GPUs dedicated; contamination invalidates a block.
"""
import argparse
import csv
import fcntl
import itertools
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import uuid
import zipfile

import saved_ar_reference as saved_ar
import run_adaptive_publication as engine_worker
import run_overnight_extensions as base
from analyze_baselines import paired_bootstrap, row_statistics, output_checks
from baseline_common import (ROOT, FAMILIES, EXCLUDED, atomic_json, compute_processes,
    digest, execution_fingerprint, file_digest, freeze_json, gpu_inventory,
    read_json, software_fingerprint, validate_manifest, validate_models)
from gpu_process_ownership import TOKEN_ENV, partition, cleanup_tagged
from run_baselines import PendingWindow, interrupted, now, require_idle, stop_worker

DEFAULT_OUT = ROOT/'results/next_stage/publication_performance_v2_2026-11-18'
GATE = ROOT/'results/next_stage/prefix_overlap_v1_2026-11-17'
WORKLOAD = ROOT/'results/next_stage/shared_uniform_v1_2026-11-17/baselines/manifest.json'
SPEC = dict(seed=0, tuning_repetitions=2, repetitions=5, output_cap=256,
    capacity=4096, kv_fraction=.70, profile_capture=False, prefix_cache=False,
    precision='bfloat16', batch_size=1, cuda_graphs=True, jit_misses=True,
    protocols=['fixed', 'natural'], tuning_protocol='fixed',
    complete_k=[4,7,10], publication_k=[7,10], publication_p=[2,4,7], overlap_windows=[1,2], f=3,
    bootstrap_samples=10000, monitor_seconds=1, dedicated_gpus=True,
    tie_rule='smaller K, then smaller P, then smaller queue bound',
    objective='lowest balanced mean full-request latency on tuning prompts only',
    test_scope='Reused frozen test prompts; not a newly unseen workload',ar_policy='saved_outputs_short_continuity_v2')


def code_files():
    names = ('run_publication_reuse_ar.py', 'saved_ar_reference.py', 'run_publication_campaign.py', 'run_adaptive_publication.py',
             'run_overnight_extensions.py', 'run_publication_study.py',
             'benchmark_shared_numerics.py', 'gpu_process_ownership.py')
    return {f'profiling/{name}': file_digest(ROOT/'profiling'/name) for name in names}


def config(method, k=7, p=0, window=1):
    label = 'ar' if method == 'ar' else f'{method}_k{k}'
    if method in ('uniform', 'overlap'): label += f'_p{p}'
    if method == 'overlap': label += f'_w{window}'
    return dict(label=label, method=method, mode='ar' if method == 'ar' else 'ssd',
        k=k, f=3, p=p, policy='overlap' if method == 'overlap' else 'uniform', window=window)


def tuning_configs():
    return ([config('complete', k) for k in SPEC['complete_k']] +
        [config('uniform', k, p) for k in SPEC['publication_k'] for p in SPEC['publication_p']] +
        [config('overlap', k, p, w) for k in SPEC['publication_k']
         for p in SPEC['publication_p'] for w in SPEC['overlap_windows']])


def validate_workload(splits):
    seen = set()
    for split, size in [('tune',8), ('heldout',25), ('warmup',1)]:
        rows = splits[split]
        if len(rows) != 3*size or {p['family'] for p in rows} != set(FAMILIES):
            raise ValueError('Wrong split size or family')
        for family in FAMILIES:
            if sum(p['family'] == family for p in rows) != size: raise ValueError('Unbalanced split')
        for p in rows:
            if p['id'] in seen or p['id'] in EXCLUDED: raise ValueError('Overlapping or excluded prompt')
            seen.add(p['id'])
            if not 1 <= len(p['token_ids']) <= 2048 or digest(p['token_ids']) != p['token_ids_sha256']:
                raise ValueError('Invalid input tokens')


def require_gate(folder):
    q = read_json(folder/'queue.json')
    if read_json(folder/'status.json')['status'] != 'complete' or q['execution'] != execution_fingerprint():
        raise ValueError('Complete correctness gate for the current engine is required')
    for name, sha in q['queue_code'].items():
        if file_digest(ROOT/name) != sha: raise ValueError('Correctness gate execution changed')
    evidence = {}
    for phase in ('correctness','stress'):
        path = folder/phase/'summary.json'; s = read_json(path)
        if not s['correctness_passed'] or not all(s['repeatability'].values()):
            raise ValueError('Prior correctness gate failed')
        if not sum(r['responses_before_tail_done'] for r in s['overlap_coverage'].values()):
            raise ValueError('Prior gate lacks actual response-before-tail evidence')
        evidence[str(path)] = file_digest(path)
    review = read_json(folder/'implementation_review.json')
    for relative, sha in review['valid_receipt_sha256'].items():
        path = folder/relative
        if file_digest(path) != sha: raise ValueError('Correctness receipt changed')
        receipt = read_json(path)
        if not receipt['valid'] or receipt['returncode'] != 0 or receipt.get('error'):
            raise ValueError('Incomplete correctness gate receipt')
        result = path.parent/'result.json'
        if file_digest(result) != receipt['output_sha256']: raise ValueError('Correctness output changed')
        evidence[str(path)] = sha; evidence[str(result)] = receipt['output_sha256']
    evidence[str(folder/'queue.json')] = file_digest(folder/'queue.json')
    evidence[str(folder/'status.json')] = file_digest(folder/'status.json')
    return evidence


def definition(q,phase,configs,selection_sha=None):
    continuity=phase=='continuity';tune=phase=='tune'
    prompts=q['continuity_prompts'] if continuity else q['splits']['tune' if tune else 'heldout']
    d=dict(queue_sha256=q['sha256'],phase=phase,configurations=configs,prompts=prompts,
        warmup=q['splits']['warmup'],repetitions=1 if continuity else 2 if tune else 5,
        protocols=['fixed'] if continuity or tune else SPEC['protocols'],kv_fraction=.70,
        profile_capture=False,output_cap=256,capacity=4096,selection_sha256=selection_sha)
    shuffled=list(configs);random.Random(0).shuffle(shuffled)
    orders=None if continuity or tune else random.Random(0).sample(list(itertools.permutations(configs)),5)
    jobs=[]
    for r in range(d['repetitions']):
        order=(shuffled if r==0 else list(reversed(shuffled))) if continuity or tune else orders[r]
        indices=random.Random(1000+r).sample(range(len(prompts)),len(prompts))
        for protocol in d['protocols'] if r%2==0 else list(reversed(d['protocols'])):
            jobs += [dict(id=f'{protocol}_r{r}_{c["label"]}',config=c,round=r,protocol=protocol,prompt_order=indices) for c in order]
    d['jobs']=jobs;d['sha256']=digest(d);return d


def prepare(folder):
    gate_evidence = require_gate(GATE)
    original = read_json(WORKLOAD); validate_manifest(original)
    validate_workload(original['splits']); validate_models(original['models'])
    q = dict(created_at=now(), execution=execution_fingerprint(), software=software_fingerprint(),
        queue_code=code_files(), hardware=gpu_inventory(), models=original['models'],
        splits=original['splits'], spec=SPEC, gate_evidence_sha256=gate_evidence,
        tuning_configs=tuning_configs(),
        continuity_prompts=[next(p for p in original['splits']['tune'] if p['family']==f) for f in FAMILIES],
        ar_reference=saved_ar.prepare_reference(WORKLOAD.parent),
        workload_source=dict(path=str(WORKLOAD), sha256=file_digest(WORKLOAD)),
        reference_contract='shared-fixed16-v1; saved validated AR plus short fresh continuity; original AR separate',
        profiler=False, shared_gpu_measurements=False,
        primary_comparisons=['overlap vs tuned complete SSD', 'overlap vs tuned original uniform publication'],
        selection='Fixed-length tuning only; same winners for secondary natural EOS',
        f_rationale='F=3 held constant from prior balanced baseline tuning; this campaign retunes K/P/window',
        anchor_rule='Also evaluate complete K7/F3 if the new complete winner has another K',
        timing_reuse='SSD measurements fresh; AR timing historical, not a contemporaneous arm',
        scope=SPEC['test_scope'])
    q['sha256'] = digest(q); freeze_json(folder/'queue.json', q)
    saved_ar.validate_reference(q['ar_reference'],models=q['models'],software=q['software'],gpus=q['hardware']['gpus'])
    freeze_json(folder/'continuity/definition.json',definition(q,'continuity',[config('ar')]))
    freeze_json(folder/'tune/definition.json', definition(q, 'tune', tuning_configs()))
    with zipfile.ZipFile(folder/'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(q['execution']['files']) | set(q['queue_code'])):
            archive.write(ROOT/name, name)
    write_protocol(folder, q)
    atomic_json(folder/'status.json', dict(status='prepared', at=now(), continuity_requests=3,tuning_requests=1008,
        heldout_requests_min=2250, heldout_requests_max=3000))
    return q


def check_runtime(q):
    if digest({k:v for k,v in q.items() if k != 'sha256'}) != q['sha256'] or q['spec'] != SPEC:
        raise ValueError('Frozen campaign specification changed')
    if q['execution'] != execution_fingerprint() or q['queue_code'] != code_files():
        raise ValueError('Execution changed; prepare a new versioned campaign')
    if q['software'] != software_fingerprint() or q['hardware']['gpus'] != gpu_inventory()['gpus']:
        raise ValueError('Software or GPU identities changed')
    saved_ar.validate_reference(q['ar_reference'],models=q['models'],software=q['software'],gpus=q['hardware']['gpus'])


def validate_sources(q):
    saved_ar.validate_reference(q['ar_reference'],models=q['models'],software=q['software'],gpus=q['hardware']['gpus'])
    source=q['workload_source']
    if file_digest(source['path']) != source['sha256']:
        raise ValueError('Frozen source workload changed')
    for path,sha in q['gate_evidence_sha256'].items():
        if file_digest(path) != sha: raise ValueError('Correctness gate evidence changed')


def load_definition(folder,q,phase):
    d=read_json(folder/phase/'definition.json')
    if phase=='continuity':expected=definition(q,phase,[config('ar')])
    elif phase=='tune':expected=definition(q,phase,tuning_configs())
    else:expected=definition(q,'heldout',evaluation_configs(validate_selection(folder,q)),file_digest(folder/'selection.json'))
    if d!=expected:raise ValueError('Phase definition changed')
    return d


def validate_continuity(folder,q):
    d=load_definition(folder,q,'continuity');blocks=available_results(folder/'continuity',d,q,True)
    checks=check_outputs(blocks,q,'continuity')
    if len(checks)!=3 or not all(v['exact'] for v in checks):raise ValueError('Saved AR continuity failed; fresh full AR required')
    atomic_json(folder/'continuity/summary.json',dict(correctness_passed=True,requests=3,checks=checks,performance_evidence=False))


def validate_result(result, j, d, q):
    base.validate_result(result, j, d, q)
    if result['software'] != q['software']: raise ValueError('Worker software changed')
    for row in result['requests']:
        first = row['first_delivery_s']
        if first is None or not math.isfinite(first) or not 0 <= first <= row['latency_s']:
            raise ValueError('Invalid first-delivery time')
        if any(type(t) is not int for t in row['output_token_ids']): raise ValueError('Invalid output IDs')
        if row['prompt_tokens'] > 2048: raise ValueError('Prompt exceeds frozen capacity')
        if row['publication_policy_stats'] is not None and j['config']['method'] != 'overlap':
            raise ValueError('Unexpected publication telemetry')
        if j['config']['method'] == 'overlap' and len(row['publication_policy_stats'] or []) != 12:
            raise ValueError('Missing overlap telemetry')


def validate_receipt(receipt, d, output):
    if (receipt.get('valid') is not True or receipt.get('definition_sha256') != d['sha256'] or
        receipt.get('returncode') != 0 or receipt.get('samples',0) < 1 or
        receipt.get('external_processes') or receipt.get('error') or
        receipt.get('dedicated_gpus') is not True or receipt.get('profiler') is not False or
        receipt.get('output_sha256') != file_digest(output)):
        raise ValueError('Invalid, incomplete, or contaminated campaign receipt')


def block_result(folder, j, d, q):
    for attempt in sorted((folder/j['id']).glob('attempt_*')):
        path = attempt/'receipt.json'
        if not path.exists(): continue
        receipt = read_json(path)
        if receipt.get('valid') is not True: continue
        validate_receipt(receipt, d, attempt/'result.json')
        result = read_json(attempt/'result.json'); validate_result(result, j, d, q)
        return result
    return None


def run_block(args, d, j, q):
    check_runtime(q)
    uuids = [g['uuid'] for g in q['hardware']['gpus']]; require_idle(uuids)
    parent = args.outdir/d['phase']/j['id']; parent.mkdir(parents=True, exist_ok=True)
    attempt = parent/f'attempt_{len(list(parent.glob("attempt_*")))}'; attempt.mkdir()
    token = uuid.uuid4().hex
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(uuids if j['config']['mode']=='ssd' else uuids[:1]))
    env[TOKEN_ENV] = token; env.pop('AO_SPAWN_EXECUTABLE',None)
    for key in list(env):
        if key.startswith('SSD_PROFILE'): env[key]='0'
    if env.get('CUDA_INJECTION64_PATH') or env.get('NVTX_INJECTION64_PATH'):
        raise ValueError('Profiler injection invalidates a performance campaign')
    command = [sys.executable, '-O', str(Path(__file__).resolve()), '--worker', '--outdir', str(args.outdir),
        '--phase', d['phase'], '--job', j['id'], '--attempt', str(attempt)]
    process=None; error=None; samples=0; foreign={}; owned=[]; started=time.monotonic()
    try:
        with (attempt/'worker.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                owned, others = partition(compute_processes(uuids), process.pid, token)
                for row in others: foreign[(row['pid'],row['uuid'])]=row
                samples+=1
                if others or process.poll() is not None: break
                if time.monotonic()-started > 1800:
                    error='Worker exceeded the declared 1800-second block budget'; break
                time.sleep(SPEC['monitor_seconds'])
    except BaseException as exc:
        error=str(exc) or type(exc).__name__; raise
    finally:
        if process is not None: stop_worker(process)
        cleanup_tagged(owned,token)
        output=attempt/'result.json'
        valid=process is not None and process.returncode==0 and samples>0 and not foreign and error is None
        if valid:
            try: validate_result(read_json(output),j,d,q)
            except (ValueError,KeyError,OSError,TypeError) as exc: valid=False;error=str(exc)
        atomic_json(attempt/'receipt.json',dict(valid=bool(valid),definition_sha256=d['sha256'],
            command=command,returncode=process.returncode if process else None,samples=samples,
            external_processes=list(foreign.values()),error=error,run_token=token,
            dedicated_gpus=True,profiler=False,at=now(),
            output_sha256=file_digest(output) if output.exists() else None))
    if not valid:
        if foreign: raise PendingWindow('Contaminated block invalidated; wait for both GPUs to be dedicated')
        raise RuntimeError(f'Campaign worker failed: {attempt}/worker.log; {error}')


def available_results(folder,d,q,complete=False):
    values=[block_result(folder,j,d,q) for j in d['jobs']]
    if complete and any(v is None for v in values): raise ValueError('All declared valid blocks are required')
    return [v for v in values if v is not None]


def check_outputs(results,q,phase):return saved_ar.check_outputs(results,q,phase)


def select(folder,q):
    validate_continuity(folder,q)
    d=load_definition(folder,q,'tune')
    results=available_results(folder/'tune',d,q,complete=True)
    checks=check_outputs(results,q,d['phase'])
    if len(checks)!=1008 or not all(c['exact'] for c in checks):
        raise ValueError('Tuning exactness failed; do not select a winner')
    scores=[]
    for c in d['configurations']:
        rows=[r for b in results if b['job']['config']==c for r in b['requests']]
        if len(rows)!=48: raise ValueError('Selection requires both complete tuning repetitions')
        scores.append(dict(config=c,requests=len(rows),mean_latency_s=sum(r['latency_s'] for r in rows)/len(rows)))
    winners={method:min((s for s in scores if s['config']['method']==method),
        key=lambda s:(s['mean_latency_s'],s['config']['k'],s['config']['p'],s['config']['window']))['config']
        for method in ('complete','uniform','overlap')}
    evidence={str(path.relative_to(folder)):file_digest(path) for path in (folder/'tune').glob('*/attempt_*/receipt.json')
              if read_json(path).get('valid') is True}
    selection=dict(queue_sha256=q['sha256'],tuning_definition_sha256=d['sha256'],winners=winners,scores=scores,
        evidence_sha256=evidence,selection_data='Tuning requests only; held-out results never read',
        objective=SPEC['objective'],tie_rule=SPEC['tie_rule'])
    selection['sha256']=digest(selection)
    return selection


def validate_selection(folder,q):
    value=read_json(folder/'selection.json')
    if value != select(folder,q): raise ValueError('Frozen selection or its tuning evidence changed')
    return value


def evaluation_configs(selection):
    values=[selection['winners'][method] for method in ('complete','uniform','overlap')]
    if selection['winners']['complete']['k'] != 7:
        anchor=config('complete',7);anchor.update(label='anchor_k7_f3',method='anchor')
        values.append(anchor)
    return values


def aggregate(rows):
    pubs=[v for r in rows for v in r['publication_records']]
    states=[r['publication_policy_stats'] for r in rows if r['publication_policy_stats'] is not None]
    lengths=[v['valid_length'] for v in pubs]
    return dict(**row_statistics(rows),mean_target_wait_s=sum(r['draft_wait_s'] for r in rows)/len(rows),
        mean_final_drain_s=sum(r['final_drain_s'] for r in rows)/len(rows),
        mean_publication_wall_s=sum(v['publication_s'] for v in pubs)/len(pubs) if pubs else None,
        mean_checkpoint_drain_s=sum(v['drain_s'] for v in pubs)/len(pubs) if pubs else None,
        mean_deferred_tail_drain_s=sum(v[11] for v in states)/len(rows) if states else 0,
        responses_before_tail_done=sum(v[10] for v in states),
        ready_length_histogram={str(n):lengths.count(n) for n in sorted(set(lengths))},
        accepted_draft_tokens=sum(r['accepted_draft_tokens'] for r in rows),
        verified_draft_tokens=sum(r['verified_draft_tokens'] for r in rows),
        tokens_per_verification=sum(sum(r['accepted_suffix_lens_with_recovery']) for r in rows)/
            max(1,sum(len(r['accepted_suffix_lens_with_recovery']) for r in rows)),
        max_target_allocated_bytes=max(r['target_peak_memory']['allocated_bytes'] for r in rows),
        max_target_reserved_bytes=max(r['target_peak_memory']['reserved_bytes'] for r in rows),
        max_draft_allocated_bytes=max([v['allocated_bytes'] for r in rows for v in r['draft_peak_memory']] or [0]),
        max_draft_reserved_bytes=max([v['reserved_bytes'] for r in rows for v in r['draft_peak_memory']] or [0]),
        output_lengths=[r['output_tokens'] for r in rows])


def tuning_report(folder,q,selection):
    d=load_definition(folder,q,'tune');results=available_results(folder/'tune',d,q,True)
    checks=check_outputs(results,q,d['phase'])
    atomic_json(folder/'tune/summary.json',dict(queue_sha256=q['sha256'],valid_blocks=len(results),
        requests=sum(len(b['requests']) for b in results),exact_comparisons=len(checks),
        exact_matches=sum(c['exact'] for c in checks),scores=selection['scores'],winners=selection['winners']))
    lines=['# Publication tuning','', 'Dedicated GPUs; 24 balanced tuning prompts × two repetitions per setting.',
        'Fixed 256 output tokens, EOS ignored. Selection uses tuning only. F=3 throughout.','',
        '| Configuration | Mean request latency (s) | Winner |','|---|---:|---|']
    for score in selection['scores']:
        c=score['config'];wins=c in selection['winners'].values()
        lines.append(f'| {c["label"]} | {score["mean_latency_s"]:.6f} | {"yes" if wins else ""} |')
    (folder/'tune/report.md').write_text('\n'.join(lines)+'\n')


def final_report(folder,q):
    selected=validate_selection(folder,q);d=load_definition(folder,q,'heldout')
    blocks=available_results(folder/'heldout',d,q,True)
    return saved_ar.report_campaign(folder,q,selected,d,blocks,aggregate,('complete','uniform'),1008,3,1)


def write_final_markdown(folder,q,summary):
    lines=['# Dedicated-GPU publication performance campaign','',
        'Fresh measurements with BF16, batch one, CUDA graphs, full prefills, 0.70 target/draft KV fractions, '
        'and final draft draining included. No profiler. Every accepted block passed monitoring of both GPUs.','',
        q['scope']+'. Configurations were selected on 24 tuning prompts; the same 75 test prompts '
        'were evaluated five times per method and protocol. Natural EOS is secondary and was not retuned.','',
        f'Exactness: {summary["exact_speculative_matches"]}/{summary["exact_speculative_comparisons"]} paired speculative requests matched fresh shared AR. '
        'Original historical AR is a separate numerical reference.','',
        '## Frozen winners','']
    if not summary['correctness_passed']:
        lines[2:2]=['**Exactness or repeatability failed. These results do not establish correctness-preserving acceleration.**','']
    lines += [f'- {method}: `{c["label"]}`, F=3.' for method,c in summary['winners'].items()]
    for protocol in ('fixed','natural'):
        lines += ['',f'## {protocol} output protocol','',
            '| Method | GPUs | Mean s | p50 s | p95 s | tok/s | First delivery ms | Wait s | Tokens/verify |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in summary['results']:
            if r['protocol']!=protocol or r['family']!='all':continue
            s=r['statistics']
            lines.append(f'| {r["config"]["label"]} | {r["gpu_count"]} | {s["mean_latency_s"]:.4f} | '
                f'{s["p50_latency_s"]:.4f} | {s["p95_latency_s"]:.4f} | {s["committed_tok_per_s"]:.2f} | '
                f'{s["mean_first_delivery_s"]*1000:.2f} | {s["mean_target_wait_s"]:.4f} | {s["tokens_per_verification"]:.3f} |')
        lines += ['', '| Overlap comparison | Family | Speed ratio (reference / overlap) | Paired 95% CI |',
                  '|---|---|---:|---|']
        for r in summary['results']:
            if r['protocol']!=protocol or r['method']!='overlap':continue
            for ref in ('complete','uniform'):
                c=r['comparison_to'][ref];lo,hi=c['speedup_ci95']
                lines.append(f'| {ref} | {r["family"]} | {c["speedup"]:.3f}× | {lo:.3f}–{hi:.3f} |')
        lines += ['', '| Method | Acceptance | Cache hit rate | Mean output tokens | Target / draft peak allocated GiB | Final drain ms | Responses before tail |',
            '|---|---:|---:|---:|---|---:|---:|']
        for r in summary['results']:
            if r['protocol']!=protocol or r['family']!='all':continue
            s=r['statistics'];a=s['draft_acceptance_rate'];h=s['mean_cache_hit_rate']
            acceptance=f'{a:.3f}' if a is not None else '—';hit=f'{h:.3f}' if h is not None else '—'
            lines.append(f'| {r["config"]["label"]} | {acceptance} | {hit} | {s["mean_output_tokens"]:.2f} | '
                f'{s["max_target_allocated_bytes"]/2**30:.2f} / {s["max_draft_allocated_bytes"]/2**30:.2f} | '
                f'{s["mean_final_drain_s"]*1000:.3f} | {s["responses_before_tail_done"]:.0f} |')
    primary=next(r for r in summary['results'] if r['protocol']=='fixed' and r['family']=='all' and r['method']=='overlap')
    c=primary['comparison_to']['complete'];lo,hi=c['speedup_ci95']
    finding=('lower latency supported by the paired interval' if lo>1 else
             'higher latency supported by the paired interval' if hi<1 else 'the paired interval includes no latency difference')
    lines += ['', '## Interpretation','',
        f'Against tuned complete SSD, the fixed-length overlap speed ratio is {c["speedup"]:.3f}×: {finding}. '
        'The original-publication comparison above is also required to assess incremental value. '
        'Overlap counts alone are not evidence of a speedup.','',
        'Intervals use 10,000 seeded paired bootstrap samples of rounds and prompts within families. '
        'They are per-comparison intervals, without a familywise multiple-comparison correction. '
        'AR uses one GPU; all SSD methods use two. Target verification retains its padded K+1 graph. '
        'Full request latency includes both prefills and final draining. Publication wall-time fields '
        'can include outstanding GPU completion and must not be interpreted as pure bookkeeping overhead.','',
        'Detailed memory, cache, acceptance, output lengths, first divergences, repeatability, '
        'and per-family regressions are recorded in summary.json. Raw block outputs and receipts are retained.']
    (folder/'report.md').write_text('\n'.join(lines)+'\n')


def plots(folder,selection,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,method in zip(axes,('complete','uniform','overlap')):
        scores=[s for s in selection['scores'] if s['config']['method']==method]
        ax.bar([s['config']['label'] for s in scores],[s['mean_latency_s'] for s in scores])
        ax.tick_params(axis='x',labelrotation=90);ax.set(title=method,ylabel='Mean tuning request latency (s)')
    fig.tight_layout();fig.savefig(folder/'tuning.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,protocol in zip(axes,('fixed','natural')):
        rows=[r for r in results if r['protocol']==protocol and r['family']=='all']
        means=[r['statistics']['mean_latency_s'] for r in rows]
        ci=[r['comparison_to']['complete']['mean_latency_ci95_s'] for r in rows]
        ax.bar([r['method'] for r in rows],means,
            yerr=[[max(0,m-a[0]) for m,a in zip(means,ci)],[max(0,a[1]-m) for m,a in zip(means,ci)]],capsize=4)
        ax.set(title=protocol,ylabel='Full-request latency (s), paired 95% CI')
    fig.tight_layout();fig.savefig(folder/'latency.png',dpi=160);plt.close(fig)


def write_protocol(folder,q):
    text=['# Publication comparison, reused AR v2','',
        'User-authorized amendment: remove repeated full AR timing arms. Historical AR outputs and receipts are hash-pinned. '
        'Target arithmetic and AR graph functions, weights/tokenizer, software and GPU identities must match.',
        'First run one short AR continuity block on three tuning prompts, one per family, with three warmups and fixed256 output. '
        'It must match saved AR exactly before tuning starts. Its timing is diagnostic and not an AR performance arm.',
        'All 21 SSD configurations remain: complete K4/7/10; uniform K7/10 × P2/4/7; '
        'overlap same K/P × queue bounds1/2. F3. Two repetitions on 24 balanced tuning prompts = 42 blocks / 1008 requests.',
        'Freeze complete/uniform/overlap winners using tuning alone. Held-out results are not used for selection. '
        'Evaluate the three SSD winners on75 prompts × five rounds × two protocols =30 blocks /2250 requests. '
        'Add completeK7 if needed:10 blocks /750 requests. Natural EOS reuses fixed-length-selected winners.',
        'Primary latency comparisons are fresh SSD versus SSD, paired by round/prompt. Saved AR timing appears only as '
        'historical context; historical ratios use independent old/new round resampling with paired prompts, not contemporaneous-run claims.',
        'AR continuity uses GPU0; SSD target GPU0 / draft GPU1. Both GPUs must be dedicated. '
        'Same global lock, UUIDs, contamination handling and block resume. BF16, batch1, graphs, JIT misses, '
        'capacity4096, full prefills,0.70 KV fractions, no prefix lookup, no profiler, final drain included.',
        'Total new work:73–83 blocks /3261–4011 scored requests including three continuity checks. '
        'Previously completed AR blocks are references only and cannot count as newly executed blocks.']
    (folder/'protocol.md').write_text('\n\n'.join(text)+'\n')


def run_phase(args,q,d):
    folder=args.outdir/d['phase']
    for j in d['jobs']:
        while block_result(folder,j,d,q) is None:
            try:
                atomic_json(args.outdir/'status.json',dict(status='running',phase=d['phase'],job=j['id'],at=now()))
                run_block(args,d,j,q)
            except PendingWindow as exc:
                atomic_json(args.outdir/'status.json',dict(status='pending_dedicated_gpus',phase=d['phase'],job=j['id'],reason=str(exc),at=now()))
                if not args.watch: return False
                time.sleep(30)
        results=available_results(folder,d,q)
        checks=check_outputs(results,q,d['phase'])
        if not all(c['exact'] for c in checks):
            atomic_json(folder/'output_failures.json',[c for c in checks if not c['exact']])
            raise ValueError('Token mismatch against fresh AR; no further performance blocks scheduled')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--phase',choices=['all','prepare','continuity','tune','select','heldout','report'],default='all')
    parser.add_argument('--watch',action='store_true');parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--job');parser.add_argument('--attempt',type=Path)
    args=parser.parse_args();args.outdir=args.outdir.resolve()
    # The existing worker imports this module as its __mp_main__ entrypoint too.
    # Only this process installs the campaign-specific validator functions.
    if args.worker:
        q=read_json(args.outdir/'queue.json');check_runtime(q)
        if args.phase not in ('continuity','tune','heldout'): raise ValueError('Invalid worker phase')
        load_definition(args.outdir,q,args.phase)
        base.check_runtime=check_runtime
        engine_worker.worker(args);return 0
    signal.signal(signal.SIGTERM,interrupted)
    args.outdir.mkdir(parents=True,exist_ok=True)
    with (ROOT/'results/baselines/eval1/.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fresh=not (args.outdir/'queue.json').exists()
        q=prepare(args.outdir) if fresh else read_json(args.outdir/'queue.json')
        check_runtime(q);validate_workload(q['splits'])
        if args.phase=='prepare' or args.dry_run:
            print('Frozen: 3 AR checks, 1008 SSD tuning, 2250–3000 SSD held-out requests. See '+str(args.outdir/'protocol.md'))
            return 0
        validate_sources(q)
        if not fresh: validate_models(q['models'])
        try:
            if args.phase in ('all','continuity'):
                if not run_phase(args,q,load_definition(args.outdir,q,'continuity')):return 2
            validate_continuity(args.outdir,q)
            if args.phase=='continuity':return 0
            if args.phase in ('all','tune'):
                if not run_phase(args,q,load_definition(args.outdir,q,'tune')):return 2
                if args.phase=='tune':
                    atomic_json(args.outdir/'status.json',dict(status='tuning_complete',at=now()));return 0
            if args.phase in ('all','select'):
                chosen=select(args.outdir,q);freeze_json(args.outdir/'selection.json',chosen)
                tuning_report(args.outdir,q,chosen)
                held=definition(q,'heldout',evaluation_configs(chosen),file_digest(args.outdir/'selection.json'))
                freeze_json(args.outdir/'heldout/definition.json',held)
                if args.phase=='select':
                    atomic_json(args.outdir/'status.json',dict(status='selection_frozen',at=now(),winners=chosen['winners']));return 0
            if args.phase in ('all','heldout'):
                if not run_phase(args,q,load_definition(args.outdir,q,'heldout')):return 2
            if args.phase in ('all','report'):
                summary=final_report(args.outdir,q)
                atomic_json(args.outdir/'status.json',dict(status='complete',at=now(),scored_requests=summary['scored_requests'],report=str(args.outdir/'report.md')))
        except BaseException as exc:
            atomic_json(args.outdir/'status.json',dict(status='stopped',reason=str(exc),at=now()));raise
    return 0


if __name__=='__main__':raise SystemExit(main())
