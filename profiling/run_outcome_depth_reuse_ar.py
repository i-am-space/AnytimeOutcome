"""Versioned unequal-outcome-depth gate, tuning and paired evaluation.

Source profiling/env.sh once. This campaign never imports old timing results.
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

SOURCE_ROOT = ROOT.parents[1]
DEFAULT_OUT = SOURCE_ROOT/'results/next_stage/outcome_depth_v2_2026-11-18'
DEFAULT_AFTER = SOURCE_ROOT/'results/next_stage/context_length_v2_2026-11-18'
GATE = SOURCE_ROOT/'results/next_stage/prefix_overlap_v1_2026-11-17'
WORKLOAD = SOURCE_ROOT/'results/next_stage/shared_uniform_v1_2026-11-17/baselines/manifest.json'
SPEC = dict(seed=0,tuning_repetitions=2,repetitions=5,output_cap=256,capacity=4096,
    kv_fraction=.70,profile_capture=False,prefix_cache=False,precision='bfloat16',batch_size=1,
    cuda_graphs=True,jit_misses=True,protocols=['fixed','natural'],tuning_protocol='fixed',
    complete_k=[4,7,10],outcome_k=[7,10],f=3,p=4,window=2,
    depths={'7':[7,4,4],'10':[10,7,4]},bootstrap_samples=10000,monitor_seconds=1,
    dedicated_gpus=True,tie_rule='smaller K, then P, then queue bound',
    objective='lowest balanced mean full-request latency on tuning prompts only',
    test_scope='Reused frozen test prompts, disjoint from tuning; not newly unseen',
    allocation='Static recovery-rank budgets, identical across acceptance positions; no learned scheduler',
    ar_policy='saved_outputs_short_gate_v2',
    comparisons='Tuned complete and overlap controls; equal-work uniform and reversed-rank ablations')


def code_files():
    names=('run_outcome_depth_reuse_ar.py','run_outcome_depth_campaign.py','run_publication_campaign.py',
        'run_publication_reuse_ar.py','saved_ar_reference.py','run_context_length_sweep_v2.py','run_adaptive_publication.py',
        'run_overnight_extensions.py','run_publication_study.py','benchmark_shared_numerics.py','gpu_process_ownership.py',
        'run_context_length_sweep.py','analyze_context_sweep.py')
    return {f'profiling/{n}':file_digest(ROOT/'profiling'/n) for n in names}


def config(method,k=7,p=0,window=2):
    depths=None
    if method in ('ranked','ranked_overlap','reverse','budget'):
        depths=list(SPEC['depths'][str(k)])
        if method=='reverse':depths.reverse()
        if method=='budget':depths=[sum(depths)//3]*3
    overlap=method in ('overlap','ranked_overlap')
    return dict(label='ar' if method=='ar' else f'{method}_k{k}',method=method,
        mode='ar' if method=='ar' else 'ssd',k=k,f=3,p=4 if overlap else 0,
        policy='overlap' if overlap else 'uniform',window=2,outcome_depths=depths)


def tuning_configs():
    return [config('complete',k) for k in SPEC['complete_k']]+[
        config(method,k) for method in ('overlap','ranked','ranked_overlap','reverse','budget') for k in SPEC['outcome_k']]


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


def validate_gate(folder,q):
    d=load_definition(folder,q,'gate');blocks=available_results(folder/'gate',d,q,True)
    checks=check_outputs(blocks,q,'gate')
    if len(checks)!=len(tuning_configs())*len(q['gate_prompts'])*2 or not all(c['exact'] for c in checks):
        raise ValueError('New engine correctness gate failed')
    stats=[r['outcome_depth_stats'] for b in blocks if b['job']['config']['outcome_depths'] for r in b['requests']]
    if not stats or not any(v['computed_query_rows']<v['dense_query_rows'] for v in stats):
        raise ValueError('Gate lacks actual reduced-row execution')
    if not any(v['short_cache_responses']>0 for v in stats):
        raise ValueError('Gate lacks actual short cached outcomes')
    result=dict(status='complete',exact_comparisons=len(checks),exact_matches=sum(c['exact'] for c in checks),
        valid_blocks=len(blocks),requests=sum(len(b['requests']) for b in blocks),
        computed_query_rows=sum(v['computed_query_rows'] for v in stats),
        dense_query_rows=sum(v['dense_query_rows'] for v in stats),reference=q['reference_contract'])
    atomic_json(folder/'gate/summary.json',result)
    return result


def definition(q,phase,configs,selection_sha=None):
    gate=phase=='gate';tune=phase=='tune'
    d=dict(queue_sha256=q['sha256'],phase=phase,configurations=configs,
        prompts=q['gate_prompts'] if gate else q['splits']['tune' if tune else 'heldout'],
        warmup=q['splits']['warmup'],repetitions=1 if gate else 2 if tune else 5,
        protocols=['fixed'] if tune else SPEC['protocols'],kv_fraction=.10 if gate else .70,
        profile_capture=False,output_cap=256,capacity=4096,selection_sha256=selection_sha)
    others=[c for c in configs if c['method']!='ar'];random.Random(0).shuffle(others)
    ar=[c for c in configs if c['method']=='ar']
    orders=random.Random(0).sample(list(itertools.permutations(configs)),5) if not gate and not tune else None
    jobs=[]
    for r in range(d['repetitions']):
        order=ar+(others if r==0 else list(reversed(others))) if gate or tune else orders[r]
        indices=random.Random(1000+r).sample(range(len(d['prompts'])),len(d['prompts']))
        for protocol in d['protocols'] if r%2==0 else list(reversed(d['protocols'])):
            jobs += [dict(id=f'{protocol}_r{r}_{c["label"]}',config=c,round=r,protocol=protocol,prompt_order=indices) for c in order]
    d['jobs']=jobs;d['sha256']=digest(d);return d


def prepare(folder,after=DEFAULT_AFTER):
    original=read_json(WORKLOAD);validate_manifest(original);validate_workload(original['splits'])
    prior=read_json(SOURCE_ROOT/'results/next_stage/overnight_extensions_v2_2026-11-17/queue.json')
    real_ids={p['id'] for p in prior['profile_prompts']}
    gate_prompts=[p for p in prior['correctness_prompts'] if p['id'] in real_ids or p['id'].endswith('-2048')]
    if len(gate_prompts)!=6:raise ValueError('Expected six real and boundary coverage inputs')
    dependencies=[after,SOURCE_ROOT/'results/next_stage/publication_performance_v2_2026-11-18']
    q=dict(created_at=now(),execution=execution_fingerprint(),software=software_fingerprint(),
        queue_code=code_files(),hardware=gpu_inventory(),models=original['models'],
        splits=original['splits'],gate_prompts=gate_prompts,spec=SPEC,tuning_configs=tuning_configs(),
        ar_reference=saved_ar.prepare_reference(WORKLOAD.parent,SOURCE_ROOT),
        after=[dict(path=str(v.resolve()),queue_sha256=file_digest(v/'queue.json')) for v in dependencies],
        source_root=str(SOURCE_ROOT),workload_source=dict(path=str(WORKLOAD),sha256=file_digest(WORKLOAD)),
        reference_contract='shared-fixed16-v1; saved validated AR plus short fresh gate; original AR separate',
        scope=SPEC['test_scope'],selection='Fixed-output tuning only; natural EOS is secondary',
        timing_reuse='SSD results fresh; AR timing historical. Only small fresh AR correctness blocks remain.',
        limitations=['Rank is a proxy for recovery likelihood, not a calibrated full-outcome probability.',
            'Budgets are static across acceptance positions; learned/context-sensitive allocation remains future work.',
            'KV reservation and target K+1 verification size remain unchanged. Fewer draft rows need not reduce latency.',
            'Overlap controls retune K with P4/window2 fixed; this is not an exhaustive P/window search.',
            'Equal-work controls match full-tree query rows, not launch count, memory traffic or per-kernel cost.'])
    q['sha256']=digest(q);freeze_json(folder/'queue.json',q)
    for phase in ('gate','tune'):
        configs=([config('ar')]+tuning_configs()) if phase=='gate' else tuning_configs()
        freeze_json(folder/phase/'definition.json',definition(q,phase,configs))
    with zipfile.ZipFile(folder/'source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(q['execution']['files'])|set(q['queue_code'])|{
                'tests/test_outcome_depth.py','tests/test_outcome_depth_gpu.py',
                'tests/test_outcome_depth_campaign.py','docs/outcome_depth.md'}):archive.write(ROOT/name,name)
    write_protocol(folder,q)
    atomic_json(folder/'status.json',dict(status='prepared',at=now(),gate_requests=168,tuning_requests=624,heldout_requests=4500))
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
    if file_digest(q['workload_source']['path'])!=q['workload_source']['sha256']:raise ValueError('Workload changed')
    for dep in q['after']:
        folder=Path(dep['path'])
        if file_digest(folder/'queue.json')!=dep['queue_sha256']:raise ValueError('Predecessor queue changed')
        old=read_json(folder/'queue.json')
        for name,sha in (old['execution']['files']|old['queue_code']).items():
            if file_digest(Path(q['source_root'])/name)!=sha:raise ValueError('Frozen predecessor code changed')


def predecessor_ready(q):
    return all((Path(v['path'])/'status.json').exists() and
        read_json(Path(v['path'])/'status.json').get('status')=='complete' for v in q['after'])


def validate_dependencies(folder,q):
    if not predecessor_ready(q):raise ValueError('Both earlier queues must complete successfully')
    validate_sources(q)
    import run_context_length_sweep_v2 as sweep
    import run_publication_reuse_ar as publication
    evidence={}
    for dep in q['after']:
        path=Path(dep['path']);old=read_json(path/'queue.json')
        is_sweep='input_lengths' in old['spec']
        for phase in (('measure','profiles') if is_sweep else ('continuity','tune','heldout')):
            d=(sweep.load_definition(path,old,phase) if is_sweep else publication.load_definition(path,old,phase))
            blocks=[(sweep if is_sweep else publication).block_result(path/phase,j,d,old) for j in d['jobs']]
            if any(b is None for b in blocks):raise ValueError('Predecessor valid blocks missing')
            if is_sweep:
                for b in blocks:
                    if b['job']['config']['method']!='ar':sweep.verify_outputs(b,sweep.reference_rows(path,old,b['job']['round']))
            elif not all(c['exact'] for c in publication.check_outputs(blocks,old,phase)):raise ValueError('Predecessor exactness failed')
            for receipt in (path/phase).glob('*/attempt_*/receipt.json'):
                if read_json(receipt).get('valid') is True:
                    evidence[str(receipt)]=file_digest(receipt)
                    evidence[str(receipt.parent/'result.json')]=file_digest(receipt.parent/'result.json')
        summary=read_json(path/'summary.json')
        if summary.get('status')!='complete' or not (path/'report.md').exists():raise ValueError('Predecessor report incomplete')
        if not is_sweep and summary.get('correctness_passed') is not True:raise ValueError('Predecessor correctness failed')
        for name in ('queue.json','summary.json','report.md'):evidence[str(path/name)]=file_digest(path/name)
    freeze_json(folder/'dependency.json',dict(artifacts_sha256=evidence,sha256=digest(evidence)))


def load_definition(folder,q,phase):
    d=read_json(folder/phase/'definition.json')
    if phase in ('gate','tune'):expected=definition(q,phase,([config('ar')]+tuning_configs()) if phase=='gate' else tuning_configs())
    else:expected=definition(q,'heldout',evaluation_configs(validate_selection(folder,q)),file_digest(folder/'selection.json'))
    if d!=expected:raise ValueError('Frozen phase definition changed')
    return d


def validate_result(result,j,d,q):
    base.validate_result(result,j,d,q)
    if result['software']!=q['software'] or result.get('outcome_depths')!=j['config']['outcome_depths']:
        raise ValueError('Worker software or outcome depths changed')
    for row in result['requests']:
        first=row['first_delivery_s']
        if first is None or not math.isfinite(first) or not 0<=first<=row['latency_s']:raise ValueError('Invalid first delivery')
        if any(type(t) is not int for t in row['output_token_ids']):raise ValueError('Invalid tokens')
        states=row['publication_policy_stats']
        if j['config']['policy']=='overlap':
            if len(states or [])!=12:raise ValueError('Missing overlap telemetry')
        elif states is not None:raise ValueError('Unexpected publication telemetry')
        stats=row.get('outcome_depth_stats')
        if j['config']['outcome_depths']:
            from ssd.engine.outcome_depth import STAT_NAMES
            if not stats or set(stats)!=set(STAT_NAMES) or any(not math.isfinite(v) or v<0 for v in stats.values()):
                raise ValueError('Missing or invalid outcome-depth execution counters')
            if stats['computed_query_rows']>stats['dense_query_rows']:raise ValueError('Invalid row accounting')
        elif stats is not None:raise ValueError('Unexpected outcome-depth execution')


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
    d=load_definition(folder,q,'tune');results=available_results(folder/'tune',d,q,True);checks=check_outputs(results,q,d['phase'])
    if len(checks)!=len(tuning_configs())*48 or not all(c['exact'] for c in checks):raise ValueError('Tuning exactness failed')
    scores=[]
    for c in d['configurations']:
        rows=[r for b in results if b['job']['config']==c for r in b['requests']]
        if len(rows)!=48:raise ValueError('Both tuning repetitions required')
        scores.append(dict(config=c,requests=len(rows),mean_latency_s=sum(r['latency_s'] for r in rows)/len(rows)))
    winners={method:min((s for s in scores if s['config']['method']==method),
        key=lambda s:(s['mean_latency_s'],s['config']['k'],s['config']['p'],s['config']['window']))['config']
        for method in ('complete','overlap','ranked','ranked_overlap')}
    evidence={str(p.relative_to(folder)):file_digest(p) for p in (folder/'tune').glob('*/attempt_*/receipt.json') if read_json(p).get('valid') is True}
    value=dict(queue_sha256=q['sha256'],tuning_definition_sha256=d['sha256'],winners=winners,scores=scores,
        evidence_sha256=evidence,selection_data='Tuning only; held-out results never read',objective=SPEC['objective'],
        tie_rule=SPEC['tie_rule'],ablations='Budget and reverse controls use the selected ranked K, without separate held-out selection')
    value['sha256']=digest(value);return value


def validate_selection(folder,q):
    value=read_json(folder/'selection.json')
    if value != select(folder,q): raise ValueError('Frozen selection or its tuning evidence changed')
    return value


def evaluation_configs(selection):
    return [selection['winners'][m] for m in ('complete','overlap','ranked','ranked_overlap')]+[
        config(m,selection['winners']['ranked']['k']) for m in ('budget','reverse')]


def base_aggregate(rows):
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


def aggregate(rows):
    result=base_aggregate(rows)
    from ssd.engine.outcome_depth import STAT_NAMES
    states=[r['outcome_depth_stats'] for r in rows if r.get('outcome_depth_stats')]
    result['outcome_depth_stats']={name:sum(r[name] for r in states) for name in STAT_NAMES}
    v=result['outcome_depth_stats']
    result['draft_query_row_reduction']=1-v['computed_query_rows']/v['dense_query_rows'] if v['dense_query_rows'] else None
    return result


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
    return saved_ar.report_campaign(folder,q,selected,d,blocks,aggregate,('complete','overlap','budget','reverse'),624,168,28)


def write_final_markdown(folder,q,summary):
    lines=['# Unequal continuation-depth experiment','',
        'Dedicated GPUs, batch one, BF16, graphs, full prefills and final draining. No profiler.',
        'AR uses one GPU. Every SSD arm uses two. Reference: '+q['reference_contract']+'.',
        q['scope']+'. Configurations selected only on fixed-output tuning; natural EOS is secondary.',
        f'Exact outputs: {summary["exact_speculative_matches"]}/{summary["exact_speculative_comparisons"]}.',
        'Query-row reductions describe actual draft model rows skipped, not measured latency or KV-memory savings.','']
    for protocol in ('fixed','natural'):
        lines += [f'## {protocol}','',
            '| Method | Depths by recovery rank | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Tokens/verify | Query rows avoided |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in summary['results']:
            if r['protocol']!=protocol or r['family']!='all':continue
            c=r['config'];v=r['statistics'];reduction=v['draft_query_row_reduction']
            reduction='—' if reduction is None else f'{100*reduction:.1f}%'
            lines.append(f'| {c["label"]} | {c["outcome_depths"]} | {r["gpu_count"]} | '
                f'{v["mean_latency_s"]:.4f} | {v["p50_latency_s"]:.4f} | {v["p95_latency_s"]:.4f} | '
                f'{v["committed_tok_per_s"]:.2f} | {v["mean_target_wait_s"]:.4f} | {v["tokens_per_verification"]:.3f} | {reduction} |')
        lines += ['','| Unequal-depth method | Family | Reference | Speed ratio | Paired 95% interval |','|---|---|---|---:|---|']
        for r in summary['results']:
            if r['protocol']!=protocol or r['method'] not in ('ranked','ranked_overlap'):continue
            for ref,v in r['comparison_to'].items():
                if ref=='ar':continue
                lo,hi=v['speedup_ci95'];lines.append(f'| {r["method"]} | {r["family"]} | {ref} | {v["speedup"]:.3f} | {lo:.3f}–{hi:.3f} |')
    lines += ['','## Interpretation limits','',
        'A win must be established against the tuned complete baseline and tuned overlap baseline. '
        'Compare ranked versus reversed allocation to test whether priority matters. '
        'Compare ranked versus uniform equal-work allocation to test whether uneven depth matters beyond reducing work. '
        'The latter two controls use the ranked winner’s K; ranked-overlap may select another K, so those comparisons are not matched-K ablations for it.',
        'Intervals use 10,000 paired bootstrap samples of rounds and prompts within families. No multiple-comparison correction. '
        'Per-family regressions, exactness, repeatability, acceptance, output lengths and memory are in summary.json.',
        *q['limitations']]
    if not summary['correctness_passed']:lines.insert(2,'**Correctness failed. No correctness-preserving speedup established.**')
    (folder/'report.md').write_text('\n'.join(lines)+'\n')


def plots(folder,selection,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5));scores=selection['scores']
    ax.bar([v['config']['label'] for v in scores],[v['mean_latency_s'] for v in scores])
    ax.set_ylabel('Mean tuning full-request latency (s)');ax.tick_params(axis='x',labelrotation=75)
    fig.tight_layout();fig.savefig(folder/'tuning.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,protocol in zip(axes,('fixed','natural')):
        selected=[r for r in results if r['family']=='all' and r['protocol']==protocol]
        means=[r['statistics']['mean_latency_s'] for r in selected]
        ci=[r['comparison_to']['complete']['mean_latency_ci95_s'] for r in selected]
        ax.bar([r['method'] for r in selected],means,yerr=[[max(0,m-b[0]) for m,b in zip(means,ci)],
            [max(0,b[1]-m) for m,b in zip(means,ci)]],capsize=3)
        ax.set(title=protocol,ylabel='Mean full-request latency (s), paired 95% CI');ax.tick_params(axis='x',labelrotation=70)
    fig.tight_layout();fig.savefig(folder/'latency.png',dpi=170);plt.close(fig)


def write_protocol(folder,q):
    text=['# Unequal-depth successor with saved AR v2','',
        'User-authorized amendment removes full AR timing repetitions. Saved shared-fixed16-v1 AR outputs '
        'are validated by receipts, input IDs, model/software/GPU identity and target numerical code. '
        'The small fresh AR correctness gate remains; its timings are diagnostic.',
        'After the publication v2 and context v2 queues complete, run the full six-input gate: fresh AR '
        'plus13 SSD settings, both protocols =28 blocks /168 requests at0.10 KV. '
        'All speculative outputs must match fresh gate AR, with actual reduced rows and short cached outcomes.',
        'Tuning retains all13 SSD settings: complete K4/7/10, overlap, ranked, ranked+overlap, reverse and '
        'uniform equal-work at K7/10; F3, P4, queue2. Two repetitions ×24 balanced prompts =26 blocks /624 requests. '
        'All tuning outputs must also match saved baseline AR. Selection reads tuning only.',
        'Freeze complete/overlap/ranked/ranked+overlap winners; budget/reverse K follows the ranked winner. '
        'Six SSD arms ×75 held-out prompts ×five repetitions ×two protocols =60 blocks /4500 requests. '
        'Saved AR supplies exact output references; no new full AR timing arm is executed.',
        'Primary SSD comparisons are fresh and paired by round/prompt. Any ratio to AR uses explicitly historical '
        'timing with independent historical/current round resampling. No contemporaneous AR claim.',
        'One worker at a time, shared main lock, dedicated UUID-pinned GPUs, contamination invalidates current block only. '
        'BF16,batch1,graphs,JIT,full prefills,capacity4096,prefix cache off,no profiler,final drain included. '
        'Performance KV0.70. AR gate GPU0; SSD targetGPU0/draftGPU1.',
        'Totals:114 blocks /5292 scored requests, plus342 warmups. This replaces v1 without deleting its artifacts. '
        'Static rank budgets are K7:[7,4,4], K10:[10,7,4]. Unequal depths do not reduce target verification graph size.']
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



def worker(args):
    import torch
    from ssd import LLM, SamplingParams
    from baseline_common import allocation
    q = read_json(args.outdir/'queue.json')
    d = load_definition(args.outdir, q, args.phase)
    check_runtime(q)
    j = next(j for j in d['jobs'] if j['id'] == args.job)
    c = j['config']; mode = c['mode']
    model = LLM(q['models']['target']['path'], draft=q['models']['draft']['path'],
        outcome_depths=c['outcome_depths'], target_numerics='shared_fixed16', num_gpus=2 if mode == 'ssd' else 1,
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
        outcome_depths=model.config.outcome_depths, actual_dtype=str(model.config.hf_config.torch_dtype), cuda_graphs=True, prefix_cache=False,
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

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=DEFAULT_OUT);parser.add_argument('--after',type=Path,default=DEFAULT_AFTER)
    parser.add_argument('--phase',choices=['all','prepare','gate','tune','select','heldout','report'],default='all')
    parser.add_argument('--watch',action='store_true');parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--job');parser.add_argument('--attempt',type=Path)
    args=parser.parse_args();args.outdir=args.outdir.resolve();args.after=args.after.resolve()
    if args.worker:worker(args);return 0
    signal.signal(signal.SIGTERM,interrupted);args.outdir.mkdir(parents=True,exist_ok=True)
    with (args.outdir/'.watcher.lock').open('a') as watcher:
        fcntl.flock(watcher,fcntl.LOCK_EX|fcntl.LOCK_NB)
        q=read_json(args.outdir/'queue.json') if (args.outdir/'queue.json').exists() else prepare(args.outdir,args.after)
        check_runtime(q);validate_workload(q['splits']);validate_sources(q)
        if q['after'][0]['path']!=str(args.after):raise ValueError('Predecessor differs from frozen queue')
        if args.phase=='prepare' or args.dry_run:
            print('Frozen: 168 gate, 624 SSD tuning, 4500 SSD held-out requests. No full AR reruns.');return 0
        try:
            with (SOURCE_ROOT/'results/baselines/eval1/.campaign.lock').open('a') as lock:
                while True:
                    check_runtime(q);validate_sources(q)
                    if predecessor_ready(q):
                        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                        except BlockingIOError:pass
                    atomic_json(args.outdir/'status.json',dict(status='waiting_for_predecessors',after=q['after'],at=now()))
                    if not args.watch:return 2
                    time.sleep(30)
                validate_dependencies(args.outdir,q);validate_models(q['models'])
                if args.phase in ('all','gate'):
                    if not run_phase(args,q,load_definition(args.outdir,q,'gate')):return 2
                validate_gate(args.outdir,q)
                if args.phase=='gate':return 0
                if args.phase in ('all','tune'):
                    if not run_phase(args,q,load_definition(args.outdir,q,'tune')):return 2
                    if args.phase=='tune':return 0
                if args.phase in ('all','select'):
                    chosen=select(args.outdir,q);freeze_json(args.outdir/'selection.json',chosen)
                    tuning_report(args.outdir,q,chosen)
                    freeze_json(args.outdir/'heldout/definition.json',definition(q,'heldout',evaluation_configs(chosen),file_digest(args.outdir/'selection.json')))
                    if args.phase=='select':return 0
                if args.phase in ('all','heldout'):
                    if not run_phase(args,q,load_definition(args.outdir,q,'heldout')):return 2
                if args.phase in ('all','report'):
                    summary=final_report(args.outdir,q)
                    atomic_json(args.outdir/'status.json',dict(status='complete',at=now(),scored_requests=summary['scored_requests'],
                        gate_requests=168,report=str(args.outdir/'report.md')))
        except BaseException as exc:
            atomic_json(args.outdir/'status.json',dict(status='stopped',reason=str(exc),at=now()));raise
    return 0


if __name__=='__main__':raise SystemExit(main())
