"""Reuse validated shared-AR outputs without claiming fresh AR measurements."""
import ast
from functools import lru_cache
from pathlib import Path

import numpy as np

from baseline_common import ROOT, digest, file_digest, read_json, validate_manifest
from run_baselines import accepted_result


AR_GRAPH_FUNCTIONS = ('run_decode_cudagraph', 'capture_cudagraph')
GRAPH_FILE = 'third_party/ssd/ssd/engine/helpers/cudagraph_helpers.py'


def graph_contract(path):
    nodes = {n.name:n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)}
    return {name:digest(ast.dump(nodes[name],include_attributes=False)) for name in AR_GRAPH_FUNCTIONS}


def numerical_files(files):
    return {name:sha for name,sha in files.items() if
        '/layers/' in name or '/models/' in name or
        name.endswith(('/engine/model_runner.py','/engine/shared_attention.py',
            '/engine/shared_numerics.py','/engine/step.py','/engine/sequence.py',
            '/engine/scheduler.py','/sampling_params.py','/utils/context.py','/utils/misc.py'))}


def prepare_reference(source, source_root=ROOT):
    """Pin all AR artifacts; do not parse held-out outputs or latency for tuning."""
    source=Path(source).resolve();manifest=read_json(source/'manifest.json');validate_manifest(manifest)
    if manifest.get('target_numerics')!='shared_fixed16':raise ValueError('Saved AR uses another numerical reference')
    paths=[source/'manifest.json',source/'tuning_jobs.json',source/'heldout_jobs.json']
    jobs={}
    for phase,filename in [('tune','tuning_jobs.json'),('heldout','heldout_jobs.json')]:
        jobs[phase]=[j for j in read_json(source/filename) if j['config']['mode']=='ar']
        expected=2 if phase=='tune' else 10
        if len(jobs[phase])!=expected:raise ValueError('Saved AR campaign is incomplete')
        for j in jobs[phase]:
            output=(source/'raw'/j['id']).with_suffix('.json')
            receipt_path=output.with_suffix('.receipt.json');r=read_json(receipt_path);m=r.get('monitoring',{})
            if (r.get('valid') is not True or r.get('job_sha256')!=j['job_sha256'] or
                r.get('execution_sha256')!=manifest['execution']['sha256'] or r.get('returncode')!=0 or
                m.get('samples',0)<1 or m.get('external_processes') or m.get('error') or
                r.get('error') or r.get('output_sha256')!=file_digest(output)):
                raise ValueError('Saved AR receipt invalid, interrupted or contaminated')
            paths.extend([output,receipt_path])
    if file_digest(Path(source_root)/GRAPH_FILE)!=manifest['execution']['files'][GRAPH_FILE]:
        raise ValueError('Cannot establish archived AR CUDA-graph contract')
    value=dict(source=str(source),artifacts_sha256={str(p):file_digest(p) for p in paths},jobs=jobs,
        models=manifest['models'],software=manifest['software'],gpus=manifest['hardware']['gpus'],
        numerical_files=numerical_files(manifest['execution']['files']),
        ar_graph_functions=graph_contract(Path(source_root)/GRAPH_FILE),
        reference_contract='shared-fixed16-v1; saved validated AR outputs; original historical AR separate',
        timing_contract='AR timings are historical context, not fresh or contemporaneous measurements')
    value['sha256']=digest(value);return value


def validate_reference(ref, root=ROOT, models=None, software=None, gpus=None):
    if digest({k:v for k,v in ref.items() if k!='sha256'})!=ref['sha256']:raise ValueError('Saved AR metadata changed')
    for name,sha in ref['artifacts_sha256'].items():
        if file_digest(name)!=sha:raise ValueError('Saved AR artifact changed: '+name)
    for name,sha in ref['numerical_files'].items():
        if file_digest(Path(root)/name)!=sha:raise ValueError('AR numerical code changed; fresh AR required')
    if graph_contract(Path(root)/GRAPH_FILE)!=ref['ar_graph_functions']:
        raise ValueError('AR CUDA graph changed; fresh AR required')
    if models is not None and models!=ref['models']:raise ValueError('AR model/tokenizer differs')
    if software is not None and software!=ref['software']:raise ValueError('AR software/environment differs')
    if gpus is not None and gpus!=ref['gpus']:raise ValueError('AR GPU identities/software differ')


@lru_cache(maxsize=128)
def _load(source, job_json, artifact_sha):
    import json
    source=Path(source);job=json.loads(job_json)
    return accepted_result(source,job,read_json(source/'manifest.json'))


def reference_block(ref,phase,protocol,round_):
    import json
    job=next(j for j in ref['jobs'][phase] if j['protocol']==protocol and j['repeat']==round_)
    path=(Path(ref['source'])/'raw'/job['id']).with_suffix('.json')
    sha=file_digest(path)
    if sha!=ref['artifacts_sha256'][str(path)]:raise ValueError('Saved AR output changed')
    return _load(ref['source'],json.dumps(job,sort_keys=True),sha)


def reference_grid(ref,phase,protocol,ids,rounds):
    result=[]
    for r in range(rounds):
        rows={row['id']:row for row in reference_block(ref,phase,protocol,r)['requests']}
        result.append([rows[id_] for id_ in ids])
    return result


def check_outputs(blocks,q,phase):
    """Gate uses its fresh small AR; tuning/test outputs use validated saved AR."""
    from baseline_common import first_difference
    fresh={(b['job']['protocol'],b['job']['round']):b for b in blocks if b['job']['config']['mode']=='ar'}
    checks=[]
    for b in blocks:
        j=b['job'];is_ar=j['config']['mode']=='ar'
        if phase=='gate':
            if is_ar:continue
            ref=fresh[(j['protocol'],j['round'])]
        else:
            if is_ar and phase!='continuity':raise ValueError('Unexpected full AR performance block')
            ref=reference_block(q['ar_reference'],'tune' if phase=='continuity' else phase,j['protocol'],j['round'])
        rows={r['id']:r for r in ref['requests']}
        for row in b['requests']:
            gold=rows[row['id']]
            if row['input_token_ids_sha256']!=gold['input_token_ids_sha256']:raise ValueError('AR reference input differs')
            position=first_difference(gold['output_token_ids'],row['output_token_ids'])
            checks.append(dict(protocol=j['protocol'],round=j['round'],config=j['config']['label'],id=row['id'],
                exact=position is None,first_difference_zero_based=position,
                reference_origin='fresh_gate' if phase=='gate' else 'saved_shared_ar'))
    return checks


def historical_comparison(reference,method,families,samples=10000,seed=0):
    """Pair prompts, but resample old and new rounds independently."""
    a,b=np.asarray(reference,float),np.asarray(method,float)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or a.shape[1]!=len(families) or np.any(a<=0) or np.any(b<=0):
        raise ValueError('Expected positive historical/current round-by-prompt matrices')
    rng=np.random.default_rng(seed)
    ar=rng.integers(0,len(a),size=(samples,len(a)));br=rng.integers(0,len(b),size=(samples,len(b)))
    labels=np.asarray(families)
    prompts=np.concatenate([rng.choice(np.flatnonzero(labels==f),size=(samples,sum(labels==f)),replace=True)
        for f in sorted(set(families))],axis=1)
    ma=a[ar[:,:,None],prompts[:,None,:]].mean((1,2));mb=b[br[:,:,None],prompts[:,None,:]].mean((1,2))
    interval=lambda x:np.quantile(x,[.025,.975]).tolist()
    return dict(speedup=float(a.mean()/b.mean()),speedup_ci95=interval(ma/mb),
        mean_latency_s=float(b.mean()),mean_latency_ci95_s=interval(mb),
        latency_reduction=float(1-b.mean()/a.mean()),latency_reduction_ci95=interval(1-mb/ma),
        bootstrap_samples=samples,bootstrap_seed=seed,
        bootstrap_unit='paired prompt IDs within families; independent historical/current rounds',
        reference_origin='historical_ar',contemporaneous=False,
        limitation='Cross-campaign timing comparison; GPU state and time-dependent drift are not controlled')


def report_campaign(folder,q,selection,d,blocks,aggregate,fresh_references,tuning_requests,prelude_requests,prelude_blocks):
    """Only real new worker blocks count; historical AR stays separately labeled."""
    import csv
    from analyze_baselines import paired_bootstrap,output_checks,row_statistics
    from baseline_common import FAMILIES,atomic_json
    ids=[p['id'] for p in d['prompts']];families=[p['family'] for p in d['prompts']]
    records={(b['job']['protocol'],b['job']['round'],b['job']['config']['method'],r['id']):r
        for b in blocks for r in b['requests']}
    results=[];historical=[]
    for protocol in d['protocols']:
        ar=reference_grid(q['ar_reference'],'heldout',protocol,ids,5)
        historical.append(dict(protocol=protocol,measurement_origin='historical_ar',gpu_count=1,
            statistics=row_statistics([r for rr in ar for r in rr])))
        grids={c['method']:[[records[(protocol,r,c['method'],id_)] for id_ in ids] for r in range(5)]
            for c in d['configurations']}
        for family in ('all',)+FAMILIES:
            indices=[i for i,f in enumerate(families) if family=='all' or f==family]
            family_labels=[families[i] for i in indices]
            gold=[[row[i] for i in indices] for row in ar]
            for c in d['configurations']:
                grid=[[row[i] for i in indices] for row in grids[c['method']]]
                latency=[[r['latency_s'] for r in row] for row in grid]
                comparisons={ref:paired_bootstrap([[row[i]['latency_s'] for i in indices] for row in grids[ref]],
                    latency,family_labels,10000,0) for ref in fresh_references}
                comparisons['historical_ar']=historical_comparison([[r['latency_s'] for r in row] for row in gold],latency,family_labels)
                results.append(dict(protocol=protocol,family=family,config=c,method=c['method'],gpu_count=2,
                    measurement_origin='fresh_dedicated',statistics=aggregate([r for rr in grid for r in rr]),
                    correctness=output_checks(grid,gold),comparison_to=comparisons,
                    latency_regression_to={k:v['speedup']<1 for k,v in comparisons.items()}))
    checks=check_outputs(blocks,q,'heldout')
    correct=all(v['exact'] for v in checks) and all(r['correctness']['repeatable_prompts']==r['correctness']['total_prompts']
        and r['correctness']['requests_matching_ar_round0']==r['correctness']['total_requests'] for r in results)
    heldout=sum(len(b['requests']) for b in blocks)
    summary=dict(status='complete' if correct else 'correctness_failed',queue_sha256=q['sha256'],
        selection_sha256=file_digest(folder/'selection.json'),valid_heldout_blocks=len(blocks),
        heldout_requests=heldout,tuning_requests=tuning_requests,prelude_requests=prelude_requests,
        scored_requests=heldout+tuning_requests+prelude_requests,
        warmup_requests=3*(2*len(q['tuning_configs'])+len(blocks)+prelude_blocks),
        correctness_passed=correct,exact_speculative_comparisons=len(checks),
        exact_speculative_matches=sum(v['exact'] for v in checks),reference=q['reference_contract'],
        scope=q['scope'],winners=selection['winners'],results=results,
        ar_reference_sha256=q['ar_reference']['sha256'],historical_ar_context=historical,
        fresh_ar_performance_blocks=0,bootstrap_samples=10000,
        bootstrap_unit='SSD: paired rounds/prompts; historical AR: independent rounds and paired prompts',
        dedicated_gpus=True,profiler=False,
        natural_eos_note='Secondary evaluation of fixed-length-selected settings; no retuning')
    atomic_json(folder/'summary.json',summary)
    rows=[]
    for r in results:
        v=r['statistics'];comp=r['comparison_to']['complete']
        rows.append(dict(protocol=r['protocol'],family=r['family'],method=r['method'],config=r['config']['label'],
            measurement_origin=r['measurement_origin'],gpu_count=2,mean_latency_s=v['mean_latency_s'],
            p50_latency_s=v['p50_latency_s'],p95_latency_s=v['p95_latency_s'],tokens_per_second=v['committed_tok_per_s'],
            first_delivery_s=v['mean_first_delivery_s'],target_wait_s=v['mean_target_wait_s'],
            speedup_vs_complete=comp['speedup'],ci95_low=comp['speedup_ci95'][0],ci95_high=comp['speedup_ci95'][1]))
    with (folder/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Fresh SSD comparisons with saved shared-AR references','',
        'AR is not rerun as a full performance arm. Saved AR outputs come from validated dedicated baseline receipts. '
        'Short fresh AR checks must pass before this campaign proceeds. Target numerical files, AR graph functions, '
        'model/tokenizer, software and GPU identities are checked against the saved reference.',
        q['scope']+'. Selection uses tuning only; natural EOS is secondary.',
        f'Exact speculative outputs: {sum(v["exact"] for v in checks)}/{len(checks)} against shared-fixed16-v1. '
        'Original historical AR is a separate reference.','',
        '## Historical AR timing context — measured earlier, not rerun','',
        '| Protocol | GPUs | Earlier mean s | Earlier p50 s | Earlier p95 s |','|---|---:|---:|---:|---:|']
    for h in historical:
        v=h['statistics'];lines.append(f'| {h["protocol"]} | 1 | {v["mean_latency_s"]:.4f} | {v["p50_latency_s"]:.4f} | {v["p95_latency_s"]:.4f} |')
    lines+=['','Historical-AR ratios in summary.json are cross-campaign context. Their bootstrap pairs prompts but '
        'resamples historical/current rounds independently. It does not control temporal GPU-state differences. '
        'Primary SSD-versus-SSD comparisons use fresh paired rounds on dedicated GPUs.']
    for protocol in d['protocols']:
        lines+=['',f'## {protocol}: fresh SSD measurements','',
            '| Method | GPUs | Mean s | p50 s | p95 s | tok/s | First delivery ms | Wait s | Tokens/verify |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in results:
            if r['protocol']!=protocol or r['family']!='all':continue
            v=r['statistics'];lines.append(f'| {r["config"]["label"]} | 2 | {v["mean_latency_s"]:.4f} | '
                f'{v["p50_latency_s"]:.4f} | {v["p95_latency_s"]:.4f} | {v["committed_tok_per_s"]:.2f} | '
                f'{v["mean_first_delivery_s"]*1000:.2f} | {v["mean_target_wait_s"]:.4f} | {v["tokens_per_verification"]:.3f} |')
        lines+=['','| Method | Family | Fresh reference | Speed ratio | Paired 95% CI |','|---|---|---|---:|---|']
        for r in results:
            if r['protocol']!=protocol:continue
            for ref in fresh_references:
                if r['method']==ref:continue
                v=r['comparison_to'][ref];lo,hi=v['speedup_ci95']
                lines.append(f'| {r["method"]} | {r["family"]} | {ref} | {v["speedup"]:.3f} | {lo:.3f}–{hi:.3f} |')
    lines+=['','## Interpretation','',
        'Lower latency must be established against the strongest fresh matched SSD comparator. '
        'Overlap counts and fewer query rows alone do not establish a speedup. All request timing includes '
        'prefills and final draining, excludes loading/warmup/serialization, and uses no profiler. '
        'Target verification retains K+1 rows. Detailed acceptance, cache hits, memory, output lengths, '
        'first divergences, repeatability and per-family regressions are in summary.json.',*q.get('limitations',[])]
    if not correct:lines.insert(2,'**Correctness failed. No correctness-preserving acceleration established.**')
    (folder/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,protocol in zip(axes,d['protocols']):
        vals=[r for r in results if r['protocol']==protocol and r['family']=='all']
        means=[r['statistics']['mean_latency_s'] for r in vals]
        ci=[r['comparison_to']['complete']['mean_latency_ci95_s'] for r in vals]
        ax.bar([r['method'] for r in vals],means,yerr=[[max(0,m-b[0]) for m,b in zip(means,ci)],
            [max(0,b[1]-m) for m,b in zip(means,ci)]],capsize=3)
        ax.set(title=protocol,ylabel='Fresh SSD full-request latency (s), paired 95% CI');ax.tick_params(axis='x',labelrotation=60)
    fig.tight_layout();fig.savefig(folder/'latency.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5));scores=selection['scores']
    ax.bar([r['config']['label'] for r in scores],[r['mean_latency_s'] for r in scores])
    ax.set_ylabel('Fresh SSD tuning latency (s)');ax.tick_params(axis='x',labelrotation=90)
    fig.tight_layout();fig.savefig(folder/'tuning.png',dpi=160);plt.close(fig)
    if not correct:raise ValueError('Exactness or repeatability failed')
    return summary
