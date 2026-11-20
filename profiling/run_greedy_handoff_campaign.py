"""Successor campaign: unchanged SSD baseline versus versioned greedy handoffs.

CPU-only controller. Waits for the unequal-depth campaign, then uses the shared
exclusive GPU lock. Every worker starts fresh, warms three requests, and resumes
only from validated whole-block receipts. No full AR timing campaign is repeated.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
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

from baseline_common import (ROOT,atomic_json,compute_processes,digest,execution_fingerprint,
    file_digest,freeze_json,gpu_inventory,read_json,software_fingerprint,validate_models,first_difference)
from run_baselines import PendingWindow,require_idle,stop_worker,interrupted
from gpu_process_ownership import TOKEN_ENV,partition,cleanup_tagged
from analyze_baselines import paired_bootstrap,row_statistics
import saved_ar_reference as saved_ar

SOURCE_ROOT=ROOT.parents[1]
DEFAULT_OUT=SOURCE_ROOT/'results/next_stage/greedy_handoff_v1_2026-11-19'
AFTER=SOURCE_ROOT/'results/next_stage/outcome_depth_v2_2026-11-18'
WORKLOAD=SOURCE_ROOT/'results/next_stage/shared_uniform_v1_2026-11-17/baselines'
SPEC=dict(seed=0,batch_size=1,precision='bfloat16',capacity=4096,output_cap=256,
    fanout=3,publication_minimum=4,queue_bound=2,tuning_repetitions=2,repetitions=5,
    protocols=['fixed','natural'],kv_fraction=.70,gate_kv_fraction=.10,prefix_cache=False,
    profiler=False,bootstrap_samples=10000,monitor_seconds=1,baseline='unchanged original checkout',
    objective='lowest balanced mean fixed256 tuning latency, then smaller K',
    timing='full request including prefills and final drain; warmups/loading/serialization excluded',
    scope='Same frozen task mix; reused test prompts, not newly unseen; new version of greedy SSD')
METHODS=('legacy_complete','legacy_overlap','ids_complete','ids_inline','ids_threaded','ids_bypass')


def now():return datetime.now(timezone.utc).isoformat()


def config(method,k=7):
    greedy=method.startswith('ids_');publication=method not in ('ar','legacy_complete','ids_complete','micro')
    return dict(method=method,label=method if method in ('ar','micro') else f'{method}_k{k}',
        mode='ar' if method=='ar' else 'ssd',k=k,f=3,p=4 if publication else 0,
        greedy=greedy,service='threaded' if method in ('ids_threaded','ids_bypass') else 'inline',
        bypass=method in ('ids_complete','ids_bypass'),
        engine_root=str(ROOT if greedy or method=='micro' else SOURCE_ROOT))


def configurations():
    return [config(m,k) for m in METHODS for k in ((4,7,10) if m in ('legacy_complete','ids_complete') else (7,10))]


def code_files():
    names=['run_greedy_handoff_campaign.py','greedy_handoff_worker.py','trace_legacy_handoff.py',
           'saved_ar_reference.py','gpu_process_ownership.py','analyze_baselines.py','baseline_common.py',
           'run_engine.py','run_baselines.py']
    result={str(ROOT/'profiling'/n):file_digest(ROOT/'profiling'/n) for n in names}
    for base in (ROOT,SOURCE_ROOT):
        result.update({str(p):file_digest(p) for p in (base/'third_party/ssd/ssd').rglob('*.py')})
    return result


def definition(q,phase,configs,selection_sha=None):
    gate=phase=='gate';tune=phase=='tune';micro=phase=='micro'
    diagnostic=phase in ('diagnostic','stress')
    prompts=[] if micro else q['gate_prompts'] if gate else q['diagnostic_prompts'] if diagnostic else q['splits']['tune' if tune else 'heldout']
    reps=2 if tune else 5 if phase=='heldout' else 1
    protocols=['fixed'] if tune or phase in ('diagnostic','micro') else ['fixed','natural']
    d=dict(queue_sha256=q['sha256'],phase=phase,configurations=configs,prompts=prompts,
        warmup=[] if micro else q['splits']['warmup'],repetitions=reps,protocols=protocols,
        kv_fraction=.10 if gate or diagnostic or micro else .70,diagnostic=diagnostic,
        selection_sha256=selection_sha,profiler=False)
    others=[c for c in configs if c['method']!='ar'];random.Random(0).shuffle(others)
    orders=random.Random(0).sample(list(itertools.permutations(configs)),5) if phase=='heldout' else None
    jobs=[]
    for r in range(reps):
        ordered=orders[r] if orders else [c for c in configs if c['method']=='ar']+(others if r==0 else list(reversed(others)))
        order=random.Random(1000+r).sample(range(len(prompts)),len(prompts))
        for protocol in protocols if r%2==0 else list(reversed(protocols)):
            jobs += [dict(id=f'{protocol}_r{r}_{c["label"]}',config=c,round=r,protocol=protocol,prompt_order=order) for c in ordered]
    d['jobs']=jobs;d['sha256']=digest(d);return d


def phase_configs(phase,selection=None):
    if phase=='micro':return [config('micro')]
    if phase=='gate':return [config('ar')]+configurations()
    if phase=='tune':return configurations()
    if phase=='stress':return [config('ar'),config('ids_bypass',10)]
    if phase=='diagnostic':return [config(m,k) for m in ('legacy_overlap','ids_inline','ids_bypass') for k in (7,10)]
    return [selection['winners'][m] for m in METHODS]


def prepare(folder):
    previous=read_json(AFTER/'queue.json');old=read_json(WORKLOAD/'manifest.json')
    q=dict(created_at=now(),spec=SPEC,code=code_files(),software=software_fingerprint(),hardware=gpu_inventory(),
        models=old['models'],splits=old['splits'],gate_prompts=previous['gate_prompts'],
        diagnostic_prompts=[next(p for p in old['splits']['tune'] if p['family']==f) for f in ('chat','code','doc')],
        ar_reference=saved_ar.prepare_reference(WORKLOAD,SOURCE_ROOT),
        after=dict(path=str(AFTER),queue_sha256=file_digest(AFTER/'queue.json'),
                   code_root=str(SOURCE_ROOT/'.worktrees/outcome-depth-v1'),
                   code=previous['execution']['files']|previous['queue_code']),
        baseline_root=str(SOURCE_ROOT),candidate_root=str(ROOT),
        limitations=['Greedy-only transport; stochastic decoding is rejected.',
            'Diagnostic event calibration and logging are excluded from performance selection.',
            'No target numerical, attention or verification-graph arithmetic changes.',
            'Historical AR is an output reference; all six SSD timing arms are freshly measured.'])
    q['sha256']=digest(q);freeze_json(folder/'queue.json',q)
    for phase in ('micro','gate','stress','diagnostic','tune'):
        freeze_json(folder/phase/'definition.json',definition(q,phase,phase_configs(phase)))
    with zipfile.ZipFile(folder/'source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for absolute in sorted(q['code']):
            p=Path(absolute);prefix='candidate' if p.is_relative_to(ROOT) else 'original'
            base=ROOT if prefix=='candidate' else SOURCE_ROOT
            archive.write(p,str(Path(prefix)/p.relative_to(base)))
        for p in sorted((ROOT/'tests').glob('test_*handoff*.py')):archive.write(p,'candidate/'+str(p.relative_to(ROOT)))
        p=ROOT/'tests/test_greedy_transport.py';archive.write(p,'candidate/'+str(p.relative_to(ROOT)))
    atomic_json(folder/'status.json',dict(status='prepared',after=q['after']['path'],at=now()))
    return q


def check_runtime(q):
    if digest({k:v for k,v in q.items() if k!='sha256'})!=q['sha256'] or q['spec']!=SPEC:
        raise ValueError('Frozen handoff specification changed')
    if q['code']!=code_files():raise ValueError('Execution code changed; use a new campaign version')
    if q['software']!=software_fingerprint() or q['hardware']['gpus']!=gpu_inventory()['gpus']:
        raise ValueError('Software environment or GPU identities changed')
    if file_digest(Path(q['after']['path'])/'queue.json')!=q['after']['queue_sha256']:
        raise ValueError('Predecessor queue changed')
    for name,sha in q['after']['code'].items():
        if file_digest(Path(q['after']['code_root'])/name)!=sha:raise ValueError('Predecessor execution changed')
    saved_ar.validate_reference(q['ar_reference'],root=SOURCE_ROOT,models=q['models'],software=q['software'],gpus=q['hardware']['gpus'])
    saved_ar.validate_reference(q['ar_reference'],root=ROOT)


def load_definition(folder,q,phase):
    selected=None
    if phase=='heldout':selected=validated_selection(folder,q)
    expected=definition(q,phase,phase_configs(phase,selected),file_digest(folder/'selection.json') if selected else None)
    actual=read_json(folder/phase/'definition.json')
    if actual!=expected:raise ValueError('Phase definition changed')
    return actual


def validate_result(b,j,d,q):
    if b.get('queue_sha256')!=q['sha256'] or b.get('definition_sha256')!=d['sha256'] or b.get('job')!=j:
        raise ValueError('Worker provenance differs')
    if b.get('software')!=q['software']:raise ValueError('Worker software differs')
    if d['phase']=='micro':
        if not b.get('micro_cuda_passed'):raise ValueError('GPU ownership tests failed')
        return
    if b.get('engine_root')!=j['config']['engine_root'] or b.get('diagnostic')!=d['diagnostic']:
        raise ValueError('Worker implementation or diagnostic state differs')
    if not Path(b.get('engine_module_path','')).is_relative_to(Path(j['config']['engine_root'])/'third_party/ssd'):
        raise ValueError('Worker imported another SSD checkout')
    expected_transport='greedy_ids_v1' if j['config']['greedy'] else 'legacy_logits'
    if (b.get('transport_version')!=expected_transport or b.get('handoff_service')!=j['config']['service'] or
            b.get('bypass_ready')!=j['config']['bypass']):raise ValueError('Transport/service/bypass differs')
    if b.get('profile_capture') or b.get('prefix_cache') or b.get('actual_dtype')!='torch.bfloat16' or not b.get('cuda_graphs') or b.get('kv_fraction')!=d['kv_fraction']:
        raise ValueError('Worker settings differ')
    rows=b.get('requests',[])
    expected=[d['prompts'][i] for i in j['prompt_order']]
    if [r['id'] for r in rows]!=[p['id'] for p in expected]:raise ValueError('Partial/reordered block')
    for r,p in zip(rows,expected):
        ids=r['output_token_ids']
        if r['input_token_ids_sha256']!=p['token_ids_sha256'] or r['prompt_tokens']!=len(p['token_ids']):raise ValueError('Input tokens differ')
        if not ids or len(ids)>256 or any(type(t) is not int for t in ids) or r['output_tokens']!=len(ids):raise ValueError('Invalid outputs')
        if j['protocol']=='fixed' and len(ids)!=256:raise ValueError('Fixed length differs')
        if not math.isfinite(r['latency_s']) or r['latency_s']<=0 or r['first_delivery_s'] is None:raise ValueError('Invalid timing')
        if not 0<=r['first_delivery_s']<=r['latency_s']:raise ValueError('Invalid delivery time')
        if not d['diagnostic'] and r.get('handoff_trace'):raise ValueError('Diagnostic instrumentation in timing block')


def block_result(folder,j,d,q):
    for path in sorted((folder/j['id']).glob('attempt_*/receipt.json')):
        r=read_json(path)
        if not r.get('valid'):continue
        out=path.parent/'result.json'
        if (r.get('definition_sha256')!=d['sha256'] or r.get('returncode')!=0 or r.get('samples',0)<1 or
                r.get('external_processes') or r.get('error') or r.get('output_sha256')!=file_digest(out)):
            raise ValueError('Invalid/contaminated receipt')
        b=read_json(out);validate_result(b,j,d,q);return b
    return None


def available(folder,d,q,complete=False):
    blocks=[block_result(folder,j,d,q) for j in d['jobs']]
    if complete and any(b is None for b in blocks):raise ValueError('Required valid blocks missing')
    return [b for b in blocks if b is not None]


def checks(blocks,q,phase):
    if phase=='micro':return []
    if phase=='stress':return saved_ar.check_outputs(blocks,q,'gate')
    if phase=='diagnostic':
        # Same tuning IDs, but no dependence on held-out results.
        return saved_ar.check_outputs(blocks,q,'tune')
    return saved_ar.check_outputs(blocks,q,phase)


def run_block(args,q,d,j):
    check_runtime(q);uuids=[g['uuid'] for g in q['hardware']['gpus']];require_idle(uuids)
    parent=args.outdir/d['phase']/j['id'];parent.mkdir(parents=True,exist_ok=True)
    attempt=parent/f'attempt_{len(list(parent.glob("attempt_*")))}';attempt.mkdir()
    token=uuid.uuid4().hex
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids[:1] if j['config']['mode']=='ar' else uuids),
             AO_HANDOFF_ENGINE_ROOT=j['config']['engine_root'])
    env[TOKEN_ENV]=token;env.pop('AO_SPAWN_EXECUTABLE',None)
    for name in list(env):
        if name.startswith('SSD_PROFILE'):env[name]='0'
        if name.startswith('AO_HANDOFF_') and name!='AO_HANDOFF_ENGINE_ROOT':env.pop(name)
    if env.get('CUDA_INJECTION64_PATH') or env.get('NVTX_INJECTION64_PATH'):raise ValueError('Profiler injection enabled')
    if d['phase']=='diagnostic' and not j['config']['greedy']:
        env['AO_HANDOFF_LEGACY_TRACE']='1';env['AO_HANDOFF_TRACE_DIR']=str(attempt/'trace')
    if d['phase']=='stress':env['AO_HANDOFF_STRESS']='1'
    command=[sys.executable,'-O',str(ROOT/'profiling/greedy_handoff_worker.py'),'--outdir',str(args.outdir),
             '--phase',d['phase'],'--job',j['id'],'--attempt',str(attempt)]
    process=None;error=None;samples=0;foreign={};owned=[];started=time.monotonic()
    try:
        with (attempt/'worker.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                owned,others=partition(compute_processes(uuids),process.pid,token)
                samples+=1
                for row in others:foreign[row['pid'],row['uuid']]=row
                if others or process.poll() is not None:break
                if time.monotonic()-started>1800:error='Worker exceeded 1800-second budget';break
                time.sleep(1)
    except BaseException as exc:error=str(exc) or type(exc).__name__;raise
    finally:
        if process is not None:stop_worker(process)
        cleanup_tagged(owned,token)
        out=attempt/'result.json'
        valid=process is not None and process.returncode==0 and samples>0 and not foreign and error is None
        if valid:
            try:validate_result(read_json(out),j,d,q)
            except (ValueError,KeyError,OSError,TypeError) as exc:valid=False;error=str(exc)
        atomic_json(attempt/'receipt.json',dict(valid=bool(valid),definition_sha256=d['sha256'],command=command,
            returncode=process.returncode if process else None,samples=samples,external_processes=list(foreign.values()),
            error=error,run_token=token,started_at_monotonic=started,wall_seconds=time.monotonic()-started,
            diagnostic=d['diagnostic'],output_sha256=file_digest(out) if out.exists() else None,at=now()))
    if not valid:
        if foreign:raise PendingWindow('Current block invalidated; waiting for dedicated GPUs')
        log=(attempt/'worker.log').read_text(errors='replace')
        if process is not None and process.returncode and 'CUDA driver initialization failed' in log:
            raise PendingWindow('CUDA initialization failed; invalid attempt saved, retry after backoff')
        raise RuntimeError(f'Worker failed: {attempt}/worker.log; {error}')


def run_phase(args,q,phase):
    d=load_definition(args.outdir,q,phase)
    for j in d['jobs']:
        while block_result(args.outdir/phase,j,d,q) is None:
            try:
                atomic_json(args.outdir/'status.json',dict(status='running',phase=phase,job=j['id'],at=now()))
                run_block(args,q,d,j)
            except PendingWindow as exc:
                atomic_json(args.outdir/'status.json',dict(status='pending_dedicated_gpus',phase=phase,job=j['id'],reason=str(exc),at=now()))
                if not args.watch:return False
                time.sleep(30)
        exact=checks(available(args.outdir/phase,d,q),q,phase)
        if not all(c['exact'] for c in exact):
            atomic_json(args.outdir/phase/'output_failures.json',[c for c in exact if not c['exact']])
            raise ValueError('Exact output mismatch; no performance work follows')
    blocks=available(args.outdir/phase,d,q,True)
    if phase=='stress':
        counts=[r.get('publication_policy_stats') for b in blocks for r in b['requests']]
        if not any(v and v[10]>0 for v in counts):raise ValueError('Stress lacks actual response-before-tail evidence')
    if phase=='diagnostic':
        rows=[dict(t,config_label=b['job']['config']['label'],prompt_id=r['id'],job=b['job']['id'])
              for b in blocks for r in b['requests'] for t in r.get('handoff_trace',[])]
        if not rows:raise ValueError('Timing diagnostic produced no trace evidence')
        atomic_json(args.outdir/'diagnostic/summary.json',dict(diagnostic_only=True,records=rows,
            classifications=dict(Counter(t['classification'] for t in rows if t.get('request_posted_ns') is not None)),
            by_configuration={label:dict(Counter(t['classification'] for t in rows if t['config_label']==label and t.get('request_posted_ns') is not None))
                              for label in sorted({t['config_label'] for t in rows})},
            unrequested_generations=sum(t.get('request_posted_ns') is None for t in rows)))
    atomic_json(args.outdir/phase/'completion.json',dict(status='complete',valid_blocks=len(blocks),
        exact_matches=sum(c['exact'] for c in checks(blocks,q,phase)),at=now()))
    return True


def select(folder,q):
    d=load_definition(folder,q,'tune');blocks=available(folder/'tune',d,q,True)
    if not all(c['exact'] for c in checks(blocks,q,'tune')):raise ValueError('Tuning exactness failed')
    scores=[]
    for c in configurations():
        rows=[r for b in blocks if b['job']['config']==c for r in b['requests']]
        if len(rows)!=48:raise ValueError('Both tuning repetitions required')
        scores.append(dict(config=c,requests=len(rows),mean_latency_s=sum(r['latency_s'] for r in rows)/len(rows)))
    winners={m:min((s for s in scores if s['config']['method']==m),key=lambda s:(s['mean_latency_s'],s['config']['k']))['config'] for m in METHODS}
    evidence={str(p.relative_to(folder)):file_digest(p) for p in (folder/'tune').glob('*/attempt_*/receipt.json') if read_json(p).get('valid')}
    result=dict(queue_sha256=q['sha256'],winners=winners,scores=scores,evidence=evidence,selection_data='Tuning only')
    result['sha256']=digest(result);return result


def validated_selection(folder,q):
    value=read_json(folder/'selection.json')
    if value!=select(folder,q):raise ValueError('Frozen selection differs from tuning')
    return value


def predecessor_ready(q):
    p=Path(q['after']['path'])
    if read_json(p/'status.json').get('status')!='complete':return False
    summary=read_json(p/'summary.json')
    if summary.get('status')!='complete' or summary.get('correctness_passed') is not True or not (p/'report.md').exists():
        raise ValueError('Predecessor final report/correctness incomplete')
    # Completed status alone is insufficient: verify all required whole blocks.
    for phase in ('gate','tune','heldout'):
        d=read_json(p/phase/'definition.json')
        blocks=[]
        for j in d['jobs']:
            valid=None
            for rp in (p/phase/j['id']).glob('attempt_*/receipt.json'):
                r=read_json(rp)
                if not r.get('valid'):continue
                out=rp.parent/'result.json'
                if r.get('returncode')!=0 or r.get('samples',0)<1 or r.get('external_processes') or r.get('error') or r.get('definition_sha256')!=d['sha256'] or r.get('output_sha256')!=file_digest(out):raise ValueError('Predecessor receipt invalid')
                valid=read_json(out)
                if valid.get('job')!=j:raise ValueError('Predecessor job differs')
                break
            if valid is None:raise ValueError('Predecessor valid blocks missing')
            blocks.append(valid)
        if not all(c['exact'] for c in saved_ar.check_outputs(blocks,read_json(p/'queue.json'),phase)):
            raise ValueError('Predecessor exactness failed')
    return True


def report(folder,q):
    selected=validated_selection(folder,q);d=load_definition(folder,q,'heldout');blocks=available(folder/'heldout',d,q,True)
    exact=checks(blocks,q,'heldout')
    if not all(c['exact'] for c in exact):raise ValueError('Held-out exactness failed')
    ids=[p['id'] for p in q['splits']['heldout']];families=[p['family'] for p in q['splits']['heldout']]
    results=[];lines=['# Versioned greedy handoff comparison','',
        'Original complete SSD and original overlap import the unchanged original checkout. '
        'Token-only transport, independent host service and bypass are separate new arms. '
        'All performance runs are fresh, unprofiled and dedicated; AR timings are not rerun. '
        'The same frozen test prompts are reused. Selection uses tuning only.','',
        f'Exact held-out outputs: {sum(c["exact"] for c in exact)}/{len(exact)} against shared-fixed16-v1.','']
    for protocol in ('fixed','natural'):
        grids={}
        for m in METHODS:
            grids[m]=[]
            for r in range(5):
                b=next(b for b in blocks if b['job']['protocol']==protocol and b['job']['round']==r and b['job']['config']['method']==m)
                rows={x['id']:x for x in b['requests']};grids[m].append([rows[id_] for id_ in ids])
        lines += [f'## {protocol}','', '| Method | GPUs | Mean s | p50 s | p95 s | tok/s | Speedup vs original | 95% CI | Speedup vs IDs-complete |', '|---|---:|---:|---:|---:|---:|---:|---|---:|']
        for m in METHODS:
            grid=grids[m];rows=[x for rr in grid for x in rr];stats=row_statistics(rows)
            to={}
            for control in ('legacy_complete','ids_complete'):
                to[control]=paired_bootstrap([[x['latency_s'] for x in rr] for rr in grids[control]],[[x['latency_s'] for x in rr] for rr in grid],families)
            repeatable=all(all(rr[j]['output_token_ids']==grid[0][j]['output_token_ids'] for rr in grid) for j in range(75))
            result=dict(protocol=protocol,method=m,config=selected['winners'][m],statistics=stats,
                comparison_to=to,repeatable=repeatable,mean_wait_s=sum(r['draft_wait_s'] for r in rows)/len(rows))
            states=[r['publication_policy_stats'] for r in rows if r.get('publication_policy_stats')]
            result.update(mean_final_drain_s=sum(r['final_drain_s'] for r in rows)/len(rows),
                tokens_per_verification=sum(sum(r['accepted_suffix_lens_with_recovery']) for r in rows)/max(1,sum(len(r['accepted_suffix_lens_with_recovery']) for r in rows)),
                full_ready_bypasses=sum(s[12] for s in states if len(s)==16),
                response_token_payload_bytes=sum(s[13] for s in states if len(s)==16),
                avoided_draft_score_payload_bytes=sum(s[14] for s in states if len(s)==16),
                target_peak_allocated_bytes=max(r['target_peak_memory']['allocated_bytes'] for r in rows),
                draft_peak_allocated_bytes=max((p['allocated_bytes'] for r in rows for p in r['draft_peak_memory']),default=0))
            c=to['legacy_complete'];lo,hi=c['speedup_ci95']
            lines.append(f'| {m} | 2 | {stats["mean_latency_s"]:.4f} | {stats["p50_latency_s"]:.4f} | {stats["p95_latency_s"]:.4f} | {stats["committed_tok_per_s"]:.2f} | {c["speedup"]:.3f} | {lo:.3f}–{hi:.3f} | {to["ids_complete"]["speedup"]:.3f} |')
            result['families']={}
            for family in ('chat','code','doc'):
                jj=[i for i,f in enumerate(families) if f==family]
                result['families'][family]=paired_bootstrap([[rr[i]['latency_s'] for i in jj] for rr in grids['legacy_complete']],[[rr[i]['latency_s'] for i in jj] for rr in grid],[family]*len(jj))
            results.append(result)
        lines.append('')
    lines += ['## Interpretation','',
        'Compare each new arm against both the unchanged original SSD and the new IDs-only complete SSD. '
        'A gain from token-only transport alone is a transport optimization; incremental publication benefit '
        'requires a further gain over IDs-only complete SSD. Shorter prefixes retain the existing K+1 target graph. '
        'All scored latency includes final draining. Diagnostic timings and artificial stress are separate artifacts.']
    summary=dict(status='complete',queue_sha256=q['sha256'],selection_sha256=selected['sha256'],results=results,
        valid_heldout_blocks=len(blocks),heldout_requests=len(exact),exact_matches=sum(c['exact'] for c in exact),
        correctness_passed=all(c['exact'] for c in exact),reference='shared-fixed16-v1',original_baseline_preserved=True,
        fresh_ar_performance_blocks=0,bootstrap_samples=10000)
    atomic_json(folder/'summary.json',summary);(folder/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,protocol in zip(axes,('fixed','natural')):
        rr=[r for r in results if r['protocol']==protocol]
        means=[r['statistics']['mean_latency_s'] for r in rr]
        ci=[r['comparison_to']['legacy_complete']['mean_latency_ci95_s'] for r in rr]
        ax.bar([r['method'] for r in rr],means,yerr=[[max(0,m-a[0]) for m,a in zip(means,ci)],[max(0,a[1]-m) for m,a in zip(means,ci)]],capsize=3)
        ax.set(title=protocol,ylabel='Mean full-request latency (s)')
        ax.tick_params(axis='x',labelrotation=65)
    fig.tight_layout();fig.savefig(folder/'latency.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,5))
    ax.bar([s['config']['label'] for s in selected['scores']],[s['mean_latency_s'] for s in selected['scores']])
    ax.set(ylabel='Mean tuning request latency (s)',title='Tuning only, fixed 256 output tokens')
    ax.tick_params(axis='x',labelrotation=75)
    fig.tight_layout();fig.savefig(folder/'tuning.png',dpi=170);plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',type=Path,default=DEFAULT_OUT)
    p.add_argument('--watch',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--phase',choices=['all','prepare','micro','gate','stress','diagnostic','tune','select','heldout','report'],default='all')
    args=p.parse_args();args.outdir=args.outdir.resolve();args.outdir.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,interrupted)
    with (args.outdir/'.watcher.lock').open('a') as watcher:
        fcntl.flock(watcher,fcntl.LOCK_EX|fcntl.LOCK_NB)
        q=read_json(args.outdir/'queue.json') if (args.outdir/'queue.json').exists() else prepare(args.outdir)
        check_runtime(q)
        if args.dry_run or args.phase=='prepare':
            print('Frozen: GPU microtests,180 gate,12 stress,18 diagnostic,672 tuning,4500 held-out requests. Original baseline preserved.');return 0
        try:
            with (SOURCE_ROOT/'results/baselines/eval1/.campaign.lock').open('a') as lock:
                while True:
                    check_runtime(q)
                    if predecessor_ready(q):
                        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                        except BlockingIOError:pass
                    atomic_json(args.outdir/'status.json',dict(status='waiting_for_predecessor',after=q['after']['path'],at=now()))
                    if not args.watch:return 2
                    time.sleep(30)
                validate_models(q['models'])
                for phase in ('micro','gate','stress','diagnostic','tune'):
                    if args.phase in ('all',phase):
                        if not run_phase(args,q,phase):return 2
                        if args.phase==phase:return 0
                    else:
                        available(args.outdir/phase,load_definition(args.outdir,q,phase),q,True)
                chosen=select(args.outdir,q);freeze_json(args.outdir/'selection.json',chosen)
                freeze_json(args.outdir/'heldout/definition.json',definition(q,'heldout',phase_configs('heldout',chosen),file_digest(args.outdir/'selection.json')))
                if args.phase=='select':return 0
                if args.phase in ('all','heldout'):
                    if not run_phase(args,q,'heldout'):return 2
                    if args.phase=='heldout':return 0
                report(args.outdir,q)
                atomic_json(args.outdir/'status.json',dict(status='complete',report=str(args.outdir/'report.md'),at=now()))
        except BaseException as exc:
            atomic_json(args.outdir/'status.json',dict(status='stopped',reason=str(exc),at=now()));raise
    return 0


if __name__=='__main__':raise SystemExit(main())
