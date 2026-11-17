"""Versioned early-publication correctness gate and dedicated paired comparison.

The study uses nine declared, previously investigated prompts. It is engineering
validation, not the 75-prompt held-out campaign. Pending GPU windows are resumable.
"""
import argparse
import fcntl
import itertools
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import run_engine
from baseline_common import (ROOT, allocation, atomic_json, compute_processes, digest,
    execution_fingerprint, external_processes, file_digest, freeze_json, gpu_inventory,
    read_json, software_fingerprint, validate_models)
from run_baselines import PendingWindow, interrupted, now, require_idle, stop_worker
from benchmark_shared_numerics import first_difference, percentile


def fingerprint():
    return dict(execution=execution_fingerprint(), software=software_fingerprint(),
                supervisor_sha256=file_digest(__file__))


def validate_definition(d):
    if digest({k:v for k,v in d.items() if k!='sha256'}) != d['sha256']:
        raise ValueError('Study definition changed')
    for key, value in fingerprint().items():
        if d[key] != value:
            raise ValueError(f'Cannot resume study: {key} changed')
    if gpu_inventory()['gpus'] != d['hardware']['gpus']:
        raise ValueError('GPU identities or driver changed')


def jobs(methods, repeats, protocols, size):
    rng=random.Random(0)
    permutations=list(itertools.permutations(methods))
    orders=rng.sample(permutations,repeats) if repeats<=len(permutations) else [rng.choice(permutations) for _ in range(repeats)]
    output=[]
    for r,order in enumerate(orders):
        prompt_order=random.Random(1000+r).sample(range(size),size)
        for protocol in (protocols if r%2==0 else list(reversed(protocols))):
            for method in order:
                output.append(dict(id=f'{protocol}_r{r}_{method}',method=method,round=r,
                                   protocol=protocol,prompt_order=prompt_order))
    return output


def prepare(args):
    source=read_json(args.source/'definition.json')
    validate_models(source['models'])
    k,f,sd_k=7,3,6
    selection=None
    if args.baseline_dir:
        from run_baselines import load_campaign,validated_selection
        manifest=load_campaign(args.baseline_dir)
        if manifest.get('target_numerics')!='shared_fixed16':
            raise ValueError('Comparison requires retuned shared-backend baseline')
        selected=validated_selection(args.baseline_dir,manifest)
        k,f=selected['winners']['ssd']['k'],selected['winners']['ssd']['f']
        sd_k=selected['winners']['sd']['k']
        selection=dict(path=str(args.baseline_dir/'selection.json'),sha256=file_digest(args.baseline_dir/'selection.json'))
    if not 1<=args.publish_tokens<=k:
        raise ValueError('Publication depth must be between 1 and the frozen SSD K')
    methods=['ar','sd','complete','early'] if args.phase=='correctness' else ['ar','complete','early']
    d=dict(created_at=now(),purpose='Previously investigated engineering prompts; not held-out headline results',
        reference_contract='shared-fixed16-v1; new AR reference; original AR control retained in prior evidence',
        phase=args.phase,models=source['models'],hardware=gpu_inventory(),prompts=source['prompts'],
        warmup=source['warmup'],methods=methods,repetitions=1 if args.phase=='correctness' else 5,
        protocols=['fixed','natural'],output_cap=256,capacity=4096,
        kv_fraction=.10 if args.phase=='correctness' else .70,k=k,f=f,sd_k=sd_k,
        publish_tokens=args.publish_tokens,selection=selection,
        configuration_status='retuned shared baseline' if selection else 'declared starting K/F; not retuned',
        source=dict(path=str(args.source/'definition.json'),sha256=file_digest(args.source/'definition.json')),
        **fingerprint())
    d['jobs']=jobs(methods,d['repetitions'],d['protocols'],len(d['prompts']))
    d['sha256']=digest(d)
    freeze_json(args.outdir/'definition.json',d)
    return d


def worker(args):
    import torch
    from ssd import LLM,SamplingParams
    d=read_json(args.outdir/'definition.json');validate_definition(d)
    j=next(j for j in d['jobs'] if j['id']==args.job)
    mode='ar' if j['method']=='ar' else 'sd' if j['method']=='sd' else 'ssd'
    model=LLM(d['models']['target']['path'],draft=d['models']['draft']['path'],
        num_gpus=2 if mode=='ssd' else 1,speculate=mode!='ar',draft_async=mode=='ssd',
        speculate_k=d['sd_k'] if mode=='sd' else d['k'],async_fan_out=d['f'],jit_speculate=True,
        target_numerics='shared_fixed16',max_num_seqs=1,max_model_len=d['capacity'],
        gpu_memory_utilization=d['kv_fraction'],draft_gpu_memory_utilization=d['kv_fraction'],
        enable_prefix_cache=False,drain_draft_on_finish=mode=='ssd',
        early_publish_tokens=d['publish_tokens'] if j['method']=='early' else 0)
    sampling=SamplingParams(temperature=0,max_new_tokens=d['output_cap'],ignore_eos=j['protocol']=='fixed')
    for p in d['warmup']:
        model.generate([p['token_ids']],[sampling],use_tqdm=False,stream_callback=lambda *a:None)
    result=dict(definition_sha256=d['sha256'],job=j,requests=[],allocation=allocation(d,mode),
        numerics=d['reference_contract'],profiler=False,cuda_graphs=True,kv_fraction=d['kv_fraction'])
    for i in j['prompt_order']:
        torch.cuda.reset_peak_memory_stats()
        row=run_engine.measure_request(model,d['prompts'][i],sampling,synchronize=torch.cuda.synchronize)
        row['target_peak_memory']=dict(allocated_bytes=torch.cuda.max_memory_allocated(),
                                       reserved_bytes=torch.cuda.max_memory_reserved())
        result['requests'].append(row)
        atomic_json(args.attempt/'result.json',result)
        print(f'{j["id"]}: {row["id"]} done',flush=True)
    model.exit(hard=True)


def validate_result(result,job,d):
    if result['definition_sha256']!=d['sha256'] or result['job']!=job:
        raise ValueError('Result belongs to another study')
    expected=[d['prompts'][i] for i in job['prompt_order']]
    if len(result['requests'])!=len(expected):raise ValueError('Incomplete block')
    for row,p in zip(result['requests'],expected):
        if (row['id']!=p['id'] or row['input_token_ids_sha256']!=digest(p['token_ids']) or
                row['output_tokens']!=len(row['output_token_ids']) or not 1<=row['output_tokens']<=d['output_cap'] or
                (job['protocol']=='fixed' and row['output_tokens']!=d['output_cap'])):
            raise ValueError('Unpaired input or invalid output length')


def block_result(folder,j,d):
    for attempt in sorted((folder/j['id']).glob('attempt_*')):
        path=attempt/'receipt.json'
        if not path.exists():continue
        receipt=read_json(path)
        if not receipt['valid']:continue
        if (receipt.get('definition_sha256')!=d['sha256'] or receipt.get('samples',0)<1 or
            receipt.get('external_processes') or receipt.get('error') or receipt.get('returncode')!=0 or
            receipt['output_sha256']!=file_digest(attempt/'result.json')):
            raise ValueError('Completed receipt or result changed')
        result=read_json(attempt/'result.json');validate_result(result,j,d)
        return result
    return None


def run_block(args,d,j):
    validate_definition(d)
    uuids=[g['uuid'] for g in d['hardware']['gpus']];require_idle(uuids)
    parent=args.outdir/j['id'];parent.mkdir(exist_ok=True)
    attempt=parent/f'attempt_{len(list(parent.glob("attempt_*")))}';attempt.mkdir()
    mode='ar' if j['method']=='ar' else 'sd' if j['method']=='sd' else 'ssd'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids if mode=='ssd' else uuids[:1]))
    env.pop('AO_SPAWN_EXECUTABLE',None)
    for key in list(env):
        if key.startswith('SSD_PROFILE'):env[key]='0'
    if env.get('CUDA_INJECTION64_PATH') or env.get('NVTX_INJECTION64_PATH'):
        raise ValueError('Profiler injection invalidates timing')
    command=[sys.executable,'-O',str(Path(__file__).resolve()),'--worker','--outdir',str(args.outdir),
             '--job',j['id'],'--attempt',str(attempt)]
    process=None;error=None;others=[];samples=0;started=time.monotonic()
    try:
        with (attempt/'worker.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                others=external_processes(compute_processes(uuids),process.pid);samples+=1
                if others or process.poll() is not None:break
                if time.monotonic()-started>900:
                    error='Worker exceeded 900-second budget';break
                time.sleep(1)
    except BaseException as exc:
        error=str(exc) or type(exc).__name__;raise
    finally:
        if process is not None:stop_worker(process)
        output=attempt/'result.json'
        valid=process is not None and process.returncode==0 and samples>0 and not others and error is None
        if valid:
            try:validate_result(read_json(output),j,d)
            except (KeyError,ValueError,OSError) as exc:valid=False;error=str(exc)
        atomic_json(attempt/'receipt.json',dict(valid=valid,definition_sha256=d['sha256'],command=command,
            samples=samples,external_processes=others,error=error,returncode=process.returncode if process else None,
            output_sha256=file_digest(output) if output.exists() else None,at=now()))
    if not valid:
        if others:raise PendingWindow('Other GPU job appeared; contaminated block invalidated')
        raise RuntimeError(f'Study worker failed: {attempt}/worker.log')


def report(folder,d):
    import statistics
    blocks=[block_result(folder,j,d) for j in d['jobs']]
    if any(b is None for b in blocks):raise ValueError('All valid study blocks required')
    records={(b['job']['protocol'],b['job']['round'],b['job']['method'],r['id']):r
             for b in blocks for r in b['requests']}
    tables={};comparisons=[];intervals={};repeatability={}
    for protocol in d['protocols']:
        table={}
        for method in d['methods']:
            rows=[r for (p,_,m,_),r in records.items() if p==protocol and m==method]
            latencies=[r['latency_s'] for r in rows]
            publications=[v for r in rows for v in r['publication_records']]
            table[method]=dict(requests=len(rows),gpu_count=1 if method in ('ar','sd') else 2,
                mean_s=statistics.mean(latencies),p50_s=percentile(latencies,.5),p95_s=percentile(latencies,.95),
                committed_tok_s=sum(r['output_tokens'] for r in rows)/sum(latencies),
                first_delivery_s=statistics.mean(r['first_delivery_s'] for r in rows),
                mean_wait_s=statistics.mean(r['draft_wait_s'] for r in rows),
                mean_final_drain_s=statistics.mean(r['final_drain_s'] for r in rows),
                mean_publication_s=statistics.mean(v['publication_s'] for v in publications) if publications else None,
                mean_interrupt_drain_s=statistics.mean(v['drain_s'] for v in publications) if publications else None,
                partial_publications=sum(v['valid_length']<d['k'] for v in publications),
                verified_draft_tokens=sum(r['verified_draft_tokens'] for r in rows),
                accepted_draft_tokens=sum(r['accepted_draft_tokens'] for r in rows),
                tokens_per_verification=sum(sum(r['accepted_suffix_lens_with_recovery']) for r in rows)/
                    max(1,sum(len(r['accepted_suffix_lens_with_recovery']) for r in rows)),
                max_target_allocated_bytes=max(r['target_peak_memory']['allocated_bytes'] for r in rows),
                max_draft_allocated_bytes=max([v['allocated_bytes'] for r in rows for v in r['draft_peak_memory']] or [0]),
                output_lengths=[r['output_tokens'] for r in rows],
                per_family_mean_s={f:statistics.mean(r['latency_s'] for r in rows if r['family']==f)
                                   for f in ('chat','code','doc')})
            repeatability[f'{protocol}/{method}']=all(len({tuple(records[(protocol,r,method,p['id'])]['output_token_ids'])
                for r in range(d['repetitions'])})==1 for p in d['prompts'])
            if method!='ar':
                for (p,r,m,id_),row in records.items():
                    if p==protocol and m==method:
                        ref=records[(p,r,'ar',id_)]['output_token_ids']
                        comparisons.append(dict(protocol=p,round=r,method=m,id=id_,
                            exact=row['output_token_ids']==ref,first_difference=first_difference(row['output_token_ids'],ref)))
        tables[protocol]=table
        if d['repetitions']>=5:
            rng=random.Random(0);ratios=[]
            families={f:[p['id'] for p in d['prompts'] if p['family']==f] for f in ('chat','code','doc')}
            for _ in range(10000):
                rounds=rng.choices(range(d['repetitions']),k=d['repetitions'])
                ids=[id_ for items in families.values() for id_ in rng.choices(items,k=len(items))]
                a=sum(records[(protocol,r,'complete',id_)]['latency_s'] for r in rounds for id_ in ids)
                b=sum(records[(protocol,r,'early',id_)]['latency_s'] for r in rounds for id_ in ids)
                ratios.append(a/b)
            intervals[protocol]=dict(low=percentile(ratios,.025),high=percentile(ratios,.975),seed=0,samples=10000,
                unit='paired rounds and prompts within families',ratio='complete-cache latency / early latency')
    out=dict(definition_sha256=d['sha256'],valid_blocks=len(blocks),requests=len(records),tables=tables,
        comparisons=comparisons,exact_matches=sum(c['exact'] for c in comparisons),
        correctness_passed=all(c['exact'] for c in comparisons),within_method_repeatability=repeatability,
        paired_speed_ratio_intervals=intervals,limitations=[d['purpose'],d['configuration_status'],
            'Fixed target K+1 graph retained; padded tail is calculated but cannot be accepted.',
            'New AR arithmetic; no original-AR exactness claim.',
            'Uniform checkpoint waiting, NCCL handoff, and final obsolete-work draining are included.'])
    atomic_json(folder/'summary.json',out)
    text=['# Uniform early publication study','',d['purpose'],'',f'Valid blocks: {len(blocks)}. Exact matches: {out["exact_matches"]}/{len(comparisons)}.','']
    for protocol,table in tables.items():
        text += [f'## {protocol}','','| Method | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Partial handoffs |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,row in table.items():
            text.append(f'| {name} | {row["gpu_count"]} | {row["mean_s"]:.4f} | {row["p50_s"]:.4f} | {row["p95_s"]:.4f} | {row["committed_tok_s"]:.2f} | {row["mean_wait_s"]:.4f} | {row["partial_publications"]} |')
        text.append('')
    text += out['limitations']
    (folder/'report.md').write_text('\n'.join(text)+'\n')
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=ROOT/'results/correctness/shared_fixed16_v1_2026-11-17_v2')
    parser.add_argument('--phase',choices=['correctness','compare','report'],default='correctness')
    parser.add_argument('--baseline-dir',type=Path)
    parser.add_argument('--publish-tokens',type=int,default=4)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--attempt',type=Path)
    parser.add_argument('--job')
    args=parser.parse_args();args.outdir=args.outdir.resolve()
    if args.worker:worker(args);return 0
    signal.signal(signal.SIGTERM,interrupted)
    args.outdir.mkdir(parents=True,exist_ok=True)
    with (ROOT/'results/baselines/eval1/.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        d=read_json(args.outdir/'definition.json') if (args.outdir/'definition.json').exists() else prepare(args)
        validate_definition(d);validate_models(d['models'])
        if args.phase=='report':report(args.outdir,d);return 0
        try:
            for j in d['jobs']:
                if block_result(args.outdir,j,d) is not None:continue
                while True:
                    try:
                        atomic_json(args.outdir/'status.json',dict(status='running',job=j['id'],at=now()))
                        print('Running '+j['id'],flush=True);run_block(args,d,j);break
                    except PendingWindow as exc:
                        atomic_json(args.outdir/'status.json',dict(status='pending_dedicated_gpus',reason=str(exc),at=now()))
                        if not args.watch:return 2
                        time.sleep(30)
            summary=report(args.outdir,d)
            atomic_json(args.outdir/'status.json',dict(status='complete' if summary['correctness_passed'] else 'correctness_failed',at=now()))
            return 0 if summary['correctness_passed'] else 1
        except BaseException as exc:
            atomic_json(args.outdir/'status.json',dict(status='stopped',reason=str(exc),at=now()));raise


if __name__=='__main__':raise SystemExit(main())
