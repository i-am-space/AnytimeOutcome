"""Append-only overnight queue; never edits the active frozen campaign.

After its dependency completes: tune publication depth on tuning prompts only,
profile the frozen winner, broaden correctness, and run a fresh paired held-out
SSD comparison. Wait for dedicated GPUs; invalidate contaminated attempts.
"""
import argparse
import fcntl
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import zipfile

import run_engine  # Also installs the optimized-Python workaround in spawned workers.
from baseline_common import (ROOT, allocation, atomic_json, compute_processes, digest,
    execution_fingerprint, external_processes, file_digest, freeze_json, gpu_inventory,
    read_json, software_fingerprint, validate_manifest, validate_models)
from run_baselines import (PendingWindow, accepted_result, interrupted, load_campaign, now,
                           require_idle, stop_worker, validated_selection)
from analyze_baselines import paired_bootstrap, row_statistics
from benchmark_shared_numerics import first_difference

PHASES=('depth_tune','profiles','correctness','heldout')


def code_files():
    names=('run_overnight_extensions.py','run_publication_study.py','benchmark_shared_numerics.py',
           'analyze_nsys.py','run_next_stage.py')
    return {f'profiling/{name}':file_digest(ROOT/'profiling'/name) for name in names}


def check_runtime(q,profile=False):
    if execution_fingerprint()!=q['execution'] or code_files()!=q['queue_code']:
        raise ValueError('Queued execution code changed; preserve results and prepare a new queue version')
    software=software_fingerprint()
    expected=q['software']
    if profile:
        # Nsight may append injection libraries. Keep all other software/env checks exact.
        software=read_copy(software);expected=read_copy(expected)
        preload=software['environment'].pop('LD_PRELOAD')
        required=expected['environment'].pop('LD_PRELOAD')
        if required and not all(p in (preload or '').split(':') for p in required.split(':')):
            raise ValueError('Profile worker lost the declared runtime libraries')
    if software!=expected:raise ValueError('Queued software/environment changed')
    if gpu_inventory()['gpus']!=q['hardware']['gpus']:
        raise ValueError('GPU identities or driver changed')
    if digest({k:v for k,v in q.items() if k!='sha256'})!=q['sha256']:
        raise ValueError('Queue manifest was modified')


def profiler_fingerprint():
    path=shutil.which('nsys')
    if not path:raise ValueError('Nsight Systems unavailable')
    return dict(executable=str(Path(path).resolve()),
                version=subprocess.check_output([path,'--version'],text=True,timeout=15).strip())


def read_copy(value):
    import copy
    return copy.deepcopy(value)


def boundary_prompts(tokenizer,bases,lengths=(2040,2047,2048)):
    """Insert synthetic filler inside user content; never truncate the original input."""
    marker=tokenizer.encode('<|im_start|>user\n',add_special_tokens=False)
    filler=tokenizer.encode(' Context for a numerical boundary test.'*2048,add_special_tokens=False)
    output=[]
    for base in bases:
        original=base['token_ids']
        start=next((i+len(marker) for i in range(len(original)-len(marker)+1)
                    if original[i:i+len(marker)]==marker),None)
        if start is None:raise ValueError('Cannot identify frozen user-content boundary')
        for length in lengths:
            extra=length-len(original)
            if extra<0 or extra>len(filler):raise ValueError('Invalid boundary filler length')
            ids=original[:start]+filler[:extra]+original[start:]
            output.append(dict(id=f'boundary-{base["id"]}-{length}',family=base['family'],
                category='synthetic_context_boundary',token_ids=ids,source_id=base['id'],
                original_input_sha256=digest(original),synthetic_filler_tokens=extra,
                description='Synthetic filler inserted inside user content; complete source prompt retained'))
    return output


def enqueue(args):
    manifest=load_campaign(args.dependency/'baselines')
    if manifest.get('target_numerics')!='shared_fixed16':raise ValueError('Require the new shared baseline')
    if software_fingerprint()!=manifest['software']:
        raise ValueError('Queue software must match the active frozen baseline')
    source_path=ROOT/'results/correctness/shared_fixed16_v1_2026-11-17_v2/definition.json'
    source=read_json(source_path)
    wanted={'specbench-148','HumanEval/36','specbench-496'}
    bases=[p for p in source['prompts'] if p['id'] in wanted]
    if len(bases)!=3 or {p['family'] for p in bases}!={'chat','code','doc'}:
        raise ValueError('Missing declared correctness/profile coverage prompts')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(manifest['models']['target']['path'],local_files_only=True)
    stress=bases+boundary_prompts(tokenizer,bases)
    q=dict(created_at=now(),dependency=str(args.dependency),baseline_manifest_sha256=manifest['manifest_sha256'],
        execution=manifest['execution'],software=software_fingerprint(),queue_code=code_files(),
        hardware=manifest['hardware'],models=manifest['models'],warmup=manifest['splits']['warmup'],
        tuning_prompts=manifest['splits']['tune'],heldout_prompts=manifest['splits']['heldout'],
        profile_prompts=bases,correctness_prompts=stress,seed=0,output_cap=256,capacity=4096,
        reference_contract='shared-fixed16-v1; new AR arithmetic; original AR remains a separate control',
        depth_tuning=dict(grid_rule='sorted unique {1,2,4,K}, plus complete-cache P=0 control',
            repetitions=2,protocol='fixed',objective='lowest balanced mean full-request latency among P>0',
            exact_tie_rule='smaller P',selection_data='24 tuning prompts only; no held-out results'),
        profiler=profiler_fingerprint(),
        profiles=dict(methods=['ar','complete','early'],requests_each=3,protocol='fixed',profiler='Nsight Systems',
                      use_for_headline_timing=False),
        correctness=dict(k_values=[4,7,10],methods=['ar','sd','complete','early'],protocols=['fixed','natural'],
            inputs='3 seen coverage prompts plus 9 synthetic boundary prompts; not held-out performance',kv_fraction=.10),
        heldout=dict(methods=['complete','early'],repetitions=5,protocols=['fixed','natural'],
            prompts=75,requests=1500,controls='fresh interleaved SSD arms; baseline AR reused for tokens only'),
        kv_fraction=.70,bootstrap_samples=10000,dedicated_gpus=True)
    q['sha256']=digest(q)
    check_runtime(q)
    freeze_json(args.outdir/'queue.json',q)
    with zipfile.ZipFile(args.outdir/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in dict(q['execution']['files'],**q['queue_code']):archive.write(ROOT/name,name)
        archive.write(args.outdir/'queue.json','queue.json')
    atomic_json(args.outdir/'status.json',dict(status='waiting_for_dependency',dependency=str(args.dependency),at=now()))
    (args.outdir/'plan.md').write_text('''# Overnight additions

Wait for the existing baseline/publication pipeline to complete successfully.

1. Tune P on 24 tuning prompts, two fixed-256 repetitions: unique P=1,2,4,K,
   plus complete-cache P=0. Freeze the best early P; exact ties choose smaller P.
2. Capture fresh shared AR / complete SSD / selected early SSD Nsight Systems
   traces on three seen prompts. Profiled timings are diagnostic only.
3. Check K=4,7,10 with AR/SD/complete SSD/early SSD, both EOS protocols, on three
   seen coverage prompts and nine synthetic inputs of 2040/2047/2048 tokens.
4. Evaluate selected P on all 75 held-out prompts: fresh paired complete/early
   SSD, both protocols, five rounds (1500 requests). Baseline AR supplies exact
   token references only. Use 10,000 paired hierarchical bootstrap draws.

All workers are serial, UUID pinned, prefix caching disabled, graphs enabled,
greedy BF16, capacity4096, with three separate warmups. Both GPUs must remain
dedicated. Contaminated attempts are invalidated; unrelated jobs are untouched.
Code/environment changes and correctness failures stop dependent measurements.
''')
    return q


def load_queue(folder):
    q=read_json(folder/'queue.json');check_runtime(q);return q


def dependency_state(q):
    path=Path(q['dependency'])/'status.json'
    if not path.exists():return 'waiting'
    status=read_json(path)['status']
    if status=='complete':return 'ready'
    if status in ('stopped','correctness_failed','complete_correctness_failed'):
        raise ValueError('Existing pipeline stopped; resolve its error before overnight additions')
    return 'waiting'


def config(label,mode,k,f,p=0):
    return dict(label=label,mode=mode,k=k,f=f,p=p)


def make_jobs(configs,repeats,protocols,size,tuning=False):
    order=list(configs);random.Random(0).shuffle(order)
    result=[]
    for r in range(repeats):
        if tuning:current=order if r==0 else list(reversed(order))
        elif len(order)==2:current=order if r%2==0 else list(reversed(order))
        else:current=random.Random(r).sample(order,len(order))
        prompt_order=random.Random(1000+r).sample(range(size),size)
        for protocol in (protocols if r%2==0 else list(reversed(protocols))):
            for cfg in current:
                result.append(dict(id=f'{protocol}_r{r}_{cfg["label"]}',config=cfg,round=r,
                                   protocol=protocol,prompt_order=prompt_order))
    return result


def phase_definition(folder,q,phase,baseline_selection,depth=None):
    ssd=baseline_selection['winners']['ssd'];k,f=ssd['k'],ssd['f']
    if phase=='depth_tune':
        ps=sorted({1,2,4,k})
        cfgs=[config('complete','ssd',k,f)]+[config(f'p{p}','ssd',k,f,p) for p in ps]
        prompts=q['tuning_prompts'];repeats=2;protocols=['fixed']
    elif phase=='profiles':
        cfgs=[config('ar','ar',k,f),config('complete','ssd',k,f),config('early','ssd',k,f,depth['winner_p'])]
        prompts=q['profile_prompts'];repeats=1;protocols=['fixed']
    elif phase=='correctness':
        cfgs=[config('ar','ar',k,f)]
        for value in (4,7,10):
            cfgs += [config(f'sd_k{value}','sd',value,f),config(f'complete_k{value}','ssd',value,f),
                     config(f'early_k{value}','ssd',value,f,min(value,depth['winner_p']))]
        prompts=q['correctness_prompts'];repeats=1;protocols=['fixed','natural']
    else:
        cfgs=[config('complete','ssd',k,f),config('early','ssd',k,f,depth['winner_p'])]
        prompts=q['heldout_prompts'];repeats=5;protocols=['fixed','natural']
    body=dict(queue_sha256=q['sha256'],phase=phase,configurations=cfgs,prompts=prompts,
        warmup=q['warmup'],repetitions=repeats,protocols=protocols,output_cap=256,capacity=4096,
        kv_fraction=.10 if phase=='correctness' else .70,profile_capture=phase=='profiles',
        baseline_selection_sha256=baseline_selection['selection_sha256'],
        depth_selection_sha256=depth['sha256'] if depth else None)
    body['jobs']=make_jobs(cfgs,repeats,protocols,len(prompts),tuning=phase=='depth_tune')
    body['sha256']=digest(body);freeze_json(folder/'definition.json',body)
    return body


def load_phase(folder,q):
    d=read_json(folder/'definition.json')
    if d['queue_sha256']!=q['sha256'] or digest({k:v for k,v in d.items() if k!='sha256'})!=d['sha256']:
        raise ValueError('Phase definition changed')
    return d


def worker(args):
    import torch
    from ssd import LLM,SamplingParams
    q=read_json(args.outdir/'queue.json')
    folder=args.outdir/args.phase;d=load_phase(folder,q);check_runtime(q,d['profile_capture'])
    j=next(j for j in d['jobs'] if j['id']==args.job);c=j['config'];mode=c['mode']
    model=LLM(q['models']['target']['path'],draft=q['models']['draft']['path'],
        target_numerics='shared_fixed16',num_gpus=2 if mode=='ssd' else 1,
        speculate=mode!='ar',draft_async=mode=='ssd',speculate_k=c['k'],async_fan_out=c['f'],
        jit_speculate=True,max_num_seqs=1,max_model_len=4096,enable_prefix_cache=False,
        gpu_memory_utilization=d['kv_fraction'],draft_gpu_memory_utilization=d['kv_fraction'],
        early_publish_tokens=c['p'],drain_draft_on_finish=mode=='ssd')
    sampling=SamplingParams(temperature=0,max_new_tokens=256,ignore_eos=j['protocol']=='fixed')
    for p in d['warmup']:
        model.generate([p['token_ids']],[sampling],use_tqdm=False,stream_callback=lambda *a:None)
    torch.cuda.synchronize()
    result=dict(queue_sha256=q['sha256'],definition_sha256=d['sha256'],job=j,requests=[],
        hardware_allocation=allocation(q,mode),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        actual_dtype=str(model.config.hf_config.torch_dtype),cuda_graphs=True,prefix_cache=False,
        kv_fraction=d['kv_fraction'],profile_capture=d['profile_capture'],software=software_fingerprint())
    if d['profile_capture']:
        torch.cuda.profiler.start();torch.cuda.nvtx.range_push('timed_region')
    try:
        for i in j['prompt_order']:
            torch.cuda.reset_peak_memory_stats()
            row=run_engine.measure_request(model,d['prompts'][i],sampling,
                synchronize=torch.cuda.synchronize,profile_capture=d['profile_capture'])
            row['target_peak_memory']=dict(allocated_bytes=torch.cuda.max_memory_allocated(),
                                           reserved_bytes=torch.cuda.max_memory_reserved())
            result['requests'].append(row)
            if not d['profile_capture']:atomic_json(args.attempt/'result.json',result)
    finally:
        if d['profile_capture']:
            torch.cuda.nvtx.range_pop();torch.cuda.profiler.stop()
    atomic_json(args.attempt/'result.json',result)
    model.exit(hard=True)


def validate_result(result,j,d,q):
    if result['definition_sha256']!=d['sha256'] or result['queue_sha256']!=q['sha256'] or result['job']!=j:
        raise ValueError('Worker result provenance differs')
    mode=j['config']['mode'];uuids=[g['uuid'] for g in q['hardware']['gpus']]
    if (result['hardware_allocation']!=allocation(q,mode) or result['actual_dtype']!='torch.bfloat16' or
        result['cuda_visible_devices']!=','.join(uuids if mode=='ssd' else uuids[:1]) or
        result['cuda_graphs'] is not True or result['prefix_cache'] is not False or
        result['kv_fraction']!=d['kv_fraction'] or result['profile_capture']!=d['profile_capture']):
        raise ValueError('Worker settings differ from frozen protocol')
    expected=[d['prompts'][i] for i in j['prompt_order']]
    if len(result['requests'])!=len(expected):raise ValueError('Incomplete worker block')
    for row,p in zip(result['requests'],expected):
        if (row['id']!=p['id'] or row['family']!=p['family'] or
            row['input_token_ids_sha256']!=digest(p['token_ids']) or
            row['output_tokens']!=len(row['output_token_ids']) or not 1<=row['output_tokens']<=256 or
            (j['protocol']=='fixed' and row['output_tokens']!=256) or
            not math.isfinite(row['latency_s']) or row['latency_s']<=0 or
            not 0<=row['accepted_draft_tokens']<=row['verified_draft_tokens']):
            raise ValueError('Incomplete or unpaired request evidence')


def block_result(folder,j,d,q):
    for attempt in sorted((folder/j['id']).glob('attempt_*')):
        path=attempt/'receipt.json'
        if not path.exists():continue
        r=read_json(path)
        if r.get('valid') is not True:continue
        if (r['definition_sha256']!=d['sha256'] or r['returncode']!=0 or r['samples']<1 or
            r['external_processes'] or r['error'] or r['output_sha256']!=file_digest(attempt/'result.json')):
            raise ValueError('Completed receipt/output changed')
        for name,sha in r.get('artifacts',{}).items():
            if file_digest(attempt/name)!=sha:raise ValueError('Completed trace/analysis changed')
        result=read_json(attempt/'result.json');validate_result(result,j,d,q)
        return result
    return None


def worker_command(args,j,d,attempt):
    command=[sys.executable,'-O',str(Path(__file__).resolve()),'--worker','--outdir',str(args.outdir),
             '--phase',d['phase'],'--job',j['id'],'--attempt',str(attempt)]
    if d['profile_capture']:
        if shutil.disk_usage(args.outdir).free<10*1024**3:
            raise ValueError('Require 10 GiB free before a new trace')
        command=['nsys','profile','--trace=cuda,nvtx,osrt','--cuda-graph-trace=node',
            '--gpu-metrics-devices=all','--gpu-metrics-frequency=20000',
            '--capture-range=cudaProfilerApi','--capture-range-end=stop',
            '--sample=none','--cpuctxsw=none','-o',str(attempt/'trace'),*command]
    return command


def analyze_profile(attempt,j):
    sqlite=attempt/'trace.sqlite'
    subprocess.run(['nsys','export','--type','sqlite','--output',str(sqlite),str(attempt/'trace.nsys-rep')],check=True)
    subprocess.run([sys.executable,str(ROOT/'profiling/analyze_nsys.py'),str(sqlite),
        '--name',j['config']['label'],'--out',str(attempt/'analysis')],check=True)
    # Do not use the old complete-tree critical-path estimator for interrupted trees.
    return {str(p.relative_to(attempt)):file_digest(p) for p in attempt.rglob('*')
            if p.is_file() and (p.suffix in ('.sqlite','.nsys-rep','.json','.png','.txt')) and p.name!='receipt.json'}


def run_block(args,d,j,q):
    check_runtime(q);uuids=[g['uuid'] for g in q['hardware']['gpus']];require_idle(uuids)
    folder=args.outdir/d['phase'];parent=folder/j['id'];parent.mkdir(parents=True,exist_ok=True)
    attempt=parent/f'attempt_{len(list(parent.glob("attempt_*")))}';attempt.mkdir()
    command=worker_command(args,j,d,attempt)
    mode=j['config']['mode']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids if mode=='ssd' else uuids[:1]))
    env.pop('AO_SPAWN_EXECUTABLE',None)
    for key in list(env):
        if key.startswith('SSD_PROFILE'):env[key]='0'
    if env.get('CUDA_INJECTION64_PATH') or env.get('NVTX_INJECTION64_PATH'):
        raise ValueError('Unexpected profiler injection in queue supervisor')
    process=None;others=[];samples=0;error=None;artifacts={};started=time.monotonic()
    limit=1800 if len(d['prompts'])>=75 else 1200
    try:
        with (attempt/'worker.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                others=external_processes(compute_processes(uuids),process.pid);samples+=1
                if others or process.poll() is not None:break
                if time.monotonic()-started>limit:error='Worker exceeded declared time budget';break
                time.sleep(1)
    except BaseException as exc:
        error=str(exc) or type(exc).__name__;raise
    finally:
        if process is not None:stop_worker(process)
        valid=process is not None and process.returncode==0 and samples>0 and not others and error is None
        output=attempt/'result.json'
        if valid:
            try:
                validate_result(read_json(output),j,d,q)
                if d['profile_capture']:artifacts=analyze_profile(attempt,j)
            except (KeyError,ValueError,OSError,subprocess.SubprocessError) as exc:valid=False;error=str(exc)
        atomic_json(attempt/'receipt.json',dict(valid=valid,definition_sha256=d['sha256'],command=command,
            returncode=process.returncode if process else None,samples=samples,external_processes=others,
            error=error,output_sha256=file_digest(output) if output.exists() else None,artifacts=artifacts,at=now()))
    if not valid:
        if others:raise PendingWindow('Contaminated block invalidated; waiting for dedicated GPUs')
        raise RuntimeError(f'Worker or trace analysis failed: {attempt}/worker.log; {error}')


def all_results(folder,d,q):
    values=[block_result(folder,j,d,q) for j in d['jobs']]
    if any(v is None for v in values):raise ValueError('Complete valid phase evidence required')
    return values


def baseline_ar(q,phase):
    """Read only AR token references from the declared split, never performance for selection."""
    baseline=Path(q['dependency'])/'baselines'
    manifest=read_json(baseline/'manifest.json')
    if manifest['manifest_sha256']!=q['baseline_manifest_sha256']:raise ValueError('Baseline manifest changed')
    if phase=='depth_tune':
        from baseline_common import tuning_jobs
        jobs=[j for j in tuning_jobs(manifest) if j['config']['mode']=='ar']
    else:
        from baseline_common import heldout_jobs
        jobs=[j for j in heldout_jobs(manifest,read_json(baseline/'selection.json')) if j['config']['mode']=='ar']
    return {(j['protocol'],j['repeat'],r['id']):r for j in jobs
            for r in accepted_result(baseline,j,manifest)['requests']}


def exact_checks(results,references):
    checks=[]
    for b in results:
        j=b['job']
        for r in b['requests']:
            ref=references[(j['protocol'],j['round'],r['id'])]
            if r['input_token_ids_sha256']!=ref['input_token_ids_sha256']:
                raise ValueError('Exactness references have different input tokens')
            index=first_difference(ref['output_token_ids'],r['output_token_ids'])
            checks.append(dict(id=r['id'],round=j['round'],protocol=j['protocol'],config=j['config']['label'],
                               exact=index is None,first_difference=index))
    return checks


def select_depth(folder,d,q):
    if d['phase']!='depth_tune':raise ValueError('Publication selection accepts tuning evidence only')
    results=all_results(folder,d,q)
    checks=exact_checks(results,baseline_ar(q,'depth_tune'))
    if not all(c['exact'] for c in checks):raise ValueError('Tuning exactness failed; do not select P')
    scores=[]
    for cfg in d['configurations']:
        rows=[r for b in results if b['job']['config']==cfg for r in b['requests']]
        scores.append(dict(config=cfg,mean_latency_s=sum(r['latency_s'] for r in rows)/len(rows),requests=len(rows)))
    winner=min((r for r in scores if r['config']['p']>0),key=lambda r:(r['mean_latency_s'],r['config']['p']))
    control=next(r for r in scores if r['config']['p']==0)
    body=dict(queue_sha256=q['sha256'],tuning_definition_sha256=d['sha256'],winner_p=winner['config']['p'],
        k=winner['config']['k'],f=winner['config']['f'],scores=scores,
        selection_rule=q['depth_tuning'],tuning_speed_ratio=control['mean_latency_s']/winner['mean_latency_s'],
        scope='Selection on tuning only; this ratio is not held-out performance evidence')
    body['sha256']=digest(body);freeze_json(folder/'selection.json',body)
    return body


def report_phase(folder,d,q):
    results=all_results(folder,d,q)
    records={(b['job']['protocol'],b['job']['round'],b['job']['config']['label'],r['id']):r
             for b in results for r in b['requests']}
    if d['phase'] in ('depth_tune','heldout'):
        checks=exact_checks(results,baseline_ar(q,d['phase']))
    else:
        refs={(p,r,id_):row for (p,r,label,id_),row in records.items() if label=='ar'}
        checks=exact_checks([b for b in results if b['job']['config']['mode']!='ar'],refs)
    correctness=all(c['exact'] for c in checks)
    statistics=[];intervals={};repeatability={}
    for protocol in d['protocols']:
        for cfg in d['configurations']:
            rows=[row for (p,_,label,_),row in records.items() if p==protocol and label==cfg['label']]
            pubs=[v for row in rows for v in row['publication_records']]
            statistics.append(dict(protocol=protocol,config=cfg,gpu_count=2 if cfg['mode']=='ssd' else 1,
                **row_statistics(rows),mean_wait_s=sum(r['draft_wait_s'] for r in rows)/len(rows),
                mean_final_drain_s=sum(r['final_drain_s'] for r in rows)/len(rows),
                mean_publication_wall_s=sum(v['publication_s'] for v in pubs)/len(pubs) if pubs else None,
                mean_interrupt_drain_s=sum(v['drain_s'] for v in pubs)/len(pubs) if pubs else None,
                partial_handoffs=sum(v['valid_length']<cfg['k'] for v in pubs),
                ready_length_histogram={str(n):sum(v['valid_length']==n for v in pubs)
                                       for n in sorted({v['valid_length'] for v in pubs})},
                accepted_draft_tokens=sum(r['accepted_draft_tokens'] for r in rows),
                verified_draft_tokens=sum(r['verified_draft_tokens'] for r in rows),
                max_target_allocated_bytes=max(r['target_peak_memory']['allocated_bytes'] for r in rows),
                max_target_reserved_bytes=max(r['target_peak_memory']['reserved_bytes'] for r in rows),
                max_draft_allocated_bytes=max([v['allocated_bytes'] for r in rows for v in r['draft_peak_memory']] or [0]),
                max_draft_reserved_bytes=max([v['reserved_bytes'] for r in rows for v in r['draft_peak_memory']] or [0]),
                output_lengths=[r['output_tokens'] for r in rows],
                per_family={family:row_statistics([r for r in rows if r['family']==family])
                            for family in ('chat','code','doc')}))
            repeatability[f'{protocol}/{cfg["label"]}']=all(len({tuple(records[(protocol,r,cfg['label'],p['id'])]['output_token_ids'])
                for r in range(d['repetitions'])})==1 for p in d['prompts'])
        if d['phase']=='heldout':
            for family in ('all','chat','code','doc'):
                prompts=[p for p in d['prompts'] if family=='all' or p['family']==family]
                a=[[records[(protocol,r,'complete',p['id'])]['latency_s'] for p in prompts] for r in range(5)]
                b=[[records[(protocol,r,'early',p['id'])]['latency_s'] for p in prompts] for r in range(5)]
                intervals[f'{protocol}/{family}']=paired_bootstrap(a,b,[p['family'] for p in prompts],10000,0)
    out=dict(queue_sha256=q['sha256'],definition_sha256=d['sha256'],phase=d['phase'],valid_blocks=len(results),
        requests=len(records),statistics=statistics,comparisons=checks,correctness_passed=correctness,
        exact_matches=sum(c['exact'] for c in checks),repeatability=repeatability,paired_intervals=intervals,
        performance_evidence=d['phase']=='heldout',limitations=[
            'Exactness is against shared-fixed16-v1 AR, not original historical AR.',
            'Profiled and synthetic boundary timings are diagnostic only.',
            'Target graph retains padded K+1 work; all final draft draining is included.',
            'Publication wall time includes outstanding GPU completion in complete-cache SSD.'])
    atomic_json(folder/'summary.json',out)
    lines=[f'# Overnight {d["phase"]} results','',f'Valid blocks: {len(results)}; requests: {len(records)}.',
        f'Exact outputs: {out["exact_matches"]}/{len(checks)} against shared AR.','',
        '| Protocol | Configuration | GPUs | Mean s | p50 s | p95 s | tok/s | Wait s | Partial handoffs |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in statistics:
        lines.append(f'| {row["protocol"]} | {row["config"]["label"]} | {row["gpu_count"]} | '
            f'{row["mean_latency_s"]:.4f} | {row["p50_latency_s"]:.4f} | {row["p95_latency_s"]:.4f} | '
            f'{row["committed_tok_per_s"]:.2f} | {row["mean_wait_s"]:.4f} | {row["partial_handoffs"]} |')
    if intervals:
        lines+=['','| Protocol / family | Complete / early latency | Paired 95% CI |','|---|---:|---|']
        for key,value in intervals.items():
            lines.append(f'| {key} | {value["speedup"]:.3f}× | {value["speedup_ci95"][0]:.3f}–{value["speedup_ci95"][1]:.3f} |')
    lines+=['',*out['limitations']]
    (folder/'report.md').write_text('\n'.join(lines)+'\n')
    plot_phase(folder,out)
    if not correctness or not all(repeatability.values()):
        raise ValueError(f'{d["phase"]} exactness/repeatability failed; dependent stages stopped')
    return out


def plot_phase(folder,out):
    if out['phase'] not in ('depth_tune','heldout'):return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=out['statistics']
    fig,axes=plt.subplots(1,2 if out['phase']=='heldout' else 1,figsize=(10,4),squeeze=False)
    protocols=['fixed','natural'] if out['phase']=='heldout' else ['fixed']
    for ax,protocol in zip(axes[0],protocols):
        selected=[r for r in rows if r['protocol']==protocol]
        ax.bar([r['config']['label'] for r in selected],[r['mean_latency_s'] for r in selected])
        ax.set(ylabel='Mean full-request latency (s)',title=f'{out["phase"]}: {protocol}')
    fig.tight_layout();fig.savefig(folder/'latency.png',dpi=160);plt.close(fig)


def run_phase(args,d,q):
    folder=args.outdir/d['phase']
    for j in d['jobs']:
        if block_result(folder,j,d,q) is not None:continue
        while True:
            try:
                atomic_json(args.outdir/'status.json',dict(status='running',phase=d['phase'],job=j['id'],at=now()))
                print(f'{d["phase"]}: {j["id"]}',flush=True);run_block(args,d,j,q);break
            except PendingWindow as exc:
                atomic_json(args.outdir/'status.json',dict(status='pending_dedicated_gpus',phase=d['phase'],reason=str(exc),at=now()))
                time.sleep(30)
    return report_phase(folder,d,q)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--dependency',type=Path,default=ROOT/'results/next_stage/shared_uniform_v1_2026-11-17')
    parser.add_argument('--enqueue',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--phase',choices=PHASES)
    parser.add_argument('--job')
    parser.add_argument('--attempt',type=Path)
    args=parser.parse_args();args.outdir=args.outdir.resolve();args.dependency=args.dependency.resolve()
    if args.worker:worker(args);return 0
    signal.signal(signal.SIGTERM,interrupted)
    args.outdir.mkdir(parents=True,exist_ok=True)
    if args.enqueue:
        enqueue(args);print('All four additions frozen in '+str(args.outdir/'queue.json'));return 0
    q=load_queue(args.outdir)
    if args.dry_run:
        print('Waiting on '+q['dependency'])
        print('Depth tuning: 192–240 requests; profiles: 9; correctness: 240; paired held-out SSD: 1500.')
        return 0
    with (args.outdir/'.queue.lock').open('a') as own_lock:
        fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        phase='dependency'
        try:
            while dependency_state(q)!='ready':
                check_runtime(q)
                atomic_json(args.outdir/'status.json',dict(status='waiting_for_dependency',at=now(),dependency=q['dependency']))
                time.sleep(30)
            from run_next_stage import require_baseline
            baseline=Path(q['dependency'])/'baselines';require_baseline(baseline)
            manifest=load_campaign(baseline)
            if manifest['manifest_sha256']!=q['baseline_manifest_sha256']:
                raise ValueError('Dependency baseline manifest changed after queue preparation')
            selection=validated_selection(baseline,manifest)
            validate_models(q['models'])
            if profiler_fingerprint()!=q['profiler']:
                raise ValueError('Nsight Systems version changed after queue preparation')
            with (ROOT/'results/baselines/eval1/.campaign.lock').open('a') as gpu_lock:
                fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                depth=None
                for phase in PHASES:
                    check_runtime(q)
                    folder=args.outdir/phase;folder.mkdir(exist_ok=True)
                    d=phase_definition(folder,q,phase,selection,depth)
                    run_phase(args,d,q)
                    if phase=='depth_tune':depth=select_depth(folder,d,q)
                atomic_json(args.outdir/'status.json',dict(status='complete',at=now(),
                    heldout_report=str(args.outdir/'heldout/report.md')))
            return 0
        except BaseException as exc:
            atomic_json(args.outdir/'status.json',dict(status='stopped',phase=phase,reason=str(exc),at=now()));raise


if __name__=='__main__':raise SystemExit(main())
