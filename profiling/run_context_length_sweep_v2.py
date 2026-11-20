"""Queue a controlled context-length diagnostic after the publication campaign.

No GPU worker starts before validated predecessor completion and acquisition of
the shared campaign lock. Existing engine/campaign files remain unchanged.
"""
import argparse
import copy
import csv
import fcntl
import itertools
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import uuid
import zipfile

import run_publication_reuse_ar as predecessor
import run_engine
from analyze_context_sweep import analyze
from analyze_baselines import paired_bootstrap
from baseline_common import (ROOT, FAMILIES, allocation, atomic_json, compute_processes,
    digest, execution_fingerprint, file_digest, freeze_json, gpu_inventory, read_json,
    software_fingerprint, validate_models)
from gpu_process_ownership import TOKEN_ENV, partition, cleanup_tagged
from run_baselines import PendingWindow, interrupted, now, require_idle, stop_worker
from run_overnight_extensions import profiler_fingerprint, validate_result as validate_engine_result

DEFAULT_OUT = ROOT/'results/next_stage/context_length_v2_2026-11-18'
DEFAULT_AFTER = predecessor.DEFAULT_OUT
SPEC = dict(seed=0, input_lengths=[256,512,1024,2048], output_cap=256,
    repetitions=3, capacity=4096, kv_fraction=.70, f=3, k=[7,10], p=4, window=2,
    precision='bfloat16', batch_size=1, cuda_graphs=True, prefix_cache=False,
    protocol='fixed', ignore_eos=True, bootstrap_samples=10000, monitor_seconds=1,
    scope='Controlled synthetic length diagnostic; three task templates, not held-out generalization',
    warmup='One separate task per family at every measured length before each worker block',
    profiling='Separate Nsight blocks; never mix profiled latency with unprofiled results')
TASKS = {
    'chat': 'Explain in detail how a student can organize a week of study. Ignore the background notes.',
    'code': 'Write a Python function that computes the length of the longest increasing subsequence. '
            'Explain its approach and complexity. Ignore the background notes.',
    'doc': 'Document: Aster Lab opened in Pune in 2019. Meera Sen leads its robotics team. '
           'Workshops run on Mondays and Wednesdays. The lab studies low-cost assistive robots. '
           'Using only this document, describe the lab, its schedule, and its research. Ignore the background notes.',
}
WARM_TASKS = {
    'chat': 'Explain several ways to prepare for an examination. Ignore the background notes.',
    'code': 'Write a Python function that merges two sorted lists. Explain its complexity. Ignore the background notes.',
    'doc': 'Document: Cedar Center opened in Kochi in 2020. Arun leads its reading group. '
           'Meetings run on Tuesdays. Using only this document, describe the center. Ignore the background notes.',
}


def code_files():
    names = ('run_context_length_sweep_v2.py','run_context_length_sweep.py','analyze_context_sweep.py',
        'run_publication_reuse_ar.py','saved_ar_reference.py','run_publication_campaign.py',
        'run_adaptive_publication.py','run_overnight_extensions.py','run_publication_study.py',
        'benchmark_shared_numerics.py','gpu_process_ownership.py')
    return {f'profiling/{name}':file_digest(ROOT/'profiling'/name) for name in names}


def configurations():
    return [predecessor.config('ar')] + [predecessor.config(method,k,0 if method=='complete' else 4,2)
        for k in SPEC['k'] for method in ('complete','uniform','overlap')]


def make_prompts(tokenizer, warm=False):
    """Generate exact complete chat inputs by adding words, never truncation."""
    result = []
    for family, task in (WARM_TASKS if warm else TASKS).items():
        for length in SPEC['input_lengths']:
            n = 0
            for _ in range(12):
                text = 'Background notes:' + ' note'*n + '\n\nTask: ' + task
                tokens = tokenizer.apply_chat_template([dict(role='user',content=text)],
                    tokenize=True,add_generation_prompt=True,enable_thinking=False)
                difference = length-len(tokens)
                if difference == 0: break
                n += difference
                if n < 0: raise ValueError('Complete task exceeds declared input length')
            else:
                raise ValueError('Cannot construct the exact context length without truncation')
            result.append(dict(id=f'{"warm" if warm else "sweep"}-{family}-{length}',
                family=family,category='synthetic_controlled_context',text=text,token_ids=tokens,
                token_ids_sha256=digest(tokens),context_length=length,synthetic_filler_words=n,
                source_id=f'synthetic-{family}-template-v1',complete_task=task))
    return result


def validate_prompts(prompts, warmup):
    seen = set()
    for rows in (prompts,warmup):
        if len(rows) != len(FAMILIES)*len(SPEC['input_lengths']):
            raise ValueError('Wrong controlled workload size')
        cells = set()
        for p in rows:
            cell = p['family'],p['context_length']
            if cell in cells or p['id'] in seen: raise ValueError('Duplicate sweep cell or ID')
            cells.add(cell);seen.add(p['id'])
            if (p['family'] not in FAMILIES or p['context_length'] not in SPEC['input_lengths'] or
                len(p['token_ids']) != p['context_length'] or digest(p['token_ids']) != p['token_ids_sha256'] or
                p['context_length']+SPEC['output_cap']+max(SPEC['k'])+1 > SPEC['capacity']):
                raise ValueError('Invalid context tokens or capacity')


def definition(q, phase):
    profiled = phase == 'profiles'
    d = dict(queue_sha256=q['sha256'],phase=phase,prompts=q['prompts'],warmup=q['warmup'],
        output_cap=256,capacity=4096,kv_fraction=.70,profile_capture=profiled,protocols=['fixed'])
    configs = configurations()
    jobs = []
    if not profiled:
        orders = random.Random(0).sample(list(itertools.permutations(configs[1:])),3)
        for r, order in enumerate(orders):
            indices = random.Random(1000+r).sample(range(len(d['prompts'])),len(d['prompts']))
            for c in [configs[0],*order]:
                jobs.append(dict(id=f'r{r}_{c["label"]}',round=r,protocol='fixed',config=c,prompt_order=indices))
    else:
        controls = [predecessor.config('complete',7),predecessor.config('complete',10),predecessor.config('overlap',10,4,2)]
        for length in SPEC['input_lengths']:
            indices = [i for i,p in enumerate(d['prompts']) if p['context_length']==length]
            for c in random.Random(length).sample(controls,len(controls)):
                jobs.append(dict(id=f'l{length}_{c["label"]}',round=0,protocol='fixed',config=c,
                    context_length=length,prompt_order=indices))
    d['jobs'] = jobs;d['sha256'] = digest(d)
    return d


def prepare(folder, after):
    old = read_json(after/'queue.json')
    predecessor.check_runtime(old)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(old['models']['target']['path'],local_files_only=True)
    q = dict(created_at=now(),execution=execution_fingerprint(),queue_code=code_files(),
        software=software_fingerprint(),hardware=gpu_inventory(),models=old['models'],
        after=dict(path=str(after),queue_sha256=file_digest(after/'queue.json')),
        spec=SPEC,profiler=profiler_fingerprint(),prompts=make_prompts(tok),warmup=make_prompts(tok,True),
        tokenizer=dict(type=type(tok).__name__,chat_template_sha256=digest(tok.chat_template)),
        reference_contract='shared-fixed16-v1; fresh AR on identical inputs; historical original AR separate',
        limitations=['Synthetic repeated filler; only one scored task template per family.',
            'Three repeated rounds measure repeatability; no claim about unseen prompts.',
            'Measured context includes input plus up to 256 generated tokens.',
            'Partial trees are not reported as full K-step drafting.',
            'Profiled wall/GPU spans may be perturbed; performance comparisons use separate unprofiled runs.'])
    validate_prompts(q['prompts'],q['warmup'])
    q['sha256'] = digest(q);freeze_json(folder/'queue.json',q)
    for phase in ('measure','profiles'):
        freeze_json(folder/phase/'definition.json',definition(q,phase))
    with zipfile.ZipFile(folder/'source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(q['execution']['files'])|set(q['queue_code'])):
            archive.write(ROOT/name,name)
    (folder/'protocol.md').write_text(
        '# Controlled context-length diagnostic\n\n'
        'Runs only after the publication performance campaign completes with valid receipts, exact outputs, '
        'frozen selection and a final report. Shares the global campaign lock; one worker at a time.\n\n'
        '- Inputs: exactly 256, 512, 1024, 2048 tokens. Three synthetic chat/code/document tasks, '
        'extended using repeated irrelevant notes; complete tasks and chat formatting remain intact.\n'
        '- Outputs: fixed 256, EOS ignored. BF16, batch one, CUDA graphs, JIT misses, shared AR arithmetic, '
        '4096 capacity, 0.70 target/draft KV fractions, prefix caching disabled, final drain included.\n'
        '- Unprofiled: fresh AR plus complete/uniform/overlap at K7 and K10; F3, P4, queue bound two. '
        'Three seeded rounds, same prompt order within a round, fresh worker per configuration/round: '
        '21 blocks / 252 scored requests. Twelve separate warmup requests per worker, covering every length/family.\n'
        '- Profiles: complete K7, complete K10, overlap K10 at each length; three family requests per block: '
        '12 blocks / 36 scored requests, three length-matched separate warmups per block. '
        'CUDA graph nodes and NVTX captured with Nsight Systems, without intrusive per-step synchronization.\n'
        '- Both GPUs must remain dedicated, including AR. UUID-pinned allocation, one-second foreign-process '
        'monitoring, invalidation and wait/resume; never terminate unrelated jobs.\n'
        '- Every speculative result must match fresh same-round AR; profiles match round-zero AR. '
        'Stop on exactness, execution, provenance, or trace-correlation errors.\n'
        '- Report full-request/decoding time, wait, acceptance, tokens per verification, partial/actual handoffs, '
        'draining, memory; separately report tree/verify GPU spans and CPU wall durations, and ready-depth histograms. '
        'Raw traces, outputs, receipts and plots are retained.\n\n'
        'This is a synthetic length-sensitivity diagnostic, not a newly unseen benchmark or an automatic speedup claim. '
        'Paired bootstrap intervals describe only these three task templates and repeated rounds. '
        'No extra output-length sweep or retuning is part of this queue.\n')
    atomic_json(folder/'status.json',dict(status='waiting_for_predecessor',after=str(after),at=now(),
        unprofiled_requests=252,profiled_requests=36))
    return q


def check_runtime(q, profile=False):
    if digest({k:v for k,v in q.items() if k!='sha256'})!=q['sha256'] or q['spec']!=SPEC:
        raise ValueError('Frozen sweep protocol changed')
    if q['execution']!=execution_fingerprint() or q['queue_code']!=code_files():
        raise ValueError('Execution changed; create a new versioned sweep')
    actual, expected = copy.deepcopy(software_fingerprint()),copy.deepcopy(q['software'])
    if profile:
        preload=actual['environment'].pop('LD_PRELOAD');required=expected['environment'].pop('LD_PRELOAD')
        if required and not all(p in (preload or '').split(':') for p in required.split(':')):
            raise ValueError('Profile worker lost required runtime libraries')
    if actual!=expected or q['hardware']['gpus']!=gpu_inventory()['gpus']:
        raise ValueError('Software, environment or GPU identities changed')
    if not profile and q['profiler']!=profiler_fingerprint(): raise ValueError('Profiler version changed')
    if file_digest(Path(q['after']['path'])/'queue.json')!=q['after']['queue_sha256']:
        raise ValueError('Predecessor queue changed')


def predecessor_ready(q):
    """Only a successful final campaign status permits dependent GPU work."""
    path=Path(q['after']['path'])/'status.json'
    return path.exists() and read_json(path).get('status')=='complete'


def validate_dependency(folder,q):
    after=Path(q['after']['path'])
    if not predecessor_ready(q): raise ValueError('Predecessor has not completed successfully')
    old=read_json(after/'queue.json');predecessor.check_runtime(old);predecessor.validate_sources(old)
    predecessor.validate_selection(after,old)
    for phase in ('continuity','tune','heldout'):
        d=predecessor.load_definition(after,old,phase)
        results=predecessor.available_results(after/phase,d,old,complete=True)
        if not all(c['exact'] for c in predecessor.check_outputs(results,old,phase)):
            raise ValueError('Predecessor output agreement failed')
    s=read_json(after/'summary.json')
    if s.get('status')!='complete' or s.get('correctness_passed') is not True or not (after/'report.md').is_file():
        raise ValueError('Predecessor final results/report missing or invalid')
    paths=[after/name for name in ['queue.json','selection.json','summary.json','report.md','continuity/definition.json','continuity/summary.json','tune/definition.json','heldout/definition.json']]
    for phase in ('continuity','tune','heldout'):
        for receipt in (after/phase).glob('*/attempt_*/receipt.json'):
            if read_json(receipt).get('valid') is True:
                paths.extend([receipt,receipt.parent/'result.json'])
    evidence=dict(predecessor=q['after'],artifacts_sha256={str(p):file_digest(p) for p in paths})
    evidence['sha256']=digest(evidence)
    freeze_json(folder/'dependency.json',evidence)


def load_definition(folder,q,phase):
    d=read_json(folder/phase/'definition.json')
    if d!=definition(q,phase): raise ValueError('Sweep definition or schedule changed')
    return d


def validate_result(result,j,d,q):
    validate_engine_result(result,j,d,q)
    for row,i in zip(result['requests'],j['prompt_order']):
        if row['prompt_tokens']!=d['prompts'][i]['context_length']:
            raise ValueError('Wrong context length')
        if not 0 <= row['first_delivery_s'] <= row['latency_s']:
            raise ValueError('Invalid first delivery')
        if j['config']['method']=='overlap' and len(row['publication_policy_stats'] or [])!=12:
            raise ValueError('Missing overlap telemetry')


def validate_receipt(receipt,d,attempt):
    if (receipt.get('valid') is not True or receipt.get('definition_sha256')!=d['sha256'] or
        receipt.get('returncode')!=0 or receipt.get('samples',0)<1 or receipt.get('external_processes') or
        receipt.get('error') or receipt.get('dedicated_gpus') is not True or
        receipt.get('profiler')!=d['profile_capture'] or
        receipt.get('output_sha256')!=file_digest(attempt/'result.json')):
        raise ValueError('Invalid, interrupted or contaminated sweep receipt')
    artifacts=receipt.get('artifacts',{})
    if d['profile_capture'] and not {'trace.nsys-rep','trace.sqlite','phase_analysis.json'} <= set(artifacts):
        raise ValueError('Profile artifacts incomplete')
    for name,sha in artifacts.items():
        if file_digest(attempt/name)!=sha: raise ValueError('Profile artifact changed')


def block_result(folder,j,d,q):
    for attempt in sorted((folder/j['id']).glob('attempt_*')):
        receipt=attempt/'receipt.json'
        if not receipt.exists() or read_json(receipt).get('valid') is not True: continue
        validate_receipt(read_json(receipt),d,attempt)
        result=read_json(attempt/'result.json');validate_result(result,j,d,q)
        if d['profile_capture']:result=dict(result,phase_analysis=read_json(attempt/'phase_analysis.json'))
        return result
    return None


def worker_command(args,d,j,attempt):
    command=[sys.executable,'-O',str(Path(__file__).resolve()),'--worker','--outdir',str(args.outdir),
        '--phase',d['phase'],'--job',j['id'],'--attempt',str(attempt)]
    if d['profile_capture']:
        if shutil.disk_usage(args.outdir).free<10*1024**3: raise ValueError('Require 10 GiB free before a trace')
        command=[shutil.which('nsys'),'profile','--trace=cuda,nvtx,osrt','--cuda-graph-trace=node',
            '--capture-range=cudaProfilerApi','--capture-range-end=stop','--sample=none','--cpuctxsw=none',
            '-o',str(attempt/'trace'),*command]
    return command


def run_block(args,d,j,q):
    check_runtime(q)
    uuids=[g['uuid'] for g in q['hardware']['gpus']];require_idle(uuids)
    parent=args.outdir/d['phase']/j['id'];parent.mkdir(parents=True,exist_ok=True)
    attempt=parent/f'attempt_{len(list(parent.glob("attempt_*")))}';attempt.mkdir()
    token=uuid.uuid4().hex
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids if j['config']['mode']=='ssd' else uuids[:1]))
    env[TOKEN_ENV]=token;env.pop('AO_SPAWN_EXECUTABLE',None)
    for key in list(env):
        if key.startswith('SSD_PROFILE'):env[key]='0'
    if env.get('CUDA_INJECTION64_PATH') or env.get('NVTX_INJECTION64_PATH'):
        raise ValueError('Unexpected profiler injection in supervisor')
    command=worker_command(args,d,j,attempt)
    process=None;foreign={};owned=[];samples=0;error=None;artifacts={};started=time.monotonic()
    try:
        with (attempt/'worker.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                owned,others=partition(compute_processes(uuids),process.pid,token)
                for r in others:foreign[r['pid'],r['uuid']]=r
                samples+=1
                if others or process.poll() is not None:break
                if time.monotonic()-started>1800:
                    error='Worker exceeded 1800-second budget';break
                time.sleep(1)
    except BaseException as exc:
        error=str(exc) or type(exc).__name__;raise
    finally:
        if process is not None:stop_worker(process)
        cleanup_tagged(owned,token)
        output=attempt/'result.json'
        valid=process is not None and process.returncode==0 and samples>0 and not foreign and error is None
        if valid:
            try:
                validate_result(read_json(output),j,d,q)
                if d['profile_capture']:
                    subprocess.run([q['profiler']['executable'],'export','--type','sqlite','--output',
                        str(attempt/'trace.sqlite'),str(attempt/'trace.nsys-rep')],check=True,timeout=300,
                        stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                    analysis=analyze(attempt/'trace.sqlite',j['config']['k'])
                    if j['config']['method']=='complete' and analysis['full_trees']==0:
                        raise ValueError('No correlated full tree in complete-SSD capture')
                    atomic_json(attempt/'phase_analysis.json',analysis)
                    artifacts={name:file_digest(attempt/name) for name in ['trace.nsys-rep','trace.sqlite','phase_analysis.json']}
            except Exception as exc:
                valid=False;error=str(exc)
        atomic_json(attempt/'receipt.json',dict(valid=bool(valid),definition_sha256=d['sha256'],
            command=command,returncode=process.returncode if process else None,samples=samples,
            external_processes=list(foreign.values()),error=error,run_token=token,dedicated_gpus=True,
            profiler=d['profile_capture'],at=now(),artifacts=artifacts,
            output_sha256=file_digest(output) if output.exists() else None))
    if not valid:
        if foreign:raise PendingWindow('Contaminated block invalidated; waiting for dedicated GPUs')
        raise RuntimeError(f'Sweep worker failed: {attempt}/worker.log; {error}')


def worker(args):
    import torch
    from ssd import LLM,SamplingParams
    q=read_json(args.outdir/'queue.json');d=load_definition(args.outdir,q,args.phase)
    check_runtime(q,d['profile_capture'])
    j=next(j for j in d['jobs'] if j['id']==args.job);c=j['config'];mode=c['mode']
    model=LLM(q['models']['target']['path'],draft=q['models']['draft']['path'],
        target_numerics='shared_fixed16',num_gpus=2 if mode=='ssd' else 1,speculate=mode!='ar',
        draft_async=mode=='ssd',speculate_k=c['k'],async_fan_out=c['f'],jit_speculate=True,
        max_num_seqs=1,max_model_len=4096,enable_prefix_cache=False,gpu_memory_utilization=.70,
        draft_gpu_memory_utilization=.70,early_publish_tokens=c['p'],publication_policy=c['policy'],
        publication_max_inflight=c['window'],drain_draft_on_finish=mode=='ssd')
    sampling=SamplingParams(temperature=0,max_new_tokens=256,ignore_eos=True)
    warm=[p for p in d['warmup'] if not d['profile_capture'] or p['context_length']==j['context_length']]
    for p in warm:model.generate([p['token_ids']],[sampling],use_tqdm=False,stream_callback=lambda *a:None)
    torch.cuda.synchronize()
    result=dict(queue_sha256=q['sha256'],definition_sha256=d['sha256'],job=j,requests=[],
        hardware_allocation=allocation(q,mode),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        actual_dtype=str(model.config.hf_config.torch_dtype),cuda_graphs=True,prefix_cache=False,
        kv_fraction=.70,profile_capture=d['profile_capture'],software=software_fingerprint(),warmup_requests=len(warm))
    if d['profile_capture']:
        torch.cuda.profiler.start();torch.cuda.nvtx.range_push('timed_region')
    try:
        for i in j['prompt_order']:
            torch.cuda.reset_peak_memory_stats()
            row=run_engine.measure_request(model,d['prompts'][i],sampling,synchronize=torch.cuda.synchronize,
                profile_capture=d['profile_capture'])
            row['target_peak_memory']=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
            result['requests'].append(row)
            if not d['profile_capture']:atomic_json(args.attempt/'result.json',result)
    finally:
        if d['profile_capture']:torch.cuda.nvtx.range_pop();torch.cuda.profiler.stop()
    atomic_json(args.attempt/'result.json',result)
    model.exit(hard=True)


def verify_outputs(result,refs):
    checks=[]
    for row in result['requests']:
        gold=refs[row['id']]
        if row['input_token_ids_sha256']!=gold['input_token_ids_sha256']:raise ValueError('Reference input changed')
        idx=predecessor.base.first_difference(gold['output_token_ids'],row['output_token_ids'])
        checks.append(dict(id=row['id'],exact=idx is None,first_difference_zero_based=idx))
    if not all(c['exact'] for c in checks): raise ValueError('Exactness failed against fresh shared AR')
    return checks


def reference_rows(folder,q,round_):
    d=load_definition(folder,q,'measure')
    j=next(j for j in d['jobs'] if j['round']==round_ and j['config']['method']=='ar')
    b=block_result(folder/'measure',j,d,q)
    if b is None:raise ValueError('Fresh AR reference is incomplete')
    return {r['id']:r for r in b['requests']}


def run_phase(args,q,d):
    for j in d['jobs']:
        while block_result(args.outdir/d['phase'],j,d,q) is None:
            try:
                atomic_json(args.outdir/'status.json',dict(status='running',phase=d['phase'],job=j['id'],at=now()))
                run_block(args,d,j,q)
            except PendingWindow as exc:
                atomic_json(args.outdir/'status.json',dict(status='pending_dedicated_gpus',phase=d['phase'],job=j['id'],reason=str(exc),at=now()))
                if not args.watch:return False
                time.sleep(30)
        b=block_result(args.outdir/d['phase'],j,d,q)
        if j['config']['method']!='ar':verify_outputs(b,reference_rows(args.outdir,q,j['round']))
    return True


def report(folder,q):
    d=load_definition(folder,q,'measure')
    blocks=[block_result(folder/'measure',j,d,q) for j in d['jobs']]
    pd=load_definition(folder,q,'profiles')
    profiles=[block_result(folder/'profiles',j,pd,q) for j in pd['jobs']]
    if any(b is None for b in blocks+profiles):raise ValueError('All valid measurement and profile blocks required')
    checks=[]
    for b in blocks+profiles:
        if b['job']['config']['method']!='ar':checks+=verify_outputs(b,reference_rows(folder,q,b['job']['round']))
    rows=[]
    for c in configurations():
        selected=[b for b in blocks if b['job']['config']==c]
        for length in SPEC['input_lengths']:
            grid=[[next(r for r in b['requests'] if r['id']==f'sweep-{f}-{length}') for f in FAMILIES]
                  for b in sorted(selected,key=lambda b:b['job']['round'])]
            flat=[r for round_ in grid for r in round_]
            metric=predecessor.aggregate(flat)
            metric['mean_decode_s']=sum(r['decode_s'] for r in flat)/len(flat)
            metric['mean_prefill_s']=sum(r['prefill_s'] for r in flat)/len(flat)
            metric['context_length']=length;metric['config']=c;metric['gpu_count']=1 if c['method']=='ar' else 2
            metric['repeatable_templates']=sum(len({tuple(round_[i]['output_token_ids']) for round_ in grid})==1 for i in range(3))
            if metric['repeatable_templates']!=3:raise ValueError('Within-method repeatability failed')
            if c['method']!='ar':
                ref=next(v for v in configurations() if v['method']=='complete' and v['k']==c['k'])
                rg=[[next(r['latency_s'] for r in b['requests'] if r['id']==f'sweep-{f}-{length}') for f in FAMILIES]
                    for b in sorted([b for b in blocks if b['job']['config']==ref],key=lambda b:b['job']['round'])]
                metric['comparison_to_matched_complete']=paired_bootstrap(rg,[[r['latency_s'] for r in rr] for rr in grid],list(FAMILIES),10000,0)
            rows.append(metric)
    phase_rows=[dict(context_length=b['job']['context_length'],config=b['job']['config'],**b['phase_analysis']) for b in profiles]
    summary=dict(status='complete',queue_sha256=q['sha256'],unprofiled_requests=252,profiled_requests=36,
        unprofiled_valid_blocks=len(blocks),profiled_valid_blocks=len(profiles),exact_comparisons=len(checks),
        exact_matches=sum(c['exact'] for c in checks),reference=q['reference_contract'],scope=SPEC['scope'],
        results=rows,profiles=phase_rows,limitations=q['limitations'])
    atomic_json(folder/'summary.json',summary)
    lines=['# Controlled context-length diagnostic','',SPEC['scope']+'.',
        'Fixed 256 output tokens, EOS ignored; three repeated rounds; BF16, batch one, graphs, full prefills and final draining.',
        'AR uses one GPU; SSD uses two. All accepted blocks passed dedicated-GPU monitoring.',
        f'Exact speculative outputs: {len(checks)}/{len(checks)} against fresh shared AR. Original historical AR is separate.','',
        '## Unprofiled request measurements','',
        '| Input tokens | Configuration | GPUs | Mean request s | Decode s | Wait s | Tokens/verify | Actual overlap handoffs |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    csv_rows=[]
    for r in rows:
        c=r['config'];csv_rows.append(dict(input_tokens=r['context_length'],config=c['label'],gpus=r['gpu_count'],
            mean_latency_s=r['mean_latency_s'],mean_decode_s=r['mean_decode_s'],mean_wait_s=r['mean_target_wait_s'],
            tokens_per_verification=r['tokens_per_verification'],responses_before_tail_done=r['responses_before_tail_done']))
        lines.append(f'| {r["context_length"]} | {c["label"]} | {r["gpu_count"]} | {r["mean_latency_s"]:.4f} | '
            f'{r["mean_decode_s"]:.4f} | {r["mean_target_wait_s"]:.4f} | {r["tokens_per_verification"]:.3f} | {r["responses_before_tail_done"]:.0f} |')
    lines+=['','## Separate profiled phase measurements','',
        '| Input tokens | Configuration | Full trees | Mean full-tree GPU span ms | Verify GPU span ms | Verify wall ms | Tree/verify GPU ratio | Hit ready-depth histogram |',
        '|---:|---|---:|---:|---:|---:|---:|---|']
    fmt=lambda v:'—' if v is None else f'{v:.3f}'
    for r in phase_rows:
        lines.append(f'| {r["context_length"]} | {r["config"]["label"]} | {r["full_trees"]} | '
            f'{fmt(r["mean_full_tree_gpu_span_ms"])} | {fmt(r["mean_verify_gpu_span_ms"])} | '
            f'{fmt(r["mean_verify_wall_ms"])} | {fmt(r["full_tree_over_verify_gpu"])} | {r["cache_hit_ready_depth_histogram"]} |')
    lines+=['','## Interpretation boundaries','',
        'Use matched K controls to assess publication; do not interpret a K10 improvement over K10 as a win over a faster K7 baseline. '
        'Examine the tree/verify ratio, ready depth, wait and accepted work together. More overlap alone is not a speedup.',
        'Profiled GPU spans include gaps; CPU launch-range duration is not GPU execution time. '
        'Incomplete or uncorrelated trees are not extrapolated to a full K-step cost. '
        'Initial context grows by up to 256 generated tokens during each request.',
        'Paired 95% intervals in summary.json use 10,000 seeded draws over rounds and templates within families. '
        'With one template per family these intervals do not estimate workload generalization.','',*q['limitations']]
    (folder/'report.md').write_text('\n'.join(lines)+'\n')
    with (folder/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    plots(folder,rows,phase_rows)
    return summary


def plots(folder,rows,profiles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for c in configurations()[1:]:
        vals=sorted([r for r in rows if r['config']==c],key=lambda r:r['context_length'])
        axes[0].plot([r['context_length'] for r in vals],[r['mean_latency_s'] for r in vals],marker='o',label=c['label'])
        axes[1].plot([r['context_length'] for r in vals],[r['mean_target_wait_s'] for r in vals],marker='o',label=c['label'])
    axes[0].set_ylabel('Unprofiled mean request latency (s)');axes[1].set_ylabel('Unprofiled mean target wait (s)')
    for ax in axes:ax.set_xlabel('Initial input tokens');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'length_latency.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for label in sorted({r['config']['label'] for r in profiles if r['config']['method']=='complete'}):
        vals=sorted([r for r in profiles if r['config']['label']==label],key=lambda r:r['context_length'])
        x=[r['context_length'] for r in vals]
        axes[0].plot(x,[r['mean_full_tree_gpu_span_ms'] for r in vals],marker='o',label=label+' draft tree')
        axes[0].plot(x,[r['mean_verify_gpu_span_ms'] for r in vals],marker='s',ls='--',label=label+' verify')
        axes[1].plot(x,[r['full_tree_over_verify_gpu'] for r in vals],marker='o',label=label)
    axes[0].set_ylabel('Profiled mean GPU span (ms)');axes[1].set_ylabel('Full-tree / verification GPU-span ratio')
    for ax in axes:ax.set_xlabel('Initial input tokens');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'draft_verify_scaling.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--after',type=Path,default=DEFAULT_AFTER)
    parser.add_argument('--phase',choices=['all','measure','profiles','report'],default='all')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--job');parser.add_argument('--attempt',type=Path)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();args.outdir=args.outdir.resolve();args.after=args.after.resolve()
    if args.worker:worker(args);return 0
    signal.signal(signal.SIGTERM,interrupted)
    args.outdir.mkdir(parents=True,exist_ok=True)
    with (args.outdir/'.watcher.lock').open('a') as watcher_lock:
        fcntl.flock(watcher_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        q=read_json(args.outdir/'queue.json') if (args.outdir/'queue.json').exists() else prepare(args.outdir,args.after)
        check_runtime(q);validate_prompts(q['prompts'],q['warmup'])
        if Path(q['after']['path'])!=args.after:raise ValueError('Predecessor path differs from frozen queue')
        if args.prepare_only or args.dry_run:
            print('Frozen: 21 unprofiled blocks / 252 requests; 12 profile blocks / 36 requests. No GPU work started.');return 0
        try:
            with (ROOT/'results/baselines/eval1/.campaign.lock').open('a') as lock:
                while True:
                    check_runtime(q)
                    if predecessor_ready(q):
                        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                        except BlockingIOError:pass
                    atomic_json(args.outdir/'status.json',dict(status='waiting_for_predecessor',after=q['after']['path'],
                        reason='Requires successful completed campaign and exclusive global campaign lock',at=now()))
                    if not args.watch:return 2
                    time.sleep(30)
                validate_dependency(args.outdir,q);validate_models(q['models'])
                for phase in ('measure','profiles'):
                    if args.phase in ('all',phase):
                        if not run_phase(args,q,load_definition(args.outdir,q,phase)):return 2
                        if args.phase==phase:
                            atomic_json(args.outdir/'status.json',dict(status=phase+'_complete',at=now()));return 0
                if args.phase in ('all','report'):
                    summary=report(args.outdir,q)
                    atomic_json(args.outdir/'status.json',dict(status='complete',at=now(),report=str(args.outdir/'report.md'),
                        unprofiled_requests=summary['unprofiled_requests'],profiled_requests=summary['profiled_requests']))
        except BaseException as exc:
            atomic_json(args.outdir/'status.json',dict(status='stopped',reason=str(exc),at=now()));raise
    return 0


if __name__=='__main__':raise SystemExit(main())
