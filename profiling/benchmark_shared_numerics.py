"""Dedicated, resumable prototype benchmark; never the frozen held-out campaign.

Three already-investigated prompts (one/family), fixed 256 tokens, five rounds,
four methods, fresh workers and three warmups. Both GPUs must be dedicated.
Original AR is a performance/output control; shared SD/SSD must match shared AR.
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

import run_engine  # Also installs optimized-Python workaround in spawned children.
from baseline_common import (ROOT,atomic_json,compute_processes,digest,execution_fingerprint,
    external_processes,file_digest,freeze_json,read_json,software_fingerprint,validate_models)
from run_baselines import PendingWindow,interrupted,now,require_idle,stop_worker

METHODS=('original_ar','shared_ar','shared_sd','shared_ssd')


def fingerprints():
    paths=[Path(__file__),ROOT/'profiling/shared_numerics.py',ROOT/'profiling/numerics_attention.py']
    return dict(execution=execution_fingerprint(),software=software_fingerprint(),
                prototype_files={str(p):file_digest(p) for p in paths})


def schedule():
    orders=random.Random(0).sample(list(itertools.permutations(METHODS)),5)
    return [dict(round=r,method=method,prompt_order=random.Random(r).sample(range(3),3))
            for r,order in enumerate(orders) for method in order]


def validate_definition(definition):
    for key,value in fingerprints().items():
        if definition[key]!=value:
            raise ValueError(f'Prototype benchmark cannot resume: {key} changed')
    if digest({k:v for k,v in definition.items() if k!='sha256'})!=definition['sha256']:
        raise ValueError('Prototype benchmark definition was modified')


def prepare(folder,gate_folder):
    source=read_json(gate_folder/'definition.json');summary=read_json(gate_folder/'summary.json')
    if not summary['paired_correctness_passed'] or summary['candidate_comparisons']!=36:
        raise ValueError('Require the complete 36-comparison shared-backend correctness check')
    if 'shared-fixed16-v1' not in source.get('reference_contract',''):
        raise ValueError('Wrong reference contract')
    if source['execution']!=execution_fingerprint() or source['software']!=software_fingerprint():
        raise ValueError('Correctness evidence does not match current execution')
    for path,sha in source['candidate_code']['files'].items():
        if file_digest(path)!=sha:
            raise ValueError('Shared backend changed since correctness validation')
    validate_models(source['models'])
    prompts=[p for p in source['prompts'] if p['id'] in {'specbench-148','HumanEval/36','specbench-553'}]
    if len(prompts)!=3 or {p['family'] for p in prompts}!={'chat','code','doc'}:
        raise ValueError('Expected three explicitly declared historical engineering prompts')
    definition=dict(created_at=now(),purpose='Dedicated prototype engineering benchmark; not held-out headline evidence',
        reference_contract=source['reference_contract'],models=source['models'],hardware=source['hardware'],
        prompts=prompts,warmup=source['warmup'],jobs=schedule(),kv_fraction=.70,
        output_cap=256,capacity=4096,prefix_cache=False,protocol='fixed',repetitions=5,
        sd_k=6,ssd_k=7,f=3,configuration_status='Starting configurations; not retuned for the new backend',
        correctness_gate=dict(path=str(gate_folder),definition_sha256=source['sha256'],
            summary_sha256=file_digest(gate_folder/'summary.json')),
        **fingerprints())
    definition['sha256']=digest(definition)
    freeze_json(folder/'definition.json',definition)
    return definition


def worker(args):
    import torch
    from ssd import LLM,SamplingParams
    definition=read_json(args.outdir/'definition.json');validate_definition(definition)
    if args.method!='original_ar':
        from shared_numerics import install_target_factory
        install_target_factory()
    mode='ssd' if args.method=='shared_ssd' else 'sd' if args.method=='shared_sd' else 'ar'
    model=LLM(definition['models']['target']['path'],draft=definition['models']['draft']['path'],
        speculate=mode!='ar',draft_async=mode=='ssd',num_gpus=2 if mode=='ssd' else 1,
        speculate_k=definition['ssd_k'] if mode=='ssd' else definition['sd_k'],
        async_fan_out=definition['f'],jit_speculate=True,max_num_seqs=1,
        max_model_len=definition['capacity'],gpu_memory_utilization=definition['kv_fraction'],
        draft_gpu_memory_utilization=definition['kv_fraction'],enable_prefix_cache=False)
    sampling=SamplingParams(temperature=0,max_new_tokens=definition['output_cap'],ignore_eos=True)
    for item in definition['warmup']:
        model.generate([item['token_ids']],[sampling],use_tqdm=False,stream_callback=lambda *a:None)
    job=next(j for j in definition['jobs'] if j['method']==args.method and j['round']==args.round)
    result=dict(definition_sha256=definition['sha256'],job=job,requests=[],
        numerics='original unchanged AR' if args.method=='original_ar' else definition['reference_contract'],
        allocation=dict(target_uuid=definition['hardware']['gpus'][0]['uuid'],
            draft_uuid=definition['hardware']['gpus'][1 if mode=='ssd' else 0]['uuid'] if mode!='ar' else None,
            gpu_count=2 if mode=='ssd' else 1),dtype=str(model.config.hf_config.torch_dtype),
        cuda_graphs=True,profiler=False,kv_fraction=.70,prefix_cache=False)
    for index in job['prompt_order']:
        # Some upstream prefill helpers print even with a delivery callback.
        # Drop console output inside measurement; retain initialization logs.
        import builtins
        original_print=builtins.print
        try:
            builtins.print=lambda *a,**k:None
            row=run_engine.measure_request(model,definition['prompts'][index],
                sampling,synchronize=torch.cuda.synchronize,profile_capture=False)
        finally:
            builtins.print=original_print
        result['requests'].append(row)
        atomic_json(args.blockdir/'result.json',result)
    model.exit(hard=True)


def block_result(folder,job,definition):
    parent=folder/f'r{job["round"]}_{job["method"]}'
    for attempt in sorted(parent.glob('attempt_*')):
        receipt=attempt/'receipt.json'
        if not receipt.exists() or not read_json(receipt).get('valid'):continue
        info=read_json(receipt)
        if info['output_sha256']!=file_digest(attempt/'result.json'):
            raise ValueError('Completed benchmark output changed')
        result=read_json(attempt/'result.json')
        validate_result(result,job,definition)
        return result
    return None


def validate_result(result,job,definition):
    if result['definition_sha256']!=definition['sha256'] or result['job']!=job:
        raise ValueError('Completed block belongs to a different benchmark')
    expected=[definition['prompts'][i] for i in job['prompt_order']]
    if [r['id'] for r in result['requests']] != [p['id'] for p in expected]:
        raise ValueError('Incomplete or unpaired benchmark output')
    for row,prompt in zip(result['requests'],expected):
        if (row['input_token_ids_sha256']!=digest(prompt['token_ids']) or
            row['output_tokens']!=definition['output_cap'] or
            len(row['output_token_ids'])!=definition['output_cap']):
            raise ValueError('Benchmark inputs or fixed output lengths differ')


def run_block(args,definition,job):
    validate_definition(definition)
    uuids=[g['uuid'] for g in definition['hardware']['gpus']]
    require_idle(uuids)
    parent=args.outdir/f'r{job["round"]}_{job["method"]}'
    parent.mkdir(exist_ok=True)
    attempt=parent/f'attempt_{len(list(parent.glob("attempt_*")))}';attempt.mkdir()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids if job['method']=='shared_ssd' else uuids[:1]))
    env.pop('AO_SPAWN_EXECUTABLE',None)
    for key in list(env):
        if key.startswith('SSD_PROFILE'):env[key]='0'
    if env.get('CUDA_INJECTION64_PATH') or env.get('NVTX_INJECTION64_PATH'):
        raise ValueError('Profiler injection is not allowed')
    command=[sys.executable,'-O',str(Path(__file__).resolve()),'--worker','--outdir',str(args.outdir),
             '--blockdir',str(attempt),'--method',job['method'],'--round',str(job['round'])]
    process=None;error=None;others=[];samples=0;started=time.monotonic()
    try:
        with (attempt/'worker.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                others=external_processes(compute_processes(uuids),process.pid);samples+=1
                if others:break
                if process.poll() is not None:break
                if time.monotonic()-started>600:
                    error='Worker exceeded 600-second budget';break
                time.sleep(1)
    except BaseException as exc:
        error=str(exc) or type(exc).__name__
        raise
    finally:
        if process is not None:stop_worker(process)
        valid=process is not None and process.returncode==0 and samples>0 and not others and error is None
        output=attempt/'result.json'
        if valid:
            result=read_json(output)
            try:validate_result(result,job,definition)
            except (KeyError,ValueError) as exc:valid=False;error=str(exc)
        atomic_json(attempt/'receipt.json',dict(valid=valid,definition_sha256=definition['sha256'],
            command=command,samples=samples,external_processes=others,error=error,
            returncode=process.returncode if process else None,at=now(),
            output_sha256=file_digest(output) if output.exists() else None))
    if not valid:
        if others:raise PendingWindow('Another GPU job appeared; block invalidated and scheduling stopped')
        raise RuntimeError('Prototype benchmark worker failed; inspect its preserved log')


def report(folder,definition):
    import statistics
    blocks=[block_result(folder,j,definition) for j in definition['jobs']]
    if any(b is None for b in blocks):raise ValueError('All 20 valid blocks are required')
    records={}
    summary={}
    for method in METHODS:
        rows=[r for b in blocks if b['job']['method']==method for r in b['requests']]
        values=sorted(r['latency_s'] for r in rows)
        summary[method]=dict(requests=len(rows),rounds=5,mean_s=statistics.mean(values),
            p50_s=statistics.median(values),p95_s=percentile(values,.95),
            committed_tok_s=sum(r['output_tokens'] for r in rows)/sum(values),
            first_delivery_mean_s=statistics.mean(r['first_delivery_s'] for r in rows))
    for b in blocks:
        for row in b['requests']:records[(b['job']['round'],b['job']['method'],row['id'])]=row
    comparisons=[]
    for (round_id,method,prompt),row in records.items():
        if method in ['shared_sd','shared_ssd']:
            reference=records[(round_id,'shared_ar',prompt)]
            comparisons.append(dict(round=round_id,method=method,id=prompt,
                exact_match=row['output_token_ids']==reference['output_token_ids'],
                first_difference=first_difference(row['output_token_ids'],reference['output_token_ids'])))
    repeatability={method:all(len({tuple(records[(r,method,p['id'])]['output_token_ids']) for r in range(5)})==1
                              for p in definition['prompts']) for method in METHODS}
    intervals={}
    for method in ['shared_sd','shared_ssd']:
        for reference in ['shared_ar','original_ar']:
            ratios=[];rng=random.Random(0)
            for _ in range(10000):
                rounds=rng.choices(range(5),k=5)
                numerator=sum(records[(r,reference,p['id'])]['latency_s'] for r in rounds for p in definition['prompts'])
                denominator=sum(records[(r,method,p['id'])]['latency_s'] for r in rounds for p in definition['prompts'])
                ratios.append(numerator/denominator)
            intervals[f'{method}_vs_{reference}']=dict(low=percentile(ratios,.025),high=percentile(ratios,.975),
                samples=10000,seed=0,unit='round',limitation='One fixed prompt per family; no estimate of prompt generalization')
    atomic_json(folder/'summary.json',dict(definition_sha256=definition['sha256'],
        scope=definition['purpose'],reference_contract=definition['reference_contract'],
        methods=summary,valid_blocks=len(blocks),paired_exact=sum(c['exact_match'] for c in comparisons),
        comparisons=comparisons,correctness_passed=all(c['exact_match'] for c in comparisons),
        within_method_repeatability=repeatability,paired_speed_ratio_intervals=intervals,limitations=[
            'Only three already-investigated prompts; not held-out generalization evidence.',
            'Starting K/F configurations were not retuned for this backend.',
            'Original AR has different arithmetic; shared comparisons use an explicitly new reference.']))


def percentile(values,p):
    values=sorted(values);position=(len(values)-1)*p;lo=int(position);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def first_difference(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)) if len(a)!=len(b) else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--gate',type=Path)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--blockdir',type=Path)
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--round',type=int)
    args=parser.parse_args();args.outdir=args.outdir.resolve()
    if args.worker:worker(args);return
    signal.signal(signal.SIGTERM,interrupted)
    args.outdir.mkdir(parents=True,exist_ok=True)
    with (ROOT/'results/baselines/eval1/.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        definition=(read_json(args.outdir/'definition.json') if (args.outdir/'definition.json').exists()
                    else prepare(args.outdir,args.gate.resolve()))
        validate_definition(definition)
        try:
            atomic_json(args.outdir/'status.json',dict(status='running',at=now()))
            for job in definition['jobs']:
                if block_result(args.outdir,job,definition) is not None:continue
                print(f'Round {job["round"]+1}/5: {job["method"]}',flush=True)
                run_block(args,definition,job)
            report(args.outdir,definition)
            passed=read_json(args.outdir/'summary.json')['correctness_passed']
            atomic_json(args.outdir/'status.json',dict(status='complete' if passed else 'complete_correctness_failed',at=now()))
        except PendingWindow as exc:
            atomic_json(args.outdir/'status.json',dict(status='pending_dedicated_gpus',at=now(),reason=str(exc)))
            print(str(exc),flush=True)
        except BaseException as exc:
            atomic_json(args.outdir/'status.json',dict(status='stopped',at=now(),reason=str(exc)))
            raise


if __name__=='__main__':main()
