"""Exact-output gate, separate from the frozen performance campaign.

Run after sourcing profiling/env.sh. Workers use CUDA graphs, fresh processes,
full prefills, greedy decoding, and fixed/natural EOS protocols. An external GPU
job invalidates a dedicated worker. With --allow-shared-gpus, record other jobs
and admit workers by free memory instead. Unrelated jobs are never terminated.
Results are correctness evidence, not five-repeat headline timings.
"""
import argparse
import fcntl
import math
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

from baseline_common import (ROOT, atomic_json, compute_processes, digest, execution_fingerprint,
                             external_processes, file_digest, freeze_json, read_json,
                             software_fingerprint, validate_models)
from run_baselines import PendingWindow, interrupted, now, require_idle, stop_worker

# Spawn imports this file as __mp_main__ without calling worker(). Apply the
# same Python 3.13 / torch 2.10 -O workaround as run_engine.py in every child.
if sys.flags.optimize:
    import torch._dynamo.symbolic_convert as _sc
    _sc._detect_and_normalize_assert_statement=lambda *a,**k: False


def first_difference(a, b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),
                min(len(a),len(b)) if len(a)!=len(b) else None)


def require_memory(uuids, mode):
    """Admission check for shared correctness runs, never an eviction policy."""
    output=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'],text=True)
    free={row.split(',')[0].strip():int(row.split(',')[1].strip())
          for row in output.strip().splitlines()}
    minimum={uuids[0]:24500 if mode=='sd' else 23000}
    if mode=='ssd': minimum[uuids[1]]=10000
    if any(free.get(uuid,0)<needed for uuid,needed in minimum.items()):
        raise PendingWindow(f'Insufficient free memory for correctness worker: {free}; required {minimum}')
    return free


def import_references(source, definition):
    """Reuse completed outputs only when their execution and inputs agree."""
    source=Path(source).resolve(); old=read_json(source/'definition.json')
    if digest({k:v for k,v in old.items() if k!='sha256'})!=old['sha256']:
        raise ValueError('Source correctness definition hash differs')
    for key in ['execution','software','models','hardware','prompts','warmup',
                'output_cap','protocols','prefix_cache','kv_fractions','capacity']:
        if old[key]!=definition[key]:
            raise ValueError(f'Cannot reuse correctness evidence: {key} differs')
    references={}
    for protocol in definition['protocols']:
        for config in definition['configurations']:
            if config not in old['configurations']: continue
            if config['mode']!='ar' and old.get('candidate_code') != definition.get('candidate_code'):
                raise ValueError('Cannot reuse correctness evidence: candidate code differs')
            name=f'{protocol}_{config["mode"]}'
            path=source/f'{name}.json'; receipt=source/f'{name}.receipt.json'
            if not receipt.exists() or not read_json(receipt).get('valid'): continue
            result=read_json(path)
            if result['definition_sha256']!=old['sha256']:
                raise ValueError('Reused result has the wrong source definition')
            if result['config']!=config or result['protocol']!=protocol:
                raise ValueError('Reused configuration differs')
            if [(r['id'],r['input_sha256']) for r in result['requests']] != [
                    (r['id'],digest(r['token_ids'])) for r in definition['prompts']]:
                raise ValueError('Reused correctness inputs are unpaired or incomplete')
            references[name]=dict(path=str(path),sha256=file_digest(path),
                receipt_path=str(receipt),receipt_sha256=file_digest(receipt),
                source_definition_path=str(source/'definition.json'),
                source_definition_sha256=old['sha256'])
    return references


def load_result(folder, definition, protocol, mode):
    name=f'{protocol}_{mode}'
    reference=definition.get('reused_results',{}).get(name)
    if reference:
        if (file_digest(reference['path'])!=reference['sha256'] or
            file_digest(reference['receipt_path'])!=reference['receipt_sha256']):
            raise ValueError('Reused evidence changed after freezing')
        return read_json(reference['path'])
    return read_json(folder/f'{name}.json')


def prepare(folder, extra_per_family, modes=None, gpu0_only=False,
            allow_shared=False, reuse_from=None, kv_fraction=0.10,
            candidate='ar_compatible', sd_k=6, ssd_k=7):
    from transformers import AutoTokenizer
    from workloads import get_prompts
    previous=read_json(ROOT/'results/baselines/eval1/manifest.json')
    tokenizer=AutoTokenizer.from_pretrained(previous['models']['target']['path'])
    historical=get_prompts(['chat','code','doc'],1,split='heldout',tokenizer=tokenizer)
    extras=[]
    for family in ['chat','code','doc']:
        extras.extend([r for r in previous['splits']['tune'] if r['family']==family][:extra_per_family])
    prompts=historical+extras
    if len({r['id'] for r in prompts})!=len(prompts):
        raise ValueError('Repeated prompt in correctness workload')
    # Recheck pinned model contents, rather than assuming old references still apply.
    validate_models(previous['models'])
    definition=dict(created_at=now(),purpose='exact-output correctness; not a performance campaign',
        models=previous['models'],hardware=previous['hardware'],
        execution=execution_fingerprint(),supervisor_sha256=file_digest(__file__),
        software=software_fingerprint(),prompts=prompts,
        # Match the exact three chat warmups used by the original saved AR runs.
        # Warmup-driven compilation can itself change the prefill arithmetic.
        warmup=get_prompts(['chat'],3,split='tune',tokenizer=tokenizer),
        output_cap=256,protocols=['fixed','natural'],
        configurations=[dict(mode='ar',numerics='batched',k=6,f=3),
                        dict(mode='sd',numerics=candidate,k=sd_k,f=3),
                        dict(mode='ssd',numerics=candidate,k=ssd_k,f=3)],
        prefix_cache=False,kv_fractions=kv_fraction,capacity=4096,
        gpu0_only_contextual=gpu0_only,
        allow_shared_gpus=allow_shared,
        legacy_reference=read_json(ROOT/'results/nsys/ar_run.json')['requests'])
    if candidate=='hybrid_bmm':
        path=ROOT/'profiling/numerics_candidates.py'
        definition['candidate_code']=dict(path=str(path),sha256=file_digest(path))
    if modes is not None:
        definition['configurations']=[c for c in definition['configurations'] if c['mode'] in modes]
    if reuse_from:
        definition['reused_results']=import_references(reuse_from,definition)
    definition['sha256']=digest(definition)
    freeze_json(folder/'definition.json',definition)
    return definition


def worker(args):
    import ssd.paths  # Establish architecture before importing flashinfer.
    import torch
    from ssd import LLM, SamplingParams
    definition=read_json(args.outdir/'definition.json')
    if execution_fingerprint()!=definition['execution'] or file_digest(__file__)!=definition['supervisor_sha256']:
        raise ValueError('Execution code changed after the correctness workload was frozen')
    if software_fingerprint()!=definition['software']:
        raise ValueError('Worker software differs from the frozen definition')
    config=next(c for c in definition['configurations'] if c['mode']==args.mode)
    if config['numerics']=='hybrid_bmm':
        candidate=definition['candidate_code']
        if file_digest(candidate['path'])!=candidate['sha256']:
            raise ValueError('Experimental candidate changed after freezing')
        # Diagnostic-only injection: the production engine and its numerical
        # defaults do not expose this rejected/unvalidated candidate.
        import ssd.engine.ar_compatible as numerical_module
        from numerics_candidates import install_hybrid_bmm
        numerical_module.install_ar_compatible=install_hybrid_bmm
    model=LLM(definition['models']['target']['path'],
        draft=definition['models']['draft']['path'],
        speculate=args.mode!='ar',draft_async=args.mode=='ssd',
        num_gpus=2 if args.mode=='ssd' else 1,
        speculate_k=config['k'],async_fan_out=config['f'],jit_speculate=True,
        max_num_seqs=1,max_model_len=definition['capacity'],kvcache_block_size=256,
        gpu_memory_utilization=definition['kv_fractions'],
        draft_gpu_memory_utilization=definition['kv_fractions'],
        enable_prefix_cache=False,target_numerics=(
            'ar_compatible' if config['numerics']=='hybrid_bmm' else config['numerics']))
    sampling=SamplingParams(temperature=0,max_new_tokens=definition['output_cap'],
                            ignore_eos=args.protocol=='fixed')
    result=dict(mode=args.mode,protocol=args.protocol,config=config,definition_sha256=definition['sha256'],
        cuda_graphs=True,hardware=definition['hardware'],requests=[],
        timing_status='Diagnostic only; one repetition, small KV allocation, no speedup claim')
    for item in definition['warmup']:
        model.generate([item['token_ids']],[sampling],use_tqdm=False,stream_callback=lambda *a:None)
    for item in definition['prompts']:
        torch.cuda.synchronize(); started=time.perf_counter()
        outputs,metrics=model.generate([item['token_ids']],[sampling],use_tqdm=False,
                                       stream_callback=lambda *a:None)
        torch.cuda.synchronize()
        result['requests'].append(dict(id=item['id'],family=item['family'],
            input_sha256=digest(item['token_ids']),output_token_ids=outputs[0]['token_ids'],
            latency_s=time.perf_counter()-started,
            accepted_lengths=list(metrics['accepted_suffix_lens_with_recovery']),
            cache_hits=list(metrics['cache_hits'])))
        atomic_json(args.outdir/f'{args.protocol}_{args.mode}.json',result)
        print(f'{args.protocol} {args.mode}: {item["id"]} complete',flush=True)
    model.exit(hard=True)


def monitor_worker(folder,definition,mode,protocol,budget_seconds):
    name=f'{protocol}_{mode}'; receipt=folder/f'{name}.receipt.json'
    if receipt.exists(): raise ValueError('Existing attempt must be preserved in a different output directory')
    if execution_fingerprint()!=definition['execution'] or file_digest(__file__)!=definition['supervisor_sha256']:
        raise ValueError('Code changed during gate; create a new output directory')
    uuids=[r['uuid'] for r in definition['hardware']['gpus']]
    monitored_uuids=uuids[:1] if definition.get('gpu0_only_contextual') else uuids
    if definition.get('gpu0_only_contextual') and mode=='ssd':
        raise ValueError('SSD cannot run in a GPU0-only diagnostic')
    shared=definition.get('allow_shared_gpus',False)
    admission_memory=require_memory(uuids,mode) if shared else None
    if not shared: require_idle(monitored_uuids)
    env=dict(os.environ)
    env['CUDA_VISIBLE_DEVICES']=','.join(uuids if mode=='ssd' else uuids[:1])
    env.pop('AO_SPAWN_EXECUTABLE',None)
    for key in list(env):
        if key.startswith('SSD_PROFILE'): env[key]='0'
    if env.get('CUDA_INJECTION64_PATH') or env.get('NVTX_INJECTION64_PATH'):
        raise ValueError('Profiler injection is not allowed')
    command=[sys.executable,'-O',str(Path(__file__).resolve()),'--worker',
             '--outdir',str(folder),'--mode',mode,'--protocol',protocol]
    started=time.monotonic(); process=None; samples=0; contamination=[]; error=None
    background_changes=[]; previous_background=None
    shared_changes=[]; previous_shared=None
    try:
        with (folder/f'{name}.log').open('w') as log:
            process=subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                rows=compute_processes(uuids)
                others=external_processes([r for r in rows if r['uuid'] in monitored_uuids],process.pid)
                if shared and others!=previous_shared:
                    shared_changes.append(dict(at=now(),processes=others)); previous_shared=others
                background=[r for r in rows if r['uuid'] not in monitored_uuids]
                if background!=previous_background:
                    background_changes.append(dict(at=now(),processes=background))
                    previous_background=background
                samples+=1
                if others and not shared:
                    contamination=others; break
                if time.monotonic()-started>budget_seconds:
                    error='Correctness worker exceeded time budget'; break
                if process.poll() is not None: break
                time.sleep(1)
    except BaseException as exc:
        error=str(exc) or type(exc).__name__
        if isinstance(exc,(KeyboardInterrupt,SystemExit)): raise
    finally:
        if process is not None: stop_worker(process)
        valid=process is not None and process.returncode==0 and samples>0 and not contamination and error is None
        if valid:
            result=read_json(folder/f'{name}.json')
            valid=(result['definition_sha256']==definition['sha256'] and
                   [r['id'] for r in result['requests']]==[r['id'] for r in definition['prompts']])
        atomic_json(receipt,dict(valid=valid,samples=samples,contamination=contamination,
            error=error,returncode=process.returncode if process else None,at=now(),
            monitored_gpu_uuids=monitored_uuids,background_process_changes=background_changes,
            shared_gpu_correctness_only=shared,external_process_changes=shared_changes,
            admission_memory_free_mib=admission_memory))
    if not valid: raise RuntimeError(f'Invalid worker {name}; inspect its log and receipt')


def summarize(folder,definition):
    comparisons=[]; legacy=[]
    for protocol in definition['protocols']:
        reference=load_result(folder,definition,protocol,'ar')['requests']
        modes=[c['mode'] for c in definition.get('configurations',
            [dict(mode='ar'),dict(mode='sd'),dict(mode='ssd')]) if c['mode']!='ar']
        for mode in modes:
            actual=load_result(folder,definition,protocol,mode)['requests']
            for ar,row in zip(reference,actual):
                if ar['id']!=row['id'] or ar['input_sha256']!=row['input_sha256']:
                    raise ValueError('Unpaired correctness records')
                index=first_difference(ar['output_token_ids'],row['output_token_ids'])
                comparisons.append(dict(protocol=protocol,mode=mode,id=row['id'],
                    exact_match=index is None,first_difference=index,
                    ar_length=len(ar['output_token_ids']),actual_length=len(row['output_token_ids'])))
    current={r['id']:r for r in load_result(folder,definition,'fixed','ar')['requests']}
    for old in definition['legacy_reference']:
        row=current[old['id']]
        index=first_difference(old['output_token_ids'],row['output_token_ids'])
        legacy.append(dict(id=old['id'],exact_match=index is None,first_difference=index))
    summary=dict(at=now(),definition_sha256=definition['sha256'],
        passed=all(r['exact_match'] for r in comparisons+legacy),
        candidate_comparisons=len(comparisons),exact_matches=sum(r['exact_match'] for r in comparisons),
        shared_gpu_correctness_only=definition.get('allow_shared_gpus',False),
        reused_results=definition.get('reused_results',{}),
        unchanged_ar_reference=legacy,comparisons=comparisons,
        limitations=['Small deterministic greedy correctness gate; not a universal proof.',
                     'Not a completed performance campaign or an acceleration claim.',
                     'Original frozen fast-mode campaign is unchanged; it still has exact-output failures.'])
    atomic_json(folder/'summary.json',summary)
    print(f'Exact matches: {summary["exact_matches"]}/{len(comparisons)}; unchanged AR: '
          f'{sum(r["exact_match"] for r in legacy)}/{len(legacy)}',flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--mode',choices=['ar','sd','ssd'])
    parser.add_argument('--protocol',choices=['fixed','natural'])
    parser.add_argument('--extra-per-family',type=int,default=2)
    parser.add_argument('--modes',nargs='+',choices=['ar','sd','ssd'],default=['ar','sd','ssd'])
    parser.add_argument('--gpu0-only-contextual',action='store_true',
                        help='Allow unrelated GPU1 work for AR/SD correctness diagnostics only')
    parser.add_argument('--allow-shared-gpus',action='store_true',
                        help='Record other jobs without invalidating correctness evidence; no performance claims')
    parser.add_argument('--reuse-from',type=Path,help='Import complete, compatible correctness results')
    parser.add_argument('--kv-fraction',type=float,default=0.10)
    parser.add_argument('--candidate',choices=['ar_compatible','hybrid_bmm'],default='ar_compatible')
    parser.add_argument('--sd-k',type=int,default=6)
    parser.add_argument('--ssd-k',type=int,default=7)
    parser.add_argument('--worker-budget-seconds',type=float,default=600)
    args=parser.parse_args(); args.outdir=args.outdir.resolve()
    if args.worker:
        worker(args); return
    if not 0<=args.extra_per_family<=8: parser.error('extra-per-family must be 0..8')
    if not 1<=args.sd_k<=10 or not 1<=args.ssd_k<=10:
        parser.error('Diagnostic K must be 1..10')
    if 'ar' not in args.modes: parser.error('AR reference must be included')
    if args.gpu0_only_contextual and 'ssd' in args.modes:
        parser.error('GPU0-only diagnostics support AR and SD only')
    if not math.isfinite(args.worker_budget_seconds) or args.worker_budget_seconds<=0:
        parser.error('worker-budget-seconds must be finite and positive')
    if not math.isfinite(args.kv_fraction) or not 0<args.kv_fraction<1:
        parser.error('kv-fraction must be between 0 and 1')
    signal.signal(signal.SIGTERM,interrupted)
    args.outdir.mkdir(parents=True,exist_ok=True)
    if (args.outdir/'definition.json').exists():
        raise ValueError('Output directory already has a frozen gate; preserve it and use a new directory')
    # Share the campaign lock so a previous benchmark watcher cannot run beside this gate.
    with (ROOT/'results/baselines/eval1/.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            definition=prepare(args.outdir,args.extra_per_family,args.modes,args.gpu0_only_contextual,
                args.allow_shared_gpus,args.reuse_from,args.kv_fraction,
                args.candidate,args.sd_k,args.ssd_k)
            atomic_json(args.outdir/'status.json',dict(status='running',at=now()))
            for protocol in definition['protocols']:
                for config in definition['configurations']:
                    if f'{protocol}_{config["mode"]}' in definition.get('reused_results',{}):
                        load_result(args.outdir,definition,protocol,config['mode'])
                        print(f'Reusing compatible {protocol} / {config["mode"]}',flush=True)
                        continue
                    print(f'Starting {protocol} / {config["mode"]}',flush=True)
                    monitor_worker(args.outdir,definition,config['mode'],protocol,args.worker_budget_seconds)
            summary=summarize(args.outdir,definition)
            atomic_json(args.outdir/'status.json',dict(status='passed' if summary['passed'] else 'failed',at=now()))
        except BaseException as exc:
            atomic_json(args.outdir/'status.json',dict(status='stopped',at=now(),error=str(exc)))
            raise


if __name__=='__main__': main()
