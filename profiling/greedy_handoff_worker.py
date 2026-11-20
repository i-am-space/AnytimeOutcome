"""Fresh monitored worker. Baseline imports the unchanged original engine."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time

NEW_ROOT=Path(__file__).resolve().parents[1]
ENGINE_ROOT=Path(os.environ.get('AO_HANDOFF_ENGINE_ROOT',NEW_ROOT))
sys.path.insert(0,str(NEW_ROOT/'profiling'))
sys.path.insert(0,str(ENGINE_ROOT/'third_party/ssd'))

# These hooks must also run when multiprocessing re-imports this entrypoint.
if os.environ.get('AO_HANDOFF_LEGACY_TRACE')=='1':
    import ssd.engine
    name='ssd.engine.handoff_state'
    spec=importlib.util.spec_from_file_location(name,NEW_ROOT/'third_party/ssd/ssd/engine/handoff_state.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    from trace_legacy_handoff import install
    install()
if os.environ.get('AO_HANDOFF_STRESS')=='1':
    import torch
    from ssd.engine.greedy_draft_runner import GreedyDraftRunner
    original_step=GreedyDraftRunner._tree_step
    def delayed(r,depth,*a,**kw):
        out=original_step(r,depth,*a,**kw)
        if depth>=r.config.early_publish_tokens:torch.cuda._sleep(4000000)
        return out
    GreedyDraftRunner._tree_step=delayed


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--phase',required=True);parser.add_argument('--job',required=True)
    parser.add_argument('--attempt',type=Path,required=True);args=parser.parse_args()
    from baseline_common import read_json,atomic_json,allocation,software_fingerprint
    import run_greedy_handoff_campaign as campaign
    q=read_json(args.outdir/'queue.json');campaign.check_runtime(q)
    d=campaign.load_definition(args.outdir,q,args.phase);j=next(j for j in d['jobs'] if j['id']==args.job)
    c=j['config']
    if Path(c['engine_root'])!=ENGINE_ROOT:raise ValueError('Engine root changed')
    if args.phase=='micro':
        import subprocess
        command=[sys.executable,'-m','pytest','tests/test_greedy_handoff_gpu.py','-q']
        completed=subprocess.run(command,cwd=NEW_ROOT)
        if completed.returncode:raise RuntimeError('GPU ownership tests failed')
        atomic_json(args.attempt/'result.json',dict(queue_sha256=q['sha256'],definition_sha256=d['sha256'],job=j,
            requests=[],micro_cuda_passed=True,software=software_fingerprint()))
        return
    import run_engine
    import torch
    from ssd import LLM,SamplingParams
    import ssd
    if not Path(ssd.__file__).resolve().is_relative_to(ENGINE_ROOT):
        raise ValueError('Wrong SSD checkout was imported')
    from ssd.engine.llm_engine import METRICS
    options=dict(target_numerics='shared_fixed16',num_gpus=1 if c['mode']=='ar' else 2,
        speculate=c['mode']!='ar',draft_async=c['mode']=='ssd',speculate_k=c['k'],async_fan_out=3,
        jit_speculate=True,max_num_seqs=1,max_model_len=4096,enable_prefix_cache=False,
        gpu_memory_utilization=d['kv_fraction'],draft_gpu_memory_utilization=d['kv_fraction'],
        early_publish_tokens=c['p'],publication_policy='overlap' if c['p'] else 'uniform',
        publication_max_inflight=2,drain_draft_on_finish=c['mode']=='ssd')
    if c['greedy']:
        options.update(greedy_transport=True,handoff_service=c['service'],
                       handoff_bypass_ready=c['bypass'],handoff_trace=d['diagnostic'])
    model=LLM(q['models']['target']['path'],draft=q['models']['draft']['path'],**options)
    sampling=SamplingParams(temperature=0,max_new_tokens=256,ignore_eos=j['protocol']=='fixed')
    for p in d['warmup']:model.generate([p['token_ids']],[sampling],use_tqdm=False,stream_callback=lambda *a:None)
    result=dict(queue_sha256=q['sha256'],definition_sha256=d['sha256'],job=j,requests=[],
        hardware_allocation=allocation(q,c['mode']),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        actual_dtype=str(model.config.hf_config.torch_dtype),cuda_graphs=True,prefix_cache=False,
        kv_fraction=d['kv_fraction'],profile_capture=False,diagnostic=d['diagnostic'],
        stress=args.phase=='stress',engine_root=str(ENGINE_ROOT),engine_module_path=ssd.__file__,software=software_fingerprint())
    result.update(transport_version='greedy_ids_v1' if getattr(model.config,'greedy_transport',False) else 'legacy_logits',
                  handoff_service=getattr(model.config,'handoff_service','inline'),
                  bypass_ready=getattr(model.config,'handoff_bypass_ready',False))
    for i in j['prompt_order']:
        p=d['prompts'][i];torch.cuda.reset_peak_memory_stats();started=time.monotonic_ns()
        row=run_engine.measure_request(model,p,sampling,synchronize=torch.cuda.synchronize)
        finished=time.monotonic_ns()
        row['target_peak_memory']=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
        row['handoff_trace']=list(METRICS.get('handoff_trace',[]))
        if os.environ.get('AO_HANDOFF_LEGACY_TRACE')=='1':
            from trace_legacy_handoff import collect
            row['handoff_trace']=[r for r in collect(os.environ['AO_HANDOFF_TRACE_DIR'])
                if r.get('delivery_ns') is not None and started<=r['delivery_ns']<=finished]
        result['requests'].append(row);atomic_json(args.attempt/'result.json',result)
        print(j['id'],p['id'],'done',flush=True)
    model.exit(hard=True)


if __name__=='__main__':main()
