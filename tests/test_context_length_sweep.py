import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
import run_context_length_sweep as sweep
from analyze_context_sweep import reconstruct


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs==dict(tokenize=True,add_generation_prompt=True,enable_thinking=False)
        text=messages[0]['content']
        # Fixed template and task overhead; one token per repeated filler word.
        return [101,102]+[10]*text.count(' note')+[201]*60+[103,104]


def queue():
    return dict(sha256='queue',reference_contract='shared-fixed16-v1',limitations=['Synthetic diagnostic'],
                prompts=sweep.make_prompts(Tokenizer()),warmup=sweep.make_prompts(Tokenizer(),True))


def test_exact_complete_contexts_and_disjoint_warmup():
    q=queue();sweep.validate_prompts(q['prompts'],q['warmup'])
    assert len(q['prompts'])==12
    for p in q['prompts']+q['warmup']:
        assert p['complete_task'] in p['text']
        assert p['token_ids'][:2]==[101,102] and p['token_ids'][-2:]==[103,104]
        assert len(p['token_ids'])==p['context_length']
    assert sweep.make_prompts(Tokenizer())==q['prompts']


@pytest.mark.parametrize('change',['tokens','hash','length','duplicate','warmup_overlap'])
def test_invalid_frozen_workload_rejected(change):
    q=queue()
    if change=='tokens':q['prompts'][0]['token_ids'].append(99)
    elif change=='hash':q['prompts'][0]['token_ids_sha256']='wrong'
    elif change=='length':q['prompts'][0]['context_length']=4096
    elif change=='duplicate':q['prompts'][1]=q['prompts'][0]
    else:q['warmup'][0]['id']=q['prompts'][0]['id']
    with pytest.raises(ValueError):sweep.validate_prompts(q['prompts'],q['warmup'])


def test_separate_profile_and_performance_schedules_preserve_pairing():
    q=queue();d=sweep.definition(q,'measure');p=sweep.definition(q,'profiles')
    assert d==sweep.definition(q,'measure')
    assert len(d['jobs'])==21 and sum(len(j['prompt_order']) for j in d['jobs'])==252
    assert len(p['jobs'])==12 and sum(len(j['prompt_order']) for j in p['jobs'])==36
    assert not d['profile_capture'] and p['profile_capture']
    for round_ in range(3):
        jobs=[j for j in d['jobs'] if j['round']==round_]
        assert jobs[0]['config']['method']=='ar'
        assert len({tuple(j['prompt_order']) for j in jobs})==1
    assert {j['config']['method'] for j in p['jobs']}=={'complete','overlap'}
    for j in p['jobs']:
        assert {d['prompts'][i]['context_length'] for i in j['prompt_order']}=={j['context_length']}
        assert len(j['prompt_order'])==3


@pytest.mark.parametrize('state',['running','pending_dedicated_gpus','stopped','selection_frozen','complete'])
def test_predecessor_gate_requires_successful_completion(tmp_path,state):
    sweep.atomic_json(tmp_path/'status.json',dict(status=state))
    q=dict(after=dict(path=str(tmp_path)))
    assert sweep.predecessor_ready(q)==(state=='complete')


def test_dependency_rejects_missing_final_report_even_with_complete_status(tmp_path,monkeypatch):
    after=tmp_path/'previous';after.mkdir()
    sweep.atomic_json(after/'status.json',dict(status='complete'))
    sweep.atomic_json(after/'queue.json',{})
    sweep.atomic_json(after/'summary.json',dict(status='complete',correctness_passed=True))
    for name in ('check_runtime','validate_sources','validate_selection'):
        monkeypatch.setattr(sweep.predecessor,name,lambda *a:None)
    monkeypatch.setattr(sweep.predecessor,'load_definition',lambda *a:{})
    monkeypatch.setattr(sweep.predecessor,'available_results',lambda *a,**k:[])
    with pytest.raises(ValueError,match='report missing'):
        sweep.validate_dependency(tmp_path,dict(after=dict(path=str(after))))
    assert not (tmp_path/'dependency.json').exists()


@pytest.mark.parametrize('profile',[False,True])
def test_receipts_reject_contamination_interruption_and_profile_mixing(tmp_path,profile):
    sweep.atomic_json(tmp_path/'result.json',{})
    artifacts={}
    if profile:
        for name in ('trace.nsys-rep','trace.sqlite','phase_analysis.json'):
            (tmp_path/name).write_text('test artifact')
            artifacts[name]=sweep.file_digest(tmp_path/name)
    d=dict(sha256='definition',profile_capture=profile)
    r=dict(valid=True,definition_sha256='definition',returncode=0,samples=4,external_processes=[],
        error=None,dedicated_gpus=True,profiler=profile,output_sha256=sweep.file_digest(tmp_path/'result.json'),artifacts=artifacts)
    sweep.validate_receipt(r,d,tmp_path)
    for field,value in [('valid',False),('returncode',-15),('samples',0),('external_processes',[dict(pid=123)]),
                        ('profiler',not profile),('dedicated_gpus',False),('output_sha256','changed')]:
        with pytest.raises(ValueError):sweep.validate_receipt(dict(r,**{field:value}),d,tmp_path)
    if profile:
        (tmp_path/'trace.sqlite').write_text('changed')
        with pytest.raises(ValueError,match='artifact changed'):sweep.validate_receipt(r,d,tmp_path)


def test_partial_outputs_cannot_count_as_complete_block(tmp_path):
    q=queue();d=sweep.definition(q,'measure');j=d['jobs'][0]
    attempt=tmp_path/j['id']/'attempt_0';attempt.mkdir(parents=True)
    sweep.atomic_json(attempt/'result.json',dict(requests=[{}]))
    assert sweep.block_result(tmp_path,j,d,q) is None
    sweep.atomic_json(attempt/'receipt.json',dict(valid=False))
    assert sweep.block_result(tmp_path,j,d,q) is None


def test_exactness_gate_checks_tokens_and_input_identity():
    gold=dict(id='x',input_token_ids_sha256='same',output_token_ids=[1,2,3])
    b=dict(requests=[copy.deepcopy(gold)])
    assert sweep.verify_outputs(b,{'x':gold})[0]['exact']
    b['requests'][0]['output_token_ids']=[1,2,4]
    with pytest.raises(ValueError,match='Exactness'):sweep.verify_outputs(b,{'x':gold})
    b['requests'][0]['input_token_ids_sha256']='changed'
    with pytest.raises(ValueError,match='input changed'):sweep.verify_outputs(b,{'x':gold})


def trace(full=True,miss=False):
    # Host tree steps enqueue quickly; GPU work finishes much later.
    # Target and drafter intentionally reuse correlation ID 1.
    nv=[(0,1000,'timed_region',1),(1,999,'request:sample',1),
        (10,50,'draft/decode_tree',2),(11,20,'draft/tree_step_0',2),
        (100,250,'SD/speculate',1),(110,111,'SSD/handshake_send',1),
        (251,500,'SD/verify',1),(252,260,'verify/target_fwd',1)]
    launches=[(12,1,2),(253,1,1)]
    kernels=[(15,90,1,2),(280,490,1,1)]
    if full:
        nv.append((21,30,'draft/tree_step_1',2));launches.append((22,2,2));kernels.append((91,200,2,2))
    if miss:nv.append((120,240,'draft/jit_speculate(miss)',2))
    return nv,launches,kernels


def test_trace_uses_gpu_completion_not_cpu_enqueue_or_other_process():
    r=reconstruct(*trace(),k=2)
    assert r['full_trees']==1 and r['partial_trees']==0
    assert r['mean_full_tree_gpu_span_ms']==pytest.approx(185/1e6)
    assert r['mean_verify_gpu_span_ms']==pytest.approx(210/1e6)
    assert r['mean_verify_wall_ms']==pytest.approx(249/1e6)
    assert r['cache_hit_ready_depth_histogram']=={'1':1}
    assert r['rounds'][0]['remaining_full_tree_ms']==pytest.approx(90/1e6)


def test_interrupted_tree_is_not_extrapolated_to_full_depth():
    r=reconstruct(*trace(full=False),k=2)
    assert r['full_trees']==0 and r['partial_trees']==1
    assert r['mean_full_tree_gpu_span_ms'] is None and r['full_tree_over_verify_gpu'] is None
    assert r['cache_hit_ready_depth_histogram']=={'1':1}


def test_miss_has_no_invented_ready_depth():
    r=reconstruct(*trace(miss=True),k=2)
    assert r['cache_hit_ready_depth_histogram']=={}
    assert r['unknown_ready_depth_rounds']==1


def test_missing_gpu_correlation_fails_instead_of_using_cpu_launch_time():
    nv,launch,kernels=trace()
    with pytest.raises(ValueError,match='verification GPU work'):
        reconstruct(nv,launch,[v for v in kernels if v[3]!=1],2)


def test_incomplete_full_depth_correlation_stays_unknown():
    nv,launch,kernels=trace()
    # Another partial tree is fully correlated, while the tree at the target's
    # request is missing a launch correlation. Do not invent its ready depth.
    nv.extend([(600,650,'draft/decode_tree',2),(610,630,'draft/tree_step_0',2)])
    launch.append((615,3,2));kernels.append((620,690,3,2))
    r=reconstruct(nv,launch,[v for v in kernels if not (v[3]==2 and v[2]==2)],2)
    assert r['full_trees']==0 and r['uncorrelated_trees']==1
    assert r['cache_hit_ready_depth_histogram']=={}


def test_aggregation_keeps_lengths_templates_and_rounds_paired(tmp_path,monkeypatch):
    q=queue();d=sweep.definition(q,'measure');pd=sweep.definition(q,'profiles')
    blocks={}
    for phase,definition in [('measure',d),('profiles',pd)]:
        for j in definition['jobs']:
            rows=[]
            for i in j['prompt_order']:
                p=q['prompts'][i]
                value=p['context_length']+10000*j['round']+100*sweep.FAMILIES.index(p['family'])
                rows.append(dict(id=p['id'],input_token_ids_sha256=p['token_ids_sha256'],family=p['family'],
                    latency_s=value,decode_s=value-.1,prefill_s=.1,output_token_ids=[1,2]))
            b=dict(job=j,requests=rows)
            if phase=='profiles':
                b['phase_analysis']=dict(full_trees=2,mean_full_tree_gpu_span_ms=2.,mean_verify_gpu_span_ms=1.,
                    mean_verify_wall_ms=1.2,full_tree_over_verify_gpu=2.,cache_hit_ready_depth_histogram={'4':2})
            blocks[phase,j['id']]=b
    monkeypatch.setattr(sweep,'load_definition',lambda f,q,phase:d if phase=='measure' else pd)
    monkeypatch.setattr(sweep,'block_result',lambda folder,j,*a:blocks[folder.name,j['id']])
    monkeypatch.setattr(sweep,'plots',lambda *a:None)
    def aggregate(rows):
        return dict(mean_latency_s=sum(r['latency_s'] for r in rows)/len(rows),mean_target_wait_s=0,
                    tokens_per_verification=4,responses_before_tail_done=0)
    monkeypatch.setattr(sweep.predecessor,'aggregate',aggregate)
    calls=[]
    def bootstrap(reference,method,families,*a):
        assert reference==method
        assert families==list(sweep.FAMILIES)
        assert len(reference)==3 and all(len(round_)==3 for round_ in reference)
        assert reference[1][0]-reference[0][0]==10000
        assert reference[0][1]-reference[0][0]==100
        calls.append(reference)
        return dict(speedup=1.)
    monkeypatch.setattr(sweep,'paired_bootstrap',bootstrap)
    summary=sweep.report(tmp_path,q)
    assert summary['exact_comparisons']==252 and len(calls)==24
    assert len(summary['results'])==28
    assert summary['results'][0]['mean_latency_s']==pytest.approx(256+10000+100)
