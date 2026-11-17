from pathlib import Path
from types import SimpleNamespace
import copy
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
import run_publication_campaign as campaign


def prompt(id_,family):
    ids=[1,2,3]
    return dict(id=id_,family=family,category=family,token_ids=ids,token_ids_sha256=campaign.digest(ids))


def queue():
    splits={}
    for split,size in [('tune',8),('heldout',25),('warmup',1)]:
        splits[split]=[prompt(f'{split}-{f}-{i}',f) for f in campaign.FAMILIES for i in range(size)]
    return dict(sha256='queue',splits=splits)


def test_grid_and_frozen_workload_match_declared_sizes():
    q=queue();campaign.validate_workload(q['splits'])
    cfgs=campaign.tuning_configs()
    assert len(cfgs)==22 and len({c['label'] for c in cfgs})==22
    counts={m:sum(c['method']==m for c in cfgs) for m in ('ar','complete','uniform','overlap')}
    assert counts==dict(ar=1,complete=3,uniform=6,overlap=12)
    assert all(c['f']==3 for c in cfgs)
    d=campaign.definition(q,'tune',cfgs)
    assert len(d['jobs'])==44 and len(d['jobs'])*len(d['prompts'])==1056


@pytest.mark.parametrize('change',['overlap','tokens','size','balance','excluded'])
def test_workload_rejects_overlap_changed_tokens_unbalanced_or_excluded_inputs(change):
    splits=queue()['splits']
    if change=='overlap':splits['heldout'][0]['id']=splits['tune'][0]['id']
    elif change=='tokens':splits['tune'][0]['token_ids']=[4]
    elif change=='size':splits['warmup'].pop()
    elif change=='balance':splits['tune'][0]['family']='code'
    else:splits['heldout'][0]['id']='HumanEval/36'
    with pytest.raises(ValueError):campaign.validate_workload(splits)


def test_tuning_order_reverses_and_preserves_prompt_pairing():
    q=queue();cfgs=campaign.tuning_configs()
    d=campaign.definition(q,'tune',cfgs)
    assert d==campaign.definition(q,'tune',cfgs)
    first=[j for j in d['jobs'] if j['round']==0]
    second=[j for j in d['jobs'] if j['round']==1]
    assert first[0]['config']['method']==second[0]['config']['method']=='ar'
    assert [j['config'] for j in first[1:]]==list(reversed([j['config'] for j in second[1:]]))
    for jobs in (first,second): assert len({tuple(j['prompt_order']) for j in jobs})==1


@pytest.mark.parametrize('complete_k,expected_count',[(7,4),(4,5)])
def test_heldout_permutations_repetitions_protocols_and_anchor(complete_k,expected_count):
    selected=dict(winners=dict(complete=campaign.config('complete',complete_k),
        uniform=campaign.config('uniform',7,4),overlap=campaign.config('overlap',10,4,2)))
    cfgs=campaign.evaluation_configs(selected)
    assert len(cfgs)==expected_count
    assert any(c['k']==7 and c['p']==0 and c['mode']=='ssd' for c in cfgs)
    d=campaign.definition(queue(),'heldout',cfgs,'selection')
    assert len(d['jobs'])==expected_count*10
    assert len(d['jobs'])*len(d['prompts'])==expected_count*750
    orders=[]
    for r in range(5):
        jobs=[j for j in d['jobs'] if j['round']==r]
        assert len({tuple(j['prompt_order']) for j in jobs})==1
        assert jobs[0]['protocol']==('fixed' if r%2==0 else 'natural')
        orders.append(tuple(j['config']['label'] for j in jobs[:expected_count]))
        for protocol in ('fixed','natural'):
            assert len([j for j in jobs if j['protocol']==protocol])==expected_count
    assert len(set(orders))==5


def tuning_evidence():
    q=queue();d=campaign.definition(q,'tune',campaign.tuning_configs())
    results=[]
    for j in d['jobs']:
        results.append(dict(job=j,requests=[dict(id=p['id'],family=p['family'],input_token_ids_sha256='same',
            latency_s=2.,output_token_ids=[10,11]) for p in q['splits']['tune']]))
    return q,d,results


def test_selection_reads_only_tuning_results_and_breaks_ties_as_declared(monkeypatch,tmp_path):
    q,d,results=tuning_evidence()
    def load(folder,q,phase):
        assert phase=='tune','Selection accessed test results'
        return d
    def available(folder,*args,**kwargs):
        assert folder==tmp_path/'tune','Selection accessed test output files'
        return results
    monkeypatch.setattr(campaign,'load_definition',load)
    monkeypatch.setattr(campaign,'available_results',available)
    selected=campaign.select(tmp_path,q)
    assert selected['winners']['complete']['k']==4
    assert selected['winners']['uniform']['k']==7 and selected['winners']['uniform']['p']==2
    c=selected['winners']['overlap']
    assert (c['k'],c['p'],c['window'])==(7,2,1)
    assert sum(s['requests'] for s in selected['scores'])==1056


def test_selection_rejects_token_discrepancies(monkeypatch,tmp_path):
    q,d,results=tuning_evidence()
    results[1]['requests'][0]['output_token_ids']=[10,99]
    monkeypatch.setattr(campaign,'load_definition',lambda *a:d)
    monkeypatch.setattr(campaign,'available_results',lambda *a,**kw:results)
    with pytest.raises(ValueError,match='exactness'):campaign.select(tmp_path,q)


def test_receipt_rejects_contamination_interruption_and_shared_gpu_results(tmp_path):
    output=tmp_path/'result.json';output.write_text('{}')
    d=dict(sha256='definition')
    receipt=dict(valid=True,definition_sha256='definition',returncode=0,samples=3,
        external_processes=[],error=None,dedicated_gpus=True,profiler=False,output_sha256=campaign.file_digest(output))
    campaign.validate_receipt(receipt,d,output)
    for field,value in [('valid',False),('returncode',-15),('samples',0),('external_processes',[dict(pid=99)]),
                        ('error','interrupted'),('dedicated_gpus',False),('profiler',True),('output_sha256','changed')]:
        with pytest.raises(ValueError):campaign.validate_receipt(dict(receipt,**{field:value}),d,output)


def test_incomplete_attempts_never_count_as_repetitions(tmp_path):
    j=dict(id='fixed_r0_overlap')
    for i in range(3):
        path=tmp_path/j['id']/f'attempt_{i}'/'receipt.json'
        campaign.atomic_json(path,dict(valid=False,returncode=-15))
    assert campaign.block_result(tmp_path,j,{}, {}) is None


def test_monitor_invalidates_whole_block_and_never_cleans_up_foreign_job(monkeypatch,tmp_path):
    args=SimpleNamespace(outdir=tmp_path)
    d=dict(phase='tune',sha256='definition')
    j=dict(id='fixed_r0_ar',config=dict(mode='ar'))
    q=dict(hardware=dict(gpus=[dict(uuid='g0'),dict(uuid='g1')]))
    owned=[dict(pid=10,uuid='g0')];foreign=[dict(pid=99,uuid='g1')]
    cleaned=[];stopped=[]
    process=SimpleNamespace(pid=10,returncode=None)
    monkeypatch.setattr(campaign,'check_runtime',lambda *a:None)
    monkeypatch.setattr(campaign,'require_idle',lambda *a:None)
    monkeypatch.setattr(campaign,'compute_processes',lambda *a:owned+foreign)
    monkeypatch.setattr(campaign,'partition',lambda *a:(owned,foreign))
    monkeypatch.setattr(campaign.subprocess,'Popen',lambda *a,**kw:process)
    def stop(p):
        stopped.append(p.pid);p.returncode=-15
    monkeypatch.setattr(campaign,'stop_worker',stop)
    monkeypatch.setattr(campaign,'cleanup_tagged',lambda rows,token:cleaned.extend(r['pid'] for r in rows))
    with pytest.raises(campaign.PendingWindow):campaign.run_block(args,d,j,q)
    receipt=campaign.read_json(tmp_path/'tune'/j['id']/'attempt_0/receipt.json')
    assert receipt['valid'] is False and receipt['external_processes']==foreign
    assert stopped==[10] and cleaned==[10]
    assert campaign.block_result(tmp_path/'tune',j,d,q) is None


def test_resume_rejects_changed_source_evidence(tmp_path):
    source=tmp_path/'manifest.json';source.write_text('{}')
    q=dict(workload_source=dict(path=str(source),sha256=campaign.file_digest(source)),gate_evidence_sha256={})
    campaign.validate_sources(q)
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='workload'):campaign.validate_sources(q)


def test_exactness_waits_for_matching_round_ar_then_detects_first_divergence():
    ar=dict(job=dict(config=dict(method='ar'),protocol='fixed',round=2),
        requests=[dict(id='p',input_token_ids_sha256='same',output_token_ids=[1,2])])
    sd=dict(job=dict(config=dict(method='overlap',label='o'),protocol='fixed',round=2),
        requests=[dict(id='p',input_token_ids_sha256='same',output_token_ids=[1,3])])
    assert campaign.check_outputs([sd])==[]
    checks=campaign.check_outputs([sd,ar])
    assert checks[0]['first_difference_zero_based']==1 and not checks[0]['exact']


def test_import_does_not_install_artificial_delay_or_mutate_other_campaigns():
    from ssd.engine.draft_runner import DraftRunner
    assert DraftRunner._decode_tree_step.__name__=='_decode_tree_step'
    assert campaign.base.check_runtime.__module__=='run_overnight_extensions'


def test_statistical_report_keeps_methods_rounds_prompts_paired(monkeypatch,tmp_path):
    cfgs=[campaign.config('ar'),campaign.config('complete',7),campaign.config('uniform',7,4),campaign.config('overlap',7,4,1)]
    q=queue();q.update(reference_contract='shared AR',scope='Reused test fixture')
    d=campaign.definition(q,'heldout',cfgs,'frozen')
    selected=dict(winners={c['method']:c for c in cfgs if c['method']!='ar'})
    (tmp_path/'selection.json').write_text('{}')
    blocks=[]
    for j in d['jobs']:
        rows=[]
        for p in d['prompts']:
            # Round and family effects are shared: pairing cancels both exactly.
            latency=(j['round']+1)*(1+campaign.FAMILIES.index(p['family']))
            if j['config']['method']!='overlap':latency*=2
            rows.append(dict(id=p['id'],family=p['family'],latency_s=float(latency),first_delivery_s=.01,
                output_tokens=256,output_token_ids=[1]*256,input_token_ids_sha256='same',tokens_per_step=4.,
                cache_hit_rate=.8,verified_draft_tokens=7,accepted_draft_tokens=3,publication_records=[],
                publication_policy_stats=[0.]*12 if j['config']['method']=='overlap' else None,
                draft_wait_s=.1,final_drain_s=.01,accepted_suffix_lens_with_recovery=[4],draft_peak_memory=[],
                target_peak_memory=dict(allocated_bytes=100,reserved_bytes=200)))
        blocks.append(dict(job=j,requests=rows))
    monkeypatch.setattr(campaign,'validate_selection',lambda *a:selected)
    monkeypatch.setattr(campaign,'load_definition',lambda *a:d)
    monkeypatch.setattr(campaign,'available_results',lambda *a:blocks)
    monkeypatch.setattr(campaign,'plots',lambda *a:None)
    summary=campaign.final_report(tmp_path,q)
    assert summary['heldout_requests']==3000 and summary['correctness_passed']
    assert summary['exact_speculative_matches']==2250
    for r in summary['results']:
        if r['method']=='overlap':
            c=r['comparison_to']['complete']
            assert c['speedup']==2 and c['speedup_ci95']==[2,2]
