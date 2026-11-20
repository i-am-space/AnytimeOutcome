from pathlib import Path
import copy
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
import saved_ar_reference as saved
import run_publication_reuse_ar as campaign
import run_context_length_sweep_v2 as context


def queue():
    splits={}
    for split,n in [('tune',8),('heldout',25),('warmup',1)]:
        splits[split]=[dict(id=f'{split}-{f}-{i}',family=f,category=f,token_ids=[1,2],
            token_ids_sha256=campaign.digest([1,2])) for f in campaign.FAMILIES for i in range(n)]
    return dict(sha256='q',splits=splits,tuning_configs=campaign.tuning_configs(),ar_reference={'sha256':'ref'},
        continuity_prompts=[next(p for p in splits['tune'] if p['family']==f) for f in campaign.FAMILIES])


def test_full_ar_removed_but_short_continuity_and_new_context_ar_remain():
    q=queue();configs=campaign.tuning_configs()
    assert len(configs)==21 and all(c['mode']=='ssd' for c in configs)
    d=campaign.definition(q,'tune',configs)
    assert len(d['jobs'])==42 and len(d['jobs'])*len(d['prompts'])==1008
    assert d==campaign.definition(q,'tune',configs)
    orders=[[j['config'] for j in d['jobs'] if j['round']==r] for r in range(2)]
    assert orders[0]==list(reversed(orders[1]))
    d=campaign.definition(q,'continuity',[campaign.config('ar')])
    assert len(d['jobs'])==1 and len(d['prompts'])==3 and {p['family'] for p in d['prompts']}==set(campaign.FAMILIES)
    selected=dict(winners={m:campaign.config(m,7,4,2) for m in ('complete','uniform','overlap')})
    configs=campaign.evaluation_configs(selected);d=campaign.definition(q,'heldout',configs,'selected')
    assert len(d['jobs'])==30 and all(j['config']['mode']=='ssd' for j in d['jobs'])
    orders=[tuple(j['config']['label'] for j in d['jobs'] if j['round']==r and j['protocol']=='fixed') for r in range(5)]
    assert len(set(orders))==5
    cq=dict(sha256='q',prompts=[dict(id=str(i)) for i in range(12)],warmup=[])
    d=context.definition(cq,'measure')
    assert sum(j['config']['mode']=='ar' for j in d['jobs'])==3


def test_saved_reference_checks_every_spec_block_without_waiting_for_fresh_ar(monkeypatch):
    gold=dict(requests=[dict(id='p',input_token_ids_sha256='same',output_token_ids=[1,2,3])])
    calls=[]
    def old(ref,phase,protocol,r):calls.append(phase);return gold
    monkeypatch.setattr(saved,'reference_block',old)
    block=dict(job=dict(config=dict(mode='ssd',label='ssd'),protocol='fixed',round=0),
        requests=[dict(id='p',input_token_ids_sha256='same',output_token_ids=[1,9,3])])
    checks=saved.check_outputs([block],dict(ar_reference={}), 'tune')
    assert calls==['tune'] and len(checks)==1 and checks[0]['first_difference_zero_based']==1
    assert checks[0]['reference_origin']=='saved_shared_ar'
    block['requests'][0]['input_token_ids_sha256']='changed'
    with pytest.raises(ValueError,match='input'):saved.check_outputs([block],dict(ar_reference={}), 'tune')


def test_ar_continuity_is_checked_against_saved_outputs(monkeypatch):
    gold=dict(requests=[dict(id='p',input_token_ids_sha256='same',output_token_ids=[1])])
    monkeypatch.setattr(saved,'reference_block',lambda *a:gold)
    block=dict(job=dict(config=dict(mode='ar',label='ar'),protocol='fixed',round=0),requests=gold['requests'])
    assert saved.check_outputs([block],dict(ar_reference={}), 'continuity')[0]['exact']
    with pytest.raises(ValueError,match='Unexpected full AR'):saved.check_outputs([block],dict(ar_reference={}), 'tune')


def test_historical_bootstrap_does_not_pretend_rounds_are_contemporaneous():
    a=[[float(r*f) for f in (1,2,3)] for r in (1,2,3,4,5)];b=[[x/2 for x in row] for row in a]
    result=saved.historical_comparison(a,b,['chat','code','doc'])
    assert result['speedup']==2 and result['speedup_ci95'][0]<2<result['speedup_ci95'][1]
    assert result['contemporaneous'] is False and result['reference_origin']=='historical_ar'
    assert 'independent' in result['bootstrap_unit']


def test_reference_artifacts_and_ar_graph_contract_cannot_drift(tmp_path):
    graph=tmp_path/saved.GRAPH_FILE;graph.parent.mkdir(parents=True)
    graph.write_text('def run_decode_cudagraph():\n    return 1\ndef capture_cudagraph():\n    return 2\n')
    artifact=tmp_path/'ar.json';artifact.write_text('{}')
    ref=dict(artifacts_sha256={str(artifact):saved.file_digest(artifact)},numerical_files={},
        ar_graph_functions=saved.graph_contract(graph),models={},software={},gpus=[])
    ref['sha256']=saved.digest(ref);saved.validate_reference(ref,root=tmp_path)
    artifact.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='artifact changed'):saved.validate_reference(ref,root=tmp_path)
    artifact.write_text('{}');graph.write_text(graph.read_text().replace('return 1','return 9'))
    with pytest.raises(ValueError,match='fresh AR required'):saved.validate_reference(ref,root=tmp_path)


def test_selection_reads_tuning_only_and_never_schedules_full_ar(monkeypatch,tmp_path):
    q=queue();d=campaign.definition(q,'tune',campaign.tuning_configs())
    blocks=[dict(job=j,requests=[dict(id=p['id'],family=p['family'],input_token_ids_sha256='same',
        output_token_ids=[1],latency_s=2.) for p in d['prompts']]) for j in d['jobs']]
    def reference(ref,phase,*a):
        assert phase=='tune','Selection accessed held-out outputs'
        return dict(requests=blocks[0]['requests'])
    monkeypatch.setattr(saved,'reference_block',reference)
    monkeypatch.setattr(campaign,'validate_continuity',lambda *a:None)
    monkeypatch.setattr(campaign,'load_definition',lambda folder,q,phase:d if phase=='tune' else pytest.fail('Held-out selection'))
    monkeypatch.setattr(campaign,'available_results',lambda folder,*a,**kw:blocks if folder==tmp_path/'tune' else pytest.fail('Held-out selection'))
    chosen=campaign.select(tmp_path,q)
    assert chosen['winners']['complete']['k']==4 and sum(v['requests'] for v in chosen['scores'])==1008


def test_report_counts_only_new_ssd_blocks_and_labels_historical_ar(monkeypatch,tmp_path):
    q=queue();q.update(reference_contract='saved shared AR',scope='synthetic fixture')
    selected=dict(winners={m:campaign.config(m,7,4,2) for m in ('complete','uniform','overlap')},scores=[])
    configs=campaign.evaluation_configs(selected);d=campaign.definition(q,'heldout',configs,'selection')
    (tmp_path/'selection.json').write_text('{}');blocks=[]
    def row(p,r,method):
        latency=float((r+1)*(campaign.FAMILIES.index(p['family'])+1))*(1 if method=='overlap' else 2)
        return dict(id=p['id'],family=p['family'],latency_s=latency,first_delivery_s=.01,output_tokens=256,
            output_token_ids=[1]*256,input_token_ids_sha256='same',tokens_per_step=4.,cache_hit_rate=.8,
            verified_draft_tokens=7,accepted_draft_tokens=3,publication_records=[],
            publication_policy_stats=[0.]*12 if method=='overlap' else None,draft_wait_s=.1,final_drain_s=.01,
            accepted_suffix_lens_with_recovery=[4],draft_peak_memory=[],target_peak_memory=dict(allocated_bytes=1,reserved_bytes=2))
    for j in d['jobs']:blocks.append(dict(job=j,requests=[row(p,j['round'],j['config']['method']) for p in d['prompts']]))
    def old(ref,phase,protocol,r):return dict(requests=[row(p,r,'ar') for p in d['prompts']])
    monkeypatch.setattr(saved,'reference_block',old)
    result=saved.report_campaign(tmp_path,q,selected,d,blocks,campaign.aggregate,('complete','uniform'),1008,3,1)
    assert result['heldout_requests']==2250 and result['scored_requests']==3261
    assert result['valid_heldout_blocks']==30 and result['fresh_ar_performance_blocks']==0
    assert all(r['measurement_origin']=='fresh_dedicated' and r['gpu_count']==2 for r in result['results'])
    assert all(r['measurement_origin']=='historical_ar' for r in result['historical_ar_context'])
    for r in result['results']:
        assert r['comparison_to']['historical_ar']['contemporaneous'] is False
        if r['method']=='overlap':assert r['comparison_to']['complete']['speedup_ci95']==[2.,2.]
    assert 'Historical AR timing context' in (tmp_path/'report.md').read_text()
