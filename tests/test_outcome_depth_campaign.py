from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
import run_outcome_depth_campaign as campaign


def queue():
    splits={}
    for split,size in [('tune',8),('heldout',25),('warmup',1)]:
        splits[split]=[dict(id=f'{split}-{f}-{i}',family=f,category=f,token_ids=[1,2],
            token_ids_sha256=campaign.digest([1,2])) for f in campaign.FAMILIES for i in range(size)]
    gate=[dict(p,id=p['id']+'-gate') for p in splits['warmup']]*2
    gate=[dict(p,id=p['id']+str(i)) for i,p in enumerate(gate)]
    return dict(sha256='queue',splits=splits,gate_prompts=gate)


def selection(k=7):
    return dict(winners={m:campaign.config(m,k) for m in ('complete','overlap','ranked','ranked_overlap')},scores=[])


def test_declared_grid_counts_and_deterministic_pairing():
    q=queue();campaign.validate_workload(q['splits']);configs=campaign.tuning_configs()
    assert len(configs)==14 and len({c['label'] for c in configs})==14
    for phase,count,requests in [('gate',28,168),('tune',28,672),('heldout',70,5250)]:
        cfgs=campaign.evaluation_configs(selection()) if phase=='heldout' else configs
        d=campaign.definition(q,phase,cfgs,'s')
        assert d==campaign.definition(q,phase,cfgs,'s')
        assert len(d['jobs'])==count and count*len(d['prompts'])==requests
        for r in range(d['repetitions']):
            jobs=[j for j in d['jobs'] if j['round']==r]
            assert len({tuple(j['prompt_order']) for j in jobs})==1
    d=campaign.definition(q,'tune',configs)
    orders=[[j['config']['label'] for j in d['jobs'] if j['round']==r] for r in range(2)]
    assert orders[0][0]==orders[1][0]=='ar' and orders[0][1:]==list(reversed(orders[1][1:]))


@pytest.mark.parametrize('k',[7,10])
def test_ablations_follow_selected_unequal_k_and_match_work(k):
    configs=campaign.evaluation_configs(selection(k))
    ranked=next(c for c in configs if c['method']=='ranked')
    controls=[c for c in configs if c['method'] in ('budget','reverse')]
    assert all(c['k']==k and sum(c['outcome_depths'])==sum(ranked['outcome_depths']) for c in controls)
    assert next(c for c in controls if c['method']=='reverse')['outcome_depths']==list(reversed(ranked['outcome_depths']))


def evidence():
    q=queue();d=campaign.definition(q,'tune',campaign.tuning_configs())
    blocks=[dict(job=j,requests=[dict(id=p['id'],family=p['family'],latency_s=2.,
        input_token_ids_sha256='same',output_token_ids=[1,2]) for p in d['prompts']]) for j in d['jobs']]
    return q,d,blocks


def test_selection_cannot_access_heldout_and_ties_choose_small_k(monkeypatch,tmp_path):
    q,d,blocks=evidence()
    def load(folder,q,phase):assert phase=='tune';return d
    def available(folder,*a,**kw):assert folder==tmp_path/'tune';return blocks
    monkeypatch.setattr(campaign,'load_definition',load);monkeypatch.setattr(campaign,'available_results',available)
    selected=campaign.select(tmp_path,q)
    assert selected['winners']['complete']['k']==4
    assert all(selected['winners'][m]['k']==7 for m in ('overlap','ranked','ranked_overlap'))
    assert sum(s['requests'] for s in selected['scores'])==672


def test_mismatch_or_missing_gate_coverage_stops_performance(monkeypatch,tmp_path):
    q,d,blocks=evidence();blocks[1]['requests'][0]['output_token_ids']=[2,1]
    monkeypatch.setattr(campaign,'load_definition',lambda *a:d)
    monkeypatch.setattr(campaign,'available_results',lambda *a,**kw:blocks)
    with pytest.raises(ValueError,match='exactness'):campaign.select(tmp_path,q)
    with pytest.raises(ValueError,match='gate'):campaign.validate_gate(tmp_path,q)


def test_corrupt_or_contaminated_receipt_and_incomplete_attempts_cannot_count(tmp_path):
    output=tmp_path/'result.json';output.write_text('{}');d=dict(sha256='d')
    receipt=dict(valid=True,definition_sha256='d',returncode=0,samples=1,external_processes=[],error=None,
        dedicated_gpus=True,profiler=False,output_sha256=campaign.file_digest(output))
    campaign.validate_receipt(receipt,d,output)
    for name,value in [('returncode',-15),('external_processes',[dict(pid=123)]),('valid',False),
                       ('samples',0),('dedicated_gpus',False),('output_sha256','bad')]:
        with pytest.raises(ValueError):campaign.validate_receipt(dict(receipt,**{name:value}),d,output)
    campaign.atomic_json(tmp_path/'job/attempt_0/receipt.json',dict(valid=False))
    assert campaign.block_result(tmp_path,dict(id='job'),d,{}) is None


def test_one_completed_predecessor_does_not_release_queue(tmp_path):
    deps=[tmp_path/'a',tmp_path/'b'];q=dict(after=[dict(path=str(p)) for p in deps])
    campaign.atomic_json(deps[0]/'status.json',dict(status='complete'))
    campaign.atomic_json(deps[1]/'status.json',dict(status='pending_dedicated_gpus'))
    assert not campaign.predecessor_ready(q)
    campaign.atomic_json(deps[1]/'status.json',dict(status='complete'))
    assert campaign.predecessor_ready(q)


def test_paired_report_groups_rounds_prompts_and_families(monkeypatch,tmp_path):
    q=queue();q.update(reference_contract='shared AR',scope='synthetic test fixture',limitations=[])
    selected=selection();cfgs=campaign.evaluation_configs(selected);d=campaign.definition(q,'heldout',cfgs,'s')
    (tmp_path/'selection.json').write_text('{}');blocks=[]
    for j in d['jobs']:
        rows=[]
        for p in d['prompts']:
            latency=float((j['round']+1)*(campaign.FAMILIES.index(p['family'])+1))
            if j['config']['method']!='ranked':latency*=2
            rows.append(dict(id=p['id'],family=p['family'],latency_s=latency,first_delivery_s=.01,
                output_tokens=256,output_token_ids=[1]*256,input_token_ids_sha256='same',tokens_per_step=4.,
                cache_hit_rate=.8,verified_draft_tokens=7,accepted_draft_tokens=3,publication_records=[],
                publication_policy_stats=[0.]*12 if j['config']['policy']=='overlap' else None,
                outcome_depth_stats=None,draft_wait_s=.1,final_drain_s=.01,
                accepted_suffix_lens_with_recovery=[4],draft_peak_memory=[],
                target_peak_memory=dict(allocated_bytes=100,reserved_bytes=200)))
        blocks.append(dict(job=j,requests=rows))
    monkeypatch.setattr(campaign,'validate_selection',lambda *a:selected)
    monkeypatch.setattr(campaign,'load_definition',lambda *a:d)
    monkeypatch.setattr(campaign,'available_results',lambda *a:blocks)
    monkeypatch.setattr(campaign,'plots',lambda *a:None)
    summary=campaign.final_report(tmp_path,q)
    assert summary['heldout_requests']==5250 and summary['tuning_requests']==672
    assert summary['exact_speculative_matches']==4500 and summary['correctness_passed']
    for r in summary['results']:
        if r['method']=='ranked':assert r['comparison_to']['complete']['speedup_ci95']==[2.,2.]
