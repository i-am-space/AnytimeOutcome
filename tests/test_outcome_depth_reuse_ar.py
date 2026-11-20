from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
import run_outcome_depth_reuse_ar as campaign


def queue():
    splits={}
    for split,n in [('tune',8),('heldout',25),('warmup',1)]:
        splits[split]=[dict(id=f'{split}-{f}-{i}',family=f,category=f,token_ids=[1,2],
            token_ids_sha256=campaign.digest([1,2])) for f in campaign.FAMILIES for i in range(n)]
    return dict(sha256='q',splits=splits,gate_prompts=[dict(id=str(i)) for i in range(6)],ar_reference={})


def test_small_fresh_ar_gate_retained_but_full_ar_timing_arms_removed():
    q=queue();configs=campaign.tuning_configs()
    assert len(configs)==13 and all(c['mode']=='ssd' for c in configs)
    gate=campaign.definition(q,'gate',[campaign.config('ar')]+configs)
    assert len(gate['jobs'])==28 and sum(j['config']['mode']=='ar' for j in gate['jobs'])==2
    tune=campaign.definition(q,'tune',configs)
    assert len(tune['jobs'])==26 and len(tune['prompts'])*len(tune['jobs'])==624
    assert not any(j['config']['mode']=='ar' for j in tune['jobs'])
    selected=dict(winners={m:campaign.config(m,7) for m in ('complete','overlap','ranked','ranked_overlap')})
    configs=campaign.evaluation_configs(selected)
    held=campaign.definition(q,'heldout',configs,'s')
    assert len(held['jobs'])==60 and len(held['jobs'])*len(held['prompts'])==4500
    assert not any(j['config']['mode']=='ar' for j in held['jobs'])
    assert sum(len(d['jobs']) for d in (gate,tune,held))==114
    assert gate==campaign.definition(q,'gate',[campaign.config('ar')]+campaign.tuning_configs())
    for d in (tune,held):
        for r in range(d['repetitions']):
            assert len({tuple(j['prompt_order']) for j in d['jobs'] if j['round']==r})==1


def test_selection_checks_saved_tuning_ar_only(monkeypatch,tmp_path):
    q=queue();d=campaign.definition(q,'tune',campaign.tuning_configs())
    blocks=[dict(job=j,requests=[dict(id=p['id'],family=p['family'],latency_s=2.,
        input_token_ids_sha256='same',output_token_ids=[1]) for p in d['prompts']]) for j in d['jobs']]
    def reference(ref,phase,*a):
        assert phase=='tune'
        return dict(requests=blocks[0]['requests'])
    monkeypatch.setattr(campaign.saved_ar,'reference_block',reference)
    monkeypatch.setattr(campaign,'load_definition',lambda folder,q,phase:d if phase=='tune' else None)
    monkeypatch.setattr(campaign,'available_results',lambda folder,*a,**kw:blocks if folder==tmp_path/'tune' else None)
    value=campaign.select(tmp_path,q)
    assert sum(v['requests'] for v in value['scores'])==624
    assert value['winners']['complete']['k']==4 and value['winners']['ranked']['k']==7
