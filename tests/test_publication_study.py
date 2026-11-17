from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
import baseline_common as common
import run_baselines
import run_publication_study as study


def test_new_baseline_jobs_pin_backend_and_draining():
    m=dict(manifest_sha256='manifest',execution=dict(sha256='code'),
        splits=dict(tune=[dict(id='a')]),target_numerics='shared_fixed16',drain_draft_on_finish=True)
    for mode in ('ar','sd','ssd'):
        config=dict(mode=mode,k=None if mode=='ar' else 7,f=3 if mode=='ssd' else None)
        job=common.make_job(m,'tune',config,0,'fixed')
        settings=common.worker_settings(job)
        assert settings['target_numerics']=='shared_fixed16'
        assert settings['drain_draft_on_finish']==(mode=='ssd')
        m['models']=dict(target=dict(path='/target'),draft=dict(path='/draft'))
        cmd=run_baselines.worker_command(Path('/campaign'),m,job,Path('/job'),Path('/out'))
        assert cmd[cmd.index('--target-numerics')+1]=='shared_fixed16'
        assert ('--drain-draft-on-finish' in cmd)==(mode=='ssd')


def test_paired_schedule_preserves_prompts_rounds_and_alternates_protocols():
    methods=['ar','complete','early']
    jobs=study.jobs(methods,5,['fixed','natural'],9)
    assert jobs==study.jobs(methods,5,['fixed','natural'],9) and len(jobs)==30
    orders=[]
    for r in range(5):
        group=[j for j in jobs if j['round']==r]
        assert len({tuple(j['prompt_order']) for j in group})==1
        assert group[0]['protocol']==('fixed' if r%2==0 else 'natural')
        orders.append(tuple(j['method'] for j in group[:3]))
    assert len(set(orders))==5


def test_contaminated_or_interrupted_study_blocks_never_count(tmp_path):
    j=dict(id='fixed_r0_early');d=dict(sha256='definition')
    for i in range(2):
        attempt=tmp_path/j['id']/f'attempt_{i}';attempt.mkdir(parents=True)
        common.atomic_json(attempt/'receipt.json',dict(valid=False,external_processes=[123] if i==0 else [],returncode=-15))
    assert study.block_result(tmp_path,j,d) is None


def test_study_rejects_changed_tokens_and_incomplete_requests():
    p=dict(id='a',token_ids=[1]);d=dict(sha256='definition',prompts=[p],output_cap=256)
    j=dict(id='j',prompt_order=[0],protocol='fixed')
    result=dict(definition_sha256='definition',job=j,requests=[dict(id='a',input_token_ids_sha256='changed',
        output_tokens=256,output_token_ids=[2]*256)])
    with pytest.raises(ValueError,match='Unpaired'):study.validate_result(result,j,d)
    result['requests']=[]
    with pytest.raises(ValueError,match='Incomplete'):study.validate_result(result,j,d)


def test_integrated_backend_matches_frozen_prototype_source():
    # A drift in arithmetic during production integration needs a new reference version.
    root=Path(__file__).resolve().parents[1]
    original=(root/'profiling/shared_numerics.py').read_text().replace(
        'from numerics_attention import parallel_ar_attention',
        'from ssd.engine.shared_attention import parallel_ar_attention')
    assert original==(root/'third_party/ssd/ssd/engine/shared_numerics.py').read_text()
    assert (root/'profiling/numerics_attention.py').read_bytes()==(root/'third_party/ssd/ssd/engine/shared_attention.py').read_bytes()


def test_hierarchical_bootstrap_keeps_methods_paired_and_reports_output_failures(monkeypatch,tmp_path):
    prompts=[dict(id=f'{f}{i}',family=f) for f in ('chat','code','doc') for i in range(2)]
    d=dict(sha256='definition',methods=['ar','complete','early'],repetitions=5,
        protocols=['fixed','natural'],prompts=prompts,k=7,purpose='Synthetic test fixture',configuration_status='fixture')
    d['jobs']=study.jobs(d['methods'],5,d['protocols'],len(prompts))
    def result(folder,job,definition):
        rows=[]
        for p in prompts:
            # Large round effects cancel under paired resampling: true ratio=2.
            latency=(job['round']+1)*(1 if job['method']=='early' else 2)
            rows.append(dict(id=p['id'],family=p['family'],latency_s=latency,first_delivery_s=.01,
                output_tokens=256,output_token_ids=[99 if job['method']=='early' and job['round']==4 else 2]*256,
                publication_records=[],draft_wait_s=.1,final_drain_s=.01,
                verified_draft_tokens=7,accepted_draft_tokens=4,accepted_suffix_lens_with_recovery=[5],
                target_peak_memory=dict(allocated_bytes=100),draft_peak_memory=[]))
        return dict(job=job,requests=rows)
    monkeypatch.setattr(study,'block_result',result)
    report=study.report(tmp_path,d)
    assert report['correctness_passed'] is False
    assert report['within_method_repeatability']['fixed/early'] is False
    assert report['within_method_repeatability']['natural/complete'] is True
    assert all(ci['low']==ci['high']==2 for ci in report['paired_speed_ratio_intervals'].values())
    assert report['valid_blocks']==30 and report['requests']==180


def test_pipeline_stops_before_performance_on_incorrect_or_unexercised_feature(tmp_path):
    import run_next_stage
    summary=dict(correctness_passed=True,within_method_repeatability=dict(early=True),
        tables=dict(fixed=dict(early=dict(partial_publications=0))))
    common.atomic_json(tmp_path/'summary.json',summary)
    with pytest.raises(ValueError,match='No actual early'):run_next_stage.require_gate(tmp_path)
    summary['tables']['fixed']['early']['partial_publications']=1
    common.atomic_json(tmp_path/'summary.json',summary)
    run_next_stage.require_gate(tmp_path)
    summary['correctness_passed']=False
    common.atomic_json(tmp_path/'summary.json',summary)
    with pytest.raises(ValueError,match='correctness failed'):run_next_stage.require_gate(tmp_path)
