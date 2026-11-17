"""Dedicated-window admission, pairing and resumable prototype statistics."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
import benchmark_shared_numerics as bench
from baseline_common import atomic_json,digest,file_digest,read_json


def fixture_definition():
    return dict(sha256='frozen',jobs=bench.schedule(),output_cap=2,
        prompts=[dict(id=str(i),family=f,token_ids=[i]) for i,f in enumerate(['chat','code','doc'])],
        reference_contract='shared-fixed16-v1',purpose='prototype')


def save_block(folder,definition,job,valid=True):
    path=folder/f'r{job["round"]}_{job["method"]}'/'attempt_0';path.mkdir(parents=True)
    latency={'original_ar':2.,'shared_ar':2.2,'shared_sd':1.,'shared_ssd':.8}[job['method']]
    latency*=job['round']+1  # Large round effects cancel only with paired resampling.
    tokens=[8,9] if job['method']=='original_ar' else [1,2]
    result=dict(definition_sha256=definition['sha256'],job=job,requests=[dict(
        id=definition['prompts'][i]['id'],input_token_ids_sha256=digest([i]),output_tokens=2,
        output_token_ids=tokens,latency_s=latency,first_delivery_s=.1) for i in job['prompt_order']])
    atomic_json(path/'result.json',result)
    atomic_json(path/'receipt.json',dict(valid=valid,output_sha256=file_digest(path/'result.json')))
    return path,result


def test_schedule_pairs_prompts_and_uses_five_distinct_method_orders():
    jobs=bench.schedule()
    assert jobs==bench.schedule() and len(jobs)==20
    orders=[]
    for r in range(5):
        rows=[j for j in jobs if j['round']==r]
        assert len({tuple(j['prompt_order']) for j in rows})==1
        assert set(j['method'] for j in rows)==set(bench.METHODS)
        orders.append(tuple(j['method'] for j in rows))
    assert len(set(orders))==5


def test_resume_rejects_changed_execution_fingerprint(monkeypatch):
    original=dict(execution='code1',software='env',prototype_files={'candidate.py':'hash'})
    monkeypatch.setattr(bench,'fingerprints',lambda:original)
    definition=dict(original);definition['sha256']=digest(definition)
    bench.validate_definition(definition)
    monkeypatch.setattr(bench,'fingerprints',lambda:dict(original,execution='code2'))
    with pytest.raises(ValueError,match='execution changed'):
        bench.validate_definition(definition)


def test_invalid_block_cannot_count_and_completed_output_is_immutable(tmp_path):
    definition=fixture_definition();job=definition['jobs'][0]
    path,result=save_block(tmp_path,definition,job,valid=False)
    assert bench.block_result(tmp_path,job,definition) is None
    atomic_json(path/'receipt.json',dict(valid=True,output_sha256=file_digest(path/'result.json')))
    assert bench.block_result(tmp_path,job,definition)==result
    result['requests'][0]['input_token_ids_sha256']='wrong'
    atomic_json(path/'result.json',result)
    with pytest.raises(ValueError,match='output changed'):
        bench.block_result(tmp_path,job,definition)
    with pytest.raises(ValueError,match='inputs'):
        bench.validate_result(result,job,definition)


def test_external_job_invalidates_only_own_worker_and_stops_scheduling(monkeypatch,tmp_path):
    class Process:
        pid=12345
        returncode=None
        def poll(self):return self.returncode
    process=Process();stopped=[]
    monkeypatch.setattr(bench,'validate_definition',lambda d:None)
    monkeypatch.setattr(bench,'require_idle',lambda u:None)
    monkeypatch.setattr(bench.subprocess,'Popen',lambda *a,**k:process)
    monkeypatch.setattr(bench,'compute_processes',lambda u:[])
    unrelated=[dict(pid=999,uuid='gpu1')]
    monkeypatch.setattr(bench,'external_processes',lambda rows,pid:unrelated)
    def stop(p):stopped.append(p.pid);p.returncode=-15
    monkeypatch.setattr(bench,'stop_worker',stop)
    definition=fixture_definition();definition['hardware']=dict(gpus=[dict(uuid='gpu0'),dict(uuid='gpu1')])
    args=type('Args',(),dict(outdir=tmp_path))()
    with pytest.raises(bench.PendingWindow,match='Another GPU job'):
        bench.run_block(args,definition,definition['jobs'][0])
    assert stopped==[12345]
    receipt=next(tmp_path.rglob('receipt.json'))
    assert not read_json(receipt)['valid'] and read_json(receipt)['external_processes']==unrelated


def test_report_preserves_round_pairing_and_original_ar_is_a_separate_control(tmp_path):
    definition=fixture_definition()
    for job in definition['jobs']:save_block(tmp_path,definition,job)
    bench.report(tmp_path,definition)
    summary=read_json(tmp_path/'summary.json')
    assert summary['valid_blocks']==20 and summary['paired_exact']==30
    assert summary['correctness_passed']
    assert all(summary['within_method_repeatability'].values())
    ci=summary['paired_speed_ratio_intervals']['shared_ssd_vs_shared_ar']
    assert ci['unit']=='round' and ci['samples']==10000
    assert ci['low']==pytest.approx(2.75) and ci['high']==pytest.approx(2.75)
    assert all(v['requests']==15 and v['rounds']==5 for v in summary['methods'].values())
    # A changed token in one repeated run must fail paired correctness/repeatability.
    job=next(j for j in definition['jobs'] if j['method']=='shared_sd')
    path=tmp_path/f'r{job["round"]}_{job["method"]}'/'attempt_0'
    result=read_json(path/'result.json');result['requests'][0]['output_token_ids']=[5,6]
    atomic_json(path/'result.json',result)
    atomic_json(path/'receipt.json',dict(valid=True,output_sha256=file_digest(path/'result.json')))
    bench.report(tmp_path,definition)
    summary=read_json(tmp_path/'summary.json')
    assert not summary['correctness_passed'] and summary['paired_exact']==29
    assert not summary['within_method_repeatability']['shared_sd']
