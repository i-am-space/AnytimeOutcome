"""Append-only queue, tuning leakage, boundary preservation, and valid receipts."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
import baseline_common as common
import run_overnight_extensions as ext


def prompts(n):
    return [dict(id=f'{f}-{i}',family=f,category='fixture',token_ids=[1,2,3])
            for f in ('chat','code','doc') for i in range(n)]


@pytest.fixture
def queue():
    return dict(sha256='queue',hardware=dict(gpus=[dict(uuid='GPU-a'),dict(uuid='GPU-b')]),
        warmup=prompts(1),tuning_prompts=prompts(8),heldout_prompts=prompts(25),
        profile_prompts=prompts(1),correctness_prompts=prompts(4),depth_tuning=dict(objective='tuning only'))


def test_boundary_filler_retains_complete_original_prompt_and_template():
    class Tokenizer:
        def encode(self,text,**kw):return [10,11] if text.startswith('<|im_start|>') else [99]*100
    original=dict(id='x',family='doc',token_ids=[8,10,11,20,21,22,30,31])
    rows=ext.boundary_prompts(Tokenizer(),[original],lengths=(12,16))
    assert [len(r['token_ids']) for r in rows]==[12,16]
    for row in rows:
        n=row['synthetic_filler_tokens']
        assert row['token_ids'][:3]==original['token_ids'][:3]
        assert row['token_ids'][3:3+n]==[99]*n
        assert row['token_ids'][3+n:]==original['token_ids'][3:]
        assert row['original_input_sha256']==common.digest(original['token_ids'])
    with pytest.raises(ValueError,match='user-content'):
        ext.boundary_prompts(Tokenizer(),[dict(original,token_ids=[1,2,3])],lengths=(12,))


@pytest.mark.parametrize('k,blocks,requests',[(4,8,192),(7,10,240),(10,10,240)])
def test_depth_grid_has_deduplicated_k_control_and_two_balanced_repetitions(tmp_path,queue,k,blocks,requests):
    selection=dict(winners=dict(ssd=dict(k=k,f=3)),selection_sha256='baseline')
    d=ext.phase_definition(tmp_path,queue,'depth_tune',selection)
    assert [c['p'] for c in d['configurations']]==[0,*sorted({1,2,4,k})]
    assert len(d['jobs'])==blocks and sum(len(j['prompt_order']) for j in d['jobs'])==requests
    half=len(d['configurations'])
    assert [j['config'] for j in d['jobs'][:half]]==[j['config'] for j in reversed(d['jobs'][half:])]
    assert len({tuple(j['prompt_order']) for j in d['jobs'][:half]})==1


def test_all_four_phase_sizes_and_fixed_selected_p(tmp_path,queue):
    selection=dict(winners=dict(ssd=dict(k=7,f=3)),selection_sha256='baseline')
    depth=dict(winner_p=2,sha256='depth')
    expected=dict(profiles=(3,9),correctness=(20,240),heldout=(20,1500))
    for phase,(blocks,requests) in expected.items():
        folder=tmp_path/phase;folder.mkdir()
        d=ext.phase_definition(folder,queue,phase,selection,depth)
        assert len(d['jobs'])==blocks and sum(len(j['prompt_order']) for j in d['jobs'])==requests
        assert d['depth_selection_sha256']=='depth'
        assert d['profile_capture']==(phase=='profiles')
        assert d['kv_fraction']==(.10 if phase=='correctness' else .70)
        if phase=='heldout':
            assert [c['p'] for c in d['configurations']]==[0,2]
            for r in range(5):
                group=[j for j in d['jobs'] if j['round']==r]
                assert len({tuple(j['prompt_order']) for j in group})==1
                assert group[0]['protocol']==('fixed' if r%2==0 else 'natural')


def test_selection_uses_tuning_only_breaks_ties_by_smaller_p(monkeypatch,tmp_path,queue):
    class NoHeldout(dict):
        def __getitem__(self,key):
            if key=='heldout_prompts':raise AssertionError('Selection accessed held-out inputs')
            return super().__getitem__(key)
    queue=NoHeldout(queue)
    configs=[ext.config('complete','ssd',7,3),ext.config('p1','ssd',7,3,1),ext.config('p4','ssd',7,3,4)]
    d=dict(phase='depth_tune',sha256='tune',configurations=configs)
    reference={('fixed',r,'x'):dict(input_token_ids_sha256='input',output_token_ids=[1]) for r in (0,1)}
    values=[dict(job=dict(config=c,round=r,protocol='fixed'),requests=[dict(id='x',
        input_token_ids_sha256='input',output_token_ids=[1],latency_s=3 if c['p']==0 else 2)])
        for c in configs for r in (0,1)]
    monkeypatch.setattr(ext,'all_results',lambda *args:values)
    calls=[]
    monkeypatch.setattr(ext,'baseline_ar',lambda q,phase:calls.append(phase) or reference)
    winner=ext.select_depth(tmp_path,d,queue)
    assert winner['winner_p']==1 and calls==['depth_tune']
    assert winner['tuning_speed_ratio']==1.5
    with pytest.raises(ValueError,match='tuning evidence only'):
        ext.select_depth(tmp_path,dict(d,phase='heldout'),queue)
    values[0]['requests'][0]['output_token_ids']=[99]
    with pytest.raises(ValueError,match='Tuning exactness'):
        ext.select_depth(tmp_path,d,queue)


def test_contaminated_and_interrupted_receipts_never_count(tmp_path,queue):
    j=dict(id='job');d=dict(sha256='phase')
    for i in range(2):
        a=tmp_path/'job'/f'attempt_{i}';a.mkdir(parents=True)
        common.atomic_json(a/'receipt.json',dict(valid=False,returncode=-15,
            external_processes=[dict(pid=123)] if i==0 else []))
    assert ext.block_result(tmp_path,j,d,queue) is None


def test_process_monitor_stops_own_worker_and_invalidates_contamination(monkeypatch,tmp_path,queue):
    monkeypatch.setattr(ext,'check_runtime',lambda *args:None)
    monkeypatch.setattr(ext,'require_idle',lambda *args:None)
    monkeypatch.setattr(ext,'worker_command',lambda *args:['fake-worker'])
    worker=SimpleNamespace(pid=100,returncode=None,poll=lambda:None)
    monkeypatch.setattr(ext.subprocess,'Popen',lambda *a,**kw:worker)
    monkeypatch.setattr(ext,'compute_processes',lambda *args:[dict(pid=999,uuid='GPU-b')])
    monkeypatch.setattr(ext,'external_processes',lambda rows,group:rows)
    stopped=[]
    def stop(process):stopped.append(process.pid);process.returncode=-15
    monkeypatch.setattr(ext,'stop_worker',stop)
    args=SimpleNamespace(outdir=tmp_path)
    d=dict(phase='heldout',sha256='phase',prompts=prompts(1),profile_capture=False)
    j=dict(id='job',config=ext.config('early','ssd',7,3,2))
    with pytest.raises(ext.PendingWindow,match='Contaminated'):
        ext.run_block(args,d,j,queue)
    assert stopped==[100]
    receipt=common.read_json(tmp_path/'heldout/job/attempt_0/receipt.json')
    assert receipt['valid'] is False and receipt['external_processes'][0]['pid']==999


def test_dependency_waits_and_stops_on_failed_upstream(tmp_path):
    q=dict(dependency=str(tmp_path))
    assert ext.dependency_state(q)=='waiting'
    common.atomic_json(tmp_path/'status.json',dict(status='running'))
    assert ext.dependency_state(q)=='waiting'
    common.atomic_json(tmp_path/'status.json',dict(status='stopped'))
    with pytest.raises(ValueError,match='pipeline stopped'):ext.dependency_state(q)
    common.atomic_json(tmp_path/'status.json',dict(status='complete'))
    assert ext.dependency_state(q)=='ready'


def test_heldout_report_bootstraps_same_rounds_and_prompts_for_both_methods(monkeypatch,tmp_path,queue):
    cfgs=[ext.config('complete','ssd',7,3),ext.config('early','ssd',7,3,2)]
    ps=prompts(2)
    d=dict(phase='heldout',sha256='phase',configurations=cfgs,prompts=ps,
        repetitions=5,protocols=['fixed','natural'])
    values=[];refs={}
    for protocol in d['protocols']:
        for r in range(5):
            for cfg in cfgs:
                rows=[]
                for p in ps:
                    row=dict(id=p['id'],family=p['family'],input_token_ids_sha256='input',
                        output_tokens=256,output_token_ids=[1]*256,latency_s=(r+1)*(2 if cfg['p']==0 else 1),
                        first_delivery_s=.01,cache_hit_rate=.8,verified_draft_tokens=7,
                        accepted_draft_tokens=4,tokens_per_step=5,publication_records=[],
                        draft_wait_s=.1,final_drain_s=.01,target_peak_memory=dict(allocated_bytes=100,reserved_bytes=200),
                        draft_peak_memory=[])
                    rows.append(row);refs[(protocol,r,p['id'])]=row
                values.append(dict(job=dict(config=cfg,round=r,protocol=protocol),requests=rows))
    monkeypatch.setattr(ext,'all_results',lambda *args:values)
    monkeypatch.setattr(ext,'baseline_ar',lambda *args:refs)
    monkeypatch.setattr(ext,'plot_phase',lambda *args:None)
    summary=ext.report_phase(tmp_path,d,queue)
    assert summary['correctness_passed'] and summary['requests']==120
    assert all(v['speedup_ci95']==[2,2] for v in summary['paired_intervals'].values())
    assert all(v['bootstrap_unit'].startswith('paired repetition') for v in summary['paired_intervals'].values())


def test_queue_source_is_separately_pinned_from_engine_execution():
    files=common.execution_fingerprint()['files']
    assert 'profiling/run_overnight_extensions.py' not in files
    assert 'profiling/run_overnight_extensions.py' in ext.code_files()
    assert 'third_party/ssd/ssd/engine/draft_runner.py' in files


def test_profile_capture_is_separate_and_ordinary_workers_are_unprofiled(monkeypatch,tmp_path):
    monkeypatch.setattr(ext.shutil,'disk_usage',lambda path:SimpleNamespace(free=20*1024**3))
    args=SimpleNamespace(outdir=tmp_path)
    job=dict(id='job')
    ordinary=ext.worker_command(args,job,dict(phase='heldout',profile_capture=False),tmp_path)
    profiled=ext.worker_command(args,job,dict(phase='profiles',profile_capture=True),tmp_path)
    assert ordinary[0]==sys.executable and 'nsys' not in ordinary
    assert profiled[:2]==['nsys','profile']
    assert '--capture-range=cudaProfilerApi' in profiled and '--phase' in profiled
    assert profiled[profiled.index('--phase')+1]=='profiles'
    monkeypatch.setattr(ext.shutil,'disk_usage',lambda path:SimpleNamespace(free=1024))
    with pytest.raises(ValueError,match='10 GiB'):
        ext.worker_command(args,job,dict(phase='profiles',profile_capture=True),tmp_path)


def test_enqueue_rejects_software_drift_from_active_baseline(monkeypatch,tmp_path):
    monkeypatch.setattr(ext,'load_campaign',lambda folder:dict(target_numerics='shared_fixed16',software={'version':'old'}))
    monkeypatch.setattr(ext,'software_fingerprint',lambda:{'version':'new'})
    with pytest.raises(ValueError,match='software must match'):
        ext.enqueue(SimpleNamespace(dependency=tmp_path,outdir=tmp_path))
