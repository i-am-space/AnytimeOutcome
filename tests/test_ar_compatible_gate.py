"""Correctness-gate pairing and dedicated-window monitoring, without GPU work."""
from pathlib import Path
import sys
import subprocess
import os

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
import run_ar_compatible_gate as gate
from baseline_common import atomic_json, digest, read_json, file_digest


@pytest.mark.parametrize('a,b,expected',[
    ([],[],None),([1,2],[1,2],None),([1,2],[1,3],1),
    ([1,2],[1],1),([],[1],0),([1],[1,2],1)])
def test_first_difference_checks_length_as_well_as_content(a,b,expected):
    assert gate.first_difference(a,b)==expected


def test_summary_keeps_original_ar_and_eos_protocols_separate(tmp_path):
    old=dict(id='old',output_token_ids=[1,2,3])
    definition=dict(sha256='version',protocols=['fixed','natural'],legacy_reference=[old])
    for protocol in definition['protocols']:
        tokens=[1,2,3] if protocol=='fixed' else [1,2]
        for mode in ['ar','sd','ssd']:
            row=dict(id='old',input_sha256='input',output_token_ids=tokens)
            atomic_json(tmp_path/f'{protocol}_{mode}.json',dict(requests=[row]))
    summary=gate.summarize(tmp_path,definition)
    assert summary['passed'] and summary['exact_matches']==4
    assert all(r['exact_match'] for r in summary['unchanged_ar_reference'])
    # Making all new modes agree on a changed reference still fails the gate.
    for mode in ['ar','sd','ssd']:
        atomic_json(tmp_path/f'fixed_{mode}.json',dict(requests=[dict(
            id='old',input_sha256='input',output_token_ids=[9,2,3])]))
    assert not gate.summarize(tmp_path,definition)['passed']


def test_summary_rejects_unpaired_inputs(tmp_path):
    definition=dict(sha256='version',protocols=['fixed'],legacy_reference=[])
    for mode in ['ar','sd']:
        atomic_json(tmp_path/f'fixed_{mode}.json',dict(requests=[dict(
            id='p',input_sha256=mode,output_token_ids=[1])]))
    with pytest.raises(ValueError,match='Unpaired'):
        gate.summarize(tmp_path,definition)


def test_contaminated_worker_is_stopped_and_invalidated(monkeypatch,tmp_path):
    class Process:
        pid=12345
        returncode=None
        def poll(self): return self.returncode
    process=Process(); stopped=[]
    monkeypatch.setattr(gate.subprocess,'Popen',lambda *a,**k:process)
    monkeypatch.setattr(gate,'execution_fingerprint',lambda:'code')
    monkeypatch.setattr(gate,'file_digest',lambda p:'supervisor')
    monkeypatch.setattr(gate,'require_idle',lambda uuids:None)
    monkeypatch.setattr(gate,'compute_processes',lambda uuids:[])
    external=[dict(pid=999,uuid='GPU-1',name='unrelated')]
    monkeypatch.setattr(gate,'external_processes',lambda rows,pid:external)
    def stop(p):
        stopped.append(p.pid); p.returncode=-15
    monkeypatch.setattr(gate,'stop_worker',stop)
    definition=dict(execution='code',supervisor_sha256='supervisor',
                    hardware=dict(gpus=[dict(uuid='GPU-0'),dict(uuid='GPU-1')]))
    with pytest.raises(RuntimeError,match='Invalid worker'):
        gate.monitor_worker(tmp_path,definition,'sd','fixed',30)
    receipt=read_json(tmp_path/'fixed_sd.receipt.json')
    assert not receipt['valid'] and receipt['contamination']==external
    assert stopped==[12345]  # Never act on the unrelated PID.


def test_shared_admission_checks_memory_without_terminating_jobs(monkeypatch):
    monkeypatch.setattr(gate.subprocess,'check_output',lambda *a,**k:'GPU-0, 30000\nGPU-1, 12000\n')
    assert gate.require_memory(['GPU-0','GPU-1'],'ssd')['GPU-1']==12000
    monkeypatch.setattr(gate.subprocess,'check_output',lambda *a,**k:'GPU-0, 30000\nGPU-1, 2000\n')
    with pytest.raises(gate.PendingWindow,match='Insufficient'):
        gate.require_memory(['GPU-0','GPU-1'],'ssd')


def test_reuse_rejects_changed_execution_and_detects_modified_outputs(tmp_path):
    source=tmp_path/'old';source.mkdir()
    prompt=dict(id='p',token_ids=[1,2])
    config=dict(mode='ar',numerics='batched',k=6,f=3)
    definition=dict(execution='code1',software='software',models='weights',hardware='hardware',
        prompts=[prompt],warmup=[],output_cap=256,protocols=['fixed'],prefix_cache=False,
        kv_fractions=.1,capacity=4096,configurations=[config])
    old=dict(definition);old['sha256']=digest(old)
    atomic_json(source/'definition.json',old)
    atomic_json(source/'fixed_ar.json',dict(definition_sha256=old['sha256'],config=config,
        protocol='fixed',requests=[dict(id='p',input_sha256=digest([1,2]),output_token_ids=[3])]))
    atomic_json(source/'fixed_ar.receipt.json',dict(valid=True))
    refs=gate.import_references(source,definition)
    new=dict(definition,reused_results=refs)
    assert gate.load_result(tmp_path,new,'fixed','ar')['requests'][0]['output_token_ids']==[3]
    with pytest.raises(ValueError,match='execution differs'):
        gate.import_references(source,dict(definition,execution='code2'))
    atomic_json(source/'fixed_ar.json',dict(requests=[]))
    with pytest.raises(ValueError,match='changed after freezing'):
        gate.load_result(tmp_path,new,'fixed','ar')


def test_optimized_spawn_import_installs_torch_workaround():
    script=str(Path(gate.__file__).resolve())
    code=('import runpy; runpy.run_path('+repr(script)+',run_name="__mp_main__"); '
          'import torch._dynamo.symbolic_convert as sc; '
          'result=sc._detect_and_normalize_assert_statement(None,None,None); '
          'print("spawn_workaround="+str(result))')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    env['PYTHONPATH']=str(Path(script).parent)+os.pathsep+env.get('PYTHONPATH','')
    result=subprocess.run([sys.executable,'-O','-c',code],env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert 'spawn_workaround=False' in result.stdout


def test_candidate_code_changes_cannot_reuse_candidate_but_can_reuse_ar(tmp_path):
    source=tmp_path/'source';source.mkdir()
    ar=dict(mode='ar',numerics='batched',k=6,f=3)
    sd=dict(mode='sd',numerics='hybrid_bmm',k=6,f=3)
    base=dict(execution='same',software='same',models='same',hardware='same',
        prompts=[dict(id='p',token_ids=[1])],warmup=[],output_cap=256,
        protocols=['fixed'],prefix_cache=False,kv_fractions=.1,capacity=4096,
        configurations=[ar,sd],candidate_code=dict(path='candidate.py',sha256='old'))
    frozen=dict(base,sha256=digest(base))
    atomic_json(source/'definition.json',frozen)
    for config in [ar,sd]:
        atomic_json(source/f'fixed_{config["mode"]}.json',dict(
            definition_sha256=frozen['sha256'],config=config,protocol='fixed',
            requests=[dict(id='p',input_sha256=digest([1]),output_token_ids=[2])]))
        atomic_json(source/f'fixed_{config["mode"]}.receipt.json',dict(valid=True))
    changed=dict(base,candidate_code=dict(path='candidate.py',sha256='new'))
    with pytest.raises(ValueError,match='candidate code differs'):
        gate.import_references(source,changed)
    ar_only=dict(changed,configurations=[ar])
    assert list(gate.import_references(source,ar_only))==['fixed_ar']


def test_shared_reference_reports_original_ar_drift_separately(tmp_path):
    definition=dict(sha256='v1',protocols=['fixed','natural'],
        reference_contract='shared-fixed16-v1',legacy_reference=[dict(id='p',output_token_ids=[1,2,3])],
        original_ar_controls={})
    for protocol in definition['protocols']:
        original=tmp_path/f'original_{protocol}.json';receipt=tmp_path/f'original_{protocol}.receipt.json'
        atomic_json(original,dict(requests=[dict(id='p',input_sha256='same',output_token_ids=[9,2,3])]))
        atomic_json(receipt,dict(valid=True))
        definition['original_ar_controls'][protocol]=dict(path=str(original),sha256=file_digest(original),
            receipt_path=str(receipt),receipt_sha256=file_digest(receipt))
        for mode in ['ar','sd','ssd']:
            atomic_json(tmp_path/f'{protocol}_{mode}.json',dict(requests=[dict(
                id='p',input_sha256='same',output_token_ids=[8,8,8])]))
    summary=gate.summarize(tmp_path,definition)
    assert summary['paired_correctness_passed'] and summary['exact_matches']==4
    assert not summary['historical_reference_passed'] and not summary['passed']
    assert len(summary['original_ar_comparisons'])==2
    assert all(not r['exact_match'] for r in summary['original_ar_comparisons'])
    atomic_json(original,dict(requests=[]))
    with pytest.raises(ValueError,match='controls changed'):
        gate.summarize(tmp_path,definition)
