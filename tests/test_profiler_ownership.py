from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
import gpu_process_ownership as own
import run_overnight_extensions as original
import run_overnight_extensions_v2 as corrected


def test_profiler_worker_in_new_group_is_owned_only_with_matching_token(monkeypatch):
    monkeypatch.setattr(own.os,'getpgid',lambda pid:{1:100,2:200,3:300}[pid])
    monkeypatch.setattr(own,'has_token',lambda pid,token:pid==2 and token=='our-nonce')
    rows=[dict(pid=i,name='python') for i in (1,2,3)]
    ours,foreign=own.partition(rows,100,'our-nonce')
    assert [r['pid'] for r in ours]==[1,2]
    assert [r['pid'] for r in foreign]==[3]


def test_token_match_is_exact_and_handles_missing_process(tmp_path):
    folder=tmp_path/'123';folder.mkdir()
    (folder/'environ').write_bytes(b'FOO=bar\0AO_GPU_RUN_TOKEN=ours-plus-extra\0')
    assert not own.has_token(123,'ours',tmp_path)
    (folder/'environ').write_bytes(b'FOO=bar\0AO_GPU_RUN_TOKEN=ours\0')
    assert own.has_token(123,'ours',tmp_path)
    assert not own.has_token(999,'ours',tmp_path)
    assert not own.has_token(123,'',tmp_path)


def test_cleanup_rechecks_token_and_never_signals_foreign_or_reused_pid(monkeypatch):
    monkeypatch.setattr(own,'has_token',lambda pid,token:pid==10)
    signals=[];monkeypatch.setattr(own.os,'kill',lambda pid,sig:signals.append(pid))
    own.cleanup_tagged([dict(pid=10),dict(pid=20),dict(pid=10)],'token')
    assert signals==[10]


def test_importing_new_controller_does_not_mutate_original_runtime():
    assert original.__file__.endswith('run_overnight_extensions.py')
    assert original.run_block is corrected._original_run_block
    assert original.worker_command is corrected._original_worker_command


def test_new_profile_disables_device_event_tracing(monkeypatch,tmp_path):
    monkeypatch.setattr(corrected,'_original_worker_command',lambda *args:['nsys','profile','python'])
    cmd=corrected.worker_command(None,None,dict(profile_capture=True),tmp_path)
    assert cmd==['nsys','profile','--cuda-event-trace=false','python']
    assert corrected.worker_command(None,None,dict(profile_capture=False),tmp_path)==['nsys','profile','python']


def test_reused_calibration_is_rejected_if_an_artifact_changed(monkeypatch,tmp_path):
    artifact=tmp_path/'result.json';artifact.write_text('{}')
    q=dict(reused_depth=dict(artifact_sha256={str(artifact):'wrong'}))
    with pytest.raises(ValueError,match='evidence changed'):
        corrected.block_result(tmp_path,None,dict(phase='depth_tune'),q)
