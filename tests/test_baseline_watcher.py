"""Dedicated-window retry and safe shutdown without real GPU jobs."""
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
import watch_baselines as watcher
import run_baselines as campaign


def test_wait_then_restart_only_contaminated_attempts(tmp_path,monkeypatch):
    gates=iter([False,True,True])
    attempts=iter([2,0])
    calls=[]
    def idle(uuids):
        assert uuids==['zero','one']
        if not next(gates):
            raise campaign.PendingWindow('Other jobs are running')
    monkeypatch.setattr(watcher,'require_idle',idle)
    monkeypatch.setattr(watcher,'run_attempt',lambda folder:calls.append(folder) or next(attempts))
    sleeps=[]
    monkeypatch.setattr(watcher.time,'sleep',sleeps.append)
    assert watcher.watch(tmp_path,['zero','one'],30)==0
    assert calls==[tmp_path,tmp_path] and sleeps==[30,30]
    assert 'complete' in (tmp_path/'watcher_status.json').read_text()


def test_errors_stop_instead_of_retrying(tmp_path,monkeypatch):
    monkeypatch.setattr(watcher,'require_idle',lambda uuids:None)
    monkeypatch.setattr(watcher,'run_attempt',lambda folder:1)
    monkeypatch.setattr(watcher.time,'sleep',lambda n:pytest.fail('Must not retry an execution error'))
    assert watcher.watch(tmp_path,['zero','one'])==1
    assert 'stopped' in (tmp_path/'watcher_status.json').read_text()


@pytest.mark.parametrize('busy',[False,True])
def test_check_does_not_start_jobs(tmp_path,monkeypatch,busy):
    def idle(uuids):
        if busy:
            raise campaign.PendingWindow('busy')
    monkeypatch.setattr(watcher,'require_idle',idle)
    monkeypatch.setattr(watcher,'run_attempt',lambda folder:pytest.fail('Check must never run campaign'))
    assert watcher.watch(tmp_path,['zero','one'],check=True)==(2 if busy else 0)
    assert not (tmp_path/'watcher_status.json').exists()


def test_stuck_worker_escalates_only_its_own_group(monkeypatch):
    calls=[]
    waits=iter([subprocess.TimeoutExpired('worker',15),None])
    def wait(timeout=None):
        event=next(waits)
        if event:
            raise event
    monkeypatch.setattr(campaign.os,'killpg',lambda pid,sig:calls.append((pid,sig)))
    campaign.stop_worker(SimpleNamespace(pid=987654321,wait=wait))
    assert calls==[(987654321,signal.SIGTERM),(987654321,signal.SIGKILL)]


def test_ctrl_c_reaches_campaign_cleanup(tmp_path,monkeypatch):
    events=[]
    waits=iter([KeyboardInterrupt(),0])
    class Child:
        def wait(self,timeout=None):
            result=next(waits)
            if isinstance(result,BaseException):
                raise result
            return result
        def send_signal(self,sig):
            events.append(sig)
    def start(command,**kwargs):
        assert '--resume' in command and '--phase' in command
        assert kwargs['start_new_session'] is True
        return Child()
    monkeypatch.setattr(watcher.subprocess,'Popen',start)
    with pytest.raises(KeyboardInterrupt):
        watcher.run_attempt(tmp_path)
    assert events==[signal.SIGINT]
