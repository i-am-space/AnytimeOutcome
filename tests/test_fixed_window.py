"""The reduced window retains all fixed evidence and never invents natural runs."""
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
from test_baseline_campaign import manifest, tuned, save_block
import baseline_common as common
import run_fixed_window as window
import run_baselines as campaign


def test_fixed_scope_keeps_original_jobs_and_all_five_rounds(tuned,manifest):
    folder,selection=tuned
    original=common.heldout_jobs(manifest,selection)
    jobs=window.fixed_jobs(manifest,selection)
    assert jobs==[j for j in original if j['protocol']=='fixed']
    assert len(jobs)==15 and sum(len(j['prompt_ids']) for j in jobs)==1125
    assert {j['repeat'] for j in jobs}==set(range(5))


def test_report_requires_every_fixed_block_but_no_natural_blocks(tuned,manifest):
    folder,selection=tuned
    jobs=window.fixed_jobs(manifest,selection)
    for j in jobs[:-1]:save_block(folder,manifest,j)
    with pytest.raises(ValueError,match='Missing completed'):
        window.write_report(folder,manifest,selection)
    assert not (folder/'fixed_window_report').exists()
    save_block(folder,manifest,jobs[-1])
    report=window.write_report(folder,manifest,selection)
    data=common.read_json(report/'summary.json')
    assert data['fixed_blocks']==15 and data['fixed_requests']==1125
    assert data['status']=='fixed_length_complete_natural_eos_deferred'
    assert len(data['results'])==12
    assert not (folder/'report.md').exists()
    assert not (folder/'raw/heldout/natural').exists()
    assert window.progress(folder,manifest,selection)['valid_fixed_blocks']==15


def test_budget_interrupt_cleans_only_our_worker_and_does_not_count(tmp_path,manifest,monkeypatch):
    job=common.tuning_jobs(manifest)[0]
    killed=[]
    class Worker:
        pid=987654321
        returncode=None
        def poll(self):return None
        def wait(self,timeout=None):self.returncode=-15;return -15
    monkeypatch.setattr(campaign,'require_idle',lambda uuids:None)
    monkeypatch.setattr(campaign.subprocess,'Popen',lambda *args,**kw:Worker())
    def alarm(uuids):raise window.BudgetExpired('deadline')
    monkeypatch.setattr(campaign,'compute_processes',alarm)
    monkeypatch.setattr(campaign.os,'killpg',lambda pid,sig:killed.append(pid))
    with pytest.raises(window.BudgetExpired):campaign.run_block(tmp_path,manifest,job,True)
    assert killed==[Worker.pid]
    with pytest.raises(ValueError,match='Missing completed'):
        campaign.accepted_result(tmp_path,job,manifest)
