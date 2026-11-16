"""GPU1 activity is disclosed and never made eligible for dedicated results."""
import sys,time
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
from test_baseline_campaign import manifest,result_for
import baseline_common as common
import run_gpu0_sd_diagnostic as single


@pytest.mark.parametrize('gpu0_contaminated',[False,True])
def test_gpu0_gate_and_dedicated_ineligibility(tmp_path,manifest,monkeypatch,gpu0_contaminated):
    folder=tmp_path/'results/baselines/eval1';folder.mkdir(parents=True)
    common.atomic_json(folder/'fixed_window_request.json',dict(deadline_unix=time.time()+3600))
    monkeypatch.setattr(single,'ROOT',tmp_path)
    monkeypatch.setattr(single,'load_campaign',lambda f:manifest)
    selected=dict(winners={'ar':dict(mode='ar',k=None,f=None),
        'sd':dict(mode='sd',k=6,f=None),'ssd':dict(mode='ssd',k=7,f=3)},selection_sha256='frozen')
    monkeypatch.setattr(single,'validated_selection',lambda f,m:selected)
    def idle(uuids):assert uuids==['GPU-zero']
    monkeypatch.setattr(single,'require_idle',idle)
    monkeypatch.setattr(single,'worker_command',lambda *a:['fake'])
    class Worker:
        pid=987654321;returncode=0;polls=0
        def poll(self):
            self.polls+=1
            return None if self.polls==1 else 0
    def spawn(*args,**kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='GPU-zero'
        job=common.read_json(folder/'gpu0_only_sd_r0/job.json')
        common.atomic_json(folder/'gpu0_only_sd_r0/run.json',result_for(manifest,job))
        return Worker()
    monkeypatch.setattr(single.subprocess,'Popen',spawn)
    rows=[dict(uuid='GPU-one',pid=123,name='unrelated')]
    if gpu0_contaminated:rows.append(dict(uuid='GPU-zero',pid=124,name='unrelated'))
    monkeypatch.setattr(single,'compute_processes',lambda uuids:rows)
    monkeypatch.setattr(single,'external_processes',lambda rows,group:rows)
    killed=[]
    monkeypatch.setattr(single,'stop_worker',lambda worker:killed.append(worker.pid))
    monkeypatch.setattr(single.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(single.signal,'signal',lambda *a:None)
    monkeypatch.setattr(single.signal,'setitimer',lambda *a:None)
    assert single.main()==(1 if gpu0_contaminated else 0)
    receipt=common.read_json(folder/'gpu0_only_sd_r0/receipt.json')
    assert receipt['valid'] is False and receipt['valid_for_dedicated_campaign'] is False
    assert receipt['diagnostic_valid'] is (not gpu0_contaminated)
    assert receipt['monitoring']['gpu1_process_changes'][0]['processes']==[rows[0]]
    assert killed==[Worker.pid]
