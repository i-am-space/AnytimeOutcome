"""Persist audit evidence before the inference engine's hard process exit."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
import audit_benchmark_compilation as audit
from baseline_common import read_json


def test_success_is_saved_before_worker_hard_exit(monkeypatch,tmp_path):
    class HardExit(BaseException):pass
    destination=tmp_path/'audit'
    monkeypatch.setattr(sys,'argv',['audit','--benchmark',str(tmp_path),
                                    '--outdir',str(destination),'--method','original_ar'])
    monkeypatch.setattr(audit.benchmark.run_engine,'measure_request',lambda *a,**k:{})
    monkeypatch.setattr(audit.torch._dynamo.utils,'counters',{'stats':{'unique_graphs':1}})
    def worker(args):
        for i in range(3):
            audit.benchmark.run_engine.measure_request(None,dict(id=str(i)),None)
        raise HardExit()
    monkeypatch.setattr(audit.benchmark,'worker',worker)
    with pytest.raises(HardExit):audit.main()
    result=read_json(destination/'audit.json')
    assert result['passed'] and len(result['requests'])==3
    assert all(r['before']==r['after'] for r in result['requests'])
