from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
import run_greedy_handoff_campaign as c


def prompt(i):
    return dict(id=str(i),family=('chat','code','doc')[i%3],token_ids=[i+1],token_ids_sha256=c.digest([i+1]))


def queue():
    return dict(sha256='queue',gate_prompts=[prompt(i) for i in range(6)],
        diagnostic_prompts=[prompt(i) for i in range(3)],splits=dict(tune=[prompt(i) for i in range(24)],
        heldout=[prompt(i+100) for i in range(75)],warmup=[prompt(i+200) for i in range(3)]))


def test_original_engine_is_separate_and_new_transport_is_named_explicitly():
    values=c.configurations();assert len(values)==14
    for x in values:
        if x['method'].startswith('legacy'):
            assert x['engine_root']==str(c.SOURCE_ROOT) and not x['greedy']
        else:assert x['engine_root']==str(c.ROOT) and x['greedy']
    assert c.config('ids_complete')['p']==0
    assert c.config('ids_bypass')['service']=='threaded' and c.config('ids_bypass')['bypass']
    assert not c.config('ids_threaded')['bypass']


def test_workload_counts_pairing_and_unprofiled_performance():
    q=queue();total=0
    for phase,jobs,requests in [('micro',1,0),('gate',30,180),('stress',4,12),('diagnostic',6,18),('tune',28,672)]:
        d=c.definition(q,phase,c.phase_configs(phase))
        assert len(d['jobs'])==jobs and len(d['prompts'])*jobs==requests
        assert d==c.definition(q,phase,c.phase_configs(phase))
        total+=requests
    winners={m:c.config(m) for m in c.METHODS}
    d=c.definition(q,'heldout',list(winners.values()),'selection')
    assert len(d['jobs'])==60 and len(d['prompts'])*60==4500
    assert not d['diagnostic'] and not d['profiler']
    for r in range(5):
        jobs=[j for j in d['jobs'] if j['round']==r]
        assert len({tuple(j['prompt_order']) for j in jobs})==1
        assert {j['config']['method'] for j in jobs}==set(c.METHODS)
    assert len({tuple(j['config']['method'] for j in d['jobs'] if j['round']==r and j['protocol']=='fixed') for r in range(5)})==5


def test_selection_never_reads_heldout(monkeypatch,tmp_path):
    q=queue();configs=c.configurations()
    d=c.definition(q,'tune',configs)
    def load(folder,q,phase):assert phase=='tune';return d
    monkeypatch.setattr(c,'load_definition',load)
    monkeypatch.setattr(c,'checks',lambda *a:[dict(exact=True)])
    def available(folder,d,q,complete):
        assert folder.name=='tune'
        return [dict(job=dict(config=x),requests=[dict(latency_s=x['k']) for _ in range(48)]) for x in configs]
    monkeypatch.setattr(c,'available',available)
    chosen=c.select(tmp_path,q)
    assert chosen['winners']['legacy_complete']['k']==4
    assert chosen['winners']['ids_bypass']['k']==7


def test_code_changes_reject_resume_before_workers(monkeypatch):
    q=dict(spec=c.SPEC,code={'old':'hash'});q['sha256']=c.digest(q)
    monkeypatch.setattr(c,'code_files',lambda:{'new':'hash'})
    with pytest.raises(ValueError,match='Execution code changed'):c.check_runtime(q)


def test_incomplete_or_contaminated_attempt_does_not_complete_a_block(tmp_path):
    q=queue();d=c.definition(q,'micro',c.phase_configs('micro'));j=d['jobs'][0]
    attempt=tmp_path/j['id']/'attempt_0';attempt.mkdir(parents=True)
    c.atomic_json(attempt/'receipt.json',dict(valid=False,returncode=-15,external_processes=[dict(pid=123)]))
    assert c.block_result(tmp_path,j,d,q) is None
    with pytest.raises(ValueError,match='missing'):c.available(tmp_path,d,q,True)


def test_completed_status_without_predecessor_receipts_cannot_release_queue(tmp_path):
    c.atomic_json(tmp_path/'status.json',dict(status='complete'))
    c.atomic_json(tmp_path/'summary.json',dict(status='complete',correctness_passed=True))
    (tmp_path/'report.md').write_text('report')
    c.atomic_json(tmp_path/'gate/definition.json',dict(jobs=[dict(id='missing')]))
    with pytest.raises(ValueError,match='blocks missing'):
        c.predecessor_ready(dict(after=dict(path=str(tmp_path))))


def test_diagnostic_stress_uses_its_small_fresh_ar_in_both_protocols(monkeypatch):
    phases=[]
    monkeypatch.setattr(c.saved_ar,'check_outputs',lambda b,q,p:phases.append(p) or [])
    c.checks([],{},'stress');c.checks([],{},'diagnostic')
    assert phases==['gate','tune']
    d=c.definition(queue(),'stress',c.phase_configs('stress'))
    assert len([j for j in d['jobs'] if j['config']['mode']=='ar'])==2
