from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
import run_gpu0_ar_control as control


def test_contextual_receipt_cannot_masquerade_as_dedicated_campaign(tmp_path):
    output=tmp_path/'result.json';output.write_text('{}')
    d=dict(sha256='definition')
    r=dict(valid=True,definition_sha256='definition',returncode=0,samples=4,external_processes=[],
        error=None,dedicated_gpus=False,allocated_gpu_dedicated=True,contextual_only=True,
        headline_performance_evidence=False,monitor_scope='physical GPU0 only',profiler=False,
        output_sha256=control.file_digest(output),nonallocated_gpu_processes=[dict(uuid='gpu1',pid=456)])
    control.validate_receipt(r,d,output)
    with pytest.raises(ValueError):control.ORIGINAL_VALIDATE_RECEIPT(r,d,output)
    assert r['dedicated_gpus'] is False
    for field,value in [('valid',False),('dedicated_gpus',True),('allocated_gpu_dedicated',False),
                        ('contextual_only',False),('headline_performance_evidence',True),
                        ('external_processes',[dict(uuid='gpu0',pid=123)]),('returncode',-15)]:
        with pytest.raises(ValueError):control.validate_receipt(dict(r,**{field:value}),d,output)


def test_monitor_records_gpu1_activity_without_invalidating_gpu0(monkeypatch,tmp_path):
    q=dict(hardware=dict(gpus=[dict(uuid='gpu0'),dict(uuid='gpu1')]))
    j=dict(id='ar');d=dict(phase='tune')
    from types import SimpleNamespace
    args=SimpleNamespace(outdir=tmp_path)
    rows=[dict(uuid='gpu0',pid=10),dict(uuid='gpu1',pid=20)]
    monkeypatch.setattr(control,'compute_processes',lambda uuids:rows)
    idle=[];monkeypatch.setattr(control,'require_idle',lambda uuids:idle.append(uuids))
    def run(args,d,j,q):
        control.campaign.require_idle(['gpu0','gpu1'])
        assert control.campaign.compute_processes(['gpu0','gpu1'])==rows[:1]
        path=tmp_path/'tune/ar/attempt_0/receipt.json'
        control.atomic_json(path,dict(valid=True,dedicated_gpus=True))
    monkeypatch.setattr(control,'ORIGINAL_RUN_BLOCK',run)
    # Track process-local overrides so they cannot leak into other test modules.
    for name in ('check_runtime','require_idle','compute_processes','__file__'):
        monkeypatch.setattr(control.campaign,name,getattr(control.campaign,name))
    control.run_block(args,d,j,q)
    assert idle==[['gpu0']]
    r=control.read_json(tmp_path/'tune/ar/attempt_0/receipt.json')
    assert r['valid'] and not r['dedicated_gpus'] and r['contextual_only']
    assert r['nonallocated_gpu_processes']==rows[1:]
