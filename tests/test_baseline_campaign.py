"""Protocol tests use synthetic evidence; no benchmark timings are fabricated."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/ssd'))
import baseline_common as common
import run_baselines as campaign
from analyze_baselines import output_checks, paired_bootstrap, write_report


def fake_loader(families,n,split,seed,tokenizer):
    return [dict(id=f'{families[0]}-{split}-{i}',family=families[0],category='synthetic',
                 text='test fixture',token_ids=[i+1]*20) for i in range(40)]


@pytest.fixture
def manifest():
    return common.make_manifest(SimpleNamespace(chat_template='test template'),
        dict(target=dict(path='/target'),draft=dict(path='/draft')),
        dict(sha256='execution',files={}),dict(python='test',packages={}),
        dict(gpus=[dict(index=0,uuid='GPU-zero',name='L40S',driver='test'),
                   dict(index=1,uuid='GPU-one',name='L40S',driver='test')],topology='SYS'),fake_loader)


def result_for(manifest,job,latency=None):
    cfg=job['config']
    if latency is None:
        latency=5 if cfg['mode']=='ar' else 3 if cfg['mode']=='sd' and cfg['k'] in (4,6) else (
            2 if cfg['mode']=='ssd' and cfg['k']==7 else 8)
    source={r['id']:r for r in manifest['splits']['tune' if job['phase']=='tune' else 'heldout']}
    rows=[dict(id=rid,family=source[rid]['family'],category='synthetic',
        prompt_tokens=len(source[rid]['token_ids']),input_token_ids_sha256=source[rid]['token_ids_sha256'],
        output_tokens=256,output_token_ids=[1]*256,latency_s=latency,first_delivery_s=0.5,
        prefill_s=0.1,decode_s=latency-0.1,num_steps=100,mean_step_ms=10,
        tokens_per_step=1.0,accepted_draft_tokens=0,verified_draft_tokens=0,
        cache_hit_rate=0.5 if cfg['mode']=='ssd' else None) for rid in job['prompt_ids']]
    return dict(job=job,manifest_sha256=manifest['manifest_sha256'],requests=rows,
        args=dict(mode=cfg['mode'],k=cfg['k'],f=cfg['f'],profile_capture=False,eager=False,
            disable_prefix_cache=True,gpu_mem_util=0.7,draft_gpu_mem_util=0.7,backup='jit',
            output_len=256,max_model_len=4096,max_prompt_tokens=2048,warmup=3,order_seed=job['order_seed'],
            ignore_eos=job['protocol']=='fixed'),
        hardware_allocation=common.allocation(manifest,cfg['mode']),hardware=manifest['hardware'],actual_dtype='torch.bfloat16',
        prefix_cache_enabled=False,software=manifest['software'],
        cuda_visible_devices='GPU-zero,GPU-one' if cfg['mode']=='ssd' else 'GPU-zero')


def save_block(folder,manifest,job):
    output,receipt,_,_=campaign.paths(folder,job)
    common.atomic_json(output,result_for(manifest,job))
    common.atomic_json(receipt,dict(valid=True,job_sha256=job['job_sha256'],
        execution_sha256=manifest['execution']['sha256'],output_sha256=common.file_digest(output),
        returncode=0,monitoring=dict(samples=1,external_processes=[],error=None)))


@pytest.fixture
def tuned(tmp_path,manifest):
    for job in common.tuning_jobs(manifest):
        save_block(tmp_path,manifest,job)
    selection=campaign.select_winners(tmp_path,manifest)
    common.freeze_json(tmp_path/'selection.json',selection)
    return tmp_path,selection


def test_frozen_disjoint_complete_manifest(manifest):
    common.validate_manifest(manifest)
    assert [len(manifest['splits'][s]) for s in ('tune','heldout','warmup')]==[24,75,3]
    assert len({r['id'] for rows in manifest['splits'].values() for r in rows})==102
    second=common.make_manifest(SimpleNamespace(chat_template='test template'),manifest['models'],
        manifest['execution'],manifest['software'],manifest['hardware'],fake_loader)
    assert second==manifest
    manifest['splits']['heldout'][0]['token_ids'][0]=999
    with pytest.raises(ValueError,match='hash'):
        common.validate_manifest(manifest)


def test_manifest_rejects_truncation_and_overlap(manifest):
    def too_long(*args,**kwargs):
        rows=fake_loader(*args,**kwargs)
        rows[0]['token_ids']=[1]*2049
        return rows
    with pytest.raises(ValueError,match='Complete prompt exceeds'):
        common.make_manifest(SimpleNamespace(chat_template='test'),manifest['models'],manifest['execution'],
                             manifest['software'],manifest['hardware'],too_long)
    body={k:v for k,v in manifest.items() if k!='manifest_sha256'}
    body['splits']['warmup'][0]=dict(body['splits']['tune'][0])
    with pytest.raises(ValueError,match='overlapping'):
        common.validate_manifest(common.seal_manifest(body))


def test_schedule_sizes_balance_and_pairing(manifest):
    tune=common.tuning_jobs(manifest)
    assert len(tune)==20 and sum(len(j['prompt_ids']) for j in tune)==480
    assert [j['config'] for j in tune[:10]]==[j['config'] for j in reversed(tune[10:])]
    selected=dict(winners={mode:next(c for c in common.configurations() if c['mode']==mode)
                           for mode in ('ar','sd','ssd')},selection_sha256='frozen')
    jobs=common.heldout_jobs(manifest,selected)
    assert len(jobs)==30 and sum(len(j['prompt_ids']) for j in jobs)==2250
    for repeat in range(5):
        block=[j for j in jobs if j['repeat']==repeat]
        assert all(j['prompt_ids']==block[0]['prompt_ids'] for j in block)
        assert block[0]['protocol']==('fixed' if repeat%2==0 else 'natural')
    orders=[[j['config']['mode'] for j in jobs if j['repeat']==r and j['protocol']=='fixed'] for r in range(5)]
    assert len({tuple(x) for x in orders})==5
    for position in range(3):
        assert all(sum(order[position]==mode for order in orders) in (1,2) for mode in ('ar','sd','ssd'))


def test_selection_is_tuning_only_and_breaks_ties(tuned,manifest):
    folder,selection=tuned
    assert selection['winners']['sd']==dict(mode='sd',k=4,f=None)
    assert selection['winners']['ssd']==dict(mode='ssd',k=7,f=1)
    (folder/'raw/heldout').mkdir(parents=True)
    (folder/'raw/heldout/unreadable.json').write_text('not JSON; must never be read by selection')
    assert campaign.select_winners(folder,manifest)==selection
    first=common.tuning_jobs(manifest)[0]
    campaign.paths(folder,first)[1].unlink()
    with pytest.raises(ValueError,match='Missing completed'):
        campaign.select_winners(folder,manifest)


@pytest.mark.parametrize('mutation',['contamination','returncode','partial','altered_output','wrong_setting'])
def test_invalid_evidence_never_counts(tmp_path,manifest,mutation):
    job=common.tuning_jobs(manifest)[0]
    save_block(tmp_path,manifest,job)
    output,receipt,_,_=campaign.paths(tmp_path,job)
    r=common.read_json(receipt)
    if mutation=='contamination':
        r['monitoring']['external_processes']=[dict(pid=123)]
    elif mutation=='returncode':
        r['returncode']=1
    elif mutation=='partial':
        receipt.unlink()
    elif mutation=='altered_output':
        output.write_text('{}')
    elif mutation=='wrong_setting':
        data=common.read_json(output)
        data['args']['profile_capture']=True
        common.atomic_json(output,data)
        r['output_sha256']=common.file_digest(output)
    if mutation!='partial':
        common.atomic_json(receipt,r)
    with pytest.raises(ValueError):
        campaign.accepted_result(tmp_path,job,manifest)


def test_busy_gpu_gate_does_not_start_workers(monkeypatch):
    monkeypatch.setattr(campaign,'compute_processes',lambda uuids:[dict(pid=123,name='unrelated')])
    monkeypatch.setattr(campaign,'external_processes',lambda rows:rows)
    with pytest.raises(campaign.PendingWindow,match='unrelated GPU'):
        campaign.require_idle(['GPU-zero','GPU-one'])


def test_midblock_contention_invalidates_only_our_group(tmp_path,manifest,monkeypatch):
    job=common.tuning_jobs(manifest)[0]
    output,receipt,_,_=campaign.paths(tmp_path,job)
    class Worker:
        pid=987654321
        returncode=0
        polls=0
        def poll(self):
            self.polls+=1
            return None  # A contaminated worker must be stopped before it finishes.
        def wait(self,timeout=None):
            return 0
    killed=[]
    def start(command,**kwargs):
        assert command[1]=='-O' and '--profile-capture' not in command
        common.atomic_json(output,result_for(manifest,job))
        return Worker()
    monkeypatch.setattr(campaign,'require_idle',lambda uuids:None)
    monkeypatch.setattr(campaign.subprocess,'Popen',start)
    monkeypatch.setattr(campaign,'compute_processes',lambda uuids:[dict(pid=123,name='unrelated')])
    monkeypatch.setattr(campaign,'external_processes',lambda rows,group=None:rows)
    monkeypatch.setattr(campaign.time,'sleep',lambda n:pytest.fail('Must stop immediately after contamination'))
    monkeypatch.setattr(campaign.os,'killpg',lambda group,sig:killed.append(group))
    with pytest.raises(campaign.PendingWindow,match='invalidated'):
        campaign.run_block(tmp_path,manifest,job,False)
    assert killed==[Worker.pid]
    assert common.read_json(receipt)['valid'] is False
    with pytest.raises(ValueError):
        campaign.accepted_result(tmp_path,job,manifest)


def test_resume_checks_evidence_before_skipping(tmp_path,manifest,monkeypatch):
    job=common.tuning_jobs(manifest)[0]
    save_block(tmp_path,manifest,job)
    monkeypatch.setattr(campaign,'require_idle',lambda uuids:pytest.fail('Completed block must be skipped'))
    campaign.run_block(tmp_path,manifest,job,True)
    with pytest.raises(ValueError,match='use --resume'):
        campaign.run_block(tmp_path,manifest,job,False)


def test_changed_model_and_execution_reject_resume(tmp_path,manifest,monkeypatch):
    model=tmp_path/'model'
    model.mkdir()
    (model/'weights.bin').write_bytes(b'abc')
    frozen=common.model_fingerprint(model)
    (model/'weights.bin').write_bytes(b'abd')
    with pytest.raises(ValueError,match='changed'):
        common.validate_models(dict(target=frozen))
    common.atomic_json(tmp_path/'manifest.json',manifest)
    monkeypatch.setattr(campaign,'execution_fingerprint',lambda:dict(sha256='different'))
    with pytest.raises(ValueError,match='Execution code'):
        campaign.load_campaign(tmp_path)


def test_paired_bootstrap_preserves_pairing_and_family_mix():
    a=np.array([[1,1,100,100],[2,2,200,200]],dtype=float)
    result=paired_bootstrap(a,a/2,['chat','chat','doc','doc'])
    assert result['speedup']==2 and result['speedup_ci95']==[2,2]
    assert result['mean_latency_ci95_s'][0]<result['mean_latency_ci95_s'][1]
    # With one round and constant values per family, stratification fixes the mix.
    r=paired_bootstrap([[1,1,100,100]],[[1,1,10,10]],['chat','chat','doc','doc'])
    assert r['speedup_ci95'][0]==r['speedup_ci95'][1]
    assert paired_bootstrap(a,a/2,['chat','chat','doc','doc'])==result


def test_output_checks_detect_length_and_repeatability():
    reference=[[dict(id='x',output_token_ids=[1,2])] for _ in range(5)]
    method=[[dict(id='x',output_token_ids=[1,2])] for _ in range(5)]
    method[4][0]['output_token_ids']=[1]
    check=output_checks(method,reference)
    assert check['requests_matching_ar_round0']==4 and check['repeatable_prompts']==0
    assert check['first_differences'][0]['first_difference_zero_based']==1


def test_full_synthetic_report_and_incomplete_rejection(tuned,manifest):
    folder,selection=tuned
    jobs=common.heldout_jobs(manifest,selection)
    for job in jobs[:-1]:
        save_block(folder,manifest,job)
    with pytest.raises(ValueError,match='Missing completed'):
        write_report(folder,manifest,selection)
    assert not (folder/'report.md').exists()
    save_block(folder,manifest,jobs[-1])
    write_report(folder,manifest,selection)
    summary=common.read_json(folder/'summary.json')
    assert summary['valid_blocks']==50
    assert summary['tuning_requests']==480 and summary['heldout_requests']==2250
    assert 'not established correctness-preserving' in (folder/'report.md').read_text()
    assert (folder/'latency.png').exists() and (folder/'tuning.png').exists()
