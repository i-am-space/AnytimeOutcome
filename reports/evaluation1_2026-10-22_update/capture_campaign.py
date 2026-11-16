"""Freeze validated completed evidence for a report while the campaign may continue."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys,shutil,json
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'profiling'))
from baseline_common import read_json,atomic_json,validate_manifest,validate_block,tuning_jobs,heldout_jobs,config_id,file_digest,execution_fingerprint
from run_baselines import validated_selection,accepted_result,paths
from analyze_baselines import row_statistics,output_checks
folder=ROOT/'results/baselines/eval1';snap=OUT/'campaign_snapshot';snap.mkdir(exist_ok=True)
m=read_json(folder/'manifest.json');validate_manifest(m)
assert execution_fingerprint()==m['execution']
s=validated_selection(folder,m)
for name in ['manifest.json','selection.json','status.json','fixed_window_status.json','fixed_window_request.json','fixed_window_definition.json','fixed_window_protocol_note.json','tuning_jobs.json','validation.json','validation_tests.txt','fixed_window_tests.txt']:
 if (folder/name).exists():shutil.copy2(folder/name,snap/name)
completed=[];excluded=[]
for job in tuning_jobs(m)+heldout_jobs(m,s):
 try:r=accepted_result(folder,job,m)
 except (ValueError,KeyError,OSError):continue
 for source in paths(folder,job)[:2]:
  dest=snap/source.relative_to(folder);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
 completed.append(r)
for p in list((folder/'invalid_attempts').rglob('*.receipt.json'))+list((folder/'raw').rglob('*.receipt.json')):
 d=read_json(p)
 if d.get('valid'):continue
 excluded.append(dict(path=str(p),reason=d.get('error'),monitoring=d.get('monitoring'),returncode=d.get('returncode')))
 target=snap/'excluded'/p.relative_to(folder);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
tune=[r for r in completed if r['job']['phase']=='tune'];held=[r for r in completed if r['job']['phase']=='heldout']
assert len(tune)==20
ids=[p['id'] for p in m['splits']['tune']]
byar={r['job']['repeat']:{x['id']:x for x in r['requests']} for r in tune if r['job']['config']['mode']=='ar'}
argrid=[[byar[i][rid] for rid in ids] for i in range(2)]
tuning=[]
for score in s['scores']:
 blocks=[r for r in tune if r['job']['config']==score['config']]
 by={r['job']['repeat']:{x['id']:x for x in r['requests']} for r in blocks}
 grid=[[by[i][rid] for rid in ids] for i in range(2)]
 stats=row_statistics([x for row in grid for x in row]);check=output_checks(grid,argrid)
 assert abs(stats['mean_latency_s']-score['mean_latency_s'])<1e-12
 tuning.append(dict(config=score['config'],config_id=config_id(score['config']),statistics=stats,
  selected=score['config']==s['winners'][score['config']['mode']],correctness=check,
  repeat_mean_latency_s=[sum(x['latency_s'] for x in by[i].values())/24 for i in range(2)]))
held_stats=[];paired=None
for r in held:
 if r['job']['protocol']!='fixed':continue
 for family in ['all','chat','code','doc']:
  rows=[x for x in r['requests'] if family=='all' or x['family']==family]
  held_stats.append(dict(mode=r['job']['config']['mode'],config=r['job']['config'],repeat=r['job']['repeat'],family=family,statistics=row_statistics(rows)))
held0={r['job']['config']['mode']:r for r in held if r['job']['protocol']=='fixed' and r['job']['repeat']==0}
if 'ar' in held0 and 'ssd' in held0:
 hids=[p['id'] for p in m['splits']['heldout']]
 grids={mode:[[{x['id']:x for x in block['requests']}[rid] for rid in hids]] for mode,block in held0.items()}
 paired=output_checks(grids['ssd'],grids['ar'])
 paired['repeatable_prompts']=None
 paired['repeatability_note']='Unavailable: only one valid held-out repetition per available method.'
 paired['families']={}
 for family in ['chat','code','doc']:
  ix=[i for i,p in enumerate(m['splits']['heldout']) if p['family']==family]
  c=output_checks([[grids['ssd'][0][i] for i in ix]],[[grids['ar'][0][i] for i in ix]])
  paired['families'][family]=dict(exact=c['requests_matching_ar_round0'],total=len(ix))
context=None
context_dir=folder/'gpu0_only_sd_r0'
if (context_dir/'receipt.json').exists():
 receipt=read_json(context_dir/'receipt.json')
 if receipt.get('diagnostic_valid') is True:
  assert receipt['valid'] is False and receipt['valid_for_dedicated_campaign'] is False
  assert not receipt['monitoring']['gpu0_external'] and receipt['monitoring']['error'] is None
  assert receipt['output_sha256']==file_digest(context_dir/'run.json')
  run=read_json(context_dir/'run.json');validate_block(run,run['job'],m)
  for p in context_dir.iterdir():
   if p.suffix=='.json':
    target=snap/'gpu0_only_sd_r0'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
  rmap={r['id']:r for r in run['requests']}
  checks=output_checks([[rmap[rid] for rid in hids]],grids['ar'])
  checks['repeatable_prompts']=None
  context=dict(config=run['job']['config'],scope=receipt['scope'],statistics=row_statistics(run['requests']),
   correctness=checks,monitoring=receipt['monitoring'],valid_for_dedicated_campaign=False,
   families={f:row_statistics([r for r in run['requests'] if r['family']==f]) for f in ['chat','code','doc']})
body=dict(captured_at=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
 manifest_sha256=m['manifest_sha256'],selection_sha256=s['selection_sha256'],
 valid_tuning_blocks=len(tune),tuning_requests=sum(len(r['requests']) for r in tune),
 valid_heldout_blocks=len(held),heldout_requests=sum(len(r['requests']) for r in held),
 natural_eos_requests=sum(len(r['requests']) for r in held if r['job']['protocol']=='natural'),
 tuning=tuning,heldout=held_stats,heldout_ssd_vs_ar=paired,contextual_sd=context,excluded_attempts=excluded,
 remaining_fixed_blocks=15-sum(r['job']['protocol']=='fixed' for r in held),
 remaining_fixed_requests=1125-sum(len(r['requests']) for r in held if r['job']['protocol']=='fixed'),
 remaining_natural_requests=1125-sum(len(r['requests']) for r in held if r['job']['protocol']=='natural'),
 caveats=['Tuning is configuration selection, not unbiased held-out evidence.','Held-out results are partial; do not report completed five-repeat results or repeated-run CIs.','All comparisons use fast modes with unresolved strict exactness.','One contaminated SD attempt is excluded; repeatability cannot be estimated from one held-out round.'])
atomic_json(OUT/'completed_campaign_snapshot.json',body)
print(json.dumps({k:body[k] for k in ['captured_at','tuning_requests','heldout_requests','valid_tuning_blocks','valid_heldout_blocks','remaining_fixed_requests','remaining_natural_requests']},indent=2))
