"""Freeze saved evidence and audit token outputs. Never starts a benchmark worker."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,shutil,sys,ast
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SNAP=OUT/'evidence_snapshot'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
SOURCES={
 'baseline':'results/next_stage/shared_uniform_v1_2026-11-17/baselines/summary.json',
 'baseline_selection':'results/next_stage/shared_uniform_v1_2026-11-17/baselines/selection.json',
 'workload':'results/next_stage/shared_uniform_v1_2026-11-17/baselines/manifest.json',
 'publication':'results/next_stage/publication_performance_v2_2026-11-18/summary.json',
 'publication_selection':'results/next_stage/publication_performance_v2_2026-11-18/selection.json',
 'publication_tune':'results/next_stage/publication_performance_v2_2026-11-18/tune/summary.json',
 'context':'results/next_stage/context_length_v2_2026-11-18/summary.json',
 'depth_selection':'results/next_stage/outcome_depth_v2_2026-11-18/selection.json',
 'depth_gate':'results/next_stage/outcome_depth_v2_2026-11-18/gate/summary.json',
 'depth_status':'results/next_stage/outcome_depth_v2_2026-11-18/status.json',
 'greedy_status':'results/next_stage/greedy_handoff_v1_2026-11-19/status.json',
 'greedy_validation':'results/next_stage/greedy_handoff_v1_2026-11-19/validation.json',
 'uniform':'results/next_stage/overnight_extensions_v2_2026-11-17/heldout/summary.json',
 'uniform_tune':'results/next_stage/overnight_extensions_v2_2026-11-17/depth_tune/summary.json',
 'adaptive':'results/next_stage/adaptive_publication_v1_2026-11-17/correctness/summary.json',
 'overlap_gate':'results/next_stage/prefix_overlap_v1_2026-11-17/correctness/summary.json',
 'overlap_stress':'results/next_stage/prefix_overlap_v1_2026-11-17/stress/summary.json',
 'shared_gate':'results/correctness/shared_fixed16_v1_2026-11-17_v2/summary.json',
 'numerical_benchmark':'results/correctness/shared_fixed16_prototype_benchmark_2026-11-17/summary.json',
 'critical_path':'results/nsys/ssd_k7_f3_critical_path.json',
 'roofline':'results/roofline/machine_roofline.json',
 'old_ar':'results/nsys/ar_analysis.json','old_sd':'results/nsys/sd_k6_analysis.json',
 'old_ssd':'results/nsys/ssd_k7_f3_analysis.json',
}

def freeze():
 if (OUT/'evidence_manifest.json').exists():
  m=read(OUT/'evidence_manifest.json')
  for name,entry in m['files'].items():assert sha(SNAP/name)==entry['sha256'],name
  return m
 SNAP.mkdir(exist_ok=True)
 inventory={}
 def capture(p):
  p=Path(p);p=p if p.is_absolute() else ROOT/p
  name=str(p.relative_to(ROOT))
  if name not in inventory:
   data=p.read_bytes();q=SNAP/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
   inventory[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
  return name
 for name in SOURCES.values():capture(name)
 campaigns={};checks=[]
 base=ROOT/'results/next_stage/shared_uniform_v1_2026-11-17/baselines'
 manifest=read(base/'manifest.json');baseline=[]
 for phase,fn in [('tune','tuning_jobs.json'),('heldout','heldout_jobs.json')]:
  capture(base/fn)
  for j in read(base/fn):
   p=(base/'raw'/j['id']).with_suffix('.json');rp=p.with_suffix('.receipt.json');r=read(rp);mon=r['monitoring']
   assert r['valid'] and r['returncode']==0 and not r.get('error')
   assert mon['samples']>0 and not mon['external_processes'] and not mon.get('error')
   assert r['job_sha256']==j['job_sha256'] and r['execution_sha256']==manifest['execution']['sha256']
   assert sha(p)==r['output_sha256'];b=read(p)
   assert b['job']==j and b['manifest_sha256']==manifest['manifest_sha256']
   assert b['prefix_cache_enabled'] is False
   assert len(b['requests'])==(24 if phase=='tune' else 75)
   baseline.append({'phase':phase,'data':b})
   capture(p);capture(rp);checks.append({'campaign':'baseline','phase':phase,'receipt':str(rp.relative_to(ROOT)),'requests':len(b['requests'])})
 campaigns['baseline']=baseline
 for name,phases in [('publication_performance_v2_2026-11-18',['continuity','tune','heldout']),('context_length_v2_2026-11-18',['measure','profiles']),('outcome_depth_v2_2026-11-18',['gate','tune','heldout']),('greedy_handoff_v1_2026-11-19',['micro','gate','stress','diagnostic','tune','heldout'])]:
  root=ROOT/'results/next_stage'/name;blocks=[]
  for p in root.glob('*.json'):capture(p)
  for phase in phases:
   dp=root/phase/'definition.json'
   if not dp.exists():continue
   capture(dp);definition=read(dp)
   seen=set()
   for rp in sorted((root/phase).glob('*/attempt_*/receipt.json')):
    r=read(rp)
    if not r.get('valid'):capture(rp);continue
    assert rp.parent.parent.name not in seen;seen.add(rp.parent.parent.name)
    p=rp.parent/'result.json'
    assert r['returncode']==0 and r['samples']>0 and not r['external_processes'] and not r.get('error')
    assert r['definition_sha256']==definition['sha256'] and sha(p)==r['output_sha256']
    assert r.get('dedicated_gpus') is True
    b=read(p);assert b['definition_sha256']==definition['sha256']
    assert b['job'] in definition['jobs']
    blocks.append({'phase':phase,'data':b});capture(p);capture(rp)
    checks.append({'campaign':name,'phase':phase,'receipt':str(rp.relative_to(ROOT)),'requests':len(b.get('requests',[]))})
  campaigns[name]=blocks
 # Freeze executed sources and reject drift without importing any GPU module.
 for name in ['outcome_depth_v2_2026-11-18','greedy_handoff_v1_2026-11-19']:
  q=read(ROOT/'results/next_stage'/name/'queue.json')
  if name.startswith('outcome'):
   wr=ROOT/'.worktrees/outcome-depth-v1';files={**q['execution']['files'],**q['queue_code']}
   for n,h in files.items():assert sha(wr/n)==h;capture(wr/n)
  else:
   for n,h in q['code'].items():assert sha(n)==h;capture(n)
 for p in (ROOT/'docs').glob('*.md'):capture(p)
 for p in (ROOT/'.worktrees/outcome-depth-v1/docs').glob('*.md'):
  if p.name=='outcome_depth.md':capture(p)
 for n in ['fast_numerics_all_layers_2026-11-17.json','attention_boundaries_2026-11-17.json','fixed_gemm_controls_2026-11-17.json','summary.json']:
  capture(ROOT/'results/correctness'/n)
 for p in (ROOT/'results/ncu').glob('*ncu.json'):capture(p)
 capture('results/ncu/summary_tables.md')
 capture('profiling/analyze_baselines.py');capture('profiling/workloads.py')
 capture('reports/evaluation1_2026-10-22_update/evidence_manifest.json')
 for p in (ROOT/'results/next_stage').glob('*v1*/supersession.json'):capture(p)
 write(OUT/'block_audit.json',checks)
 m={'captured_at_ist':datetime.now(ZoneInfo('Asia/Calcutta')).isoformat(),'sources':SOURCES,'files':inventory,'validated_blocks':len(checks),'snapshot_policy':'Only valid completed blocks count. Failed receipts retained but excluded. No GPU workers started.'}
 write(OUT/'evidence_manifest.json',m)
 return m

def analyze(m):
 def load(key):return read(SNAP/m['sources'][key])
 base=load('baseline');pub=load('publication');ctx=load('context');work=load('workload')
 # Reuse the exact saved analysis function, without importing its engine dependencies.
 code=(SNAP/'profiling/analyze_baselines.py').read_text();node=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='paired_bootstrap')
 ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'saved_paired_bootstrap','exec'),ns);bootstrap=ns['paired_bootstrap']
 allbase=[]
 for p in (SNAP/'results/next_stage/shared_uniform_v1_2026-11-17/baselines/raw').rglob('*.receipt.json'):
  allbase.append(read(p.with_name(p.name.replace('.receipt.json','.json'))))
 ar={};base_exact=0;base_total=0
 for b in allbase:
  j=b['job']
  if j['config']['mode']=='ar':
   for r in b['requests']:ar[(j['phase'],j['protocol'],j['repeat'],r['id'])]=r
 for b in allbase:
  j=b['job']
  if j['config']['mode']=='ar':continue
  for r in b['requests']:
   gold=ar[(j['phase'],j['protocol'],j['repeat'],r['id'])]
   assert gold['input_token_ids_sha256']==r['input_token_ids_sha256']
   assert gold['output_token_ids']==r['output_token_ids']
   if j['phase']=='heldout':base_exact+=1;base_total+=1
 def newblocks(name,phase):
  return [read(p.parent/'result.json') for p in (SNAP/'results/next_stage'/name/phase).glob('*/attempt_*/receipt.json') if read(p).get('valid')]
 def check_saved(blocks,phase):
  total=0
  for b in blocks:
   j=b['job']
   for r in b['requests']:
    gold=ar[(phase,j['protocol'],j['round'],r['id'])]
    assert gold['input_token_ids_sha256']==r['input_token_ids_sha256']
    assert gold['output_token_ids']==r['output_token_ids'];total+=1
  return total
 pb=newblocks('publication_performance_v2_2026-11-18','heldout');pubexact=check_saved(pb,'heldout')
 assert pubexact==pub['exact_speculative_matches']==2250
 db=newblocks('outcome_depth_v2_2026-11-18','heldout');depthexact=check_saved(db,'heldout')
 ids=[p['id'] for p in work['splits']['heldout']];families=[p['family'] for p in work['splits']['heldout']]
 methods=['complete','overlap','ranked','ranked_overlap','budget','reverse'];partial=[]
 for protocol in ['fixed','natural']:
  records={(b['job']['round'],b['job']['config']['method']):{r['id']:r for r in b['requests']} for b in db if b['job']['protocol']==protocol}
  rounds=[i for i in range(5) if all((i,method) in records for method in methods)]
  if not rounds:continue
  for method in methods:
   rows=[records[(i,method)][rid] for i in rounds for rid in ids]
   ref=[[records[(i,'complete')][rid]['latency_s'] for rid in ids] for i in rounds]
   values=[[records[(i,method)][rid]['latency_s'] for rid in ids] for i in rounds]
   comp=bootstrap(ref,values,families,10000,0)
   partial.append({'protocol':protocol,'method':method,'rounds':rounds,'requests':len(rows),'mean_latency_s':float(np.mean([r['latency_s'] for r in rows])),'mean_wait_s':float(np.mean([r['draft_wait_s'] for r in rows])),'comparison':comp})
 # Context matches against fresh AR, including separate profiled requests.
 cb=newblocks('context_length_v2_2026-11-18','measure');cp=newblocks('context_length_v2_2026-11-18','profiles')
 car={}
 for b in cb:
  if b['job']['config']['method']=='ar':
   for r in b['requests']:
    k=(r['id'],r['input_token_ids_sha256'])
    if k in car:assert car[k]==r['output_token_ids']
    car[k]=r['output_token_ids']
 context_exact=0
 for b in cb+cp:
  if b['job']['config']['method']=='ar':continue
  for r in b['requests']:
   assert car[(r['id'],r['input_token_ids_sha256'])]==r['output_token_ids'];context_exact+=1
 assert context_exact==ctx['exact_matches']==252
 for r in base['results']:
  rows=[row for b in allbase if b['job']['phase']=='heldout' and b['job']['protocol']==r['protocol'] and b['job']['config']['mode']==r['mode'] for row in b['requests'] if r['family']=='all' or row['family']==r['family']]
  assert abs(np.mean([x['latency_s'] for x in rows])-r['statistics']['mean_latency_s'])<1e-10
 audit={'baseline_heldout_exact':base_exact,'baseline_heldout_comparisons':base_total,'publication_heldout_exact':pubexact,'context_speculative_exact':context_exact,'depth_heldout_exact':depthexact,'depth_valid_blocks':len(db),'depth_required_blocks':60,'depth_status':'complete' if len(db)==60 else 'partial','depth_results':partial,'reference':'shared-fixed16-v1; original AR remains separate','stale_header_note':'Baseline generic correctness_status text is stale. Claims here were rechecked directly against validated raw token IDs.'}
 write(OUT/'analysis.json',audit)
 return audit
if __name__=='__main__':
 m=freeze();a=analyze(m);print(json.dumps({'captured':m['captured_at_ist'],'files':len(m['files']),'blocks':m['validated_blocks'],'exact':{k:v for k,v in a.items() if 'exact' in k},'depth_blocks':a['depth_valid_blocks']},indent=2))
