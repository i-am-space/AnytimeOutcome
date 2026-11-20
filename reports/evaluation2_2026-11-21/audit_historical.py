"""Supplement the primary frozen audit with older completed engineering studies."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;SNAP=OUT/'evidence_snapshot'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit():
 inventory={};results=[]
 def capture(p):
  rel=str(p.relative_to(ROOT));data=p.read_bytes();q=SNAP/rel
  if q.exists():assert q.read_bytes()==data
  else:q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
  inventory[rel]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
  return rel
 ar={}
 root=SNAP/'results/next_stage/shared_uniform_v1_2026-11-17/baselines/raw'
 for p in root.rglob('ar.json'):
  b=read(p);j=b['job']
  for r in b['requests']:ar[(j['phase'],j['protocol'],j['repeat'],r['id'])]=r
 studies=[('overnight_extensions_v2_2026-11-17',['heldout','depth_tune','correctness']),('adaptive_publication_v1_2026-11-17',['correctness']),('prefix_overlap_v1_2026-11-17',['correctness','stress'])]
 for name,phases in studies:
  root=ROOT/'results/next_stage'/name
  for phase in phases:
   phase_root=root/phase
   if phase=='depth_tune':
    reuse=read(root/'queue.json')['reused_depth'];capture(root/'queue.json')
    for n,h in reuse['artifact_sha256'].items():assert sha(n)==h
    phase_root=Path(reuse['folder'])/phase;capture(phase_root.parent/'queue.json')
   d=read(phase_root/'definition.json');capture(phase_root/'definition.json');blocks=[];shared=False
   for rp in sorted(phase_root.glob('*/attempt_*/receipt.json')):
    r=read(rp)
    if not r.get('valid'):capture(rp);continue
    p=rp.parent/'result.json';assert r['returncode']==0 and r['samples']>0 and not r.get('error')
    assert r['definition_sha256']==d['sha256'] and r['output_sha256']==sha(p)
    # Shared correctness is intentionally permitted only in marked diagnostic receipts.
    if r.get('external_processes'):assert r.get('shared_correctness') and r.get('performance_evidence') is False;shared=True
    b=read(p);assert b['job'] in d['jobs'];blocks.append(b);capture(p);capture(rp)
   fresh={(b['job']['protocol'],b['job']['round']):{r['id']:r for r in b['requests']} for b in blocks if b['job']['config']['mode']=='ar'}
   comparisons=0
   for b in blocks:
    j=b['job']
    if j['config']['mode']=='ar':continue
    for r in b['requests']:
     gold=(fresh[(j['protocol'],j['round'])][r['id']] if fresh else ar[('tune' if phase=='depth_tune' else 'heldout',j['protocol'],j['round'],r['id'])])
     assert r['input_token_ids_sha256']==gold['input_token_ids_sha256']
     assert r['output_token_ids']==gold['output_token_ids'];comparisons+=1
   results.append({'study':name,'phase':phase,'valid_blocks':len(blocks),'speculative_exact_matches':comparisons,'performance_evidence':phase=='heldout','shared_activity_diagnostic':shared})
 out={'audited_at_ist':datetime.now(ZoneInfo('Asia/Calcutta')).isoformat(),'note':'Supplementary copies of completed Nov17 evidence; primary report cutoff remains unchanged. No new measurements. Shared correctness receipts do not establish performance.','files':inventory,'studies':results}
 (OUT/'historical_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':audit()
