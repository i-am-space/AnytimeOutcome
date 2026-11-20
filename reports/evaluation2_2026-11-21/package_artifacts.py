"""Package frozen raw evidence and generated artifacts; omit disposable page PNGs."""
from pathlib import Path
import hashlib,json,zipfile
OUT=Path(__file__).resolve().parent
entries=[]
for p in sorted(OUT.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts:continue
 if p.name in ['Evaluation2_Package.zip','package_manifest.json']:continue
 if p.parent.name=='previews' and p.suffix=='.png':continue
 entries.append(p)
manifest={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in entries}
(OUT/'package_manifest.json').write_text(json.dumps({'files':manifest,'policy':'Frozen data; completed valid blocks only count; unpacked raw evidence included here instead of duplicating it in Git.'},indent=2)+'\n')
entries.append(OUT/'package_manifest.json')
with zipfile.ZipFile(OUT/'Evaluation2_Package.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in entries:z.write(p,OUT.name+'/'+str(p.relative_to(OUT)))
with zipfile.ZipFile(OUT/'Evaluation2_Package.zip') as z:
 assert z.testzip() is None
 for name,v in manifest.items():assert hashlib.sha256(z.read(OUT.name+'/'+name)).hexdigest()==v['sha256']
print('Archive:',len(entries),'files;',round((OUT/'Evaluation2_Package.zip').stat().st_size/1024**2,2),'MiB; all member hashes verified')
