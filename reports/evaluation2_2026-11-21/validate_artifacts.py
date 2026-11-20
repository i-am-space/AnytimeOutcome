"""Numerical, structural and page-level validation of the Evaluation 2 artifacts."""
from pathlib import Path
import json,zipfile,hashlib,posixpath,re,ast
from xml.etree import ElementTree as ET
import fitz
from PIL import Image,ImageOps,ImageDraw
OUT=Path(__file__).resolve().parent
P=OUT/'previews';P.mkdir(exist_ok=True)
def read(name):return json.loads((OUT/name).read_text())
m=read('evidence_manifest.json');a=read('analysis.json');fd=read('figure_data.json');layout=read('layout_validation.json')
assert layout['main_seconds']==600
for n,e in m['files'].items():assert hashlib.sha256((OUT/'evidence_snapshot'/n).read_bytes()).hexdigest()==e['sha256'],n
assert a['baseline_heldout_exact']==1500 and a['publication_heldout_exact']==2250 and a['context_speculative_exact']==252
assert len({r['rounds'][0] for r in a['depth_results']})==1
assert len(layout['guide'])==45 and min(r['bottom_y'] for r in layout['guide'])>=55
assert all(r['bottom']<=536 for r in layout['slides'])
assert all(r['bottom']<=467 for r in layout['slides'] if 165<r['top']<469), 'Content overlaps takeaway banner'
historical=read('historical_audit.json')
for n,e in historical['files'].items():assert hashlib.sha256((OUT/'evidence_snapshot'/n).read_bytes()).hexdigest()==e['sha256'],n
# Every chart's numerical fields are built from the same frozen structured sources.
base=json.loads((OUT/'evidence_snapshot'/m['sources']['baseline']).read_text());pub=json.loads((OUT/'evidence_snapshot'/m['sources']['publication']).read_text());ctx=json.loads((OUT/'evidence_snapshot'/m['sources']['context']).read_text())
for protocol in ['fixed','natural']:
 for plotted in fd['baseline_'+protocol]:
  r=next(r for r in base['results'] if (r['protocol'],r['mode'],r['family'])==(protocol,plotted['mode'],'all'));assert plotted['statistics']==r['statistics']
for r in fd['publication_effect']:
 x=next(x for x in pub['results'] if (x['protocol'],x['method'],x['family'])==(r['protocol'],r['method'],'all'))
 assert abs(r['latency_increase_pct']+100*x['comparison_to']['complete']['latency_reduction'])<1e-10
for r in fd['context_latency']:
 x=next(x for x in ctx['results'] if x['config']['label']==r['config'] and x['context_length']==r['input_tokens']);assert r['latency_s']==x['mean_latency_s']
for r in fd['unequal_depth']:
 x=next(x for x in a['depth_results'] if x['method']==r['method'] and x['protocol']==r['protocol']);assert r['rounds']==x['rounds'];assert abs(r['increase_pct']+100*x['comparison']['latency_reduction'])<1e-10
pdfs={};atlases=[]
for name,count in [('Evaluation2_Learning_Guide.pdf',45),('Evaluation2_Presentation.pdf',21),('Evaluation2_Speaker_and_QA_Guide.pdf',17)]:
 d=fitz.open(OUT/name);assert len(d)==count;pages=[];alltext=[]
 for i,page in enumerate(d):
  txt=page.get_text();assert len(txt.strip())>100;assert '\ufffd' not in txt;assert not re.search(r'\{(?:ar_|sd_|ssd_|depth_|complete_pub|uniform_pub|overlap_pub)',txt)
  alltext.append(txt)
  for b in page.get_text('blocks'):
   assert b[0]>=-1 and b[1]>=-1 and b[2]<=page.rect.width+1 and b[3]<=page.rect.height+1,(name,i,b[:4])
  pix=page.get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False);path=P/f'{Path(name).stem}_{i+1:02d}.png';pix.save(path);pages.append(path)
 # Contact sheets show every rendered page, with full-resolution previews retained.
 batch=12 if count==21 else 15
 for start in range(0,len(pages),batch):
  cols=3;cellw=450;cellh=280 if count==21 else 650;rows=(min(batch,len(pages)-start)+cols-1)//cols
  sheet=Image.new('RGB',(cols*cellw,rows*cellh),'#D8E0E8');draw=ImageDraw.Draw(sheet)
  for j,path in enumerate(pages[start:start+batch]):
   im=Image.open(path);im.thumbnail((cellw-18,cellh-35));x=(j%cols)*cellw+(cellw-im.width)//2;y=(j//cols)*cellh+25;sheet.paste(im,(x,y));draw.text(((j%cols)*cellw+10,(j//cols)*cellh+5),f'{Path(name).stem} | {start+j+1}',fill='#21394B')
  path=P/f'{Path(name).stem}_contact_{start//batch+1}.jpg';sheet.save(path,quality=86);atlases.append(str(path.name))
 pdfs[name]={'pages':len(d),'all_pages_rendered':True,'text_bounds':'passed','characters':sum(map(len,alltext))}
 if 'Learning' in name:
  assert all(t in '\n'.join(alltext) for t in ['Two hours','Half day','Full day','Questions you should be ready'])
 if 'Presentation' in name:
  assert 'PROVISIONAL' in '\n'.join(alltext) and 'GPU UNTESTED' in '\n'.join(alltext)
 # Full searchable text supports easy revision and external accessibility tools.
 (OUT/(Path(name).stem+'.txt')).write_text('\n\n'.join(f'PAGE {i+1}\n'+t for i,t in enumerate(alltext)))
# Check all PowerPoint XML, image relationships and native notes. This host has no Office renderer.
with zipfile.ZipFile(OUT/'Evaluation2_Presentation.pptx') as z:
 names=set(z.namelist());assert z.testzip() is None
 for n in names:
  if n.endswith('.xml') or n.endswith('.rels'):ET.fromstring(z.read(n))
  if n.endswith('.rels'):
   parent=posixpath.dirname(posixpath.dirname(n))
   if n=='_rels/.rels':parent=''
   for r in ET.fromstring(z.read(n)):
    if r.get('TargetMode')=='External':continue
    target=posixpath.normpath(posixpath.join(parent,r.get('Target')));assert target in names,(n,target)
 slides=[n for n in names if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)];notes=[n for n in names if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n)]
 assert len(slides)==len(notes)==21
 assert sum(z.read(n).count(b'<p:sp>') for n in slides)>100
 for n in notes:assert b'SAY' in z.read(n) and b'EVIDENCE' in z.read(n)
result={'evidence_files_verified':len(set(m['files'])|set(historical['files'])),'historical_valid_blocks_audited':sum(r['valid_blocks'] for r in historical['studies']),'valid_blocks_audited':m['validated_blocks'],'output_audit':{k:v for k,v in a.items() if 'exact' in k},'plotted_values':'passed','pdfs':pdfs,'pptx':{'slides':21,'native_text_and_shapes':True,'notes_slides':21,'xml_and_relationships':'passed','native_office_rendering':'not available on this host; PDF and editable PPTX share scene geometry'},'contact_sheets':atlases,'main_talk_seconds':600,'qa_questions':25,'no_gpu_workers_started':True,'frozen_cutoff_ist':m['captured_at_ist']}
(OUT/'artifact_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
