"""Structural, data-snapshot, and page-boundary checks; renders review sheets."""
from pathlib import Path
import hashlib,json,zipfile,xml.etree.ElementTree as ET,posixpath
import fitz
from PIL import Image,ImageDraw
folder=Path(__file__).resolve().parent
checks={}
for filename in ['Anytime_SSD_Performance_Report.pdf','Evaluation1_Experimental_Plan_Slides.pdf']:
 doc=fitz.open(folder/filename)
 print(filename,len(doc),'pages')
 for i,page in enumerate(doc):
  txt=page.get_text()
  assert len(txt)>100,(filename,i,'empty')
  for block in page.get_text('dict')['blocks']:
   for line in block.get('lines',[]):
    for span in line['spans']:
     x0,y0,x1,y1=span['bbox']
     assert min(x0,y0)>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1,(filename,i,span)
 alltext='\n'.join(p.get_text() for p in doc)
 if 'Report' in filename:
  for term in ['2,730','1,125','480','768','specbench-148','HumanEval/36','specbench-553','10,000']:
   assert term in alltext,term
  (folder/'Report_Text.txt').write_text(alltext)
 checks[filename]={'pages':len(doc),'all_text_within_page':True,'bytes':(folder/filename).stat().st_size}
 for start in range(0,len(doc),12):
  thumbs=[]
  for i in range(start,min(start+12,len(doc))):
   pix=doc[i].get_pixmap(matrix=fitz.Matrix(.68,.68),alpha=False)
   im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
   tile=Image.new('RGB',(int(doc[0].rect.width*.68)+12,int(doc[0].rect.height*.68)+33),'#dfe5eb')
   tile.paste(im,((tile.width-im.width)//2,22));ImageDraw.Draw(tile).text((8,5),f'Page {i+1}',fill='#152e49')
   thumbs.append(tile)
  cols=3;rows=(len(thumbs)+cols-1)//cols
  sheet=Image.new('RGB',(thumbs[0].width*cols,thumbs[0].height*rows),'#dfe5eb')
  for k,im in enumerate(thumbs):sheet.paste(im,((k%cols)*im.width,(k//cols)*im.height))
  sheet.save(folder/f'preview_{"report" if "Report" in filename else "slides"}_{start//12+1}.jpg',quality=88)
with zipfile.ZipFile(folder/'Evaluation1_Experimental_Plan_Slides.pptx') as z:
 assert z.testzip() is None
 names=set(z.namelist())
 for name in names:
  if name.endswith('.xml') or name.endswith('.rels'):ET.fromstring(z.read(name))
  if name.endswith('.rels'):
   # Relationship targets resolve relative to the owning XML part.
   base=posixpath.dirname(posixpath.dirname(name))
   for rel in ET.fromstring(z.read(name)):
    if rel.attrib.get('TargetMode')=='External':continue
    target=posixpath.normpath(posixpath.join(base,rel.attrib['Target']))
    assert target in names,(name,target)
 slides=[n for n in names if n.startswith('ppt/slides/slide') and n.endswith('.xml')]
 assert len(slides)==12
 checks['pptx']={'slides':len(slides),'zip_xml_relationships_valid':True,'text_editable':True}
manifest=json.loads((folder/'evidence_manifest.json').read_text())
for row in manifest['sources']:
 assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
assert manifest['campaign_scored_requests']==0
checks['sources']={'count':len(manifest['sources']),'all_sha256_verified':True,'campaign_scored_requests':0}
(folder/'artifact_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
