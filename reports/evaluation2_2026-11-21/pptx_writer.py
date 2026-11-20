"""Native PowerPoint text/shapes and embedded figures; no benchmark dependencies."""
from pathlib import Path
import zipfile
from xml.sax.saxutils import escape
OUT=Path(__file__).resolve().parent
def write_pptx(pages):
 """Minimal standards-based OOXML: native text/shape elements and chart PNGs."""
 A='http://schemas.openxmlformats.org/drawingml/2006/main';P='http://schemas.openxmlformats.org/presentationml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
 def emu(pt):return str(round(pt*12700))
 def fill(c):return f'<a:solidFill><a:srgbClr val="{c.lstrip(chr(35))}"/></a:solidFill>'
 def rels(items):return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{i}" Type="{R}/{typ}" Target="{escape(target)}"/>' for i,typ,target in items)+'</Relationships>'
 def tree():return '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
 with zipfile.ZipFile(OUT/'Evaluation2_Presentation.pptx','w',zipfile.ZIP_DEFLATED) as z:
  override=[('/ppt/presentation.xml','application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml'),('/ppt/slideMasters/slideMaster1.xml','application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml'),('/ppt/slideLayouts/slideLayout1.xml','application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml'),('/ppt/theme/theme1.xml','application/vnd.openxmlformats-officedocument.theme+xml')]+[(f'/ppt/slides/slide{i+1}.xml','application/vnd.openxmlformats-officedocument.presentationml.slide+xml') for i in range(len(pages))]
  z.writestr('[Content_Types].xml','<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'+''.join(f'<Override PartName="{path}" ContentType="{typ}"/>' for path,typ in override)+'</Types>')
  z.writestr('_rels/.rels',rels([('rId1','officeDocument','ppt/presentation.xml')]))
  presentation=f'<p:presentation xmlns:a="{A}" xmlns:r="{R}" xmlns:p="{P}"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>'+''.join(f'<p:sldId id="{256+i}" r:id="rId{i+2}"/>' for i in range(len(pages)))+f'</p:sldIdLst><p:sldSz cx="{emu(960)}" cy="{emu(540)}" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>'
  z.writestr('ppt/presentation.xml',presentation)
  z.writestr('ppt/_rels/presentation.xml.rels',rels([('rId1','slideMaster','slideMasters/slideMaster1.xml')]+[(f'rId{i+2}','slide',f'slides/slide{i+1}.xml') for i in range(len(pages))]))
  z.writestr('ppt/slideMasters/slideMaster1.xml',f'<p:sldMaster xmlns:a="{A}" xmlns:r="{R}" xmlns:p="{P}"><p:cSld><p:spTree>{tree()}</p:spTree></p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>')
  z.writestr('ppt/slideMasters/_rels/slideMaster1.xml.rels',rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','theme','../theme/theme1.xml')]))
  z.writestr('ppt/slideLayouts/slideLayout1.xml',f'<p:sldLayout xmlns:a="{A}" xmlns:r="{R}" xmlns:p="{P}" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>{tree()}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
  z.writestr('ppt/slideLayouts/_rels/slideLayout1.xml.rels',rels([('rId1','slideMaster','../slideMasters/slideMaster1.xml')]))
  cs=''.join(f'<a:{key}><a:srgbClr val="{value}"/></a:{key}>' for key,value in [('dk1','152E49'),('lt1','FFFFFF'),('dk2','26394B'),('lt2','EFF4F8'),('accent1','008C88'),('accent2','4778AE'),('accent3','D98225'),('accent4','C64D56'),('accent5','8065A8'),('accent6','697B8E'),('hlink','4778AE'),('folHlink','8065A8')])
  fonts=''.join(f'<a:{kind}><a:latin typeface="DejaVu Sans"/><a:ea typeface=""/><a:cs typeface=""/></a:{kind}>' for kind in ['majorFont','minorFont'])
  solid='<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
  fmt='<a:fillStyleLst>'+solid*3+'</a:fillStyleLst><a:lnStyleLst>'+''.join(f'<a:ln w="{w}">{solid}<a:prstDash val="solid"/></a:ln>' for w in [6350,12700,19050])+'</a:lnStyleLst><a:effectStyleLst>'+('<a:effectStyle><a:effectLst/></a:effectStyle>'*3)+'</a:effectStyleLst><a:bgFillStyleLst>'+solid*3+'</a:bgFillStyleLst>'
  z.writestr('ppt/theme/theme1.xml',f'<a:theme xmlns:a="{A}" name="Anytime Evidence"><a:themeElements><a:clrScheme name="Evidence">{cs}</a:clrScheme><a:fontScheme name="DejaVu">{fonts}</a:fontScheme><a:fmtScheme name="Evidence">{fmt}</a:fmtScheme></a:themeElements></a:theme>')
  media={}
  for index,page in enumerate(pages,1):
   elements=[];relationships=[('rId1','slideLayout','../slideLayouts/slideLayout1.xml')]
   for sid,scene in enumerate(page['scenes'],2):
    s=scene;x,y,w,h=[emu(s[k]) for k in ['x','y','w','h']];xf=f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
    if s['kind']=='image':
     path=s['path']
     if path not in media:
      media[path]=f'image{len(media)+1}.png';z.write(path,'ppt/media/'+media[path])
     rid=f'rId{len(relationships)+1}';relationships.append((rid,'image','../media/'+media[path]))
     elements.append(f'<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="Chart {sid}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr>{xf}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')
    else:
     rect=s['kind']=='rect';spfill=fill(s['color']) if rect else '<a:noFill/>'
     body=''
     if not rect:
      paras=[]
      for line in s['text'].split('\n'):
       paras.append(f'<a:p><a:pPr><a:lnSpc><a:spcPct val="132000"/></a:lnSpc><a:spcAft><a:spcPts val="0"/></a:spcAft></a:pPr><a:r><a:rPr lang="en-US" sz="{round(s["size"]*100)}" b="{1 if s["bold"] else 0}">{fill(s["color"])}<a:latin typeface="DejaVu Sans"/></a:rPr><a:t xml:space="preserve">{escape(line)}</a:t></a:r></a:p>')
      body='<p:txBody><a:bodyPr wrap="square" lIns="0" rIns="0" tIns="0" bIns="0" anchor="t"><a:normAutofit/></a:bodyPr><a:lstStyle/>'+''.join(paras)+'</p:txBody>'
     elements.append(f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{("Rectangle" if rect else "Text")} {sid}"/><p:cNvSpPr txBox="{0 if rect else 1}"/><p:nvPr/></p:nvSpPr><p:spPr>{xf}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{spfill}<a:ln><a:noFill/></a:ln></p:spPr>{body}</p:sp>')
   z.writestr(f'ppt/slides/slide{index}.xml',f'<p:sld xmlns:a="{A}" xmlns:r="{R}" xmlns:p="{P}"><p:cSld><p:spTree>{tree()}{"".join(elements)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')
   z.writestr(f'ppt/slides/_rels/slide{index}.xml.rels',rels(relationships))

def embed_notes(pages):
 """Add standard OOXML notes slides and a notes master to the native deck."""
 from xml.etree import ElementTree as ET
 file=OUT/'Evaluation2_Presentation.pptx'
 with zipfile.ZipFile(file) as z:parts={n:z.read(n) for n in z.namelist()}
 P='http://schemas.openxmlformats.org/presentationml/2006/main';A='http://schemas.openxmlformats.org/drawingml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships';PKG='http://schemas.openxmlformats.org/package/2006/relationships';CT='http://schemas.openxmlformats.org/package/2006/content-types'
 content=ET.fromstring(parts['[Content_Types].xml'])
 group='<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
 for i,page in enumerate(pages,1):
  paras=''.join('<a:p><a:r><a:rPr lang="en-US" sz="1200"/><a:t xml:space="preserve">'+escape(t)+'</a:t></a:r></a:p>' for t in page['notes'].split('\n'))
  body='<p:sp><p:nvSpPr><p:cNvPr id="2" name="Speaker notes"/><p:cNvSpPr txBox="1"/><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>'+paras+'</p:txBody></p:sp>'
  parts[f'ppt/notesSlides/notesSlide{i}.xml']=f'<p:notes xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:cSld><p:spTree>{group}{body}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>'.encode()
  rel=ET.Element('Relationships',xmlns=PKG)
  ET.SubElement(rel,'Relationship',Id='rId1',Type=R+'/slide',Target=f'../slides/slide{i}.xml')
  ET.SubElement(rel,'Relationship',Id='rId2',Type=R+'/notesMaster',Target='../notesMasters/notesMaster1.xml')
  parts[f'ppt/notesSlides/_rels/notesSlide{i}.xml.rels']=ET.tostring(rel)
  key=f'ppt/slides/_rels/slide{i}.xml.rels';rel=ET.fromstring(parts[key]);ET.SubElement(rel,'{'+PKG+'}Relationship',Id='rIdNotes',Type=R+'/notesSlide',Target=f'../notesSlides/notesSlide{i}.xml');parts[key]=ET.tostring(rel)
  ET.SubElement(content,'{'+CT+'}Override',PartName=f'/ppt/notesSlides/notesSlide{i}.xml',ContentType='application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml')
 parts['ppt/notesMasters/notesMaster1.xml']=f'<p:notesMaster xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:cSld><p:spTree>{group}</p:spTree></p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/><p:notesStyle/></p:notesMaster>'.encode()
 parts['ppt/notesMasters/_rels/notesMaster1.xml.rels']=f'<Relationships xmlns="{PKG}"><Relationship Id="rId1" Type="{R}/theme" Target="../theme/theme1.xml"/></Relationships>'.encode()
 ET.SubElement(content,'{'+CT+'}Override',PartName='/ppt/notesMasters/notesMaster1.xml',ContentType='application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml')
 pres=ET.fromstring(parts['ppt/presentation.xml']);lst=ET.Element('{'+P+'}notesMasterIdLst');ET.SubElement(lst,'{'+P+'}notesMasterId',{'{'+R+'}id':'rIdNotesMaster'});pres.insert(1,lst);parts['ppt/presentation.xml']=ET.tostring(pres)
 rel=ET.fromstring(parts['ppt/_rels/presentation.xml.rels']);ET.SubElement(rel,'{'+PKG+'}Relationship',Id='rIdNotesMaster',Type=R+'/notesMaster',Target='notesMasters/notesMaster1.xml');parts['ppt/_rels/presentation.xml.rels']=ET.tostring(rel);parts['[Content_Types].xml']=ET.tostring(content)
 with zipfile.ZipFile(file,'w',zipfile.ZIP_DEFLATED) as z:
  for n,data in parts.items():z.writestr(n,data)
