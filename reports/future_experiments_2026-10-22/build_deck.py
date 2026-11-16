"""Generate a high-level future experimentation deck from the proposal and saved status."""
from pathlib import Path
import hashlib, json, re, zipfile, html
from xml.sax.saxutils import escape
from datetime import datetime
from zoneinfo import ZoneInfo
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
NAVY='#152E49'; TEAL='#008C88'; BLUE='#4778AE'; ORANGE='#D98225'; GRAY='#697B8E'; INK='#26394B'; LIGHT='#EFF4F8'; RED='#B74750'; WHITE='#FFFFFF'
for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/dejavu/'+file))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
def clean(text): return html.unescape(re.sub('<[^>]+>','',text))

def write_pptx(pages):
 """Minimal standards-based OOXML: native text/shape elements and chart PNGs."""
 A='http://schemas.openxmlformats.org/drawingml/2006/main';P='http://schemas.openxmlformats.org/presentationml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
 def emu(pt):return str(round(pt*12700))
 def fill(c):return f'<a:solidFill><a:srgbClr val="{c.lstrip(chr(35))}"/></a:solidFill>'
 def rels(items):return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{i}" Type="{R}/{typ}" Target="{escape(target)}"/>' for i,typ,target in items)+'</Relationships>'
 def tree():return '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
 with zipfile.ZipFile(OUT/'Future_Experimentation_Plan.pptx','w',zipfile.ZIP_DEFLATED) as z:
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


class Deck:
    def __init__(self):
        self.w=960; self.h=540; self.pages=[]; self.scenes=[]
        self.c=canvas.Canvas(str(OUT/'Future_Experimentation_Plan.pdf'),pagesize=(960,540))
        self.c.setTitle('Anytime Outcome-Cache | Future Experimentation Plan')
        self.c.setAuthor('Ishaan Romil; Vishal Rao; Eshaan Sharma')

    def rect(self,x,y,w,h,color):
        self.c.setFillColor(HexColor(color)); self.c.rect(x,540-y-h,w,h,fill=1,stroke=0)
        self.scenes.append(dict(kind='rect',x=x,y=y,w=w,h=h,color=color))

    def text(self,text,x,y,w,size=18,color=INK,bold=False,maxh=None):
        p=Paragraph(text,ParagraphStyle('text',fontName='Bold' if bold else 'Body',fontSize=size,
                    leading=size*1.28,textColor=HexColor(color)))
        _,h=p.wrap(w,10000)
        if maxh is not None and h>maxh:
            raise ValueError(f'Slide {len(self.pages)} overflow {h:.1f}>{maxh}: {text}')
        if y+h>501: raise ValueError(f'Slide {len(self.pages)} bottom overflow: {text}')
        p.drawOn(self.c,x,540-y-h)
        self.scenes.append(dict(kind='text',text=clean(text.replace('<br/>','\n')),x=x,y=y,w=w,h=h+5,
                                size=size,color=color,bold=bold))
        return h

    def start(self,title,subtitle,notes,source='Proposal pp. 1–2 | Planned experiments; results pending'):
        if self.pages: self.c.showPage()
        self.scenes=[]; self.pages.append(dict(scenes=self.scenes,notes=notes,title=title))
        self.rect(0,0,960,8,NAVY)
        self.text('ANYTIME OUTCOME-CACHE  /  FUTURE EXPERIMENTS',36,23,850,10,TEAL,True)
        self.text(title,36,49,888,29,NAVY,True,maxh=76)
        self.text(subtitle,36,129,888,15,GRAY,maxh=40)
        self.rect(36,507,888,1,'#D7E0E8')
        # Footer uses a separate primitive so it may extend below the content area.
        self.c.setFont('Body',8); self.c.setFillColor(HexColor(GRAY))
        self.c.drawString(36,17,source)
        self.c.drawRightString(924,17,f'{len(self.pages):02d}  /  22 OCT 2026')
        self.scenes.append(dict(kind='text',text=source,x=36,y=514,w=770,h=15,size=8,color=GRAY,bold=False))
        self.scenes.append(dict(kind='text',text=f'{len(self.pages):02d}  /  22 OCT 2026',x=806,y=514,w=120,h=15,size=8,color=GRAY,bold=False))

    def card(self,x,y,w,h,label,title,body,color=TEAL,size=18):
        self.rect(x,y,w,h,LIGHT); self.rect(x,y,5,h,color)
        self.text(label,x+18,y+15,w-36,10,color,True,maxh=18)
        th=self.text(title,x+18,y+36,w-36,21,NAVY,True,maxh=64)
        if body:
            self.text(body,x+18,y+43+th,w-36,size,INK,maxh=h-th-51)

    def banner(self,text,y=454,color=TEAL):
        self.rect(36,y,888,32 if y>=465 else 40,LIGHT)
        self.text(text,48,y+9,864,15,color,True,maxh=27)

    def table(self,headers,rows,widths,y=183,rowh=43,size=16):
        self.rect(36,y,888,35,NAVY)
        x=36
        for t,w in zip(headers,widths):
            self.text(t,x+10,y+7,w-20,13,WHITE,True,maxh=24); x+=w
        y+=35
        for i,row in enumerate(rows):
            self.rect(36,y,888,rowh,LIGHT if i%2==0 else WHITE)
            x=36
            for t,w in zip(row,widths):
                self.text(t,x+10,y+8,w-20,size,INK,maxh=rowh-12); x+=w
            y+=rowh
        return y

    def finish(self):
        self.c.save(); write_pptx(self.pages)
        (OUT/'Speaker_Notes.txt').write_text('\n\n'.join(
            f'SLIDE {i+1}: {p["title"]}\n{p["notes"]}' for i,p in enumerate(self.pages))+'\n')
        (OUT/'slide_content.json').write_text(json.dumps(self.pages,indent=2))


d=Deck()

d.start('Future experimentation plan',
        'From the current baseline to a correct, measured anytime decoder.',
        'Scope: all remaining project experiments, based on the uploaded two-page proposal. '
        'This presentation is a plan, not a new results report. The user requested removal of a dedicated '
        'named fan-out comparator; the comparisons shown follow the proposal. Unequal fan-out alone is '
        'established work and is not claimed as our contribution. The proposed contribution concerns '
        'incremental publication and unequal continuation depths, judged against simple early handoff.')
d.card(36,191,280,219,'QUESTION','Can we stop waiting?','Use a completed draft prefix before the entire continuation is ready.')
d.card(340,191,280,219,'EXPERIMENT','Which policy helps?','Compare early publication, fixed depths, and adaptive scheduling.',BLUE)
d.card(644,191,280,219,'SUCCESS TARGET','At least 10% lower latency','Against the strongest matched baseline, with exact greedy output.',ORANGE)
d.banner('The 10% improvement is a target to test, not a result already achieved.')

d.start('What is done, and what remains?',
        'Current evidence supports the experiment design; the project prototype is still ahead.',
        'Current status is grounded in the saved baseline status and the immutable completed-results snapshot. '
        'All 480 tuning requests are valid. Dedicated held-out evidence consists of 75 AR and 75 SSD fixed-length '
        'requests, one repetition each. A separate 75-request SD run on clean GPU0 with GPU1 occupied is contextual '
        'and does not replace a dedicated campaign repetition. The two-hour window ended with the campaign incomplete. '
        'Profiling and microbenchmarks exist. Fast-mode exact agreement with unchanged AR still fails; the slow '
        'serial diagnostic reference is not an optimized correctness fix.',
        'Saved campaign status + completed-results snapshot | Current evidence, not final claims')
d.card(36,186,280,246,'DONE','Baseline foundation','Profiling and bottleneck analysis.<br/>480 tuning requests.<br/>SD K=6; SSD K=7/F=3.',TEAL,17)
d.card(340,186,280,246,'PARTIAL','Held-out comparison','75 AR + 75 SSD requests.<br/>75 SD requests for context.<br/>Repeated testing is unfinished.',ORANGE,17)
d.card(644,186,280,246,'LEFT','The core project','Exactness, safe early publication, scheduling, ablations, and final evaluation.',BLUE,17)
d.banner('Present fast-mode timings as performance evidence; exactness remains unresolved.',color=RED)

d.start('The remaining work has four stages',
        'Each stage answers a question before the next stage adds complexity.',
        'Correctness work and completion of unprofiled baseline measurements can proceed as separate tracks. '
        'No correctness-preserving acceleration claim is allowed until the unchanged target outputs match. '
        'After uniform early publication, continue the adaptive scheduler only if measured headroom remains. '
        'The timeline is ordered by evidence gates, not a promise of GPU availability or calendar dates.')
for x,label,title,body,col in [
    (36,'01','Trust the baseline','Finish controlled runs and establish exact output.',TEAL),
    (264,'02','Publish early','Test the simplest safe partial-draft handoff.',BLUE),
    (492,'03','Allocate depth','Compare fixed and adaptive schedules.',ORANGE),
    (720,'04','Validate broadly','Run final held-out tests, profiles, and replication.',TEAL)]:
    d.card(x,196,204,230,label,title,body,col,16)
d.banner('Keep the same executor so policy changes explain the differences.')

d.start('What will we compare?',
        'Simple controls first; stronger policies reveal whether extra complexity earns its cost.',
        'AR and synchronous SD are context controls using one GPU; SSD-based policy comparisons use the same '
        'two-GPU target/drafter allocation. Fixed global depth is covered by K tuning. Add an online global '
        'depth/fan-out comparator to test whether simple adaptation is sufficient. Uniform early publication '
        'with immediate handoff is the key comparator. Unequal fixed continuation depths isolate allocation. '
        'Probability-first and SwiftSpec-inspired timing policies are same-executor adaptations, not claims '
        'to reproduce their entire original systems. Candidate construction and ranking stay fixed initially. '
        'Match candidate/work budgets, memory constraints, hardware, and tuning effort across policy comparisons.')
d.table(['Method or policy','Question it answers'],[
    ['AR / synchronous SD (1 GPU)','What do standard decoding methods achieve?'],
    ['Tuned complete-cache SSD (2 GPUs)','What does asynchronous drafting already provide?'],
    ['Online global depth / fan-out','Is simple adaptation enough?'],
    ['Uniform early publication','Does earlier handoff alone help?'],
    ['Unequal fixed continuation depths','Does changing depth allocation help?'],
    ['Probability-first / timing-rule adaptations','Do simpler scheduling rules suffice?'],
    ['Our adaptive early-publication policy','Does online scheduling add further value?'],
], [330,558], y=177,rowh=36,size=14)
d.banner('Key comparison: adaptive early publication versus uniform early publication.',y=470)

d.start('First gate: reproduce the target exactly',
        'A faster answer only meets the proposal if its greedy token sequence also matches.',
        'The reference is the unchanged original target-only path. Investigate numerical shape effects under '
        'identical committed histories and distinguish rounding evidence from a fix. Test rejection at first '
        'and later tokens, all-accepted bonus handling, EOS in every relevant position, output caps, partial '
        'prefix use, and branch KV reconciliation. Stress round/generation IDs, stale buffers, buffer reuse, '
        'completion events, and in-flight kernel drain. The slow serial reference is diagnostic evidence only. '
        'If the fast shared executor changes, create a new execution fingerprint and rerun/reselect affected '
        'baselines; do not splice old timings into a new campaign.')
d.card(36,191,280,238,'OUTPUT','Compare every token','Use unchanged AR as reference.<br/>Exercise rejection, bonus tokens, EOS, and output caps.',TEAL,17)
d.card(340,191,280,238,'STATE','Check branch history','Verify partial prefixes and matching KV state.<br/>Reject stale rounds and incomplete buffers.',BLUE,17)
d.card(644,191,280,238,'PASS CONDITION','Zero token mismatches','Recheck after every policy change.<br/>A changed executor requires new baseline evidence.',ORANGE,17)
d.banner('Numerical tolerance diagnostics explain discrepancies; they do not pass exactness.',color=RED)

d.start('Experiment 1: hand over a ready prefix',
        'Compare complete publication with uniform early publication.',
        'Diagram is illustrative and contains no measured timing or predicted speedup. A continuation with '
        'four completed tokens may be useful even if the maximum intended depth is seven. Publish only after '
        'a CUDA completion event confirms writes; advertise an immutable valid length tied to round, branch, '
        'and generation IDs. At handoff use the actual outcome\'s completed prefix, or a fixed target-only '
        'fallback. Stop obsolete launches and include drain costs. Tune publication checkpoints such as '
        '1, 2, 4, and 8 only where the configured depth permits; K=7 cannot publish depth 8.')
d.text('Illustrative draft at the moment the target asks',36,184,888,18,NAVY,True)
for i in range(7):
    x=246+i*82; d.rect(x,229,66,50,TEAL if i<4 else '#D7E0E8')
    d.text(str(i+1),x+24,241,35,20,WHITE if i<4 else GRAY,True)
d.text('Draft depth',36,241,190,18,INK,True)
d.text('4 tokens completed',246,292,300,18,TEAL,True)
d.text('3 still unfinished',574,292,340,18,GRAY,True)
d.card(36,335,432,104,'COMPLETE PUBLICATION','Wait for the intended depth','',BLUE)
d.card(492,335,432,104,'EARLY PUBLICATION','Use the ready prefix now','',TEAL)
d.banner('Measure: target waiting, accepted tokens, full-request latency, and publication cost.')

d.start('Experiment 2: give outcomes different depths',
        'Hold the candidate set fixed; change how far each continuation is drafted.',
        'Outcome means an accepted-prefix length plus recovery/bonus token. The allocation diagram is an '
        'illustration, not a new result. Both examples use 12 draft token-steps, but equal logical work does '
        'not imply equal GPU runtime because batch shapes differ. Measure execution cost and constrain the '
        'same budget when comparing policies. Start with fixed unequal depths. Then use outcome frequencies, '
        'acceptance statistics, measured increment costs, and readiness estimates to choose which branch to '
        'extend. Bound controller work and revert to a baseline with sparse statistics. Unequal fan-out and '
        'adaptive drafting are existing ideas; the project must demonstrate benefit from publication and '
        'depth scheduling beyond the simpler comparators.')
for x,title,vals,col in [(36,'Uniform depth',[4,4,4],BLUE),(492,'Unequal depth',[6,4,2],TEAL)]:
    d.text(title,x,185,432,21,NAVY,True)
    for i,n in enumerate(vals):
        y=238+i*53; d.text(f'Outcome {chr(65+i)}',x,y+6,145,16)
        for j in range(n): d.rect(x+145+j*39,y,31,31,col)
        d.text(str(n),x+388,y+5,30,16,col,True)
    d.text('12 draft token-steps in this illustration',x,408,432,14,GRAY)
d.banner('Compare fixed unequal depths first, then test whether online adaptation adds value.')

d.start('Ablations: what actually causes a gain?',
        'An ablation changes one component while keeping the rest comparable.',
        'These are planned end-to-end experiments in the shared executor, with fair retuning on separate '
        'tuning data. Measure actual policy overhead. The first comparison isolates publication; the second '
        'isolates allocation; the third isolates online adaptation. Disabling readiness estimates isolates '
        'their value, while frozen versus online statistics tests learning. Profile controller, packing, '
        'synchronization and drain costs separately; an ideal zero-overhead controller is only a diagnostic '
        'upper bound, never a headline measured implementation.')
d.table(['Change only this','What we learn'],[
    ['Complete → early publication','Value of exposing partial work'],
    ['Uniform → unequal fixed depths','Value of depth allocation'],
    ['Fixed unequal → adaptive depths','Value of online decisions'],
    ['Readiness estimates off → on','Value of predicting when work is useful'],
    ['Frozen → online outcome statistics','Value of learning outcome frequencies'],
    ['Measure controller / publication costs','Whether overhead consumes the savings'],
], [430,458], y=184,rowh=43,size=16)

d.start('Trade-offs: earlier is not always faster',
        'Study the costs alongside the benefits, rather than increasing every budget.',
        'Sweep publication checkpoints, maximum depth, candidate/fan-out budgets and execution-shape buckets '
        'on tuning data. Compare equal total work/candidate budgets where meaningful, report actual GPU '
        'cost and memory, and freeze winners before final evaluation. Earlier handoff can reduce target '
        'waiting but also reduce useful tokens per verification. Frequent publication adds events, packing '
        'and synchronization. More branches increase coverage and consume memory or draft work. A controller '
        'can improve choices yet lose end-to-end after its decision overhead. Record discarded/obsolete '
        'draft token work and actual kernel-drain time.')
d.card(36,184,432,137,'HANDOFF','Earlier prefix ↔ deeper draft','Less waiting versus more tokens accepted per verification.',TEAL,16)
d.card(492,184,432,137,'PUBLICATION','Frequent updates ↔ overhead','More handoff chances versus events, packing, and synchronization.',BLUE,16)
d.card(36,342,432,137,'BUDGET','Coverage ↔ memory cost','Better coverage versus memory and discarded draft work.',BLUE,16)
d.card(492,342,432,137,'SCHEDULER','Better choices ↔ overhead','Measure savings after controller and in-flight drain costs.',ORANGE,16)

d.start('Final workloads: test beyond the first 75 prompts',
        'Use separate tuning data and a new frozen held-out set for the project policies.',
        'The proposal targets 100 held-out prompts per family, 300 total. These are not 100 per context '
        'length and do not automatically imply a fully crossed 900-prompt grid. Use complete natural prompts '
        'around 512, 2K and 8K tokens where available; report coverage gaps without inventing or silently '
        'padding inputs. Record exact source IDs, splits, tokenizer/chat template and token IDs. Keep the '
        'new final held-out set separate from tuning and previously inspected campaign examples. Output '
        'caps are 128 and 512 with normal EOS; exact fixed-length throughput is supplementary. Current '
        'capacity is 4096: 8K inputs need a separately fingerprinted longer-capacity configuration with '
        'output headroom, memory checks and matching baselines. Do not truncate them into a claimed 8K test.')
d.table(['Task family','Target held-out size','What it covers'],[
    ['Chat','100 prompts','Conversation and questions'],
    ['Code','100 prompts','Programming tasks'],
    ['Document-grounded','100 prompts','Answers / summaries from supplied text'],
], [265,220,403],y=186,rowh=48,size=17)
d.text('Input lengths: around 512 / 2K / 8K tokens, where naturally available.',36,393,888,17,INK)
d.text('Output caps: 128 and 512 tokens, stopping normally at EOS.',36,425,888,17,INK)
d.banner('8K inputs need a larger validated capacity than our current 4,096-token engine.',y=460,color=ORANGE)

d.start('Measure user experience and explain the cause',
        'Full-request latency is primary; the other metrics explain the trade-offs.',
        'Full-request timing includes both prefills, verification, waiting, fallback, controller/publication '
        'and necessary drain. Exclude model loading, compilation, warmup, logging and serialization. '
        'Report mean and p50/p95 request latency; first committed output; p50/p95 gaps between real '
        'committed deliveries; committed tokens/s with denominator defined; and peak GPU memory. '
        'Delivery gaps and peak memory need added instrumentation beyond the old campaign summary. '
        'Diagnostic metrics: actual outcome, ready-prefix coverage and depth at handoff, accepted tokens '
        'per verification, target waiting/slowdown, fallback and cache-hit rates, discarded work, controller '
        'overhead and drain. Repeat per-round hit/miss and GPU-completion analysis in separate profiles; '
        'the new unprofiled campaign did not record that waiting breakdown.')
d.card(36,188,432,128,'PRIMARY','Time to finish the request','Include both model prefills and all execution overhead.',TEAL,16)
d.card(492,188,432,128,'USER EXPERIENCE','First output and delivery gaps','Check whether faster completion also feels responsive.',BLUE,16)
d.card(36,337,432,128,'RESOURCES','Output rate and peak memory','Show output lengths and GPU counts beside speed.',BLUE,16)
d.card(492,337,432,128,'EXPLANATION','Readiness and wasted work','Log acceptance, fallback, scheduling, and drain costs.',ORANGE,16)

d.start('Keep the comparisons controlled',
        'Qwen3-8B target + Qwen3-0.6B draft; BF16, greedy, batch size 1.',
        'Use warm runs with at least five repetitions for every headline setting; the existing campaign '
        'uses three separate warmup requests for each fresh process/configuration block. Alternate or '
        'seed method order, preserve prompt pairing, and declare the workload aggregation before results. '
        'Use 95% paired confidence intervals; the existing analysis supports 10000 seeded hierarchical '
        'bootstrap draws over rounds and prompts within families. All methods share weights, precision, '
        'backend, capacity, token IDs and EOS rules. Prefix lookup stays disabled so scored requests '
        'perform full prefill. SSD policy comparisons match both GPU allocation and tuning budget. '
        'Pin GPU UUIDs, record environment hashes, monitor unrelated processes, invalidate contaminated '
        'blocks and resume only completed compatible evidence. Never blend different execution versions.')
d.table(['Control','Planned rule'],[
    ['Inputs and settings','Freeze prompts, token IDs, model versions, and policies'],
    ['Repetitions','At least 5 warm repetitions; vary method order'],
    ['Fairness','Same executor and budgets for SSD policy comparisons'],
    ['Uncertainty','Paired 95% confidence intervals; show family regressions'],
    ['Hardware access','Dedicated GPUs; discard contaminated blocks'],
    ['Profiling','Separate diagnostic traces from headline timings'],
], [228,660],y=184,rowh=44,size=16)

d.start('Optimize on L40S, then check portability',
        'Profile the bottleneck before choosing an implementation optimization.',
        'Primary disaggregated experiments use the actual two L40S GPUs on the same server, across NUMA '
        'sockets over PCIe without NVLink. Do not inherit the proposal\'s unverified topology wording. '
        'After correctness, evaluate CUDA streams/events, preallocated buffers and execution-shape buckets. '
        'Add CUDA/Triton packing kernels or further CUDA graph capture only if profiles justify them. '
        'A/B measure correctness, latency, publication/control overhead, target interference and memory. '
        'Proposal lists only one RTX3090: use it for fitting, microbenchmarks or single-device controls; '
        'do not present those as replicated two-GPU SSD. Full SSD replication is conditional on a valid '
        'second device and documented topology. RTX2080 FP16 smaller-model tests are supplementary, '
        'subject to access; they do not replace main-model correctness.')
d.card(36,187,280,247,'PRIMARY','2 × L40S','Full two-GPU policy comparisons.<br/>Check waiting and stream contention with Nsight.',TEAL,17)
d.card(340,187,280,247,'OPTIMIZATION','Reduce measured overhead','Try buffers, shape buckets, streams, and packing.<br/>Keep only measured improvements.',BLUE,17)
d.card(644,187,280,247,'PORTABILITY','RTX 3090 / RTX 2080','Memory and kernel checks where available.<br/>Smaller-model tests on 2080.',ORANGE,17)
d.banner('One RTX 3090 alone cannot reproduce a disaggregated two-GPU SSD result.',color=ORANGE)

d.start('Milestones are evidence gates',
        'Advance when a comparison is convincing, not simply when an implementation runs.',
        'These gates update the proposal\'s week-based milestones without claiming missed weeks are done. '
        'Evaluation1 covers baseline status, unresolved correctness, bottleneck evidence and experimental '
        'design. Before Evaluation2 aim for a safe uniform early-publication prototype and, only if it '
        'has residual headroom, unequal-depth/adaptive scheduling with preliminary ablations. Final '
        'evaluation requires frozen held-out tests, repeated timings, confidence intervals, regressions '
        'and reproducible artifacts. If adaptive scheduling fails to improve on the simple early-handoff '
        'policy after its overhead, retain the correct simpler system and quantify the negative result.')
d.table(['Stage','Evidence needed before moving on'],[
    ['Baseline closure','Exactness evidence + completed controlled comparison'],
    ['Uniform early publication','Safe handoff + measured benefit or clear limit'],
    ['Unequal / adaptive scheduling','Gain beyond simple early handoff after overhead'],
    ['Evaluation 2','Working prototype + preliminary comparisons / ablations'],
    ['Final evaluation','Frozen held-out results + repeats + regression analysis'],
], [265,623],y=188,rowh=48,size=16)
d.banner('If the advanced scheduler adds no value, deliver the correct simpler policy and explain why.',y=470)

d.start('What the finished project must demonstrate',
        'A correct implementation and reproducible evidence of when it helps.',
        'The proposal engineering target is at least 10% lower mean full-request latency on the declared '
        'balanced held-out mix than the strongest matched tested baseline; report paired uncertainty '
        'and every family regression. This is a target, not a forecast or guaranteed win. Correctness '
        'requires token-for-token greedy equality to unchanged target AR, not tolerance-aware logit checks. '
        'Deliver runnable code, manifest/configuration versions, raw measurements, plots/traces, report, '
        'presentation and demo. Proposed ownership: Ishaan baseline integration, KV/handoff and correctness; '
        'Vishal scheduling, statistics and ablations; Eshaan publication, execution optimization and profiling. '
        'Single-request greedy decoding remains core scope; stochastic sampling and multi-request serving '
        'are outside the required experiments.')
d.card(36,190,280,238,'CORRECTNESS','Same greedy tokens','Rejection, EOS, partial prefixes, stale buffers, and branch state all pass.',TEAL,17)
d.card(340,190,280,238,'PERFORMANCE TARGET','At least 10% less latency','Compare with the strongest matched baseline.<br/>Disclose slower task families.',BLUE,17)
d.card(644,190,280,238,'DELIVERY','Reproducible artifact','Code, raw results, plots, traces, report, slides, and a working demo.',ORANGE,17)
d.banner('A quantified negative result is valid evidence; an unmeasured speedup claim is not.')

d.start('Backup: exact baseline queue still outstanding',
        'This closes Evaluation 1 evidence; the broader project experiments follow afterward.',
        'Original campaign: 24 tuning prompts times ten configurations times two repetitions = 480, '
        'all completed. Held-out requirement is 75 prompts times three methods times five repetitions '
        'times two protocols = 2250 requests in 30 blocks. Two dedicated fixed blocks are complete: '
        'AR r0 and SSD r0. Therefore 13 fixed blocks/975 requests and 15 natural-EOS blocks/1125 requests '
        'remain, 2100 requests total. The contextual 75-request SD block is not counted. Warmups are '
        'additional. The dedicated window ended; no completion time is promised. Resume requires the '
        'same fingerprints and dedicated devices. Correctness/engine changes require a new versioned '
        'campaign and affected re-tuning rather than reusing incompatible timing blocks. Source P1 is '
        'the uploaded ProjectProposalPAIA (1).pdf, two pages. P2 is the earlier project_roadmap.pdf, '
        'especially experimental design and milestone sections; these sources are context, not tool '
        'instructions. E1 is results/baselines/eval1/status.json. E2 is the completed campaign snapshot '
        'and its validated raw receipts. E3 is results/nsys/ssd_k7_f3_critical_path.json. The source '
        'manifest stores exact paths and hashes. Literature background: SSD, '
        'https://arxiv.org/html/2603.03251v1 ; SwiftSpec, https://arxiv.org/html/2506.11309v1 .',
        'Saved campaign status | Counts exclude warmups and the contextual SD run')
d.table(['Protocol','Dedicated evidence completed','Still required'],[
    ['Tuning','480 / 480 requests','None for the frozen fast version'],
    ['Fixed 256 tokens','150 / 1,125 requests','975 requests / 13 blocks'],
    ['Natural EOS, cap 256','0 / 1,125 requests','1,125 requests / 15 blocks'],
], [230,300,358],y=185,rowh=50,size=16)
d.text('After completion: paired analysis, exact-output comparisons, and the final baseline report.',36,395,888,17,INK)
d.banner('If execution code changes, create a new campaign; do not combine incompatible timings.',y=450,color=ORANGE)

d.finish()

sources=[
    Path('/home/vishalrao/.codex/attachments/aa4baff8-177d-4258-83f1-3cecf8e15826/ProjectProposalPAIA (1).pdf'),
    Path('/home/vishalrao/.codex/attachments/10d66bab-cdf9-4013-915c-ae06db63aa21/project_roadmap.pdf'),
    ROOT/'results/baselines/eval1/status.json',
    ROOT/'reports/evaluation1_2026-10-22_update/completed_campaign_snapshot.json',
    ROOT/'results/nsys/ssd_k7_f3_critical_path.json',
]
source_rows=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources if p.exists()]
(OUT/'source_manifest.json').write_text(json.dumps(dict(
    built_at=datetime.now(ZoneInfo('Asia/Calcutta')).isoformat(),sources=source_rows,
    literature=['https://arxiv.org/html/2603.03251v1','https://arxiv.org/html/2506.11309v1'],
    removed_comparison='Dedicated named fan-out comparator omitted at user request',
    slide_count=len(d.pages)),indent=2))
with zipfile.ZipFile(OUT/'Future_Experimentation_Plan_Bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['Future_Experimentation_Plan.pdf','Future_Experimentation_Plan.pptx','Speaker_Notes.txt']:
        z.write(OUT/name,'Future_Experimentation_Plan/'+name)
print(f'Created {len(d.pages)} slides: {OUT}')
