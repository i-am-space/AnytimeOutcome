"""Build all Evaluation 2 artifacts from the frozen snapshot. No GPU work."""
from pathlib import Path
import json,re,html
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,white
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image
from content import PAGES,QA
from slides_content import SLIDES
from pptx_writer import write_pptx,embed_notes
OUT=Path(__file__).resolve().parent
M=json.loads((OUT/'evidence_manifest.json').read_text());A=json.loads((OUT/'analysis.json').read_text());FM=json.loads((OUT/'figure_manifest.json').read_text())
def load(k):return json.loads((OUT/'evidence_snapshot'/M['sources'][k]).read_text())
B,P=load('baseline'),load('publication')
V={'depth_blocks':A['depth_valid_blocks'],'depth_exact':A['depth_heldout_exact']}
for protocol in ['fixed','natural']:
 for mode in ['ar','sd','ssd']:
  r=next(r for r in B['results'] if (r['protocol'],r['mode'],r['family'])==(protocol,mode,'all'));st=r['statistics'];V[f'{mode}_{protocol}']=st['mean_latency_s']
  for short,full in [('p50','p50_latency_s'),('p95','p95_latency_s'),('tps','committed_tok_per_s')]:V[f'{mode}_{protocol}_{short}']=st[full]
  V[f'{mode}_{protocol}_first_ms']=st['mean_first_delivery_s']*1000
for method in ['complete','uniform','overlap']:
 V[method+'_pub']=next(r for r in P['results'] if (r['protocol'],r['method'],r['family'])==('fixed',method,'all'))['statistics']['mean_latency_s']
def fmt(s):return s.format_map(V)
for score in load('baseline_selection')['scores']:
 cfg=score['config'];key='tune_'+cfg['mode']
 if cfg['k'] is not None:key+='_k'+str(cfg['k'])
 if cfg['f'] is not None:key+='_f'+str(cfg['f'])
 V[key]=score['mean_latency_s']
NAVY='#152E49';INK='#26394B';TEAL='#008C88';BLUE='#315F9E';ORANGE='#D78B27';GRAY='#64788D';LIGHT='#EDF3F7';RED='#B74E59'
for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/dejavu/'+file))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
def para(text,size=10.6,color=INK,bold=False,leading=None):return Paragraph(escape(fmt(text)).replace('\n','<br/>'),ParagraphStyle('p',fontName='Bold' if bold else 'Body',fontSize=size,leading=leading or size*1.38,textColor=HexColor(color),spaceAfter=0))
def drawp(c,text,x,y,w,size=10.6,color=INK,bold=False):
 p=para(text,size,color,bold);_,h=p.wrap(w,2000);p.drawOn(c,x,y-h);return h

def footer(c,n,label,w,h):
 c.setStrokeColor(HexColor('#D6E0E8'));c.line(44,43,w-44,43)
 c.setFont('Body',8);c.setFillColor(HexColor(GRAY));c.drawString(44,28,label+' · frozen '+M['captured_at_ist'][:16].replace('T',' ')+' IST');c.drawRightString(w-44,28,f'{n:02d}')
def drawtable(c,rows,x,y,w,size=10):
 rows=[[para(str(cell),size,"#FFFFFF" if i==0 else INK,i==0) for cell in row] for i,row in enumerate(rows)]
 cols=len(rows[0]);widths=[w/cols]*cols
 if cols==2:widths=[w*.34,w*.66]
 t=Table(rows,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor(NAVY)),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor(LIGHT),white]),('LINEBELOW',(0,0),(-1,0),.4,HexColor('#CAD6E0'))]));_,hh=t.wrap(w,2000);t.drawOn(c,x,y-hh);return hh

def guide():
 c=canvas.Canvas(str(OUT/'Evaluation2_Learning_Guide.pdf'),pagesize=A4);c.setTitle('Evaluation 2 | Learning guide | Three study routes');c.setAuthor('Ishaan Romil; Vishal Rao; Eshaan Sharma')
 W,H=A4;w=W-88;bounds=[];text=[]
 for n,p in enumerate(PAGES,1):
  c.bookmarkPage(f'page{n}');c.addOutlineEntry(f'{n}. '+p['title'],f'page{n}',0)
  c.setFillColor(HexColor(TEAL));c.rect(0,H-9,W,9,fill=1,stroke=0)
  y=H-42;y-=drawp(c,p['topic'],44,y,w,9,TEAL,True)+12;y-=drawp(c,p['title'],44,y,w,23,NAVY,True)+15
  # Measure the complete page before drawing its body.
  chosen=None
  for size in [10.6,10.3,10.0]:
   fig_h=155 if p['figure'] else 0
   total=0
   for heading,body in p['body']:
    for txt,sz,bold in [(heading,size+.5,True),(body,size,False)]:total+=para(txt,sz,bold=bold).wrap(w,2000)[1]+(4 if bold else 9)
   if p['table']:
    # table height calculation uses a throwaway object, never draws it.
    rr=[[para(str(cell),9.4) for cell in row] for row in p['table']];cols=len(rr[0]);widths=[w/cols]*cols
    if cols==2:widths=[w*.34,w*.66]
    t=Table(rr,colWidths=widths);t.setStyle(TableStyle([('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8)]));total+=t.wrap(w,2000)[1]+14
   if p['figure']:total+=fig_h+para(FM[p['figure']]['caption'],8.5,GRAY).wrap(w,2000)[1]+14
   tail=para('Takeaway: '+p['takeaway'],10.2,bold=True).wrap(w-20,2000)[1]+20+para('Self-check: '+p['check'],9.7,TEAL).wrap(w,2000)[1]+14+para('Evidence: '+p['sources'],7.7,GRAY).wrap(w,2000)[1]
   if total+tail<=y-78:chosen=(size,fig_h);break
  if chosen is None:raise ValueError(f'Learning page {n} would overflow: {p["title"]}')
  size,fig_h=chosen
  if p['figure']:
   path=OUT/'figures'/(p['figure']+'.png');im=Image.open(path);iw,ih=im.size;scale=min(w/iw,fig_h/ih);dw,dh=iw*scale,ih*scale;c.drawImage(str(path),44+(w-dw)/2,y-dh,width=dw,height=dh);y-=dh+7;y-=drawp(c,FM[p['figure']]['caption'],44,y,w,8.5,GRAY)+12
  for heading,body in p['body']:
   y-=drawp(c,heading,44,y,w,size+.5,NAVY,True)+4;y-=drawp(c,body,44,y,w,size)+9
  if p['table']:y-=drawtable(c,p['table'],44,y,w,9.4)+14
  th=para('Takeaway: '+p['takeaway'],10.2,bold=True).wrap(w-20,2000)[1]+18
  c.setFillColor(HexColor(LIGHT));c.roundRect(44,y-th,w,th,5,fill=1,stroke=0);drawp(c,'Takeaway: '+p['takeaway'],54,y-9,w-20,10.2,BLUE,True);y-=th+12
  y-=drawp(c,'Self-check: '+p['check'],44,y,w,9.7,TEAL)+12;y-=drawp(c,'Evidence: '+p['sources'],44,y,w,7.7,GRAY)
  if y<55:raise ValueError(f'Guide bottom overflow p{n}: {y}')
  bounds.append({'page':n,'bottom_y':y,'body_font_pt':size});footer(c,n,'LEARNING GUIDE',W,H);c.showPage()
  text.extend([f'PAGE {n} · '+p['title'],p['topic'],*sum(([h,fmt(b)] for h,b in p['body']),[]),fmt(p['takeaway']),fmt(p['check']),''])
 c.save();(OUT/'Learning_Guide_Text.txt').write_text('\n'.join(text));return bounds

class Deck:
 def __init__(self):self.pages=[];self.c=canvas.Canvas(str(OUT/'Evaluation2_Presentation.pdf'),pagesize=(960,540));self.c.setTitle('Anytime Outcome-Cache Construction | Evaluation 2');self.c.setAuthor('Ishaan Romil; Vishal Rao; Eshaan Sharma');self.bounds=[]
 def rect(self,x,y,w,h,col):self.c.setFillColor(HexColor(col));self.c.rect(x,540-y-h,w,h,fill=1,stroke=0);self.scenes.append(dict(kind='rect',x=x,y=y,w=w,h=h,color=col))
 def text(self,text,x,y,w,size=19,color=INK,bold=False):
  text=fmt(text);p=para(text,size,color,bold);_,h=p.wrap(w,3000)
  if x<0 or y<0 or x+w>961 or y+h>536:raise ValueError(f'Slide {len(self.pages)} text overflow: {text[:50]} bottom={y+h}')
  p.drawOn(self.c,x,540-y-h);self.scenes.append(dict(kind='text',text=text,x=x,y=y,w=w,h=h+2,size=size,color=color,bold=bold));self.bounds.append({'slide':len(self.pages),'text':text,'top':y,'bottom':y+h,'font_pt':size});return h
 def image(self,name,x,y,w,h):
  path=OUT/'figures'/(name+'.png');im=Image.open(path);iw,ih=im.size;s=min(w/iw,h/ih);dw,dh=iw*s,ih*s;x+=(w-dw)/2;y+=(h-dh)/2
  self.c.drawImage(str(path),x,540-y-dh,width=dw,height=dh);self.scenes.append(dict(kind='image',path=str(path),x=x,y=y,w=dw,h=dh))
 def table(self,rows,x,y,w,size=16):
  cols=len(rows[0]);cw=w/cols
  widths=[cw]*cols
  if cols==2:widths=[w*.34,w*.66]
  for i,row in enumerate(rows):
   sizes=[self.measure(str(v),widths[j]-18,size) for j,v in enumerate(row)];hh=max(sizes)+12
   self.rect(x,y,w,hh,NAVY if i==0 else LIGHT if i%2 else '#FFFFFF')
   xx=x
   for j,v in enumerate(row):self.text(str(v),xx+9,y+6,widths[j]-18,size,'#FFFFFF' if i==0 else INK,i==0);xx+=widths[j]
   y+=hh
  return y
 def table_height(self,rows,w,size):
  cols=len(rows[0]);widths=[w/cols]*cols
  if cols==2:widths=[w*.34,w*.66]
  return sum(max(self.measure(str(v),widths[j]-18,size) for j,v in enumerate(row))+12 for row in rows)
 def measure(self,text,w,size):return para(fmt(text),size).wrap(w,3000)[1]
 def build(self):
  for i,s in enumerate(SLIDES,1):
   if i>1:self.c.showPage()
   self.scenes=[];notes=f"{'MAIN' if i<=11 else 'BACKUP'} SLIDE {i} | {s['seconds']} seconds\n\nSAY\n{fmt(s['say'])}\n\nTECHNICAL DETAIL\n{fmt(s['technical'])}\n\nTRANSITION\n{fmt(s['transition'])}\n\nEVIDENCE\n{s['source']}"
   self.pages.append(dict(title=s['title'],scenes=self.scenes,notes=notes))
   self.rect(0,0,960,7,TEAL);self.text('ANYTIME OUTCOME-CACHE CONSTRUCTION',34,19,650,10,TEAL,True)
   self.text(s['title'],34,43,890,29,NAVY,True);self.text(s['subtitle'],34,126,890,14.5,GRAY)
   if s['figure']:
    self.image(s['figure'],30,180,565,278)
    bs=next((z for z in [18.5,18,17.5,17,16.5,16] if sum(self.measure(b,298,z)+14 for b in s['bullets'])<=279),None)
    if bs is None:raise ValueError(f'Slide {i} needs shorter bullets')
    yy=180
    for b in s['bullets']:
     self.rect(609,yy+7,5,5,TEAL);hh=self.text(b,626,yy,298,bs);yy+=hh+14
    if yy>459:raise ValueError(f'Slide {i} bullet overflow {yy}')
   elif s['table']:
    if i in (4,11):
     fs=next(z for z in [14.5,14,13.5,13,12.5] if self.table_height(s['table'],558,z)<=274)
     self.table(s['table'],34,182,558,fs);yy=181
     bs=next(z for z in [17.5,17,16.5,16] if sum(self.measure(b,299,z)+15 for b in s['bullets'])<=278)
     for b in s['bullets']:yy+=self.text(b,625,yy,299,bs)+15
    elif i==7:
     fs=next(z for z in [16,15,14.5,14,13.5] if self.table_height(s['table'],510,z)<=272)
     self.table(s['table'],34,184,510,fs);yy=184
     for b in s['bullets']:yy+=self.text(b,574,yy,350,17.5)+17
    else:
     extra=sum(self.measure(b,890,14.5)+8 for b in s['bullets'])+(12 if s['bullets'] else 0)
     fs=next((z for z in [16,15,14.5,14,13,12.5,12] if self.table_height(s['table'],890,z)+extra<=274),None)
     if fs is None:raise ValueError(f'Slide {i}: table needs less text')
     end=self.table(s['table'],34,185,890,fs)
     if s['bullets']:
      yy=end+12
      for b in s['bullets']:yy+=self.text(b,34,yy,890,14.5)+8
   else:
    if i==1:
     self.text('WHAT WE BUILT',34,186,280,13,TEAL,True);self.text('WHAT WE LEARNED',350,186,280,13,BLUE,True);self.text('WHAT REMAINS',666,186,260,13,ORANGE,True)
     for j,b in enumerate(s['bullets']):self.rect(34+j*316,219,288,215,LIGHT);self.text(b,49+j*316,239,255,23,NAVY,j==2)
     self.text('Ishaan Romil · Vishal Rao · Eshaan Sharma',34,447,890,11,GRAY)
   self.rect(34,469,890,30,LIGHT);self.text(s['takeaway'],44,476,870,12.4,BLUE,True)
   self.text(s['status']+' · '+('MAIN TALK' if i<=11 else 'Q&A BACKUP'),34,506,750,8.6,TEAL if 'PROVISIONAL' not in s['status'] else ORANGE,True)
   self.text(f'{i:02d} / 21'+(f' · {s["seconds"]}s' if s['seconds'] else ''),825,506,100,9,GRAY)
   self.text('Evidence: '+s['source'],34,521,890,7.1,GRAY)
  self.c.save();write_pptx(self.pages);embed_notes(self.pages)
  def bind(x):
   if isinstance(x,str):return fmt(x)
   if isinstance(x,list):return [bind(v) for v in x]
   if isinstance(x,dict):return {k:bind(v) for k,v in x.items()}
   return x
  (OUT/'deck_content.json').write_text(json.dumps(bind(SLIDES),indent=2)+'\n')
  return self.bounds

def notes():
 c=canvas.Canvas(str(OUT/'Evaluation2_Speaker_and_QA_Guide.pdf'),pagesize=A4);c.setTitle('Evaluation 2 | Speaker notes and Q&A');W,H=A4;w=W-88;n=0
 def start(title,sub):
  nonlocal n
  n+=1;c.bookmarkPage('note'+str(n));c.addOutlineEntry(title,'note'+str(n),0);c.setFillColor(HexColor(TEAL));c.rect(0,H-9,W,9,fill=1,stroke=0);y=H-45;y-=drawp(c,title,44,y,w,22,NAVY,True)+13;y-=drawp(c,sub,44,y,w,10.3,GRAY)+20;return y
 def finish():footer(c,n,'SPEAKER + Q&A GUIDE',W,H);c.showPage()
 y=start('Your ten-minute presentation','11 main slides · 600 seconds · 10 backup slides for questions')
 for h,b in [('How to rehearse','Give the short conclusion first. For each figure: identify its baseline, state what is fixed, read the direction and magnitude, then name the limitation. The spoken scripts are starting points: pause to point to the plotted evidence.'),('Safe central sentence','We built safe incremental publication and unequal-depth drafting, established reproducible baselines, and measured when these mechanisms help or add overhead. Beating tuned complete SSD remains unestablished.'),('Questions to expect','Why no speedup? Why K7 versus K10? Why does numerical equivalence use a new AR? Does reduced work help? How were timing and GPU allocation controlled? The five Q&A pages provide 25 short answers.')]:
  y-=drawp(c,h,44,y,w,12,TEAL,True)+5;y-=drawp(c,b,44,y,w,11)+15
 y-=drawtable(c,[['Rubric criterion','Where the talk provides evidence'],['Implementation & progress · 8','Slides 2–3, 7, 10; ownership/control backups'],['Initial results & analysis · 8','Slides 5–10; measured plots and limits'],['Comparisons / tradeoffs / ablations · 8','Slides 4, 8–11; matched K and six-arm controls'],['Presentation & technical clarity · 6','Timed narrative, explicit units/reference, readable figures']],44,y,w,10)+12;finish()
 for i,s in enumerate(SLIDES[:11],1):
  y=start(f'Slide {i} · {s["title"]}',f'{s["seconds"]} seconds · cumulative {sum(x["seconds"] for x in SLIDES[:i])//60}:{sum(x["seconds"] for x in SLIDES[:i])%60:02d} · {s["status"]}')
  if s['figure']:
   p=OUT/'figures'/(s['figure']+'.png');im=Image.open(p);iw,ih=im.size;sc=min(w/iw,150/ih);dw,dh=iw*sc,ih*sc;c.drawImage(str(p),44+(w-dw)/2,y-dh,width=dw,height=dh);y-=dh+15
  for h,b in [('Say this',s['say']),('Technical detail if asked',s['technical']),('Transition',s['transition']),('Evidence',s['source'])]:
   if not b:continue
   y-=drawp(c,h,44,y,w,11,TEAL,True)+5;y-=drawp(c,b,44,y,w,10.6)+15
  if y<55:raise ValueError(f'Speaker page {n} overflow {y}')
  finish()
 for group in range(5):
  y=start(f'Q&A practice · {group+1}/5','Answer directly, then give one measurement and one limitation.')
  for j,(q,a,rubric) in enumerate(QA[group*5:(group+1)*5],group*5+1):
   y-=drawp(c,f'{j}. '+q,44,y,w,11,TEAL,True)+5;y-=drawp(c,a,44,y,w,10.6)+16
  if y<55:raise ValueError('Q&A overflow')
  finish()
 c.save();return n
if __name__=='__main__':
 g=guide();d=Deck();bounds=d.build();npages=notes();(OUT/'layout_validation.json').write_text(json.dumps({'guide':g,'slides':bounds,'speaker_pages':npages,'main_seconds':sum(s['seconds'] for s in SLIDES)},indent=2)+'\n');print('Built 45 learning pages, 21 slides,',npages,'speaker/Q&A pages')
