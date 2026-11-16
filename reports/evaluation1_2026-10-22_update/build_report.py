"""Build the Evaluation 1 evidence report and slide deck from saved results only."""
from pathlib import Path
import hashlib, json, math, re, zipfile
from xml.sax.saxutils import escape
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image
import campaign_additions as additions

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
DATE='22 October 2026'
NAVY='#152E49'; TEAL='#008C88'; ORANGE='#D98225'; RED='#C64D56'; BLUE='#4778AE'; GRAY='#697B8E'; LIGHT='#EFF4F8'; INK='#26394B'; PURPLE='#8065A8'
COLORS=[GRAY,TEAL,BLUE]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#B5C0CA','xtick.color':INK,'ytick.color':INK,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf'),('Italic','DejaVuSans-Oblique.ttf')]:
 pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/dejavu/'+file))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Italic',boldItalic='Bold')

def read(path):return json.loads((ROOT/path).read_text())
runs={m:read('results/nsys/'+n+'_run.json') for m,n in [('AR','ar'),('SD','sd_k6'),('SSD','ssd_k7_f3')]}
nsys={m:read('results/nsys/'+n+'_analysis.json') for m,n in [('AR','ar'),('SD','sd_k6'),('SSD','ssd_k7_f3')]}
roof=read('results/roofline/machine_roofline.json')
widths=read('results/roofline/forward_vs_tokens.json')
critical=read('results/nsys/ssd_k7_f3_critical_path.json')
ncu={n:read('results/ncu/'+n+'_ncu.json') for n in ['ar_decode','sd_verify','sd_draft','ssd_tree','ar_prefill']}
correct=read('results/correctness/summary.json')
backends=read('results/correctness/attention_backends/summary.json')
manifest=json.loads((OUT/'campaign_snapshot/manifest.json').read_text())
completed=json.loads((OUT/'completed_campaign_snapshot.json').read_text())
selection=json.loads((OUT/'campaign_snapshot/selection.json').read_text())
TUNING=completed['tuning']
CONTEXT=completed.get('contextual_sd')
SNAPTIME=completed['captured_at']
assert completed['tuning_requests']==480 and completed['heldout_requests']==150

def save(fig,name):
 fig.savefig(FIG/(name+'.png'),dpi=200,bbox_inches='tight');plt.close(fig)
 return str(FIG/(name+'.png'))

newfigs=additions.figures(completed,FIG)

# Each plot is derived directly from saved numeric evidence. No future scores are plotted.
fig,axs=plt.subplots(1,2,figsize=(9,4))
ms=list(runs)
throughput=[runs[m]['summary']['committed_tok_per_s'] for m in ms]
latency=[runs[m]['summary']['mean_latency_s'] for m in ms]
for ax,values,title,unit in [(axs[0],throughput,'Committed output / request time','tokens / second'),(axs[1],latency,'Full-request latency','seconds / request')]:
 bars=ax.bar(['AR\n1 GPU','SD K6\n1 GPU','SSD K7/F3\n2 GPUs'],values,color=COLORS,width=.6)
 ax.set_title(title,fontsize=12,pad=16);ax.set_ylabel(unit);ax.set_ylim(0,max(values)*1.25)
 for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+max(values)*.025,f'{v:.2f}',ha='center',fontweight='bold')
fig.tight_layout(w_pad=2);perfplot=save(fig,'profiled_performance')

fig,ax=plt.subplots(figsize=(8,4))
ids=[r['id'] for r in runs['AR']['requests']]
x=np.arange(3)
for i,m in enumerate(ms):
 rows={r['id']:r for r in runs[m]['requests']}
 ax.bar(x+(i-1)*.24,[rows[k]['latency_s'] for k in ids],width=.22,label=m+(' (2 GPUs)' if m=='SSD' else ' (1 GPU)'),color=COLORS[i])
ax.set(xticks=x,xticklabels=['Chat\nspecbench-148','Code\nHumanEval/36','Document\nspecbench-553'],ylabel='Full-request latency (seconds)')
ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.17));ax.set_ylim(0,7)
fig.tight_layout();familyplot=save(fig,'profiled_prompts')

fig,axs=plt.subplots(2,1,figsize=(8,5),sharex=True)
for ax,model,color in zip(axs,['qwen3-8b','qwen3-0.6b'],[BLUE,TEAL]):
 vals=sorted([r for r in widths if r['model']==model],key=lambda r:r['T'])
 ax.plot([r['T'] for r in vals],[r['linear_forward_us']/1000 for r in vals],marker='o',color=color)
 ax.axvline(roof['ridge_bf16_flop_per_byte_measured'],ls='--',color=ORANGE,label='Measured ridge ~283 FLOP/B')
 ax.set_xscale('log',base=2);ax.set_ylabel('Linear layers (ms)');ax.set_title(model,loc='left',fontsize=11)
 ax.set_ylim(bottom=0)
axs[0].legend(frameon=False,fontsize=9);axs[-1].set_xlabel('Tokens per forward (log scale)');fig.tight_layout();widthplot=save(fig,'forward_width')

fig,axs=plt.subplots(1,2,figsize=(8,4))
sd=nsys['SD']['nvtx_phases'];ss=nsys['SSD']['nvtx_phases']
a=sd['SD/speculate']['mean_ms'];b=sd['SD/verify']['mean_ms']
axs[0].barh(['SD engine step'],[a],color=TEAL,label='Serial drafting')
axs[0].barh(['SD engine step'],[b],left=[a],color=BLUE,label='Target verify')
axs[0].text(a/2,0,f'{a:.1f} ms',ha='center',va='center',color='white',fontweight='bold')
axs[0].text(a+b/2,0,f'{b:.1f} ms',ha='center',va='center',color='white',fontweight='bold')
axs[0].set_xlabel('Mean phase wall time (ms)');axs[0].legend(frameon=False,loc='upper left',bbox_to_anchor=(0,1.28));axs[0].set_xlim(0,55)
axs[1].bar(['Draft forward\nwall time','GPU kernel\nwork estimate'],[sd['SD/draft_fwd_*']['mean_ms'],2.42],color=[ORANGE,TEAL],width=.5)
axs[1].set_ylabel('Milliseconds');axs[1].set_ylim(0,4.5)
axs[1].text(.5,4.1,'~1.1 ms host / launch gap',ha='center',fontsize=10)
fig.tight_layout(w_pad=2);sdplot=save(fig,'sync_draft_cost')

fig,ax=plt.subplots(figsize=(8,4))
labels=['AR target','SD target + draft','SSD target','SSD draft']
useful=[];nccl=[];idle=[]
for method,gpu in [('AR','0'),('SD','0'),('SSD','0'),('SSD','1')]:
 d=nsys[method]['devices'][gpu];spin=100*d['category_ms'].get('nccl',0)/nsys[method]['window_ms']
 useful.append(d['busy_pct']-spin);nccl.append(spin);idle.append(100-d['busy_pct'])
for vals,left,color,label in [(useful,np.zeros(4),TEAL,'Kernel busy excluding NCCL'),(nccl,np.array(useful),ORANGE,'NCCL kernels (includes waiting)'),(idle,np.array(useful)+np.array(nccl),'#D7E0E8','Idle')]:
 ax.barh(labels,vals,left=left,color=color,label=label)
 for y,(v,l) in enumerate(zip(vals,left)):
  if v>6:ax.text(l+v/2,y,f'{v:.1f}%',ha='center',va='center',fontsize=10,color='white' if color==TEAL else INK)
ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Share of each profiled wall-time window (%)');ax.legend(loc='upper left',bbox_to_anchor=(0,-.19),frameon=False,fontsize=9,ncol=2)
fig.tight_layout();utilplot=save(fig,'gpu_time_states')

fig,axs=plt.subplots(1,2,figsize=(9,4))
c=critical['summary']; hit=c['mean_wait_ms_hit'];miss=c['mean_wait_ms_miss'];verify=c['mean_verify_ms']
axs[0].bar(['Cache hit\n166 rounds','Cache miss\n64 rounds'],[hit,miss],color=[TEAL,ORANGE],width=.55)
axs[0].axhline(verify,color=BLUE,ls='--',label=f'Verify: {verify:.1f} ms')
for i,v in enumerate([hit,miss]):axs[0].text(i,v+1,f'{v:.1f} ms',ha='center',fontweight='bold')
axs[0].set_ylabel('Target wait at handoff (ms)');axs[0].set_ylim(0,38);axs[0].legend(frameon=False,fontsize=9)
hist=c['ready_depth_at_request_hist'];depth=[int(k) for k in hist]
axs[1].bar([str(d) for d in depth],[hist[str(d)] for d in depth],color=[TEAL if d==5 else GRAY for d in depth],width=.6)
axs[1].set(xlabel='Completed draft depth when target asks',ylabel='Cache-hit rounds (n=166)')
for i,d in enumerate(depth):axs[1].text(i,hist[str(d)]+3,str(hist[str(d)]),ha='center',fontweight='bold')
axs[1].set_ylim(0,180);fig.tight_layout(w_pad=2);criticalplot=save(fig,'critical_path_evidence')

fig,ax=plt.subplots(figsize=(8,4))
names=['AR decode','Target verify','Sync draft','SSD tree step','Target prefill']; left=np.zeros(5)
for key,color in [('memory-bound',BLUE),('compute-bound',TEAL),('latency-bound',ORANGE)]:
 vals=[ncu[n]['overall']['bound_split_pct'].get(key,0) for n in ncu]
 ax.barh(names,vals,left=left,label=key,color=color)
 for y,(v,l) in enumerate(zip(vals,left)):
  if v>8:ax.text(l+v/2,y,f'{v:.0f}%',va='center',ha='center',color='white',fontweight='bold')
 left+=vals
ax.invert_yaxis();ax.set_xlabel('Share of captured kernel time (%)');ax.set_xlim(0,100);ax.legend(frameon=False,ncol=3,bbox_to_anchor=(.5,-.18),loc='upper center',fontsize=9)
fig.tight_layout();boundplot=save(fig,'kernel_bound_classification')

fig,ax=plt.subplots(figsize=(8,4))
first=[r['first_divergence_zero_based'] for r in correct['requests']]
ax.barh(['Chat','Code','Document'],[v for v in first],color=RED,height=.5)
for i,v in enumerate(first):ax.text(v+4,i,f'index {v} / token {v+1}',va='center',fontsize=10)
ax.set(xlim=(0,280),xlabel='First differing output index (zero-based; 256 tokens generated)');ax.invert_yaxis();fig.tight_layout();divplot=save(fig,'first_divergences')

fig,ax=plt.subplots(figsize=(8,4))
positions=['Chat\nposition 13','Code\nposition 181','Document\nposition 170']; margins=[]
logit_rows=[]
for r in correct['requests']:
 arid=r['ar_token_id'];sdid=r['speculative_token_id']
 ar={t['token_id']:t['logit'] for t in r['captures']['ar']['witnesses'][0]['actual_forward']['top']}
 sp={t['token_id']:t['logit'] for t in r['captures']['sd']['witnesses'][0]['actual_forward']['top']}
 margins.append(ar[arid]-ar[sdid]);logit_rows.append([r['id'],str(r['first_divergence_zero_based']),f'{ar[arid]:g} / {ar[sdid]:g}',f'{sp[arid]:g} / {sp[sdid]:g}'])
ax.bar(np.arange(3)-.13,margins,.26,color=BLUE,label='AR: winning-token margin')
ax.scatter(np.arange(3)+.13,[0,0,0],s=85,color=RED,marker='D',label='SD / SSD: tied logits',zorder=3)
ax.set(xticks=np.arange(3),xticklabels=positions,ylabel='Logit(AR token) - logit(speculative token)',ylim=(-.035,.32));ax.axhline(0,color=GRAY,lw=.8);ax.legend(frameon=False,loc='upper left',fontsize=9)
fig.tight_layout();marginplot=save(fig,'bf16_logit_margins')

fig,ax=plt.subplots(figsize=(8,3.8))
rows=['Original fast SD vs AR','Native Flash SD vs its AR','Math SD vs its AR (chat only)','Serial numerical SD vs original AR']
vals=[0,0,1,3];totals=[3,3,1,3]
ax.barh(rows,[100*v/t for v,t in zip(vals,totals)],color=[RED,RED,ORANGE,TEAL],height=.55)
for i,(v,t) in enumerate(zip(vals,totals)):ax.text(max(2,100*v/t-3),i,f'{v}/{t} exact prompts'+(' (64 tokens)' if i==2 else ' (256 tokens)'),va='center',ha='right' if v else 'left',color='white' if v else INK,fontsize=10)
ax.set(xlim=(0,108),xlabel='Within-control exact prompt agreement (%)');ax.invert_yaxis();fig.tight_layout();backendplot=save(fig,'correctness_controls')

# Source references are document-local and point at saved evidence, not unsupported external claims.
SOURCES={
'E01':['results/nsys/ar_run.json','results/nsys/sd_k6_run.json','results/nsys/ssd_k7_f3_run.json'],
'E02':['results/roofline/machine_roofline.json','results/roofline/gemm_sweep.json','results/roofline/roofline_gemm.png'],
'E03':['results/roofline/forward_vs_tokens.json'],
'E04':['results/nsys/ar_analysis.json','results/nsys/ar_timeline.png'],
'E05':['results/nsys/sd_k6_analysis.json','results/nsys/sd_k6_timeline.png'],
'E06':['results/nsys/ssd_k7_f3_analysis.json','results/nsys/ssd_k7_f3_timeline.png'],
'E07':['results/nsys/ssd_k7_f3_critical_path.json','results/nsys/ssd_k7_f3_critical_path.png'],
'E08':['results/ncu/'+n+'_ncu.json' for n in ncu]+['results/phase_roofline.png'],
'E09':['results/correctness/summary.json','results/correctness/ar.json','results/correctness/sd.json','results/correctness/ssd.json','results/correctness/sd_eager.json'],
'E10':['results/correctness/prefix_controls.json','results/correctness/schedule_controls.json'],
'E11':['results/correctness/attention_backends/summary.json','results/correctness/attention_math_controls.json','results/correctness/attention_flash_controls.json']+[str(p.relative_to(ROOT)) for p in (ROOT/'results/correctness/attention_backends').glob('*.json') if p.name!='summary.json'],
'E12':['results/correctness/greedy_verify_tests.txt','results/correctness/attention_validation.txt','results/correctness/attention_backends/tests.txt','results/baselines/eval1/validation.json','results/baselines/eval1/validation_tests.txt'],
'B01':['results/baselines/eval1/manifest.json','results/baselines/eval1/status.json','results/baselines/eval1/tuning_jobs.json','results/baselines/eval1/watcher_check.txt']}
ATTACH=Path('/home/vishalrao/.codex/attachments/10d66bab-cdf9-4013-915c-ae06db63aa21')
SOURCES['P01']=[str(ATTACH/'ProjectProposalPAIA (1).pdf')]
SOURCES['P02']=[str(ATTACH/'project_roadmap.pdf')]
SOURCES['P03']=[str(ATTACH/'Anytime_SSD_High_Level_Explainer.pdf')]
SOURCES['B01']=[str(p) for p in (OUT/'campaign_snapshot').glob('*.json')]
SOURCES['B02']=[str(p) for p in (OUT/'campaign_snapshot/raw/tune').rglob('*.json')]
SOURCES['B03']=[str(p) for p in (OUT/'campaign_snapshot/raw/heldout').rglob('*.json')]
SOURCES['L01-L04']=[str(OUT/'batch_size_sources.json')]
SOURCES['B04']=[str(OUT/'completed_campaign_snapshot.json')]+[str(p) for base in ['gpu0_only_sd_r0','excluded'] for p in (OUT/'campaign_snapshot'/base).rglob('*.json')]


def clean(text):return re.sub('<[^>]+>','',text).replace('&amp;','&').replace('&lt;','<').replace('&gt;','>')

class PDF:
 def __init__(self,path,width=842,height=595,label='PERFORMANCE EVIDENCE REPORT'):
  self.c=canvas.Canvas(str(path),pagesize=(width,height));self.w=width;self.h=height;self.page=0;self.label=label
  self.c.setTitle('Anytime Outcome-Cache Construction | Evaluation 1');self.c.setAuthor('Ishaan Romil; Vishal Rao; Eshaan Sharma')
 def para(self,text,x,y,w,size=10.2,color=INK,bold=False,maxh=None):
  st=ParagraphStyle('p',fontName='Bold' if bold else 'Body',fontSize=size,leading=size*1.42,textColor=HexColor(color),spaceAfter=0)
  p=Paragraph(text,st);pw,ph=p.wrap(w,10000)
  if maxh is not None and ph>maxh:raise RuntimeError(f'Page {self.page} text overflow: {ph:.1f}>{maxh}: {text[:75]}')
  p.drawOn(self.c,x,self.h-y-ph);return ph
 def start(self,title,subtitle='',tag='MEASURED / SAVED EVIDENCE'):
  if self.page:self.c.showPage()
  self.page+=1
  self.c.setFillColor(HexColor(NAVY));self.c.rect(0,self.h-10,self.w,10,fill=1,stroke=0)
  self.para('ANYTIME OUTCOME-CACHE CONSTRUCTION',32,24,self.w-64,8.3,TEAL,True)
  self.para(title,32,46,self.w-64,23,NAVY,True,maxh=65)
  if subtitle:self.para(subtitle,32,84,self.w-64,9.5,GRAY,maxh=32)
  self.c.setFillColor(HexColor(LIGHT));self.c.roundRect(32,self.h-133,self.w-64,24,5,fill=1,stroke=0)
  self.para(tag,42,115,self.w-84,8.1,RED if 'EXACTNESS' in tag else TEAL,True,maxh=16)
  self.c.setStrokeColor(HexColor('#D6E0E8'));self.c.line(32,34,self.w-32,34)
  self.para(self.label+'  /  '+DATE,32,self.h-25,self.w-110,7.2,GRAY)
  self.para(f'{self.page:02d}',self.w-59,self.h-26,28,8,GRAY,True)
 def image(self,path,x,y,w,h):
  im=Image.open(path);iw,ih=im.size;scale=min(w/iw,h/ih);dw,dh=iw*scale,ih*scale
  self.c.drawImage(str(path),x+(w-dw)/2,self.h-y-dh-(h-dh)/2,width=dw,height=dh,mask='auto')
 def blocks(self,blocks,x,y,w,size=10.3,maxy=526):
  for heading,text in blocks:
   y+=self.para(heading,x,y,w,size,TEAL,True)+4
   y+=self.para(text,x,y,w,size,INK)+13
  if y>maxy:raise RuntimeError(f'Page {self.page} blocks overflow {y}')
  return y
 def source(self,text):self.para('Evidence: '+text,32,self.h-51,self.w-64,7,GRAY,maxh=20)
 def table(self,headers,rows,x=32,y=155,w=None,widths=None,size=9):
  w=w or self.w-64
  st=ParagraphStyle('cell',fontName='Body',fontSize=size,leading=size*1.3,textColor=HexColor(INK))
  hs=ParagraphStyle('head',parent=st,fontName='Bold',textColor=white)
  data=[[Paragraph(str(v),hs) for v in headers]]+[[Paragraph(str(v),st) for v in row] for row in rows]
  tw=[v*w for v in widths] if widths else [w/len(headers)]*len(headers)
  t=Table(data,colWidths=tw,hAlign='LEFT')
  t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor(NAVY)),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor(LIGHT)]),('LINEBELOW',(0,0),(-1,0),.7,HexColor(NAVY))]))
  _,h=t.wrap(w,10000)
  if y+h>self.h-64:raise RuntimeError(f'Page {self.page} table overflow {y+h}')
  t.drawOn(self.c,x,self.h-y-h);return y+h
 def chartpage(self,title,subtitle,path,blocks,source,tag='PROFILED DIAGNOSTIC / EXACTNESS UNRESOLVED',note=None):
  self.start(title,subtitle,tag)
  self.image(path,32,150,485,315)
  self.blocks(blocks,539,158,270,10.2,maxy=516)
  if note:self.para(note,32,475,485,9.2,GRAY,maxh=47)
  self.source(source)
 def finish(self):self.c.save()

pdf=PDF(OUT/'Anytime_SSD_Performance_Report.pdf')
pdf.start('Performance evidence and experimental plan','Evaluation 1 | Completed experiments, interpretation, and remaining runs','UPDATED: 480 VALID TUNING + 150 VALID HELD-OUT MEASUREMENTS / PARTIAL EVALUATION')
pdf.blocks([
 ('The problem','Complete-cache SSD can leave the target waiting while the drafter finishes work that is only partly needed. Our project asks whether safely publishing ready prefixes and assigning unequal branch depths can reduce full-request latency.'),
 ('What the measurements establish','On this L40S setup, target decode/verify is mostly memory-bound. The SSD trace shows 9.2 ms of target waiting on cache hits and 30.0 ms on misses. Five of seven draft depths are ready in 155 of 166 hit rounds.'),
 ('What is still open','The anytime scheduler has no measured results yet. Fast SD and SSD differ from the unchanged AR reference on all three saved prompts. A slow synchronous reference matches 768 AR tokens, but fast correctness remains unresolved.'),
 ],32,155,475,11,maxy=505)
pdf.blocks([('Included','Machine roofline, model-width sweeps, AR/SD/SSD Nsight traces, five Nsight Compute phase captures, first-divergence controls, backend isolation, numerical reference, and harness validation.'),('New results','All 480 tuning measurements are complete; SD K6 and SSD K7/F3 are frozen winners. AR and SSD each have one valid 75-prompt held-out round. A separate 75-prompt SD GPU0-only round is complete and labeled contextual. Five-repeat held-out validation is incomplete.'),('Authors','Ishaan Romil / Vishal Rao / Eshaan Sharma')],548,155,260,10.4,maxy=510)
pdf.source('P01-P03; E01-E12; B01. Data snapshot taken '+DATE+' (Asia/Kolkata).')

pdf.start('How to read the evidence','Separate observations, explanations, and research hypotheses.','EVIDENCE BOUNDARIES / RESOURCE ALLOCATION IS EXPLICIT')
pdf.table(['Evidence class','What has been run','What it can support'],[
 ['Machine microbenchmarks','BF16/FP32 GEMM, DRAM/L2 bandwidth, token-width sweep','A hardware cost model; not end-to-end project speedups.'],
 ['Profiled inference','AR; SD K=6; SSD K=7/F=3, JIT backup. One chat, one code, one document prompt; 256 outputs each; three warmups.','Mechanism and diagnostic timing. No repeated unprofiled held-out claim.'],
 ['Numerical diagnostics','Fresh reproductions, eager mode, common-history replay, alternate attention, serial numerical reference.','Evidence for shape-sensitive BF16 differences. Diagnostic times are excluded from performance comparisons.'],
 ['New unprofiled campaign','480 tuning measurements complete; 150 dedicated held-out measurements complete (AR/SSD, one round each). SD contextual single-GPU run is separate.','Valid tuning selection and partial descriptive results. No completed five-repeat headline comparison or repeated-run confidence intervals yet.'],
 ],widths=[.2,.42,.38],size=9.4)
pdf.para('<b>Hardware:</b> Qwen3-8B target / Qwen3-0.6B draft, BF16, greedy, batch one. AR: target on GPU0. SD: both models on GPU0. SSD: target GPU0 and draft GPU1. Two L40S GPUs, PCIe across NUMA sockets, no NVLink. One extra GPU contributes to SSD\'s result.',32,433,778,10.3,maxh=65)
pdf.para('Historical scored prompt IDs: chat specbench-148; code HumanEval/36; document specbench-553. All three are excluded from the new held-out campaign.',32,503,778,8.6,maxh=28)
pdf.source('E01-E08; B01. Proposal attachments provide context and targets, not completed measurements.')

additions.report_pages(pdf,completed,newfigs,OUT)

pdf.chartpage('Observed request performance','One profiled run per method; three prompts x 256 output tokens.',perfplot,[
 ('Measured',f'AR: {latency[0]:.2f} s/request and {throughput[0]:.1f} tokens/s.<br/>SD: {latency[1]:.2f} s and {throughput[1]:.1f} tokens/s.<br/>SSD: {latency[2]:.2f} s and {throughput[2]:.1f} tokens/s.'),
 ('Interpretation','Observed throughput ratios are 1.42x SD/AR and 1.83x SSD/AR. They are descriptive ratios from this trace, not validated general speedups.'),
 ('For our project','Keep these as baseline mechanism evidence. The dedicated, unprofiled campaign is needed before presenting headline timing; exactness failures must remain visible.')], 'E01. Request time includes the saved driver\'s generation boundary; no new timing is performed.')

pdf.chartpage('The workload changes the apparent benefit','These are three individual prompts, not per-family averages over a dataset.',familyplot,[
 ('Measured','Each family has exactly one saved sample. The code example runs substantially faster with speculation than the chat example.'),
 ('Interpretation','Acceptance and branch predictability can vary by task. A mixed average can hide a family regression or overrepresent one easy prompt.'),
 ('For our project','Use a balanced tuning mix and the same held-out prompt IDs across methods. Report each family, pairing, and five repetitions; do not infer statistical confidence from this three-prompt sample.')], 'E01. All three historical prompt IDs are excluded from the new held-out campaign.')

pdf.chartpage('The measured L40S roofline','Independent machine calibration explains why wider verification can be economical.',ROOT/'results/roofline/roofline_gemm.png',[
 ('Measured',f'BF16 tensor peak: {roof["bf16_tensor_tflops_measured"]:.1f} TFLOP/s (median {roof["bf16_tensor_tflops_median"]:.1f}).<br/>DRAM read: {roof["dram_read_GBps"]:.1f} GB/s.<br/>DRAM copy: {roof["dram_copy_GBps"]:.1f} GB/s.'),
 ('Interpretation',f'Measured ridge: {roof["ridge_bf16_flop_per_byte_measured"]:.0f} FLOP/byte. Below the ridge, streaming model weights limits useful throughput. These are empirical ceilings under measured conditions.'),
 ('For our project','Small speculative verification batches can share weight movement across tokens. The important tradeoff is useful accepted work versus sequential draft and control costs.')], 'E02. FP32 peak: 48.6 TFLOP/s; L2 copy: 2.98 TB/s. Nominal vendor peaks are not substituted.')

pdf.chartpage('Increasing token width is initially inexpensive','Linear-layer sweep only; attention, cache management, sampling, and transfer are excluded.',widthplot,[
 ('Measured','Target linear layers stay around 22-24 ms from small widths through T=128. T=256 is about 27 ms. The 0.6B drafter shows a similar broad flat region.'),
 ('Interpretation','At low token counts, weight traffic dominates. More rows do not immediately multiply latency. T=1 uses a distinct GEMV path; wider batches use tensor-core GEMM.'),
 ('For our project','Increasing fan-out may be cheaper than increasing sequential depth. This is a tuning hypothesis, not evidence that arbitrary F or K improves request latency.')], 'E03; E02. AI ~ T is an approximation for weight-dominated BF16 linear layers.')

pdf.chartpage('AR: the target is already close to its bandwidth limit','The original trace is shown beside its interpretation.',ROOT/'results/nsys/ar_timeline.png',[
 ('Measured','GPU0 is kernel-busy 94.3% of the trace. cuBLAS GEMV accounts for about 96% of GPU kernel time. Decode DRAM-read activity is 89.3%; tensor activity is about 0.1%.'),
 ('Interpretation','Single-token decode spends most time moving weights. Low tensor activity is expected for this small batch; it does not imply a large compute optimization opportunity.'),
 ('For our project','Our main opportunity is amortizing target work across verified tokens and avoiding waiting, rather than expecting large target-kernel speedups.')], 'E04; E08. Timeline axes and sample window come from the saved Nsight Systems analysis.')

pdf.chartpage('Synchronous SD pays for serial drafting','Phase wall-time and GPU-kernel-time estimates answer different questions.',sdplot,[
 ('Measured','A decoding round spends ~24.9 ms drafting and ~24.9 ms verifying. Each of seven drafter forwards averages 3.49 ms wall time, with ~2.42 ms of GPU work.'),
 ('Interpretation','~1.1 ms per drafter forward is host/launch gap in the trace analysis. The target verification cost is amortized, but sequential draft depth adds directly to the critical path.'),
 ('For our project','Study K and F jointly with acceptance and elapsed time. Reduce drafter planning/synchronization overhead only after measuring it; CUDA graphs already capture the model forward.')], 'E05. The ~2.42 ms kernel-work estimate is a trace-derived summary in docs/perf_analysis.md, not the isolated NCU 3.10 ms capture.')

pdf.chartpage('SSD overlaps work, but the target still waits','Original two-GPU trace: target and drafter activity on the same wall-time axis.',ROOT/'results/nsys/ssd_k7_f3_timeline.png',[
 ('Measured','Target verification averages 24.9 ms/round. Draft glue is 4.3 ms and seven tree steps total 25.4 ms. That ~29.7 ms draft sequence can outlast verification.'),
 ('Interpretation','Overlapping two GPUs does not remove the slower side of a dependency. Host planning and serial tree depth still determine when the response becomes usable.'),
 ('For our project','Expose a completed prefix earlier, then measure any lost acceptance, controller costs, stream contention, and in-flight drain costs. No such modified-policy result is measured yet.')], 'E06; E07. This timeline is mechanism evidence, not an anytime-cache implementation benchmark.')

pdf.chartpage('GPU busy time can include waiting','NCCL kernel activity is separated from other kernels and idle gaps.',utilplot,[
 ('Measured','SSD target: 92.4% kernel-busy, including 35.2% of the full window in NCCL kernels. Non-NCCL kernel-busy time is ~57.2%. Drafter GPU is 72.8% busy.'),
 ('Interpretation','The trace attributes much target NCCL activity to waiting for the drafter. Kernel-busy percentage alone overstates useful target computation. Non-NCCL busy is a proxy, not a guarantee of useful work.'),
 ('For our project','Track target waiting and publication readiness, not GPU utilization alone. Low host receive-call time does not by itself prove low transport latency; trace context matters.')], 'E04-E06. Percentages are normalized by each method\'s own timed window, not by summed kernel duration.')

pdf.chartpage('Critical path: useful prefixes exist before the full tree','230 matched request/verification rounds in the SSD trace.',criticalplot,[
 ('Measured','Cache miss rate: 27.8%. Mean wait: 9.2 ms on hits, 30.0 ms on misses. Total wait is 3.45 s versus 5.72 s verify time (~37.6% wait share of their sum).'),
 ('Measured readiness','Depth 5 is ready in 155/166 hit rounds (93.4%). Other hit depths: 1 in one round, 4 in eight, 6 in two. Full-tree completion trails the request by ~8.0 ms on average.'),
 ('For our project','This motivates early publication. It does not measure the benefit: shallow prefixes may accept fewer tokens, and publication or draining can consume the saved wait.')], 'E07. Readiness is a reconstruction from the trace, not an end-to-end early-publication experiment.')

pdf.chartpage('Kernel counters identify two different optimization regimes','Five isolated steady-state forward captures; time-weighted classification.',boundplot,[
 ('Measured','AR decode and target verify: ~95% memory-bound kernel time. Prefill: 85.3% compute-bound. Drafter forward/tree: 29.3% / 37.3% latency-bound kernel time.'),
 ('Interpretation','The target streams large weights efficiently; small drafter kernels leave more room for launch and control overhead. A phase can mix memory-bound and latency-bound kernels.'),
 ('For our project','Optimize publication and drafter scheduling before adding a custom target compute kernel. NCU replay suppresses overlap, so its phase times must not be added to predict request time.')], 'E08. Another user\'s light (~3% utilization) GPU1 job overlapped the drafter NCU capture; these are diagnostic captures.')

pdf.start('Kernel measurements, with units and scope','These are per-capture aggregates, not independent repeated request benchmarks.','MEASURED / NCU REPLAY / ABSOLUTE TIMES ARE DIAGNOSTIC')
rows=[]
for name,label in zip(ncu,['AR decode (target)','SD verify (target, K6)','Sync draft forward','SSD tree step','Target prefill (631 tokens)']):
 o=ncu[name]['overall'];rows.append([label,str(o['n_kernels']),f'{o["total_kernel_time_us"]/1000:.2f}',f'{o["ai_flop_per_byte"]:.2f}',f'{o["achieved_dram_GBps"]:.0f}',f'{o["flop"]/o["total_kernel_time_us"]/1e6:.2f}'])
y=pdf.table(['Capture','Kernels','Sum time (ms)','AI (FLOP/B)','DRAM (GB/s)','Achieved TFLOP/s'],rows,widths=[.29,.1,.15,.15,.15,.16],size=9.7)
pdf.para('<b>Why verify is attractive:</b> the captured target verify takes 22.21 ms versus 23.06 ms for one AR decode forward. More tokens reuse weight traffic. This is a kernel-level observation, not a guarantee of complete request acceleration.',32,y+20,778,10.5,maxh=60)
pdf.para('<b>Counter caveats:</b> measured AI can include MMA tile padding, so measured verify/tree AI is larger than useful token-width AI. NCU bandwidth may exceed the empirical DRAM microbenchmark because conditions and counter denominators differ; it is not evidence that the roofline calibration is wrong.',32,y+90,778,10.3,maxh=60)
pdf.source('E08; E02. FLOP/s is aggregate recorded FLOPs / aggregate recorded kernel time.')

pdf.chartpage('The phases occupy different roofline regions','Saved phase roofline aggregates the five Nsight Compute captures.',ROOT/'results/phase_roofline.png',[
 ('Measured','Decode, verify, and small drafter phases lie left of the measured ridge. Prefill has aggregate AI ~365 FLOP/B and lies in the compute-oriented regime.'),
 ('Interpretation','A compute-heavy prefill can coexist with a memory-limited decode. Prefill and decode therefore need separate explanations and metrics.'),
 ('For our project','Use full-request latency including both model prefills as primary. Keep decode-only throughput as a secondary diagnostic and study input/output length sensitivity later.')], 'E08; E02. Kernel padding, cache effects, and isolated replay affect exact positions.')

for phase,label,interpretation in [
 ('ar_decode','AR decode: kernel-family detail','GEMV dominates time and saturates memory. Low occupancy and low issue activity do not by themselves imply a compute bottleneck; the dominant long-scoreboard stalls indicate waiting on memory.'),
 ('sd_verify','Target verify: kernel-family detail','Tensor-core GEMM still spends most time limited by memory at this small verification width. Multi-token verification amortizes weight traffic; it does not make the target compute-bound.'),
 ('sd_draft','Synchronous draft: kernel-family detail','The small drafter mixes memory-oriented GEMV with many short elementwise and attention kernels. Host gaps seen in Nsight Systems are additional to these isolated kernel measurements.'),
 ('ssd_tree','SSD tree step: kernel-family detail','Wide draft GEMMs improve parallel work, but short attention, pointwise, and bookkeeping kernels retain a latency component. The lightly shared GPU1 capture is diagnostic.'),
 ('ar_prefill','Target prefill: kernel-family detail','Large prefill GEMMs use the tensor pipeline much more effectively. The same request can have compute-oriented prefill and memory-oriented decode; full-request timing must include both.')]:
 pdf.start(label,'Nsight Compute aggregates; every captured kernel family is retained.','MEASURED / ISOLATED KERNEL CAPTURE / COUNTER PERCENTAGES ARE TIME-WEIGHTED')
 def num(value):return '—' if value is None else f'{value:.1f}'
 rows=[]
 for family,data in ncu[phase]['families'].items():
  rows.append([family,num(data.get('time_pct')),num(data.get('sol_compute_pct')),num(data.get('sol_memory_pct')),num(data.get('occ_achieved_pct')),num(data.get('issue_active_pct')),num(data.get('ai_flop_per_byte')),', '.join(data.get('top_stalls',[])[:3])])
 end=pdf.table(['Kernel family','Time %','Compute SOL %','Memory SOL %','Occupancy %','Issue active %','AI FLOP/B','Top warp stalls'],rows,widths=[.22,.08,.09,.09,.09,.09,.09,.25],size=7.6)
 pdf.para('<b>Interpretation:</b> '+interpretation,32,end+17,778,10,maxh=65)
 pdf.source('E08. SOL = Speed of Light (% of counter-reported peak). These counters do not include host idle gaps or overlap.')

pdf.chartpage('Strict greedy agreement currently fails in fast modes','Positions below are zero-based output indices. SD and SSD match each other.',divplot,[
 ('Measured','All three 256-token AR outputs differ from fast SD and SSD: chat at index 13, code at 181, document at 170. SD equals SSD for all 768 saved output tokens.'),
 ('Interpretation','Target verification is present, but its batched BF16 numerical path can choose a different greedy token than the declared sequential AR path. Agreement between SD and SSD is not agreement with AR.'),
 ('For our project','Keep performance observations separate from correctness-preserving acceleration claims. The proposal requires token-for-token agreement with the unchanged target; this gate remains open.')], 'E09. Fresh AR/SD/SSD reproductions match their saved outputs; eager SD reproduces the first 192 tokens.')

pdf.start('The first divergence is a numerical decision change','Candidate columns are ordered AR choice / speculative choice.','DIAGNOSTIC / SAME COMMITTED HISTORY / NOT A TOLERANCE-BASED PASS')
y=pdf.table(['Prompt / zero-based index','AR choice / speculative choice','AR logits','SD and SSD logits'],[
 ['Chat: specbench-148 / 13','integrating (53852) / careful (16585)','30.75 / 30.625','30.5 / 30.5'],
 ['Code: HumanEval/36 / 181','or (476) / ** (334)','37.0 / 36.75','37.0 / 37.0'],
 ['Doc: specbench-553 / 170','IDs (28360) / identification (21892)','29.125 / 29.0','29.0 / 29.0']],widths=[.24,.36,.2,.2],size=10)
pdf.image(marginplot,32,y+14,460,210)
pdf.blocks([('Why the chosen word flips','AR has a unique winner by one BF16 spacing. Speculative logits tie; argmax picks the smaller vocabulary index in these observed cases.'),('What this establishes','Forward grouping and cached numerical history affect the result. It does not isolate one faulty operation or certify all engine state transitions.')],520,y+22,287,10,maxy=518)
pdf.source('E09-E10. BF16 spacings at these values are 0.125 / 0.25 / 0.125.')

pdf.start('Controls narrow the cause without claiming a fast fix','Completed ablations of execution shape, graph replay, projection, and state history.','DIAGNOSTIC CONTROLS / TIMES ARE NOT PERFORMANCE RESULTS')
pdf.table(['Completed control','Result','Implication / boundary'],[
 ['Disable CUDA graphs','Eager SD reproduces all three saved prefixes through 192 tokens.','Graph replay is not necessary for the mismatches.'],
 ['Replay the same verification block sequentially','Chat and code return to the AR choice; document retains the tie.','Current query grouping explains some cases; historical grouping also matters.'],
 ['Clean, single-token target histories; widths 1/7/8','All AR histories reproduce. Wider query changes flip the code decision, while chat/document retain the AR winner in this control.','Identical token history is necessary, but execution history and KV numerics matter too.'],
 ['One-row vocabulary head on the same hidden states','All three tied decisions persist in eager SD.','Changing only the final projection is not a demonstrated remedy.'],
 ['Independent document target-schedule replay','Fresh contiguous KV, no drafter/verifier/rollback/NCCL: reproduces the tied 29.0 / 29.0 decision.','Historical target grouping is sufficient to reproduce this case; an SSD state bug is not required.'],
 ],widths=[.25,.39,.36],size=9.4)
pdf.source('E09-E10. Controls use matched histories; rows after rejected proposals are not treated as same-prefix witnesses.')

pdf.chartpage('Alternate attention and the slow numerical reference','Each control has a different comparison scope; equal bars do not imply equal coverage.',backendplot,[
 ('Completed backend tests','Native FlashAttention AR/SD diverge on all three prompts (indices 52/181/125). Math attention agrees on one 64-token chat sample, but changes the original AR output.'),
 ('Completed slow reference','Preserving single-token attention, linear, norm, RoPE, and activation shapes matches all 768 original AR tokens. Attention + linear shape control alone still fails on chat/code.'),
 ('Boundary','The expanded reference is synchronous SD only, eager, EOS ignored. It loses batching efficiency. Asynchronous SSD, natural EOS, and broad held-out equivalence remain untested.')], 'E11. Backend agreement is against its own AR where stated; the expanded serial reference is against original saved AR.')

pdf.start('What this means for the project','The proposal has a measured motivation; its optimization has not yet been validated.','INTERPRETATION AND HYPOTHESES / NO ANYTIME-POLICY SPEEDUP IS MEASURED')
pdf.table(['Finding already measured','Project decision it motivates','What must be measured next'],[
 ['Ready depth 5 exists in 93.4% of hit rounds while full completion trails the request.','Implement safe incremental prefix publication.','End-to-end uniform early handoff versus tuned complete-cache SSD, with publication and drain overhead included.'],
 ['Misses are 27.8% of rounds and wait about 30 ms.','Consider wider coverage and unequal depths across likely outcomes.','Miss rate, accepted work, and latency versus a uniform early-publication baseline with matched candidate and GPU budgets.'],
 ['Draft work and host overhead are on the critical path.','Use measured costs and readiness when scheduling.','Adaptive versus unequal fixed depths, global adaptation, probability-first, and a timing-rule adaptation.'],
 ['Fast modes fail exact AR agreement on three prompts.','Keep a strict original-reference correctness gate and the slow oracle separate.','Fast numerical remedy and EOS/rejection/bonus/partial/stale/KV tests before a correctness-preserving claim.'],
 ],widths=[.31,.27,.42],size=9.6)
pdf.source('E02-E11; P01-P03. An ~8 ms observed wait is not an ~8 ms measured gain from a new policy.')

pdf.start('Comparisons we will study','The key novelty comparison is adaptive scheduling versus uniform early publication.','EXPERIMENTAL PLAN / ONLY THE FIRST THREE METHODS HAVE PROFILED RESULTS')
pdf.table(['Comparator','Status now','Question it answers'],[
 ['AR target-only','Profiled (1 GPU)','How much time does ordinary greedy generation take?'],
 ['Synchronous SD','Profiled at K6 (1 GPU)','What does ordinary speculative verification buy?'],
 ['Complete-cache SSD','Profiled; tuned K7/F3 (2 GPUs); held-out partial','How far does overlap with complete-draft publication get us?'],
 ['Fixed global K/F; online global adaptation','Fixed grid complete; online policy pending','Can simple globally shared depth/fan-out choices explain the gain?'],
 ['Uniform early publication + immediate handoff','Not yet run','Does exposing partial work alone improve latency?'],
 ['Unequal fixed branch depths','Not yet run','Does unequal allocation help without online adaptation?'],
 ['Adaptive anytime scheduler','Not yet run','Does online scheduling improve over both early-publication and fixed allocation?'],
 ['Probability-first; SwiftSpec-inspired timing rule','Not yet run; shared-executor adaptations','Do readiness/cost estimates beat simpler allocation rules?'],
 ],widths=[.32,.30,.38],size=9.0)
pdf.source('P01-P02; E01; B01. Match two-GPU allocation, candidate construction/budget, and tuning effort for SSD-policy comparisons.')

pdf.start('Trade-offs to measure, not assume','Each knob should be judged by committed output per full elapsed time.','EXPERIMENTAL PLAN / DIRECTIONS BELOW ARE HYPOTHESES')
pdf.table(['Trade-off','Possible benefit','Possible cost','Measurements'],[
 ['Greater draft depth K','More tokens accepted per verification','More sequential draft work and later readiness','Request latency, accepted draft fraction, ready depth, wait, discarded work'],
 ['Greater fan-out F / branch coverage','Fewer outcome-cache misses','More KV/branch memory; larger kernels and control work','Miss rate, peak memory, execution shapes, target/draft slowdown'],
 ['Earlier / more frequent publication','Shorter target wait; useful partial prefix','More event/metadata checks; fewer accepted tokens; more drain costs','Wait, committed tokens/step, publication time, drain time, latency'],
 ['Unequal versus uniform allocation','Spend more work on useful outcomes','Mispredicted outcomes may have short or missing prefixes','Ready-prefix coverage, actual outcomes, fallback rate, per-family regression'],
 ['Online scheduling versus fixed policy','Adapt to prompt and execution costs','Estimator errors, cold statistics, controller overhead','Decision time, calibration, fixed-statistics control, full-request latency'],
 ['Exact numerical path versus batched fast path','Preserve unchanged AR output','Per-token operations lose batching efficiency','Exact tokens, first divergence, repeatability; benchmark a practical fix separately'],
 ],widths=[.22,.23,.25,.30],size=9.2)
pdf.source('P01-P02; E02-E11. Current traces motivate these trade-offs but do not measure the proposed-policy outcomes.')

pdf.start('Ablations that isolate the contribution','Change one component at a time in the shared executor.','EXPERIMENTAL PLAN / ALL POLICY ABLATIONS BELOW REMAIN TO RUN')
pdf.table(['Controlled comparison','Hold fixed','What it isolates'],[
 ['Complete publication vs uniform early publication','Candidate generation, K/F budget, hardware, fallback','Value of incremental publication itself'],
 ['Uniform early depths vs unequal fixed depths','Publication rule, average/declared candidate budget, tuning effort','Value of unequal allocation without online learning'],
 ['Unequal fixed depths vs adaptive depths','Executor, candidates, hardware; controller cost included','Value of online scheduling'],
 ['Probability-first vs timing-aware selection','Available outcomes and tuning data','Value of measured execution costs and readiness'],
 ['Fixed vs online outcome statistics','Scheduling rule and smoothing/fallback contract','Value and reliability of learned frequencies'],
 ['Readiness estimates on vs off','Other policy logic and actual GPU event safety','Value of readiness prediction; event validation always remains on'],
 ['Publication depths 1/2/4/8, clipped to available K','Candidate budget and safety checks','Frequency versus prefix depth and control overhead'],
 ['Measured scheduler / no-op controller / idealized ceiling','Same invocation points and workload; label counterfactuals','Whether real decision overhead erases savings; an idealized controller is diagnostic only'],
 ],widths=[.35,.35,.30],size=8.8)
pdf.source('P01-P02. Publication events, valid lengths, round IDs, immutable buffers, and KV reconciliation are correctness requirements, not optional safety ablations.')

pdf.start('Completed experiment inventory','All saved experiment families are accounted for; software tests are a separate kind of evidence.','COMPLETED / SOURCES ARE INDEXED IN THE EVIDENCE REGISTER')
pdf.table(['Completed family','Extent / result','Evidence'],[
 ['Machine and model microbenchmarks','GEMM sweep; BF16/FP32 roof; DRAM/L2 bandwidth; 38 target/draft token-width records.','E02-E03'],
 ['AR / SD / SSD Nsight Systems','Three scored prompts per method; fixed 256 outputs; request/phase timing, GPU metrics, idle and NCCL attribution.','E01, E04-E06'],
 ['SSD critical-path reconstruction','230 matched rounds; ready-depth distribution, hit/miss wait, tree completion lag.','E07'],
 ['Five Nsight Compute phase captures','AR decode, target verify, sync draft, SSD tree step, target prefill; kernel families, roofline, occupancy/stalls.','E08'],
 ['First-divergence diagnosis','Three full reproductions per mode; eager SD prefixes; common-history, query-width, head, and document schedule controls.','E09-E10'],
 ['Attention and numerical-path controls','Flash AR/SD 3x256; math chat AR/SD 1x64; replay controls; failed attention/linear reference (chat/code); expanded SD reference 3x256 exact.','E11'],
 ['Software validation','Latest full suite: 62 passed. New reduced-window (3) and GPU0-only isolation (2) CPU tests pass separately; saved logs remain available.','E12'],
 ['Benchmark and waiting harness','Frozen manifests/grid; 480 tuning and 150 dedicated held-out requests complete. One contaminated SD block excluded; GPU0-only SD uses a separately disclosed window.','B01, E12'],
 ],widths=[.27,.62,.11],size=8.7)
pdf.source('No diagnostic reference timings or synthetic test fixtures are presented as measured application performance.')

pdf.start('Experiment set: completed, partial, and deferred','The original design is retained; the two-hour window prioritizes fixed output.','480 TUNING DONE / 150 OF 1,125 DEDICATED FIXED REQUESTS DONE / NATURAL EOS DEFERRED')
y=pdf.table(['Phase / setting','Prompts and repeats','Scored requests','Why it is needed'],[
 ['Tune AR control - COMPLETE','24 balanced tuning x 2','48 / 48','Provides target-only control on the tuning mix.'],
 ['Tune SD: K=4,6,8 - COMPLETE','3 configurations x 24 x 2','144 / 144','Selected K6 by lowest mean latency.'],
 ['Tune SSD: K=4,7,10; F=1,3 - COMPLETE','6 configurations x 24 x 2','288 / 288','Selected K7/F3; all six combinations measured.'],
 ['Held-out: fixed 256 - PARTIAL','AR + frozen SD + frozen SSD; 75 x 5 planned','150 / 1,125','975 dedicated requests remain; SD contextual run does not replace a dedicated block.'],
 ['Held-out: natural EOS - DEFERRED','Same three methods; 75 x 5 planned','0 / 1,125','Secondary realistic stopping; do not retune using it.'],
 ],widths=[.29,.26,.13,.32],size=9.3)
pdf.para('<b>Protocol:</b> 8 tuning and 25 held-out prompts per chat/code/document family, plus one separate warmup per family. Seed 0; earlier investigated held-out IDs excluded; exact input IDs frozen. Winners use lowest mean tuning request latency, tie-breaking smaller K then F, before held-out results are read.',32,y+18,778,10,maxh=60)
pdf.para('<b>Execution:</b> three warmups for every fresh worker; BF16, batch one, CUDA graphs, JIT fallback, 4,096 capacity, target/draft KV fractions 0.70. No profiler; no prompt-prefix reuse. Dedicated GPU0/1 required. Contamination discards the block and the watcher retries it later.',32,y+82,778,9.8,maxh=60)
pdf.source('B01-B04. 22 dedicated valid blocks complete; 13 fixed and 15 natural blocks remain. GPU0-only SD does not count as a dedicated repetition.')

pdf.start('What the campaign will establish, and what it cannot','A repeated timing campaign does not repair numerical equivalence.','PENDING / ANALYSIS DELIVERABLES AND CORRECTNESS GATES')
pdf.blocks([
 ('Completed versus remaining','Tuning tables/curves and frozen winners now exist. Partial held-out timing and exactness are available. Five valid repetitions per method, natural-EOS tables, paired repeated-run confidence intervals, and final family-regression conclusions remain incomplete.'),
 ('Statistical and output evidence','10,000 seeded paired bootstrap draws resample rounds and prompt IDs within families. Save all output IDs; compare exact agreement and first divergence against AR, including within-method repeatability. Any fast-mode mismatch remains prominent.'),
 ('Why none of this is a fast correctness fix','More repeated timings can quantify the current implementation and expose failure rates, but they cannot satisfy the proposal\'s unchanged-target token contract if mismatches persist. A practical numerical remedy must pass exact comparisons and be benchmarked anew.')],32,154,475,10.9,maxy=515)
pdf.blocks([
 ('Next correctness work','Extend validation to asynchronous SSD and natural EOS. Check rejection at every position, all-accepted/bonus, EOS in draft/correction/bonus, output-cap boundaries, partial prefixes, stale generations, branch-local KV reconciliation, and drained in-flight work.'),
 ('Scope still to freeze','The slow numerical reference currently covers only synchronous SD on three fixed-length prompts. Broader reference runs and a fast remedy need an explicit implementation/configuration before their run counts can be frozen.')],540,154,270,10.3,maxy=514)
pdf.source('P01-P02; E09-E12; B01. Exact matching of 768 diagnostic tokens is useful, but not the full project correctness suite.')

pdf.start('Remaining prototype and matched-policy experiments','These are required follow-ups, not measured outcomes or a frozen worker schedule.','PENDING / IMPLEMENTATION MUST PRECEDE THESE RUNS')
pdf.table(['Run family still required','Concrete comparison / measurement','Why'],[
 ['Safe uniform early publication','Completed-prefix handoff versus tuned complete-cache SSD; depths 1/2/4/8 where feasible; charge event, fallback, and drain costs.','Establish whether publication alone helps before claiming a scheduler contribution.'],
 ['Fixed allocation and simple adaptation','Uniform versus unequal fixed depths; fixed global K/F versus online global K/F.','Test whether simple tuning or unequal allocation is sufficient.'],
 ['Bounded adaptive controller','Adaptive versus uniform early handoff and unequal fixed depths; probability-first and SwiftSpec-inspired timing adaptations in the same executor.','Isolate scheduling value beyond simpler policies; match two-GPU and candidate budgets and tuning effort.'],
 ['Mechanism and overhead ablations','Readiness estimates; fixed/online statistics; publication cadence; no-op/real controller; shape and budget sensitivity.','Explain why a gain or regression occurs and account for real controller and stream costs.'],
 ['Publication and KV correctness stress','Partial readiness, stale metadata, rejected suffixes, EOS/bonus cases, recycled buffers, fallback and drains.','Ensure target-verified output is based on valid, committed-history-consistent state.'],
 ],widths=[.27,.45,.28],size=9.4)
pdf.para('The proposal defines the comparator families, but these policies are not yet implemented or registered as a concrete measurement grid. Their exact block counts are therefore not honestly knowable yet. Freeze configurations and counts before measuring; do not reuse evaluation prompts for tuning.',32,462,778,10,maxh=55)
pdf.source('P01-P02. Current traces establish a reason to prototype; they do not yet establish headroom beyond an implemented uniform early-handoff baseline.')

pdf.start('Broader proposal runs and remaining measurement gaps','The small Evaluation 1 campaign is a deliberately narrower scope than the final proposal.','PENDING / FINAL-PROJECT VALIDATION AND DECISION RULE')
pdf.table(['Remaining run / metric','Declared scope or current gap','Why it is needed'],[
 ['Broader workload / lengths','Proposal targets ~100 held-out prompts per family; natural contexts near 512/2K/8K where available; natural-EOS output caps 128/512. Current campaign: 25 per family, input <=2,048, cap 256.','Generalization and input/output-length sensitivity. An 8K input cannot fit a 4,096-capacity model; change capacity/memory policy and revalidate first.'],
 ['Delivery gaps and peak memory','p50/p95 gaps between actual committed deliveries; total peak memory including branch KV, workspaces, and publication buffers are not yet campaign outputs.','Latency burstiness and memory costs can offset an average-latency gain. Add instrumentation and rerun affected settings.'],
 ['Readiness, waste, and contention','Per-outcome ready-prefix coverage, fallback, discarded work, controller/drain cost, target slowdown under overlap.','Explain policy behavior and include the full optimization cost.'],
 ['RTX 3090 replication','Use matched feasible model/context settings; small-model RTX 2080 checks are optional supporting tests.','Hardware/memory sensitivity. Single-device results cannot establish disaggregated two-GPU SSD performance.'],
 ['Final decision','At least 10% lower full-request latency than the strongest matched baseline on the declared held-out mix, with family regressions and exact correctness disclosed.','This is the proposal\'s target, not an achieved or predicted result. If adaptation adds no value beyond uniform handoff, report the correct simpler policy and negative result.'],
 ],widths=[.24,.43,.33],size=9.0)
pdf.source('P01-P02; B01. No numerical claim is made for experiments that have not been run.')

pdf.start('Evidence register and reproduction','Every numerical chart in this report comes from the saved evidence indexed below.','AUDIT TRAIL / RAW HASHES AND METRICS ARE PROVIDED WITH THE ARTIFACTS')
pdf.table(['ID','Saved evidence / source','Where to reproduce or interpret'],[
 ['E01','results/nsys/{ar,sd_k6,ssd_k7_f3}_run.json','Saved request IDs, timings, output IDs and configuration.'],
 ['E02-E03','results/roofline/{machine_roofline,gemm_sweep,forward_vs_tokens}.json','profiling/roofline_microbench.py; empirical hardware/linear-layer sweeps.'],
 ['E04-E06','results/nsys/*_analysis.json and *_timeline.png','profiling/run_nsys.sh; analyze_nsys.py.'],
 ['E07','results/nsys/ssd_k7_f3_critical_path.json and .png','profiling/ssd_critical_path.py.'],
 ['E08','results/ncu/*_ncu.json; results/phase_roofline.png','profiling/run_ncu.sh; analyze_ncu.py; summarize.py.'],
 ['E09-E10','results/correctness/summary.json; ar/sd/ssd/sd_eager; prefix/schedule_controls.json','investigate_correctness.py; replay_correctness_prefix.py; summarize_correctness.py.'],
 ['E11','results/correctness/attention_backends/*.json; attention_math/flash_controls.json','attention_controls.py; summarize_attention_controls.py.'],
 ['E12 / B01','Saved test logs; frozen campaign_snapshot/{manifest,selection,status,tuning_jobs}.json','Original campaign settings, selection evidence, snapshot progress and isolation rules.'],
 ['B02-B04','campaign_snapshot/raw/{tune,heldout}; contextual SD and excluded-attempt receipts','All 22 dedicated valid blocks; contextual SD kept distinct; completed_campaign_snapshot.json summarizes.'],
 ['L01-L04','Primary SD, SSD, EAGLE-3 v1 and SwiftSpec papers; batch_size_sources.json','Literature context for request concurrency versus internal token batching.'],
 ['P01-P03','Attached proposal, project roadmap, high-level explainer PDFs','Planning context and success criteria; not experimental evidence.'],
 ],widths=[.13,.47,.40],size=8.0)
pdf.source('Snapshot date uses Asia/Kolkata. Profiling caveats and exactness limitations are retained even where older summaries were more optimistic.')
pdf.finish()

# A common scene description produces the presentation PDF and native editable PPTX.
class Deck:
 def __init__(self):
  self.w=960;self.h=540;self.pages=[];self.scenes=[]
  self.c=canvas.Canvas(str(OUT/'Evaluation1_Experimental_Plan_Slides.pdf'),pagesize=(self.w,self.h))
  self.c.setTitle('Evaluation 1 | Comparisons, trade-offs, and ablations')
  self.c.setAuthor('Ishaan Romil; Vishal Rao; Eshaan Sharma')
 def rect(self,x,y,w,h,color):
  self.c.setFillColor(HexColor(color));self.c.rect(x,self.h-y-h,w,h,fill=1,stroke=0)
  self.scenes.append(dict(kind='rect',x=x,y=y,w=w,h=h,color=color))
 def text(self,text,x,y,w,size=18,color=INK,bold=False,maxh=None):
  st=ParagraphStyle('deck',fontName='Bold' if bold else 'Body',fontSize=size,leading=size*1.32,textColor=HexColor(color))
  p=Paragraph(text,st);_,h=p.wrap(w,10000)
  if maxh is not None and h>maxh:raise RuntimeError(f'Slide {len(self.pages)} text overflow: {h}>{maxh}: {text[:60]}')
  p.drawOn(self.c,x,self.h-y-h)
  self.scenes.append(dict(kind='text',text=clean(text.replace('<br/>','\n')),x=x,y=y,w=w,h=h+7,size=size,color=color,bold=bold))
  return h
 def image(self,path,x,y,w,h):
  iw,ih=Image.open(path).size;scale=min(w/iw,h/ih);dw,dh=iw*scale,ih*scale
  x+=(w-dw)/2;y+=(h-dh)/2
  self.c.drawImage(str(path),x,self.h-y-dh,width=dw,height=dh,mask='auto')
  self.scenes.append(dict(kind='image',path=str(path),x=x,y=y,w=dw,h=dh))
 def start(self,title,kicker,source,notes):
  if self.pages:self.c.showPage()
  self.scenes=[];self.pages.append(dict(scenes=self.scenes,notes=notes))
  self.rect(0,0,960,9,NAVY)
  self.text('ANYTIME OUTCOME-CACHE  /  EVALUATION 1',34,24,850,10,TEAL,True)
  self.text(title,34,48,892,27,NAVY,True,maxh=72)
  self.rect(34,116,892,26,LIGHT);self.text(kicker,44,121,872,10,RED if 'EXACTNESS' in kicker else TEAL,True,maxh=18)
  self.rect(34,504,892,1,'#D7E0E8')
  self.text(source,34,514,810,7.5,GRAY,maxh=20)
  self.text(f'{len(self.pages):02d}',893,512,32,10,GRAY,True)
 def blocks(self,blocks,x=620,y=165,w=306,size=16):
  for head,body in blocks:
   y+=self.text(head,x,y,w,size,TEAL,True)+5
   y+=self.text(body,x,y,w,size,INK)+16
  if y>492:raise RuntimeError(f'Slide {len(self.pages)} blocks overflow {y}')
 def table(self,headers,rows,widths=None,y=161,size=15,rowh=58):
  widths=widths or [1/len(headers)]*len(headers);total=892
  self.rect(34,y,total,36,NAVY)
  x=34
  for hd,frac in zip(headers,widths):self.text(hd,x+10,y+8,total*frac-20,size-1,'#FFFFFF',True,maxh=26);x+=total*frac
  y+=36
  for i,row in enumerate(rows):
   self.rect(34,y,total,rowh, '#FFFFFF' if i%2 else LIGHT);x=34
   for text,frac in zip(row,widths):self.text(text,x+10,y+8,total*frac-20,size,INK,maxh=rowh-12);x+=total*frac
   y+=rowh
  if y>493:raise RuntimeError(f'Slide {len(self.pages)} table overflow {y}')
  return y
 def finish(self):
  self.c.save();write_pptx(self.pages)
  (OUT/'Slide_Speaker_Notes.txt').write_text('\n\n'.join(f'SLIDE {i+1}\n'+p['notes'] for i,p in enumerate(self.pages))+'\n')


def write_pptx(pages):
 """Minimal standards-based OOXML: native text/shape elements and chart PNGs."""
 A='http://schemas.openxmlformats.org/drawingml/2006/main';P='http://schemas.openxmlformats.org/presentationml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
 def emu(pt):return str(round(pt*12700))
 def fill(c):return f'<a:solidFill><a:srgbClr val="{c.lstrip(chr(35))}"/></a:solidFill>'
 def rels(items):return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{i}" Type="{R}/{typ}" Target="{escape(target)}"/>' for i,typ,target in items)+'</Relationships>'
 def tree():return '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
 with zipfile.ZipFile(OUT/'Evaluation1_Experimental_Plan_Slides.pptx','w',zipfile.ZIP_DEFLATED) as z:
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


deck=Deck()
deck.start('What comparisons, trade-offs, and ablations?','RESEARCH QUESTION / EXPERIMENT DESIGN', 'Sources: P01-P03; E01-E12. '+DATE,
 'Our question is whether safe incremental publication and unequal outcome depths reduce full-request latency compared with tuned complete-cache SSD and simple uniform early handoff. The contribution is not just running speculation. Ten main slides answer the experimental-plan question; the last two are backup evidence.')
deck.text('Can publishing ready prefixes earlier<br/>reduce latency without changing<br/>the target\'s greedy output?',42,169,860,29,NAVY,True,maxh=130)
deck.blocks([('Evidence we already have','Profiles expose target waiting and ready partial work.'),('Evidence we still need','Completed held-out repetitions, a correct prototype, and matched-policy ablations.')],x=42,y=314,w=850,size=19)

additions.slides(deck,completed,newfigs)

deck.start('Why early publication is worth studying','MEASURED TRACE EVIDENCE / PROPOSED BENEFIT IS UNMEASURED','Sources: E07. 230 SSD rounds; K7/F3; profiled, fixed 256 outputs.',
 'The actual trace shows 9.2 milliseconds of wait on cache hits, 30 milliseconds on misses, and depth five ready in 155 of 166 hit rounds. This motivates early publication but does not prove an eight-millisecond gain; acceptance loss and control/drain costs must be charged.')
deck.image(criticalplot,34,161,565,302)
deck.blocks([('Measured','Depth 5 ready in 93.4% of cache-hit rounds.'),('Hypothesis','Earlier handoff can reduce waiting, but may reduce accepted tokens.'),('Decisive test','Uniform early publication vs complete-cache SSD.')],size=16)

deck.start('Compare against the strongest simple alternatives','TUNING COMPLETE / HELD-OUT PARTIAL / PROPOSED POLICIES NOT RUN','Sources: P01-P02; E01; B01.',
 'First establish tuned AR, SD, and complete SSD timing. Then add uniform early publication, unequal fixed depths, and adaptive scheduling. Include fixed and online global depth/fan-out, probability-first, and a SwiftSpec-inspired timing adaptation. Match GPU/candidate budgets within the SSD policy ladder.')
deck.table(['Comparison','Question isolated','Status'],[
 ['AR → synchronous SD → complete SSD','Target baseline, ordinary speculation, overlap','Tuned; test partial'],
 ['Complete SSD → uniform early handoff','Does partial publication alone help?','Not run'],
 ['Uniform → unequal fixed depths','Does unequal allocation help?','Not run'],
 ['Unequal fixed → adaptive depths','Does online scheduling add value?','Not run'],
 ['Global adaptation / probability-first / timing rule','Can simpler heuristics match the scheduler?','Not run'],
 ],widths=[.38,.39,.23],size=14,rowh=57)

deck.start('Trade-offs: optimize useful output per elapsed time','K/F TUNING COMPLETE / OTHER POLICY TRADE-OFFS STILL TO MEASURE','Sources: E02-E08; P01-P02. Directions are hypotheses, not results.',
 'Increasing K can increase acceptance but delays readiness. Increasing F covers more outcomes but uses more memory. More publication opportunities reduce waiting but add event/control/drain work. An adaptive scheduler must improve latency after including its own decision cost.')
deck.table(['Knob','Potential benefit','Cost / risk to measure'],[
 ['Depth K','More accepted tokens per verification','Serial draft latency and later readiness'],
 ['Fan-out F / coverage','Fewer cache misses','Branch/KV memory and execution-shape cost'],
 ['Publication frequency','Earlier usable prefixes','Control/events, shorter prefixes, drain costs'],
 ['Unequal allocation','More work on likely useful outcomes','Poorly covered outcomes and fallback'],
 ['Online scheduler','Adapt to measured conditions','Estimator error and decision overhead'],
 ],widths=[.25,.33,.42],size=15,rowh=57)

deck.start('Core ablations: isolate publication and allocation','PLANNED / SHARED EXECUTOR, MATCHED BUDGETS AND TUNING','Sources: P01-P02. No proposed-policy result exists yet.',
 'These comparisons isolate the contribution in stages. Hold candidate generation and rankings fixed initially. Uniform early publication is the key novelty comparator, not just ordinary speculative decoding. Safety checks remain enabled in every deployable policy.')
deck.table(['Controlled A/B comparison','What it isolates'],[
 ['Complete publication vs uniform early publication','Value of exposing partial work'],
 ['Uniform early depths vs unequal fixed depths','Value of unequal allocation without learning'],
 ['Unequal fixed vs adaptive depths','Value of online scheduling'],
 ['Probability-first vs timing-aware selection','Value of measured costs and readiness'],
 ['Fixed vs online outcome statistics','Value and reliability of learned frequencies'],
 ],widths=[.57,.43],size=16,rowh=57)

deck.start('Mechanism ablations: account for real overhead','PLANNED / A ZERO-COST CONTROLLER IS ONLY A DIAGNOSTIC','Sources: P01-P02. Actual GPU completion checks remain enabled.',
 'Test readiness estimates on/off, fixed/online outcome statistics, and publication depth 1,2,4,8 clipped to available K. Measure real controller cost and compare to no-op invocation. An idealized zero-overhead policy is an upper bound, not a deployable result. Always charge obsolete in-flight work and stream interference.')
deck.table(['Ablation / stress','Record alongside latency'],[
 ['Readiness estimate on / off','Ready-prefix coverage, actual ready depth, wait'],
 ['Publication depths 1 / 2 / 4 / 8 (where K permits)','Accepted tokens, event cost, wait, fallback'],
 ['No-op / real controller; idealized ceiling labeled','Decision cost, drain cost, target slowdown'],
 ['Candidate budget and execution shape','Misses, branch memory, kernel efficiency'],
 ['Task family and natural lengths','Regressions, output length, p50/p95 delivery gaps'],
 ],widths=[.52,.48],size=15,rowh=57)

deck.start('Correctness: three earlier diagnostic prompts','COMPLETED DIAGNOSTICS / FAST EXACTNESS REMAINS UNRESOLVED','Sources: E09-E11. Original unchanged AR is the reference.',
 'All three fast SD/SSD prompts differ from AR, at zero-based indices 13,181,170. Same-history BF16 ties explain the immediate decisions. Attention-only replacement does not fix it. A slow synchronous numerical reference matches all 768 original AR tokens, but asynchronous SSD/natural EOS/full workload remain unvalidated. Tolerance is not a strict pass.')
deck.image(divplot,34,170,555,292)
deck.blocks([('Fast paths','SD = SSD, but both differ from AR on all three prompts.'),('Slow reference','768/768 original AR tokens match in synchronous SD.'),('Required next','Fast remedy; EOS, bonus, partial/stale and KV tests.')],size=16)

deck.start('Experiment coverage: what has actually finished','480 TUNING + 150 DEDICATED HELD-OUT REQUESTS / FIVE-REPEAT COMPARISON INCOMPLETE','Source: B01. No profiler; fresh worker + three warmups per block.',
 'The grid has ten configurations: AR, three SD K values, six SSD K/F combinations. Each sees 24 tuning prompts twice: 480 measurements. Freeze minimum mean latency winners before held-out access. Test AR and frozen winners on 75 prompts five times under fixed256 and naturalEOS capped256:2250 measurements. Total 50 worker blocks and150warmups.')
deck.table(['Stage','Declared settings','Measurements'],[
 ['Tuning: 24 prompts × 2 repeats','AR; SD K=4/6/8; SSD K=4/7/10 × F=1/3','480 / 480 done'],
 ['Winners frozen from tuning only','SD K6; SSD K7/F3; AR target-only control','Complete'],
 ['Held-out fixed 256 outputs','75 prompts × 5 rounds × 3 methods planned','150 / 1,125 done'],
 ['Held-out natural EOS, cap 256','Same configurations; secondary protocol','0 / 1,125; deferred'],
 ['GPU0-only SD while GPU1 is busy','Separate contextual run; excluded from dedicated totals','75 completed (contextual)'],
 ],widths=[.35,.44,.21],size=14,rowh=57)

deck.start('Fairness and analysis controls','TUNING COMPLETE / HELD-OUT PARTIAL / CONTEXTUAL SD LABELLED','Sources: B01; P01-P02. Fast output mismatches stay beside timing results.',
 'AR and SD use one GPU, SSD uses two. Within SSD policy comparisons hold two GPUs and candidate construction fixed. Disable prompt-prefix lookup, include both prefills, warm weights/kernels, keep profiler out of scored timing, shuffle/alternate order, pair prompt and round, report 10000-sample bootstrap intervals. Dedicated-window watcher kills only our contaminated block and retries after idle.')
deck.blocks([
 ('Match the workload','Same frozen token IDs, weights, BF16, batch one, CUDA graphs, JIT fallback. Prefix-cache lookup disabled; full prefills counted.'),
 ('Make resource use explicit','AR/SD: 1 GPU. SSD: 2 GPUs. SSD-policy comparisons share the same GPU/candidate budgets and tuning effort.'),
 ('Pair and repeat','Five seeded rounds; balanced chat/code/document mix. Report per-family regressions and 10,000-draw paired bootstrap CIs.'),
     ('Keep timing valid','No profiling in scored runs. Contaminated blocks are discarded and retried after both GPUs are free.')],x=42,y=162,w=870,size=16)

deck.start('What remains, and what would count as success?','PENDING / THE 10% IMPROVEMENT IS A TARGET, NOT A RESULT','Sources: P01-P02; B01; E09-E11.',
 'Run the frozen campaign, resolve strict correctness, implement safe uniform early handoff and the comparator ladder, then run one-component ablations. Broader final evaluation includes approximately100prompts per family, feasible512/2K/8Kinputs, naturalEOS output128/512 and3090replication. 8K requires increasing current4096capacity. Success is10percent lower full-request latency than strongest matched baseline, exact target tokens, and disclosed family regressions. If scheduler adds no value, retain simpler correct policy.')
deck.table(['Remaining work','Why it is needed'],[
 ['Finish held-out repetitions; natural EOS later','Tuning is done; repeated validation and exactness remain open'],
 ['Resolve fast correctness + publication-state tests','Meet unchanged-target token contract before acceleration claims'],
 ['Implement early publication and matched-policy ladder','Test the proposed contribution against the strongest simple policy'],
 ['Run component ablations; collect memory/delivery/waste','Explain gains or regressions and include controller/stream costs'],
 ['Expand lengths/workload; feasible device replication','Check generalization and hardware/memory sensitivity'],
 ],widths=[.49,.51],size=14,rowh=54)
deck.text('Decision: ≥10% lower request latency than the strongest matched baseline, exact output, and all family regressions disclosed.',42,474,875,11,NAVY,True,maxh=22)

deck.start('Backup: observed profiled request timing','PROFILED DIAGNOSTIC / EXACTNESS UNRESOLVED / SSD USES AN EXTRA GPU','Source: E01. Three prompts per mode, one saved timing run; no confidence intervals.',
 'The descriptive throughput is44.0 AR,62.5 SD,80.6SSD tokens/second, with means5.82,4.09,3.18 seconds/request. Ratios1.42x and1.83x are illustrative only. No unprofiled repeated heldout timing exists yet, and outputs differ.')
deck.image(perfplot,34,162,560,306)
deck.blocks([('Observed','44.0 / 62.5 / 80.6 committed tokens/s for AR / SD / SSD.'),('Interpret carefully','Only three prompts, profiled. SD/SSD outputs differ from AR.'),('Resources','AR and SD: 1 GPU.<br/>SSD: 2 GPUs.')],size=16)

deck.start('Backup: optimize the critical path, not just FLOPs','MEASURED KERNEL DIAGNOSTICS / NOT END-TO-END POLICY RESULTS','Sources: E02; E08. Five isolated Nsight Compute phase captures.',
 'Target decode and verify are95percent memorybound; prefill is85percent computebound; drafter has29to37percent latencyboundkernel time. Measured roof211TFLOPS and747GB/s gives ridge283FLOP/byte. This motivates amortized verification and reducing draft waiting/host overhead, not assuming more target compute peak will solve it.')
deck.image(boundplot,34,165,565,305)
deck.blocks([('Target decode / verify','~95% memory-bound kernel time.'),('Drafter','Small kernels plus host and launch gaps limit readiness.'),('Project focus','Publication, allocation, and measured scheduling costs.')],size=17)
deck.finish()

# Persist a compact numeric snapshot and source hashes for independent audit.
SOURCES['E12'] += [str(OUT/'campaign_snapshot'/name) for name in ['fixed_window_tests.txt','gpu0_sd_tests.txt']]
SOURCES['E05'].append('docs/perf_analysis.md')
SOURCES['E08'].append('docs/perf_analysis.md')
evidence=[]
for key,paths in SOURCES.items():
 for path in paths:
  p=Path(path) if Path(path).is_absolute() else ROOT/path
  evidence.append(dict(id=key,path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(OUT/'evidence_manifest.json').write_text(json.dumps(dict(generated_at=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),source_basis='Saved completed experiments only; report generation launches no GPU work',campaign_scored_requests=completed['tuning_requests']+completed['heldout_requests'],contextual_sd_requests=75 if CONTEXT else 0,manifest_sha256=manifest['manifest_sha256'],sources=evidence),indent=2)+'\n')
(OUT/'metrics_snapshot.json').write_text(json.dumps(dict(profiled={m:r['summary'] for m,r in runs.items()},machine_roofline=roof,critical_path=critical['summary'],ncu={k:v['overall'] for k,v in ncu.items()},first_divergences=[dict(id=r['id'],zero_based=r['first_divergence_zero_based']) for r in correct['requests']],backend_controls=backends['free_generation_pairs'],serial_reference=backends['serial_reference_vs_original_ar'],new_campaign=completed,pending=dict(tuning=0,fixed=completed['remaining_fixed_requests'],natural=completed['remaining_natural_requests'],full_campaign_complete=False)),indent=2)+'\n')
print(f'Report: {pdf.page} pages; slides: {len(deck.pages)}; output: {OUT}')
