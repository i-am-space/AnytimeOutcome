"""Publication figures from the immutable evidence snapshot; CPU only."""
from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
import numpy as np
OUT=Path(__file__).resolve().parent
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
M=json.loads((OUT/'evidence_manifest.json').read_text());A=json.loads((OUT/'analysis.json').read_text())
def load(key):return json.loads((OUT/'evidence_snapshot'/M['sources'][key]).read_text())
B,P,C=load('baseline'),load('publication'),load('context')
COL={'ar':'#75869A','sd':'#008C88','ssd':'#315F9E','complete':'#315F9E','uniform':'#A779B8','overlap':'#E49532','ranked':'#168D95','ranked_overlap':'#8164AC','budget':'#8A9580','reverse':'#BF6570'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.titlesize':16,'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,'savefig.facecolor':'white'})
D={};META={}
def save(fig,name,values,sources,caption):
 fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=190,bbox_inches='tight');plt.close(fig)
 D[name]=values;META[name]={'sources':sources,'caption':caption}
def br(protocol,mode,family='all'):return next(r for r in B['results'] if (r['protocol'],r['mode'],r['family'])==(protocol,mode,family))
def pr(protocol,method,family='all'):return next(r for r in P['results'] if (r['protocol'],r['method'],r['family'])==(protocol,method,family))
for protocol in ['fixed','natural']:
 fig,ax=plt.subplots(figsize=(8.4,4.2));rows=[br(protocol,m) for m in ['ar','sd','ssd']];ys=[r['statistics']['mean_latency_s'] for r in rows]
 ci=[r['comparison_to_ar']['mean_latency_ci95_s'] for r in rows]
 ax.bar(['Shared AR\n1 GPU','SD K=6\n1 GPU','Complete SSD K=7/F=3\n2 GPUs'],ys,color=[COL[m] for m in ['ar','sd','ssd']],width=.56,yerr=np.array([[v-l[0] for v,l in zip(ys,ci)],[l[1]-v for v,l in zip(ys,ci)]]),capsize=5)
 for i,v in enumerate(ys):ax.text(i,v+.16,f'{v:.3f} s',ha='center',weight='bold',fontsize=15)
 ax.set(ylabel='Mean full-request latency (s)',ylim=(0,7.3));ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
 save(fig,'baseline_'+protocol,[{k:r[k] for k in ['mode','gpu_count','comparison_to_ar']}|{'statistics':r['statistics']} for r in rows],['baseline'],f'{protocol}: 75 prompts × five rounds per method; shared-fixed16-v1; unprofiled dedicated runs. Error bars are hierarchical 95% bootstrap intervals for means.')
fig,ax=plt.subplots(figsize=(8.8,4.2));x=np.arange(3)
for i,m in enumerate(['ar','sd','ssd']):
 vals=[br('fixed',m,f)['statistics']['mean_latency_s'] for f in ['chat','code','doc']]
 ax.bar(x+(i-1)*.24,vals,width=.23,label=m.upper()+(' (2 GPUs)' if m=='ssd' else ' (1 GPU)'),color=COL[m])
ax.set(xticks=x,xticklabels=['Chat','Code','Document'],ylabel='Mean full-request latency (s)',ylim=(0,7.4));ax.legend(ncol=3,loc='upper center');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
save(fig,'family_latency',[{'method':m,'family':f,**br('fixed',m,f)['statistics']} for m in ['ar','sd','ssd'] for f in ['chat','code','doc']],['baseline'],'Fixed256; 25 prompts/family × five rounds. Code has higher measured draft acceptance on this mix; this is not a code-quality evaluation.')
S=load('baseline_selection');fig,axes=plt.subplots(1,2,figsize=(10,4.4))
rows=sorted([r for r in S['scores'] if r['config']['mode']=='sd'],key=lambda r:r['config']['k']);axes[0].plot([r['config']['k'] for r in rows],[r['mean_latency_s'] for r in rows],'o-',color=COL['sd'],lw=2.5);axes[0].set(xlabel='Draft depth K',ylabel='Mean request latency (s)',title='SD tuning');axes[0].set_xticks([4,6,8])
for f,col in [(1,COL['ar']),(3,COL['ssd'])]:
 rows=sorted([r for r in S['scores'] if r['config']['mode']=='ssd' and r['config']['f']==f],key=lambda r:r['config']['k']);axes[1].plot([r['config']['k'] for r in rows],[r['mean_latency_s'] for r in rows],'o-',label=f'F={f}',color=col,lw=2.5)
axes[1].set(xlabel='Draft depth K',ylabel='Mean request latency (s)',title='SSD tuning');axes[1].set_xticks([4,7,10]);axes[1].legend()
save(fig,'baseline_tuning',S['scores'],['baseline_selection'],'24 tuning prompts × two repetitions per setting; tune fixed256 only. Selected SD K6 and SSD K7/F3 before held-out evaluation.')
fig,ax=plt.subplots(figsize=(8.8,4.3));values=[]
for i,(protocol,method) in enumerate([('fixed','uniform'),('fixed','overlap'),('natural','uniform'),('natural','overlap')]):
 r=pr(protocol,method);c=r['comparison_to']['complete'];v=-100*c['latency_reduction'];lo,hi=[-100*t for t in c['latency_reduction_ci95'][::-1]]
 ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=COL[method],capsize=5,ms=9)
 values.append({'protocol':protocol,'method':method,'latency_increase_pct':v,'ci95':[lo,hi],'mean_s':r['statistics']['mean_latency_s']})
ax.axvline(0,color=COL['ssd'],ls='--',lw=1.4);ax.set(yticks=range(4),yticklabels=['Fixed: uniform P7','Fixed: overlap P4/W2','Natural: uniform P7','Natural: overlap P4/W2'],xlabel='Request latency change versus complete K=7 (%)',xlim=(-.3,1.8));ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
save(fig,'publication_effect',values,['publication'],'Positive means slower. Five paired rounds × 75 reused test prompts; all methods use two GPUs. 95% intervals from 10,000 paired bootstrap samples.')
fig,ax=plt.subplots(figsize=(9,4.5));vals=[]
for label,name,color,style in [('complete_k7','Complete K=7',COL['complete'],'-'),('complete_k10','Complete K=10',COL['ar'],'--'),('uniform_k10_p4','Uniform K=10/P=4',COL['uniform'],'-.'),('overlap_k10_p4_w2','Overlap K=10/P=4',COL['overlap'],'-')]:
 rows=sorted([r for r in C['results'] if r['config']['label']==label],key=lambda r:r['context_length']);ax.plot([r['context_length'] for r in rows],[r['mean_latency_s'] for r in rows],marker='o',ls=style,label=name,color=color,lw=2.5)
 vals.extend({'config':label,'input_tokens':r['context_length'],'latency_s':r['mean_latency_s']} for r in rows)
ax.set(xlabel='Initial input length (tokens)',ylabel='Mean full-request latency (s)',xticks=[256,512,1024,2048],ylim=(1.45,4.15));ax.legend(ncol=2,loc='upper left');ax.grid(alpha=.15)
save(fig,'context_latency',vals,['context'],'Synthetic length diagnostic: one template/family, three rounds, fixed256 outputs. Publication improves K10 but complete K7 is faster at every measured length. All curves use two GPUs.')
fig,ax=plt.subplots(figsize=(8.5,4));x=np.arange(4);vals=[]
for label,color,offset in [('complete_k7',COL['complete'],-.13),('complete_k10',COL['ar'],.13)]:
 rows=sorted([r for r in C['profiles'] if r['config']['label']==label],key=lambda r:r['context_length']);ys=[r['full_tree_over_verify_gpu'] for r in rows]
 ax.bar(x+offset,ys,width=.25,label=label.replace('_',' '),color=color);vals.extend(rows)
ax.axhline(1,color='#D28931',ls='--',label='Tree time = verify time');ax.set(xticks=x,xticklabels=['256','512','1024','2048'],xlabel='Initial input tokens',ylabel='Full-tree GPU span / verify GPU span',ylim=(0,2.05));ax.legend(ncol=2,loc='upper left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
save(fig,'phase_balance',[{k:r[k] for k in ['context_length','k','mean_full_tree_gpu_span_ms','mean_verify_gpu_span_ms','full_tree_over_verify_gpu','full_trees']} for r in vals],['context'],'Separate profiled captures. K7 tree and verify spans are similar; K10 tree spans are about 1.6× verify. Spans include gaps and are not sums of isolated kernel durations.')
fig,axes=plt.subplots(1,2,figsize=(10,4.2));vals=[]
for ax,protocol in zip(axes,['fixed','natural']):
 rows=[r for r in A['depth_results'] if r['protocol']==protocol];names=[r['method'] for r in rows if r['method']!='complete']
 for i,method in enumerate(names):
  r=next(r for r in rows if r['method']==method);c=r['comparison'];v=-c['latency_reduction']*100;lo,hi=[-t*100 for t in c['latency_reduction_ci95'][::-1]]
  ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=COL[method],capsize=4,ms=7);vals.append({'protocol':protocol,'method':method,'increase_pct':v,'ci95':[lo,hi],'rounds':r['rounds'],'requests':r['requests']})
 ax.axvline(0,color=COL['complete'],ls='--');ax.set(yticks=range(len(names)),yticklabels=[s.replace('_',' + ') for s in names],xlabel='Latency change vs complete (%)',title=protocol.title()+f' · {len(rows[0]["rounds"])} paired rounds');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
save(fig,'unequal_depth',vals,['analysis.json','depth_selection'],'PROVISIONAL: only rounds with all six methods completed; 75 prompts/round, two GPUs per method. Not the final five-round result.')
fig,ax=plt.subplots(figsize=(8.3,4.1));fixed=[r for r in A['depth_results'] if r['protocol']=='fixed' and r['method']!='complete']
for i,r in enumerate(fixed):
 c=r['comparison'];v=-c['latency_reduction']*100;lo,hi=[-t*100 for t in c['latency_reduction_ci95'][::-1]]
 ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=COL[r['method']],capsize=5,ms=9)
ax.axvline(0,color=COL['complete'],ls='--');ax.set(yticks=range(len(fixed)),yticklabels=[r['method'].replace('_',' + ') for r in fixed],xlabel='Request latency change vs complete K7 (%)',xlim=(-1,19));ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
save(fig,'unequal_depth_fixed',fixed,['analysis.json'],'PROVISIONAL fixed256: three fully paired rounds, 75 prompts/round. Positive latency change means slower. Natural EOS shows the same direction in four complete paired rounds.')
fig,ax=plt.subplots(figsize=(8.5,3.8));ax.set_xlim(0,8);ax.set_ylim(-.7,3.7)
for y,(label,depths) in enumerate([('Complete',[7,7,7]),('Ranked',[7,4,4]),('Equal work',[5,5,5]),('Reverse',[4,4,7])]):
 for rank,depth in enumerate(depths):
  yy=3-y;ax.barh(yy+(rank-1)*.18,depth,height=.16,color=[COL['ssd'],COL['uniform'],COL['overlap']][rank],label=f'Recovery rank {rank+1}' if y==0 else None)
ax.set(yticks=[3,2,1,0],yticklabels=['Complete [7,7,7]','Ranked [7,4,4]','Uniform [5,5,5]','Reverse [4,4,7]'],xlabel='Prepared continuation tokens per recovery rank',xticks=range(8));ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.2));ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
save(fig,'depth_allocation',{'K':7,'F':3,'depths':[[7,7,7],[7,4,4],[5,5,5],[4,4,7]],'dense_rows':168,'ranked_rows':120},['.worktrees/outcome-depth-v1/docs/outcome_depth.md'],'Schematic allocation repeated at every acceptance position. Equal-work controls match query-row budgets, not measured latency or memory traffic.')
roof=load('roofline');fig,ax=plt.subplots(figsize=(8.6,4.8));xx=np.logspace(-1,3.4,200);peak=roof['bf16_tensor_tflops_measured'];bw=roof['dram_read_GBps'];ridge=roof['ridge_bf16_flop_per_byte_measured'];ax.loglog(xx,np.minimum(peak,xx*bw/1000),color='#26394B',lw=2.5,label='Measured L40S envelope')
points=[]
for n,label,col,marker in [('ar_decode','AR decode',COL['ar'],'o'),('sd_verify','Verify',COL['ssd'],'s'),('sd_draft','SD draft',COL['sd'],'^'),('ssd_tree','SSD tree step',COL['overlap'],'D'),('ar_prefill','Prefill','#8D6BA8','*')]:
 o=json.loads((OUT/'evidence_snapshot/results/ncu'/f'{n}_ncu.json').read_text())['overall'];ai=o['ai_flop_per_byte'];tf=o['flop']/o['total_kernel_time_us']/1e6;ax.scatter(ai,tf,s=95,color=col,marker=marker,label=label,zorder=5);points.append({'phase':n,'ai':ai,'tflops':tf,'bound_split':o['bound_split_pct']})
ax.axvline(ridge,color='#C85959',ls='--',lw=1.2);ax.text(ridge*1.07,.15,f'Ridge {ridge:.0f} FLOP/B',rotation=90,color='#B84B4B',fontsize=11);ax.set(xlabel='Arithmetic intensity (FLOP / DRAM byte)',ylabel='Achieved TFLOP/s',ylim=(.1,380));ax.legend(ncol=2,loc='upper left');ax.grid(which='both',alpha=.12)
save(fig,'roofline',{'roof':roof,'points':points},['roofline','results/ncu/*_ncu.json'],'Historical executor: five isolated Nsight Compute captures. Target verify is left of the ridge; prefill is compute-oriented. Padding, cache effects and replay influence positions.')
cp=load('critical_path')['summary'];fig,ax=plt.subplots(figsize=(8,4.0));bars=ax.bar(['Cache hit\n166 rounds','Cache miss\n64 rounds'],[cp['mean_wait_ms_hit'],cp['mean_wait_ms_miss']],color=[COL['sd'],COL['overlap']],width=.55)
for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+.8,f'{b.get_height():.1f} ms',ha='center',weight='bold')
ax.axhline(cp['mean_verify_ms'],ls='--',color=COL['ssd'],label=f'Mean verify {cp["mean_verify_ms"]:.1f} ms');ax.set(ylabel='Target handoff wait (ms)',ylim=(0,37));ax.legend();ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
save(fig,'historical_wait',cp,['critical_path'],'Historical 230 matched rounds. Dashed line is mean verification duration, shown for scale; it is not a wait threshold or an extra stacked cost.')
rounds=load('critical_path')['rounds'][:5];origin=rounds[0]['t_req']/1e6
fig,ax=plt.subplots(figsize=(9.1,3.5))
for i,r in enumerate(rounds):
 x=r['t_req']/1e6-origin;wait=r['wait_ms'];verify=r['verify_ms']
 ax.broken_barh([(x,wait)],(.8,.32),facecolors=COL['overlap'],label='Target wait' if i==0 else None)
 ax.broken_barh([(x+wait,verify)],(.8,.32),facecolors=COL['ssd'],label='Target verify' if i==0 else None)
 if r['tree_done_after_request_ms'] is not None:ax.broken_barh([(x,r['tree_done_after_request_ms'])],(.05,.32),facecolors=COL['sd'],label='Tree remaining on hit' if i==1 else None)
 ax.axvline(x,color='#9AA9B8',lw=.7,ls=':');ax.text(x+.7,1.20,'R'+str(i)+(' miss' if r['miss'] else ' hit'),fontsize=10)
ax.set(yticks=[.96,.21],yticklabels=['Target wall phases','Known draft tail'],xlabel='Elapsed ms from first selected target request',ylim=(-.18,1.9));ax.legend(ncol=3,loc='upper left');ax.grid(axis='x',alpha=.12)
save(fig,'measured_timeline',rounds,['critical_path'],'Measured reconstruction: first five matched historical rounds. The draft row shows only remaining tree time on hits, not whole-tree duration. Target wait includes dependencies and service; it is not wire-transfer time.')
old=load('old_ssd');w=old['window_ms'];d=old['devices'];busy=d['0']['busy_pct'];nccl=100*d['0']['category_ms']['nccl']/w
fig,ax=plt.subplots(figsize=(8.6,3.4));parts=[('Non-NCCL kernel activity',[busy-nccl,d['1']['busy_pct']],COL['ssd']),('NCCL kernel time',[nccl,0],COL['overlap']),('No active kernel',[100-busy,100-d['1']['busy_pct']], '#DDE4EB')];left=np.zeros(2)
for name,ys,col in parts:
 ax.barh(['Target GPU0','Drafter GPU1'],ys,left=left,color=col,label=name,height=.5)
 for i,v in enumerate(ys):
  if v>7:ax.text(left[i]+v/2,i,f'{v:.1f}%',ha='center',va='center',color='#23394B' if name=='No active kernel' else 'white',weight='bold')
 left+=ys
ax.set(xlabel='Share of measured GPU window (%)',xlim=(0,100));ax.legend(ncol=1,loc='upper left',bbox_to_anchor=(1,1));ax.invert_yaxis()
save(fig,'gpu_states',{'target_busy_pct':busy,'target_nccl_pct_of_window':nccl,'draft_busy_pct':d['1']['busy_pct']},['old_ssd'],'Historical Nsight Systems window. Non-NCCL time is a category, not a proof that all activity is useful; NCCL can spend time waiting for peer data.')
# Schematics deliberately have no measured time scale.
fig,ax=plt.subplots(figsize=(9,3.8));ax.set_xlim(0,10);ax.set_ylim(-.7,3.0);ax.axis('off')
for y,label,tail,hand in [(2.25,'Complete SSD',7.8,8.0),(1.25,'Uniform / drain first',6.0,6.2),(.25,'Overlap publication',6.0,4.3)]:
 ax.text(0,y+.17,label,fontsize=13,weight='bold');x0=3.1
 ax.add_patch(Rectangle((x0,y-.22),1.1,.42,color=COL['sd']));ax.text(x0+.55,y-.01,'Prefix ready',ha='center',va='center',color='white',fontsize=10)
 ax.add_patch(Rectangle((4.3,y-.22),tail-4.3,.42,color='#CDD6E1'));ax.text((4.3+tail)/2,y-.01,'Queued tail',ha='center',va='center',fontsize=10)
 ax.annotate('',xy=(hand,y-.36),xytext=(hand+1.3,y-.36),arrowprops={'arrowstyle':'<-','color':COL['overlap'],'lw':2})
 ax.text(hand+.65,y-.61,'Handoff',ha='center',color='#9C671D',fontsize=11)
ax.text(3.1,2.86,'Execution order only • schematic, not measured duration',fontsize=11,color='#60758A')
save(fig,'handoff_schematic',{'kind':'schematic','states':['prefix ready','queued tail','handoff'],'warning':'No timing scale'},['docs/prefix_publication.md','docs/adaptive_publication.md'],'Uniform finishes submitted work before delivery. Overlap delivers an independent completed snapshot while that work finishes, then joins before buffer reuse.')
fig,ax=plt.subplots(figsize=(9,3.5));ax.axis('off');boxes=[(.03,.50,.27,.34,'TARGET · GPU0','Qwen3-8B\nVerify → accept / correct',COL['ssd']),(.68,.50,.29,.34,'DRAFTER · GPU1','Qwen3-0.6B\nBuild outcome continuations',COL['sd']),(.32,.04,.35,.27,'SAFE HANDOFF','Valid prefix + context/epoch\nSnapshot ownership + drain',COL['overlap'])]
for x,y,w,h,t,b,col in boxes:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.018',facecolor=col,edgecolor='none',transform=ax.transAxes));ax.text(x+w/2,y+h*.72,t,transform=ax.transAxes,ha='center',color='white',weight='bold',fontsize=13);ax.text(x+w/2,y+h*.33,b,transform=ax.transAxes,ha='center',va='center',color='white',fontsize=11)
for a,b,label in [((.31,.72),(.66,.72),'Request / feedback'),((.69,.48),(.64,.27),'Completed prefix'),((.32,.22),(.16,.48),'Target-verified output')]:
 ax.annotate('',xy=b,xytext=a,xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':2,'color':'#718299'});ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.035,label,transform=ax.transAxes,ha='center',fontsize=10)
save(fig,'architecture',{'kind':'schematic','models':['Qwen3-8B','Qwen3-0.6B'],'gpus':2},['workload','docs/prefix_publication.md'],'SSD target and drafter use different physical L40S GPUs. AR/SD use GPU0 only. Architecture is schematic.')
(OUT/'figure_data.json').write_text(json.dumps(D,indent=2,allow_nan=False)+'\n');(OUT/'figure_manifest.json').write_text(json.dumps(META,indent=2)+'\n')
# Compact human-readable data for charts; the full structured values remain in JSON.
with (OUT/'figure_values.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['figure','record','values_json'])
 for name,values in D.items():
  for i,r in enumerate(values if isinstance(values,list) else [values]):w.writerow([name,i,json.dumps(r,sort_keys=True)])
print('Built',len(META),'figures from frozen evidence')
