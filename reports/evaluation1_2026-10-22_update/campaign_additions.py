"""New completed campaign figures and pages; all inputs are an immutable snapshot."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

NAVY='#152E49';TEAL='#008C88';BLUE='#4778AE';ORANGE='#D98225';GRAY='#697B8E';RED='#C64D56'

def held_rows(d):
 rows=[dict(mode=r['mode'],statistics=r['statistics'],scope='dedicated',family=r['family']) for r in d['heldout'] if r['repeat']==0]
 if d.get('contextual_sd'):
  ctx=d['contextual_sd'];rows.append(dict(mode='sd',statistics=ctx['statistics'],scope='GPU0-only / GPU1 occupied',family='all'))
  rows += [dict(mode='sd',statistics=s,scope='GPU0-only / GPU1 occupied',family=f) for f,s in ctx['families'].items()]
 return rows

def figures(d,folder):
 folder=Path(folder);out={}
 def save(fig,name):
  path=folder/(name+'.png');fig.savefig(path,dpi=210,bbox_inches='tight');plt.close(fig);out[name]=path
 rows=d['tuning']
 fig,axes=plt.subplots(1,2,figsize=(9,4))
 sd=sorted([r for r in rows if r['config']['mode']=='sd'],key=lambda r:r['config']['k'])
 xs=[r['config']['k'] for r in sd]
 for repeat,color in [(0,'#A4BFBD'),(1,'#C6DAD9')]:
  axes[0].plot(xs,[r['repeat_mean_latency_s'][repeat] for r in sd],color=color,marker='.',label=f'Round {repeat}')
 axes[0].plot(xs,[r['statistics']['mean_latency_s'] for r in sd],marker='o',color=TEAL,lw=2,label='Pooled mean')
 for r in sd:
  axes[0].annotate(f'{r["statistics"]["mean_latency_s"]:.3f}s',xy=(r['config']['k'],r['statistics']['mean_latency_s']),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10)
 axes[0].set(xlabel='SD depth K',ylabel='Full-request mean latency (s)',xticks=xs,ylim=(2.89,3.11),title='SD tuning: K6 selected')
 axes[0].legend(frameon=False,fontsize=8,loc='lower right')
 mat=np.array([[next(r['statistics']['mean_latency_s'] for r in rows if r['config']==dict(mode='ssd',k=k,f=f)) for f in [1,3]] for k in [4,7,10]])
 im=axes[1].imshow(mat,aspect='auto',cmap='viridis_r');axes[1].set(xticks=[0,1],xticklabels=[1,3],yticks=[0,1,2],yticklabels=[4,7,10],xlabel='SSD fan-out F',ylabel='SSD depth K',title='SSD tuning: K7/F3 selected')
 for y in range(3):
  for x in range(2):axes[1].text(x,y,f'{mat[y,x]:.3f}s'+(' *' if (y,x)==(1,1) else ''),ha='center',va='center',color='white' if mat[y,x]>2.65 else NAVY,fontweight='bold',fontsize=11)
 fig.colorbar(im,ax=axes[1],label='Mean request seconds');fig.tight_layout(w_pad=2);save(fig,'completed_tuning')

 allrows=held_rows(d);available=[m for m in ['ar','sd','ssd'] if any(r['mode']==m and r['family']=='all' for r in allrows)]
 fig,ax=plt.subplots(figsize=(8,4));x=np.arange(4);width=.23 if len(available)==3 else .3
 for i,m in enumerate(available):
  vals=[next(r['statistics']['mean_latency_s'] for r in allrows if r['mode']==m and r['family']==f) for f in ['all','chat','code','doc']]
  label={'ar':'AR (1 GPU, dedicated)','sd':'SD (1 GPU; GPU1 occupied)','ssd':'SSD (2 GPUs, dedicated)'}[m]
  ax.bar(x+(i-(len(available)-1)/2)*width,vals,width=width*.93,label=label,color={'ar':GRAY,'sd':ORANGE,'ssd':BLUE}[m])
 ax.set(xticks=x,xticklabels=['Balanced mix','Chat','Code','Document'],ylabel='Mean request seconds (one round)',ylim=(0,6.8))
 ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.17),fontsize=8.8,ncol=2)
 fig.tight_layout();save(fig,'partial_heldout')

 fig,ax=plt.subplots(figsize=(8,4));labels=[];pct=[];colors=[];count=[]
 for mode in ['sd','ssd']:
  r=next(r for r in rows if r['selected'] and r['config']['mode']==mode);c=r['correctness']
  labels.append(mode.upper()+' selected / tuning (48 requests)');count.append((c['requests_matching_ar_round0'],48));colors.append(TEAL if mode=='sd' else BLUE)
 if d.get('heldout_ssd_vs_ar'):
  c=d['heldout_ssd_vs_ar'];labels.append('SSD held-out (one round, 75 prompts)');count.append((c['requests_matching_ar_round0'],75));colors.append(BLUE)
 if d.get('contextual_sd'):
  c=d['contextual_sd']['correctness'];labels.append('SD held-out (GPU0-only contextual)');count.append((c['requests_matching_ar_round0'],75));colors.append(ORANGE)
 pct=[100*a/b for a,b in count];ax.barh(labels,pct,color=colors,height=.55)
 for i,(a,b) in enumerate(count):ax.text(pct[i]+2,i,f'{a}/{b} exact',va='center',fontsize=10)
 ax.set_xlim(0,100);ax.set_xlabel('Requests exactly matching AR round-0 output (%)');ax.invert_yaxis();fig.tight_layout();save(fig,'new_exactness')
 return out


def report_pages(pdf,d,figs,outdir):
 ctx=d.get('contextual_sd');ctxtext='75 completed' if ctx else 'not complete in this snapshot'
 pdf.start('New experiment set: unprofiled tuning and held-out','Snapshot: '+d['captured_at'].replace('T',' ')[:25],'COMPLETED: 480 TUNING + 150 DEDICATED HELD-OUT / CONTEXTUAL SD IS SEPARATE')
 y=pdf.table(['Setting','Exact experiment set','Coverage at this snapshot'],[
  ['Models and decoding','Qwen3-8B target / Qwen3-0.6B drafter; BF16; greedy; batch one; CUDA graphs; JIT fallback.','Same numerical fast paths as the frozen campaign; exactness remains unresolved.'],
  ['Inputs / outputs','Complete inputs <=2,048 tokens; model capacity 4,096; fixed 256 outputs with EOS ignored.','Same frozen token IDs and chat template across methods.'],
  ['Tuning set and grid','24 prompts: 8 chat + 8 code + 8 document. AR; SD K=4/6/8; SSD K=4/7/10 x F=1/3. Two repetitions.','All 20 blocks / 480 requests complete, 48 observations per configuration.'],
  ['Held-out experiment','75 prompts: 25 per family. Frozen winners + AR. Five repetitions per method planned.','AR 1/5; SSD 1/5; dedicated SD 0/5. GPU0-only SD: '+ctxtext+'.'],
  ['Timing and isolation','Three separate warmups per fresh worker; no profiler; prefix-cache lookup off; full prefills included; KV fractions 0.70/0.70.','AR/SD use GPU0; SSD uses GPU0/1. GPU0-only SD relaxes GPU1 exclusivity and is labeled contextual.'],
 ],widths=[.21,.45,.34],size=9.1)
 pdf.para('Tuning winner selection uses mean full-request latency across the balanced tuning mix. Configurations are frozen before held-out evaluation. Excluded previous held-out IDs: specbench-148, HumanEval/36, specbench-553. No prompt truncation is used.',32,y+16,778,9.4,maxh=55)
 pdf.source('B01-B04. Validated receipt and result hashes are copied into campaign_snapshot/; live waiting-status changes cannot alter this report snapshot.')

 pdf.start('Batch size in related decoding benchmarks','Request concurrency and internal speculative-token batching are different quantities.','LITERATURE CONTEXT / OUR INFERENCE REQUEST BATCH SIZE IS ONE')
 end=pdf.table(['Work / experiment','Request batch size','What the reported batch means'],[
  ['Original speculative decoding (Leviathan et al., 2023)','1 for measured latency','T5-XXL wall-time experiment on one TPU-v4; lookahead tokens are internal work.'],
  ['Speculative Speculative Decoding (2026)','1 by default in latency/baseline setup','The AR/SD and inference-engine baselines use a single concurrent request unless stated otherwise; separate batch-scaling studies also exist.'],
  ['EAGLE-3 (original v1)','1 in latency discussion; 2/4/8/16/24/32/48/56 in throughput Table 3','Larger-batch measurements are a distinct vLLM throughput experiment, not the same setup as its main latency results.'],
  ['SwiftSpec (2025)','Single-query focus; internal target/draft width 8','The selected internal batch sizes bs=8 and w=8 refer to tree nodes for one query; they do not mean eight concurrent users.'],
 ],widths=[.29,.30,.41],size=9.4)
 pdf.para('<b>For this project:</b> batch size one matches the low-latency research question. It does not establish throughput under multiple concurrent users. Matching request batch size alone does not make tokens/s comparable across papers: models, devices, precision, and prefill timing boundaries also differ.',32,end+17,778,10.0,maxh=66)
 pdf.para('Primary sources: <link href="https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf">Original SD, Section 4.1</link>; <link href="https://arxiv.org/html/2603.03251v1">SSD, Appendix B.2</link>; <link href="https://arxiv.org/html/2503.01840v1">EAGLE-3 v1, Section 4.3 / Table 3</link>; <link href="https://arxiv.org/html/2506.11309v1">SwiftSpec, Sections 2.3.2, 3.1 and 5.5</link>.',32,end+88,778,8.8,maxh=45)
 pdf.source('L01-L04. These are documented reporting setups, not experiments performed by our project.')

 pdf.chartpage('Completed tuning: K6 and K7/F3 win on this mix','24 tuning prompts x two repetitions x ten configurations = 480 measured requests.',figs['completed_tuning'],[
  ('Measured winners','AR control: 5.575 s/request.<br/>SD K6: 2.947 s/request.<br/>SSD K7/F3: 2.140 s/request.'),
  ('Trade-off observed','K10/F3 is ~39% slower than K7/F3 here. F3 lowers the mean relative to F1 at every tested K. More draft work does not monotonically improve latency.'),
  ('Selection limits','SD K6 is only ~1.6% ahead of K4; SSD K7/F3 is ~2.2% ahead of K4/F3. Two tuning rounds select settings; they do not establish statistically reliable held-out gains.')],
  'B02. All workers unprofiled with warmups and prefix reuse disabled. Curves are measured selection data, not the final test score.',tag='COMPLETED UNPROFILED TUNING / FAST EXACTNESS UNRESOLVED')

 pdf.start('All tuning configurations: latency, acceptance, exactness','48 request observations per row; two repeats of the same 24 tuning prompts.','COMPLETED SELECTION DATA / NOT A HELD-OUT GENERALIZATION CLAIM')
 def val(v,fmt='.3f'):return '—' if v is None else format(v,fmt)
 table=[]
 for r in d['tuning']:
  s=r['statistics'];c=r['correctness'];label=r['config_id']+(' *' if r['selected'] else '')
  table.append([label,f'{s["mean_latency_s"]:.3f} / {s["p50_latency_s"]:.3f} / {s["p95_latency_s"]:.3f}',
   f'{s["committed_tok_per_s"]:.1f}',f'{1000*s["mean_first_delivery_s"]:.1f}',val(s['draft_acceptance_rate']),val(s['mean_cache_hit_rate']),f'{c["requests_matching_ar_round0"]}/48',f'{c["repeatable_prompts"]}/24'])
 end=pdf.table(['Configuration','Mean / p50 / p95 (s)','tok/s','First ms','Draft accept','Cache hit','Exact AR','Stable IDs'],table,widths=[.15,.24,.10,.10,.10,.10,.11,.10],size=8.0)
 pdf.para('* Selected configuration for its method (AR is the control). Draft acceptance excludes recovery tokens and precedes final output-cap trimming. Cache hit is SSD speculation-cache hit, not prompt-cache reuse. Stable IDs means equal output across both tuning repetitions. All outputs have length 256.',32,end+15,778,9.0,maxh=50)
 pdf.source('B02. All fast speculative settings match AR for 16/48 request observations and repeat exactly on 24/24 prompt IDs. Selection ties use smaller K then F.')

 pdf.chartpage('Partial held-out results on unseen prompts','One round per available method; 75 prompts each. No repeated-run CI or final ranking.',figs['partial_heldout'],[
  ('Dedicated AR / SSD','AR: 5.575 s, 45.9 tok/s.<br/>SSD: 2.174 s, 117.8 tok/s.<br/>SSD uses two GPUs; AR uses one.'),
  ('SD single-GPU scope',('SD has a separate GPU0-exclusive run with GPU1 occupied; see the next table. It is excluded from dedicated-campaign totals.' if ctx else 'The dedicated SD block was contaminated and excluded. The separate GPU0-only run is not yet complete. No SD timing is invented.')),
  ('Interpretation','The AR/SSD mean ratio is 2.56x in this one round, with output mismatches. This is preliminary performance evidence, not a correctness-preserving acceleration result or five-round conclusion.')],
  'B03-B04. Per-family bars summarize 25 prompts; overall bars summarize the balanced 75-prompt mix. SD context differs if displayed.',tag='PARTIAL UNPROFILED HELD-OUT / GPU COUNTS AND ISOLATION DIFFERENCES DISCLOSED')

 pdf.start('Partial held-out metrics and the excluded SD attempt','Same 75 prompt IDs and fixed 256 output budget; the repetition count is still one.','DESCRIPTIVE ONLY / FIVE-REPEAT COMPLETENESS GATE REMAINS OPEN')
 allrows=[r for r in held_rows(d) if r['family']=='all'];table=[]
 for r in sorted(allrows,key=lambda r:['ar','sd','ssd'].index(r['mode'])):
  s=r['statistics'];table.append([r['mode'].upper()+(' *' if r['mode']=='sd' else ''),r['scope'],f'{s["mean_latency_s"]:.3f} / {s["p50_latency_s"]:.3f} / {s["p95_latency_s"]:.3f}',f'{s["committed_tok_per_s"]:.1f}',f'{1000*s["mean_first_delivery_s"]:.1f}',val(s['draft_acceptance_rate']),val(s['mean_cache_hit_rate'])])
 end=pdf.table(['Method','Window scope','Mean / p50 / p95 (s)','tok/s','First ms','Accept','Cache hit'],table,widths=[.08,.25,.25,.10,.12,.10,.10],size=9.1)
 pdf.blocks([('Latency and first delivery are different','SSD first delivery averages 97.4 ms versus AR 46.8 ms, despite lower full-request latency. Extra draft-prefill/control work is a plausible explanation, but these timings alone do not isolate the cause.'),('Contaminated and contextual SD','One dedicated SD attempt was stopped when an unrelated job appeared on GPU1. Its entire block is excluded. The user-authorized replacement monitors GPU0 exclusively and records GPU1 processes. A clean replacement is usable contextual evidence, but cannot count as a dedicated-window repetition.'),('What is not known yet','The p50/p95 values describe the prompt distribution within this single round. Repeated-run uncertainty, five-round repeatability, and final family-regression conclusions remain unmeasured.')],32,end+21,778,10.0,maxy=525)
 pdf.source('B03-B04. If a contextual SD row is present, * denotes GPU1 activity was allowed; no dedicated-GPU comparability claim is made for that row.')

 pdf.chartpage('Repeatable output can still disagree with AR','New tuning and held-out exactness results broaden the earlier three-prompt diagnosis.',figs['new_exactness'],[
  ('Tuning: deterministic mismatch','Selected SD and SSD match the AR reference in 16/48 requests (8 of 24 unique prompts). All tested settings repeat their own outputs on 24/24 prompt IDs.'),
  ('Held-out SSD','29/75 prompts match AR exactly; 46/75 differ in this one round. Per-family counts and first-divergence positions are saved in the snapshot JSON.'),
  ('What it means','Repeatability is not target equivalence. We have not diagnosed every new mismatch individually. No within-method held-out repeatability claim is possible with one round.')],
  'B02-B04; E09-E11. Tuning observations repeat prompts; the 48-request fraction is not 48 independent prompts.',tag='COMPLETED OUTPUT COMPARISONS / STRICT FAST-MODE CORRECTNESS STILL FAILS')


def slides(deck,d,figs):
 ctx=d.get('contextual_sd')
 deck.start('Experiment set: models, workloads, and isolation','MEASURED TUNING / PARTIAL HELD-OUT / SAME FROZEN INPUT TOKEN IDS','Sources: B01-B04. Snapshot '+d['captured_at'].replace('T',' ')[:25],
  'Use this as the experimental setup slide. Both models, BF16, greedy batch one, input limit2048 and model capacity4096 are fixed. Tuning is complete with24 balanced prompts and two repeats per setting. Heldout75 balanced prompts has only one AR and SSD round; SD GPU0-only context is separate from dedicated data. All workers have three warmups, full prefill, no prefix reuse and no profiler.')
 deck.table(['Component','Exact experiment set'],[
  ['Models / decoding','Qwen3-8B + Qwen3-0.6B; BF16; greedy; batch one; CUDA graphs; JIT fallback'],
  ['Lengths','Complete input ≤2,048; capacity 4,096; fixed 256 outputs; EOS ignored'],
  ['Tuning','24 prompts (8/family) × 2 repeats × 10 configurations = 480 valid measurements'],
  ['Held-out','75 prompts (25/family); 5 repeats planned. AR/SSD: 1 valid round each; SD contextual run separate'],
  ['Timing / GPUs','3 warmups; full prefills; no profiler/prefix reuse. AR/SD: 1 GPU; SSD: 2; KV fractions 0.70'],
 ],widths=[.24,.76],size=14,rowh=57)

 deck.start('Batch size one is a latency benchmark choice','REQUEST BATCH ≠ VERIFICATION WIDTH ≠ NUMBER OF DRAFT BRANCHES','Sources: L01-L04. Paper versions and direct primary-source links are in the report.',
  'Original SD measures latency at request batch one. SSD defaults to one concurrent request for its baselines. EAGLE-3 also tests multi-request throughput at batches2to56 in v1. SwiftSpec serves a single query with internal target and draft width8; those are tree nodes, not eight user requests. Our batch one answers the latency question and cannot support multiuser throughput claims.')
 deck.table(['Benchmark','Batching used in the paper'],[
  ['Original SD latency experiment','Request batch 1 on one TPU-v4'],
  ['SSD latency / baseline setup','One concurrent request by default; separate scaling studies'],
  ['EAGLE-3','Batch 1 for latency; original v1 throughput sweep 2–56 requests'],
  ['SwiftSpec','Single query; internal target/draft widths 8 are tree-node batches'],
 ],widths=[.36,.64],size=16,rowh=63)
 deck.text('Our scope: low-latency generation of one request. Multi-user serving throughput would require a separate batch/concurrency study.',42,459,870,13,NAVY,True,maxh=36)

 deck.start('Completed tuning: the measured K/F trade-off','480 VALID MEASUREMENTS / WINNERS SELECTED BEFORE HELD-OUT RESULTS','Source: B02. Two repeats ×24 balanced prompts per setting; GPU allocation disclosed.',
  'The full grid is complete. ARmean5.575s; SD K6mean2.947s; SSD K7/F3mean2.140s. SD K4is2.994andK8is3.053. SSD deeper K10is about39percent slower than K7/F3 atF3. Differences between K4andK6 SD and K4/F3andK7/F3 SSD are small, so the heldout repeated validation remains important. No novel earlypublication policy is measured.')
 deck.image(figs['completed_tuning'],34,161,565,307)
 deck.blocks([('Selected SD','K=6: 2.947 s/request.<br/>K=4: 2.994; K=8: 3.053.'),('Selected SSD','K=7, F=3: 2.140 s/request.<br/>K=10/F3: 2.978 s.'),('Boundary','Selection data only. Fast output exactness still fails.')],size=16)

 deck.start('Held-out results so far: one round, not five','AR/SSD DEDICATED / SD GPU0-ONLY CONTEXT SEPARATELY LABELED','Sources: B03-B04. 75 unseen prompts per completed method; no repeated-run CI.',
  'AR and SSD have one valid75prompt round each. ARmean5.575seconds45.9tokens/s; SSDmean2.174seconds117.8tokens/s with twoGPUs. AR/SSDmeanratio2.56 is descriptive only. SSD agrees withARon29/75prompts. The contaminated SD trial is excluded; the GPU0-only replacement is separately labeled if complete. More repeats and correctness remain open.')
 deck.image(figs['partial_heldout'],34,161,565,307)
 sdline=(f'SD: {ctx["statistics"]["mean_latency_s"]:.3f} s, GPU0-only; GPU1 occupied.' if ctx else 'SD dedicated attempt excluded; GPU0-only run not yet complete.')
 deck.blocks([('Measured so far','AR: 5.575 s; SSD: 2.174 s.<br/>'+sdline),('Exactness','SSD matches AR on 29/75 prompts; 46 differ.'),('Still incomplete','One round only. Complete five valid repetitions before headline CIs.')],size=15)
