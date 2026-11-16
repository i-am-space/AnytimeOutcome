"""Finish the frozen fixed-256 evaluation within an explicit wall-clock budget.

Reuses original workers, manifests, tuning evidence and block validation unchanged.
Natural-EOS blocks are deferred. This supervisor never marks the full campaign complete.
"""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from baseline_common import (ROOT, atomic_json, config_id, file_digest, freeze_json,
    heldout_jobs, read_json)
from run_baselines import (PendingWindow, accepted_result, interrupted, load_campaign,
    now, run_block, validated_selection)


class BudgetExpired(RuntimeError):
    pass


def expired(signum,frame):
    raise BudgetExpired('Two-hour window exhausted; valid blocks retained, interrupted block excluded.')


def fixed_jobs(manifest,selection):
    return [j for j in heldout_jobs(manifest,selection) if j['protocol']=='fixed']


def complete_results(folder,manifest,selection):
    # No report can claim five repetitions without every fixed block validating.
    return [accepted_result(folder,j,manifest) for j in fixed_jobs(manifest,selection)]


def progress(folder,manifest,selection):
    completed=[]
    for job in fixed_jobs(manifest,selection):
        try:
            accepted_result(folder,job,manifest)
        except (ValueError,KeyError,OSError):
            continue
        completed.append(job['id'])
    return dict(valid_fixed_blocks=len(completed),required_fixed_blocks=15,
                fixed_scored_requests=75*len(completed),completed=completed)


def handoff(request):
    """Let the identified existing block finish, then stop only our old watcher."""
    pid=request.get('takeover_watcher_pid')
    if pid is None:
        return
    def alive():
        try:
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
            state=Path(f'/proc/{pid}/stat').read_text().split()[2]
        except FileNotFoundError:
            return False
        if state=='Z':
            return False
        campaign_path=str(Path(request['current_block_receipt']).parents[4])
        if 'watch_baselines.py' not in cmd or campaign_path not in cmd:
            raise ValueError('Takeover PID no longer identifies our benchmark watcher')
        return True
    if not alive():
        return
    receipt=Path(request['current_block_receipt'])
    print('Waiting for the existing block receipt before handoff: '+str(receipt),flush=True)
    wait_until=min(request['deadline_unix']-60,time.time()+600)
    while alive() and not receipt.exists() and time.time()<wait_until:
        time.sleep(1)
    if alive():
        os.kill(pid,signal.SIGINT)  # Watcher forwards interrupt and releases its own worker session.
        until=time.time()+45
        while alive() and time.time()<until:
            time.sleep(1)
        if alive():
            raise ValueError('Existing watcher did not stop; refusing overlapping campaigns')
    print('Full-campaign watcher stopped; fixed-length-only supervisor taking over.',flush=True)


def write_report(folder,manifest,selection):
    from analyze_baselines import paired_bootstrap, output_checks, row_statistics
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=complete_results(folder,manifest,selection)
    records={(r['job']['config']['mode'],r['job']['repeat']):{x['id']:x for x in r['requests']} for r in results}
    prompts=manifest['splits']['heldout'];ids=[p['id'] for p in prompts];families=[p['family'] for p in prompts]
    grids={m:[[records[(m,i)][rid] for rid in ids] for i in range(5)] for m in ['ar','sd','ssd']}
    summaries=[];csvrows=[]
    for family in ['all','chat','code','doc']:
        ix=[i for i,f in enumerate(families) if family=='all' or f==family]
        reference=[[row[i] for i in ix] for row in grids['ar']]
        for mode in ['ar','sd','ssd']:
            grid=[[row[i] for i in ix] for row in grids[mode]]
            stats=row_statistics([r for row in grid for r in row])
            ci=paired_bootstrap([[r['latency_s'] for r in row] for row in reference],
                [[r['latency_s'] for r in row] for row in grid],[families[i] for i in ix],samples=10000,seed=0)
            check=output_checks(grid,reference)
            item=dict(protocol='fixed',family=family,mode=mode,gpu_count=2 if mode=='ssd' else 1,
                config=selection['winners'][mode],statistics=stats,comparison_to_ar=ci,correctness=check)
            summaries.append(item)
            csvrows.append(dict(family=family,mode=mode,gpu_count=item['gpu_count'],**stats,
                speedup=ci['speedup'],speedup_ci95_low=ci['speedup_ci95'][0],speedup_ci95_high=ci['speedup_ci95'][1],
                exact_matches_to_ar_round0=check['requests_matching_ar_round0'],total_requests=check['total_requests'],
                repeatable_prompts=check['repeatable_prompts'],total_prompts=check['total_prompts']))
    report=folder/'fixed_window_report';report.mkdir(exist_ok=True)
    atomic_json(report/'summary.json',dict(status='fixed_length_complete_natural_eos_deferred',
        manifest_sha256=manifest['manifest_sha256'],selection_sha256=selection['selection_sha256'],
        fixed_blocks=15,fixed_requests=1125,tuning_requests=480,results=summaries,
        correctness_status='Fast-mode exactness unresolved; inspect saved token comparisons.',
        deferred='Natural EOS: 15 blocks, 1125 requests. Full campaign remains incomplete.'))
    with (report/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    scores=selection['scores'];fig,axes=plt.subplots(1,2,figsize=(10,4))
    sd=sorted([r for r in scores if r['config']['mode']=='sd'],key=lambda r:r['config']['k'])
    axes[0].plot([r['config']['k'] for r in sd],[r['mean_latency_s'] for r in sd],marker='o')
    axes[0].set(xlabel='K',ylabel='Mean request latency (s)',title='Completed SD tuning')
    mat=np.array([[next(r['mean_latency_s'] for r in scores if r['config']==dict(mode='ssd',k=k,f=f)) for f in [1,3]] for k in [4,7,10]])
    im=axes[1].imshow(mat,aspect='auto');axes[1].set(xticks=[0,1],xticklabels=[1,3],yticks=[0,1,2],yticklabels=[4,7,10],xlabel='F',ylabel='K',title='Completed SSD tuning')
    for y in range(3):
        for x in range(2):axes[1].text(x,y,f'{mat[y,x]:.3f}s',ha='center',va='center',color='white')
    fig.colorbar(im,ax=axes[1]);fig.tight_layout();fig.savefig(report/'tuning.png',dpi=180);plt.close(fig)
    main=[r for r in summaries if r['family']=='all'];fig,ax=plt.subplots(figsize=(7,4))
    means=[r['statistics']['mean_latency_s'] for r in main];intervals=[r['comparison_to_ar']['mean_latency_ci95_s'] for r in main]
    ax.bar(['AR (1 GPU)','SD (1 GPU)','SSD (2 GPUs)'],means,
        yerr=[[max(0,m-i[0]) for m,i in zip(means,intervals)],[max(0,i[1]-m) for m,i in zip(means,intervals)]],capsize=4)
    ax.set(ylabel='Full-request seconds (95% paired bootstrap CI)',title='Fixed 256 tokens: fast-mode exactness unresolved')
    fig.tight_layout();fig.savefig(report/'latency.png',dpi=180);plt.close(fig)
    lines=['# Fixed-length Evaluation 1 results','',
      '**Performance measurements only: fast-mode exact-output agreement remains unresolved.**','',
      'All 480 tuning measurements and all 1,125 fixed-256 held-out measurements are complete. '
      'The 75 held-out prompts (25 per family) have five valid repetitions per method. '
      'Natural-EOS evaluation is deferred; this is not the full original campaign report.','',
      'Dedicated GPU monitoring, fresh workers, full prefills, separate warmups, no profiler, BF16, '
      'CUDA graphs, fixed model capacities and memory fractions follow the frozen original manifest. '
      'AR/SD use one GPU and SSD uses two. Confidence intervals use 10,000 paired draws over rounds '
      'and prompt IDs within families.','',
      '## Frozen tuning selection','',
      '| Configuration | Mean latency (s) | Selected |','|---|---:|---|']
    for r in scores:
        lines.append(f'| {config_id(r["config"])} | {r["mean_latency_s"]:.4f} | {"yes" if r["config"]==selection["winners"][r["config"]["mode"]] else ""} |')
    lines+=['','![Tuning](tuning.png)','','## Held-out fixed 256','',
      '| Family | Mode | GPUs | Mean / p50 / p95 seconds | tok/s | First delivery s | AR/method latency (95% CI) | Exact / requests | Repeatable / prompts |',
      '|---|---|---:|---|---:|---:|---|---|---|']
    for r in summaries:
        s,c,k=r['statistics'],r['comparison_to_ar'],r['correctness']
        lines.append(f'| {r["family"]} | {config_id(r["config"])} | {r["gpu_count"]} | '
         f'{s["mean_latency_s"]:.3f} / {s["p50_latency_s"]:.3f} / {s["p95_latency_s"]:.3f} | '
         f'{s["committed_tok_per_s"]:.2f} | {s["mean_first_delivery_s"]:.3f} | '
         f'{c["speedup"]:.3f}x ({c["speedup_ci95"][0]:.3f}–{c["speedup_ci95"][1]:.3f}) | '
         f'{k["requests_matching_ar_round0"]}/{k["total_requests"]} | {k["repeatable_prompts"]}/{k["total_prompts"]} |')
    regressions=[f'{r["family"]}/{r["mode"]}' for r in summaries if r['family']!='all' and r['mode']!='ar' and r['comparison_to_ar']['speedup']<1]
    lines+=['','![Latency](latency.png)','', 'Mean family regressions versus AR: '+(', '.join(regressions) or 'none')+'.','',
      'Acceptance, speculation-cache hits, output lengths, all first divergences, and paired-round exactness '
      'are in summary.json / summary.csv. Natural-EOS outputs may have different lengths and require their own evaluation.','',
      'Remaining original campaign: 1,125 natural-EOS requests in 15 valid blocks. '
      'The anytime-publication prototype, matched policy ablations, and strict fast correctness fix remain separate work.','']
    (report/'report.md').write_text('\n'.join(lines))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=ROOT/'results/baselines/eval1')
    args=parser.parse_args();folder=args.outdir.resolve()
    request=read_json(folder/'fixed_window_request.json')
    remaining=request['deadline_unix']-time.time()
    if remaining<=0:
        print('Requested window has already expired; no jobs started.',flush=True);return 2
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,remaining)
    manifest=None;selection=None
    def status(state,reason):
        data=dict(status=state,at=now(),reason=reason,deadline_unix=request['deadline_unix'],
            scope='fixed-256 only; 75 prompts and 5 repetitions; natural EOS deferred')
        if manifest and selection:data.update(progress(folder,manifest,selection))
        atomic_json(folder/'fixed_window_status.json',data)
        if manifest and selection:
            atomic_json(folder/'status.json',dict(data,full_campaign_complete=False,
                natural_eos='deferred',tuning_requests=480))
        print(state+': '+reason,flush=True)
    try:
        status('handoff','Preserving the current full-campaign block before changing the queue.')
        handoff(request)
        with (folder/'.campaign.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            manifest=load_campaign(folder);selection=validated_selection(folder,manifest)
            freeze_json(folder/'fixed_window_definition.json',dict(request=request,
                manifest_sha256=manifest['manifest_sha256'],selection_sha256=selection['selection_sha256'],
                supervisor_sha256=file_digest(__file__),job_ids=[j['id'] for j in fixed_jobs(manifest,selection)]))
            source_sha=file_digest(__file__)
            for job in fixed_jobs(manifest,selection):
                while True:
                    if file_digest(__file__)!=source_sha:raise ValueError('Fixed-window supervisor changed during execution')
                    try:
                        status('running','Starting or validating '+job['id'])
                        run_block(folder,manifest,job,True)
                        break
                    except PendingWindow as exc:
                        status('waiting_for_gpus',str(exc));time.sleep(min(30,max(0,request['deadline_unix']-time.time())))
            report=write_report(folder,manifest,selection)
            status('fixed_length_complete','All five fixed-length repetitions complete; report: '+str(report/'report.md'))
            atomic_json(folder/'status.json',dict(status='fixed_length_complete_natural_eos_deferred',at=now(),
                tuning_requests=480,fixed_requests=1125,deferred_natural_requests=1125,full_campaign_complete=False,
                report=str(report/'report.md')))
            return 0
    except BudgetExpired as exc:
        signal.setitimer(signal.ITIMER_REAL,0);status('budget_exhausted',str(exc));return 2
    except KeyboardInterrupt:
        signal.setitimer(signal.ITIMER_REAL,0);status('stopped','Interrupted; valid blocks retained.');return 130
    except (ValueError,KeyError,OSError,subprocess.SubprocessError) as exc:
        status('error',str(exc));return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    raise SystemExit(main())
