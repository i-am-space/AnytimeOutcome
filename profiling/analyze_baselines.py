"""Validate and report the complete Evaluation 1 campaign; never select on held-out data."""
import argparse
import csv
from pathlib import Path

import numpy as np

from baseline_common import FAMILIES, atomic_json, config_id, first_difference, heldout_jobs, tuning_jobs
from run_baselines import accepted_result, load_campaign, validated_selection


def paired_bootstrap(reference, method, families, samples=10000, seed=0):
    """Paired hierarchical bootstrap: shared round and stratified prompt draws."""
    a,b=np.asarray(reference,dtype=float),np.asarray(method,dtype=float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=len(families) or not np.all(a>0) or not np.all(b>0):
        raise ValueError('Expected positive paired round-by-prompt latency matrices')
    rng=np.random.default_rng(seed)
    rounds=rng.integers(0,a.shape[0],size=(samples,a.shape[0]))
    labels=np.asarray(families)
    indices=[]
    for family in sorted(set(families)):
        available=np.flatnonzero(labels==family)
        indices.append(rng.choice(available,size=(samples,len(available)),replace=True))
    prompts=np.concatenate(indices,axis=1)
    ma=a[rounds[:,:,None],prompts[:,None,:]].mean(axis=(1,2))
    mb=b[rounds[:,:,None],prompts[:,None,:]].mean(axis=(1,2))
    def interval(values):
        return np.quantile(values,[0.025,0.975]).tolist()
    return dict(speedup=float(a.mean()/b.mean()),speedup_ci95=interval(ma/mb),
        latency_reduction=float(1-b.mean()/a.mean()),latency_reduction_ci95=interval(1-mb/ma),
        mean_latency_s=float(b.mean()),mean_latency_ci95_s=interval(mb),
        bootstrap_samples=samples,bootstrap_seed=seed,
        bootstrap_unit='paired repetition rounds and prompt IDs, stratified by family')


def row_statistics(rows):
    latency=np.array([r['latency_s'] for r in rows])
    hits=[r['cache_hit_rate'] for r in rows if r['cache_hit_rate'] is not None]
    verified=sum(r['verified_draft_tokens'] for r in rows)
    return dict(requests=len(rows),mean_latency_s=float(latency.mean()),
        p50_latency_s=float(np.quantile(latency,0.50)),p95_latency_s=float(np.quantile(latency,0.95)),
        committed_tok_per_s=sum(r['output_tokens'] for r in rows)/float(latency.sum()),
        mean_first_delivery_s=float(np.mean([r['first_delivery_s'] for r in rows])),
        mean_output_tokens=float(np.mean([r['output_tokens'] for r in rows])),
        mean_tokens_per_step=float(np.mean([r['tokens_per_step'] for r in rows])),
        draft_acceptance_rate=sum(r['accepted_draft_tokens'] for r in rows)/verified if verified else None,
        mean_cache_hit_rate=float(np.mean(hits)) if hits else None)


def output_checks(method, reference):
    """Matrices contain request records, preserving all output-token sequences."""
    rounds,prompts=len(method),len(method[0])
    paired=golden=stable=all_rounds=0
    differences=[]
    for j in range(prompts):
        actual=[method[i][j]['output_token_ids'] for i in range(rounds)]
        ar=[reference[i][j]['output_token_ids'] for i in range(rounds)]
        stable+=all(x==actual[0] for x in actual)
        all_rounds+=all(actual[i]==ar[0] for i in range(rounds))
        for i in range(rounds):
            paired+=actual[i]==ar[i]
            golden+=actual[i]==ar[0]
            index=first_difference(ar[0],actual[i])
            if index is not None:
                differences.append(dict(id=method[i][j]['id'],repeat=i,first_difference_zero_based=index,
                    ar_length=len(ar[0]),method_length=len(actual[i])))
    return dict(total_requests=rounds*prompts,requests_matching_paired_ar=paired,
        requests_matching_ar_round0=golden,prompts_matching_ar_round0_in_all_rounds=all_rounds,
        repeatable_prompts=stable,total_prompts=prompts,first_differences=differences)


def write_report(folder,manifest,selection):
    # Validate all evidence before writing any final report or figure.
    tune=[accepted_result(folder,j,manifest) for j in tuning_jobs(manifest)]
    held=[accepted_result(folder,j,manifest) for j in heldout_jobs(manifest,selection)]
    records={}
    for result in held:
        job=result['job']
        records[(job['protocol'],job['config']['mode'],job['repeat'])]={r['id']:r for r in result['requests']}
    ids=[r['id'] for r in manifest['splits']['heldout']]
    families=[r['family'] for r in manifest['splits']['heldout']]
    summaries=[]
    csv_rows=[]
    for protocol in ('fixed','natural'):
        grids={mode:[[records[(protocol,mode,i)][rid] for rid in ids] for i in range(5)]
               for mode in ('ar','sd','ssd')}
        for family in ('all',)+FAMILIES:
            indices=[i for i,f in enumerate(families) if family=='all' or f==family]
            reference=[[row[i] for i in indices] for row in grids['ar']]
            a=[[r['latency_s'] for r in row] for row in reference]
            for mode in ('ar','sd','ssd'):
                grid=[[row[i] for i in indices] for row in grids[mode]]
                rows=[r for row in grid for r in row]
                stats=row_statistics(rows)
                comparison=paired_bootstrap(a,[[r['latency_s'] for r in row] for row in grid],
                    [families[i] for i in indices],manifest['spec']['bootstrap_samples'])
                check=output_checks(grid,reference)
                config=selection['winners'][mode]
                item=dict(protocol=protocol,family=family,mode=mode,config=config,
                    gpu_count=2 if mode=='ssd' else 1,statistics=stats,comparison_to_ar=comparison,
                    correctness=check,latency_regression=mode!='ar' and comparison['speedup']<1)
                summaries.append(item)
                csv_rows.append(dict(protocol=protocol,family=family,mode=mode,k=config['k'],f=config['f'],
                    gpu_count=item['gpu_count'],**stats,speedup=comparison['speedup'],
                    speedup_ci95_low=comparison['speedup_ci95'][0],speedup_ci95_high=comparison['speedup_ci95'][1],
                    exact_matches_to_ar_round0=check['requests_matching_ar_round0'],
                    repeatable_prompts=check['repeatable_prompts'],total_prompts=check['total_prompts']))
    summary=dict(status='complete',manifest_sha256=manifest['manifest_sha256'],
        selection_sha256=selection['selection_sha256'],hardware=manifest['hardware'],
        valid_blocks=len(tune)+len(held),tuning_requests=sum(len(x['requests']) for x in tune),
        heldout_requests=sum(len(x['requests']) for x in held),warmup_requests=150,
        correctness_status='Fast-mode exact equivalence remains unresolved; see token comparisons.',
        primary_protocol='fixed',secondary_protocol='natural',
        natural_protocol_note='Same fixed-length-tuned configurations; output differences can change EOS and length.',
        results=summaries)
    atomic_json(folder/'summary.json',summary)
    with (folder/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_figures(folder,selection,summaries)
    write_markdown(folder,manifest,selection,summary)
    print(f'Complete report: {folder / "report.md"}')


def write_figures(folder,selection,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scores=selection['scores']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    sd=sorted([x for x in scores if x['config']['mode']=='sd'],key=lambda r:r['config']['k'])
    axes[0].plot([x['config']['k'] for x in sd],[x['mean_latency_s'] for x in sd],marker='o',label='SD')
    ar=next(x['mean_latency_s'] for x in scores if x['config']['mode']=='ar')
    axes[0].axhline(ar,color='gray',linestyle='--',label='AR control')
    axes[0].set(xlabel='K',ylabel='Mean full-request latency (s)',title='Tuning: SD')
    axes[0].legend()
    matrix=np.array([[next(x['mean_latency_s'] for x in scores if x['config']==dict(mode='ssd',k=k,f=f))
                      for f in (1,3)] for k in (4,7,10)])
    image=axes[1].imshow(matrix,aspect='auto')
    axes[1].set(xticks=[0,1],xticklabels=[1,3],yticks=[0,1,2],yticklabels=[4,7,10],
                xlabel='F',ylabel='K',title='Tuning: SSD')
    for y in range(3):
        for x in range(2):
            axes[1].text(x,y,f'{matrix[y,x]:.3f}s',ha='center',va='center',color='white')
    fig.colorbar(image,ax=axes[1],label='Mean full-request latency (s)')
    fig.tight_layout()
    fig.savefig(folder/'tuning.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,protocol in zip(axes,('fixed','natural')):
        rows=[next(r for r in summaries if r['protocol']==protocol and r['family']=='all' and r['mode']==mode)
              for mode in ('ar','sd','ssd')]
        means=[r['statistics']['mean_latency_s'] for r in rows]
        intervals=[r['comparison_to_ar']['mean_latency_ci95_s'] for r in rows]
        errors=np.array([[max(0,m-i[0]) for m,i in zip(means,intervals)],
                         [max(0,i[1]-m) for m,i in zip(means,intervals)]])
        ax.bar(['AR (1 GPU)','SD (1 GPU)','SSD (2 GPUs)'],means,yerr=errors,capsize=4)
        ax.set(title=f'{protocol}: full-request latency',ylabel='Seconds (95% paired bootstrap CI)')
    fig.suptitle('Fast modes: exact-output discrepancies must be disclosed')
    fig.tight_layout()
    fig.savefig(folder/'latency.png',dpi=180)
    plt.close(fig)


def write_markdown(folder,manifest,selection,summary):
    lines=['# Evaluation 1 baseline experiments','',
        '**Performance measurements of fast implementations. Exact greedy equivalence is unresolved.**',
        'Output mismatches are reported below; these are not established correctness-preserving acceleration results.','',
        'The 24 tuning and 75 held-out prompts are balanced across chat, code, and document tasks. '
        'Warmup uses three separate tuning prompts. Previously investigated held-out prompts are excluded. '
        'Weights/kernels are warm, prompt-prefix reuse is disabled, and no profiler is attached.','',
        'AR and SD each allocate one L40S; SSD allocates two. This is a latency comparison with an additional GPU for SSD. '
        'Every accepted block passed dedicated-GPU process monitoring.','',
        f'Manifest: `{manifest["manifest_sha256"]}`. Selection: `{selection["selection_sha256"]}`.','',
        '## Tuning and frozen selection','',
        'Each setting has 48 measured requests: 24 tuning prompts × two repetitions. '
        'The minimum mean fixed-256 request latency selects one global configuration per method; ties use smaller K/F. '
        'Held-out measurements are never used for selection.','',
        '| Configuration | Mean request latency (s) | Selected |','|---|---:|---|']
    for row in selection['scores']:
        config=row['config']
        lines.append(f'| {config_id(config)} | {row["mean_latency_s"]:.4f} | '
                     f'{"yes" if config==selection["winners"][config["mode"]] else ""} |')
    lines+=['','![Tuning](tuning.png)','', '## Held-out comparisons','',
        'Each method/protocol has 375 requests: 75 prompts × five rounds. '
        'The 95% intervals use 10,000 paired bootstrap draws over rounds and prompt IDs, stratified by family. '
        'Speedup is AR mean latency divided by method mean latency.','',
        'Fixed 256-token output is primary. Natural EOS, capped at 256, is secondary and uses the same winners. '
        'Different output tokens can change EOS/length, so natural-EOS latency ratios may reflect different response lengths.','',
        'Exact matches use the AR output from round 0 as the fixed reference. Paired-round comparisons and '
        'first-divergence positions are also retained in summary.json. Repeatability compares each mode with its own round 0.','']
    lines+=['Draft acceptance is accepted draft tokens / verified draft tokens, excluding the recovery token. '
            'It describes verifier decisions before final EOS/output-cap trimming. Cache hit means SSD speculation-cache '
            'hit; prompt-prefix reuse is disabled.','']
    for protocol in ('fixed','natural'):
        lines += [f'### {protocol}','',
            '| Family | Method | GPUs | Mean / p50 / p95 (s) | tok/s | First delivery (s) | Mean output tokens | Tokens/step | Draft acceptance | Cache hit | AR / method latency (95% CI) | Exact requests / total | Repeatable prompts / total |',
            '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|']
        for row in summary['results']:
            if row['protocol']!=protocol:
                continue
            s,c,k=row['statistics'],row['comparison_to_ar'],row['correctness']
            hit=f'{s["mean_cache_hit_rate"]:.3f}' if s['mean_cache_hit_rate'] is not None else '—'
            acceptance=f'{s["draft_acceptance_rate"]:.3f}' if s['draft_acceptance_rate'] is not None else '—'
            lines.append(f'| {row["family"]} | {config_id(row["config"])} | {row["gpu_count"]} | '
                f'{s["mean_latency_s"]:.3f} / {s["p50_latency_s"]:.3f} / {s["p95_latency_s"]:.3f} | '
                f'{s["committed_tok_per_s"]:.2f} | {s["mean_first_delivery_s"]:.3f} | '
                f'{s["mean_output_tokens"]:.1f} | {s["mean_tokens_per_step"]:.2f} | {acceptance} | {hit} | '
                f'{c["speedup"]:.3f}× ({c["speedup_ci95"][0]:.3f}–{c["speedup_ci95"][1]:.3f}) | '
                f'{k["requests_matching_ar_round0"]}/{k["total_requests"]} | '
                f'{k["repeatable_prompts"]}/{k["total_prompts"]} |')
        lines+=['']
    regressions=[f'{r["protocol"]}/{r["family"]}/{r["mode"]}' for r in summary['results']
                 if r['family']!='all' and r['latency_regression']]
    lines+=['![Latency](latency.png)','', '## Regressions and limits','',
        'Families with mean latency regressions versus AR: '+(', '.join(regressions) if regressions else 'none')+'.','',
        'These measurements cover greedy batch-one inference, the declared context/output limits and this workload mix. '
        'The slow numerical reference is separate and is not substituted into speed measurements. '
        'Confidence intervals quantify measured variability; they do not resolve output-equivalence failures.','']
    temporary=folder/'report.md.tmp'
    temporary.write_text('\n'.join(lines))
    temporary.replace(folder/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    folder=parser.parse_args().outdir.resolve()
    manifest=load_campaign(folder)
    write_report(folder,manifest,validated_selection(folder,manifest))
