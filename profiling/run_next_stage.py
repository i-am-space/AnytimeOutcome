"""Resume correctness -> new shared baseline -> uniform publication comparison.

Source profiling/env.sh and run in tmux. Dedicated-window waits are automatic;
code changes, correctness failures, and worker errors stop the pipeline.
"""
import argparse
import fcntl
from pathlib import Path
import signal
import subprocess
import sys
import zipfile

from baseline_common import ROOT,atomic_json,file_digest,freeze_json,read_json
from run_baselines import interrupted,now


def run(command):
    print('Starting: '+' '.join(map(str,command)),flush=True)
    child=subprocess.Popen([sys.executable,*map(str,command)],cwd=ROOT)
    try:
        code=child.wait()
    except KeyboardInterrupt:
        child.send_signal(signal.SIGINT)
        child.wait(timeout=40)
        raise
    if code:raise RuntimeError(f'Phase exited with {code}: {command[0]}')


def snapshot(folder):
    definition_path=folder/('manifest.json' if (folder/'manifest.json').exists() else 'definition.json')
    d=read_json(definition_path)
    paths=dict(d['execution']['files'])
    for name in ('profiling/run_publication_study.py','profiling/check_publication_interrupt.py',
                 'profiling/run_next_stage.py'):
        paths[name]=file_digest(ROOT/name)
    destination=folder/'source_snapshot.zip'
    if destination.exists():return
    temporary=destination.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,sha in paths.items():
            if file_digest(ROOT/name)!=sha:raise ValueError('Execution changed before source snapshot')
            archive.write(ROOT/name,name)
        archive.write(definition_path,definition_path.name)
    temporary.replace(destination)


def require_gate(folder, require_partial=True):
    summary=read_json(folder/'summary.json')
    if not summary['correctness_passed']:
        raise ValueError('Exact-output correctness failed; later performance phases blocked')
    if not all(summary['within_method_repeatability'].values()):
        raise ValueError('Within-method repeatability failed')
    early=[table['early'] for table in summary['tables'].values()]
    if require_partial and not sum(row['partial_publications'] for row in early):
        raise ValueError('No actual early handoffs observed; feature has not been exercised')


def require_baseline(folder):
    summary=read_json(folder/'summary.json')
    if summary['status']!='complete' or summary['valid_blocks']!=50:
        raise ValueError('Baseline campaign incomplete')
    for row in summary['results']:
        c=row['correctness']
        if c['requests_matching_paired_ar']!=c['total_requests'] or c['repeatable_prompts']!=c['total_prompts']:
            raise ValueError('New baseline has output mismatches or repeatability failures; inspect report')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'results/next_stage/shared_uniform_v1_2026-11-17')
    parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--stress-gate',type=Path,default=ROOT/'results/correctness/uniform_publication_stress_v1_2026-11-17')
    parser.add_argument('--source',type=Path,default=ROOT/'results/correctness/shared_fixed16_v1_2026-11-17_v2')
    args=parser.parse_args();args.root=args.root.resolve();args.gate=args.gate.resolve()
    args.root.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,interrupted)
    baseline=args.root/'baselines';comparison=args.root/'publication'
    with (args.root/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze_json(args.root/'pipeline.json',dict(supervisor_sha256=file_digest(__file__),
            gate=str(args.gate),stress_gate=str(args.stress_gate),source=str(args.source),
            baseline=str(baseline),publication=str(comparison),publication_depth_rule='min(4, selected SSD K - 1)'))
        try:
            for phase in ('correctness','stress','baseline','publication'):
                atomic_json(args.root/'status.json',dict(status='running',phase=phase,at=now()))
                if phase=='correctness':
                    run([ROOT/'profiling/run_publication_study.py','--outdir',args.gate,
                         '--phase','correctness','--source',args.source,'--watch'])
                    require_gate(args.gate,require_partial=False)
                    snapshot(args.gate)
                elif phase=='stress':
                    run([ROOT/'profiling/check_publication_interrupt.py','--outdir',args.stress_gate,
                         '--source',args.source,'--watch'])
                    require_gate(args.stress_gate)
                    snapshot(args.stress_gate)
                elif phase=='baseline':
                    if not (baseline/'manifest.json').exists():
                        run([ROOT/'profiling/run_baselines.py','--outdir',baseline,
                             '--phase','prepare','--target-numerics','shared_fixed16'])
                    snapshot(baseline)
                    run([ROOT/'profiling/watch_baselines.py','--outdir',baseline])
                    require_baseline(baseline)
                    report_path=baseline/'report.md'
                    note=('Numerical reference: shared-fixed16-v1 (new AR arithmetic). '
                          'Exact agreement below is against this shared AR, not original historical AR. '
                          'SSD request latency includes final obsolete-work draining.\n\n')
                    if not report_path.read_text().startswith(note):
                        report_path.write_text(note+report_path.read_text())
                else:
                    # This rule is declared before tuning; P itself is never tuned.
                    selected=read_json(baseline/'selection.json')
                    depth=min(4,selected['winners']['ssd']['k']-1)
                    run([ROOT/'profiling/run_publication_study.py','--outdir',comparison,
                         '--phase','compare','--source',args.source,'--baseline-dir',baseline,
                         '--publish-tokens',str(depth),'--watch'])
                    require_gate(comparison,require_partial=False)
                    snapshot(comparison)
            atomic_json(args.root/'status.json',dict(status='complete',at=now(),
                baseline_report=str(baseline/'report.md'),publication_report=str(comparison/'report.md')))
        except BaseException as exc:
            atomic_json(args.root/'status.json',dict(status='stopped',phase=phase,at=now(),reason=str(exc)))
            raise


if __name__=='__main__':main()
