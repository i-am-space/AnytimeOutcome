"""Audit a scored request under fail_on_recompile; never headline timings."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import benchmark_shared_numerics as benchmark
import torch
from baseline_common import atomic_json,file_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--method',choices=benchmark.METHODS,default='original_ar')
    args=parser.parse_args()
    if args.outdir.exists():parser.error('Preserve earlier audit attempts')
    args.outdir.mkdir(parents=True)
    original=benchmark.run_engine.measure_request
    audit=dict(diagnostic_only=True,method=args.method,passed=False,requests=[],
               source_sha256=file_digest(__file__))
    def measured(model,item,sampling,**kwargs):
        before=dict(torch._dynamo.utils.counters['stats'])
        try:
            with torch.compiler.set_stance('fail_on_recompile'):
                result=original(model,item,sampling,**kwargs)
        except Exception as exc:
            audit['requests'].append(dict(id=item['id'],error=repr(exc),before=before,
                after=dict(torch._dynamo.utils.counters['stats'])))
            atomic_json(args.outdir/'audit.json',audit)
            raise
        audit['requests'].append(dict(id=item['id'],before=before,
            after=dict(torch._dynamo.utils.counters['stats'])))
        # ModelRunner.exit(hard=True) terminates the worker process, so persist
        # the conclusion before returning the final scored request.
        audit['passed']=(len(audit['requests'])==3 and all(
            'error' not in row and row['before']==row['after'] for row in audit['requests']))
        atomic_json(args.outdir/'audit.json',audit)
        return result
    benchmark.run_engine.measure_request=measured
    benchmark.worker(SimpleNamespace(outdir=args.benchmark.resolve(),blockdir=args.outdir.resolve(),
                                     method=args.method,round=0))
    audit['passed']=True
    atomic_json(args.outdir/'audit.json',audit)


if __name__=='__main__':main()
