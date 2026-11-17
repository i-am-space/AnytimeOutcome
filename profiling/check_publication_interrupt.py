"""Artificially slowed drafting exercises partial-prefix safety; never performance.

Uses the same monitored worker/report machinery as the real study. Only the
process-local test hook delays each tree step. Engine defaults are untouched.
"""
from pathlib import Path
import time

import run_publication_study as study
from baseline_common import ROOT,digest,file_digest,freeze_json,read_json,validate_models
from run_baselines import now
from ssd.engine.draft_runner import DraftRunner

BASE_STUDY=Path(study.__file__).resolve()
DELAY_SECONDS=.004
_original_step=DraftRunner._decode_tree_step


def delayed_step(self,*args,**kwargs):
    time.sleep(DELAY_SECONDS)
    return _original_step(self,*args,**kwargs)


# Spawn imports this entry point too, so the draft child receives the test hook.
DraftRunner._decode_tree_step=delayed_step
_original_fingerprint=study.fingerprint


def fingerprint():
    return dict(_original_fingerprint(),base_study_sha256=file_digest(BASE_STUDY))


def prepare(args):
    source=read_json(args.source/'definition.json');validate_models(source['models'])
    ids={'specbench-148','HumanEval/36','specbench-496'}
    prompts=[p for p in source['prompts'] if p['id'] in ids]
    if len(prompts)!=3:raise ValueError('Stress check requires the three declared coverage prompts')
    d=dict(created_at=now(),purpose='Artificial 4 ms/tree-step delay: safety stress test; no performance claim',
        reference_contract='shared-fixed16-v1; new AR reference',phase='correctness_stress',
        models=source['models'],hardware=study.gpu_inventory(),prompts=prompts,warmup=source['warmup'],
        methods=['ar','early'],repetitions=1,protocols=['fixed','natural'],output_cap=256,capacity=4096,
        kv_fraction=.10,k=7,f=3,sd_k=6,publish_tokens=4,selection=None,
        artificial_tree_step_delay_s=DELAY_SECONDS,configuration_status='Stress test only; fixed K7 F3 P4',
        source=dict(path=str(args.source/'definition.json'),sha256=file_digest(args.source/'definition.json')),
        **fingerprint())
    d['jobs']=study.jobs(d['methods'],1,d['protocols'],len(prompts));d['sha256']=digest(d)
    freeze_json(args.outdir/'definition.json',d)
    return d


study.fingerprint=fingerprint
study.prepare=prepare
# The common supervisor launches this entry point so its worker and spawned
# draft child both install the isolated delay hook. The base file is also hashed.
study.__file__=__file__

if __name__=='__main__':raise SystemExit(study.main())
