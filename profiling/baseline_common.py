"""CPU-only protocol, provenance, and validation for the Evaluation 1 campaign."""
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('chat', 'code', 'doc')
EXCLUDED = {'specbench-148', 'HumanEval/36', 'specbench-553'}
SPEC = dict(seed=0, tune_per_family=8, heldout_per_family=25, tuning_repetitions=2,
            repetitions=5, output_len=256, max_prompt_tokens=2048, max_model_len=4096,
            target_kv_fraction=0.70, draft_kv_fraction=0.70, batch_size=1,
            precision='bfloat16', cuda_graphs=True, backup='jit', prefix_cache=False,
            protocols=['fixed', 'natural'], bootstrap_samples=10000)
SPEC['tuning_grid']=dict(sd_k=[4,6,8],ssd_k=[4,7,10],ssd_f=[1,3])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Frozen artifact differs: {path}')
    else:
        atomic_json(path, value)


def execution_fingerprint():
    paths = sorted((ROOT / 'third_party/ssd/ssd').rglob('*.py'))
    paths += [ROOT / 'profiling' / name for name in
              ('run_engine.py', 'workloads.py', 'baseline_common.py', 'run_baselines.py', 'analyze_baselines.py',
               'watch_baselines.py')]
    files = {str(p.relative_to(ROOT)): file_digest(p) for p in paths}
    return dict(sha256=digest(files), files=files)


def software_fingerprint():
    packages = {}
    for name in ('torch', 'transformers', 'triton', 'flashinfer-python', 'numpy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment={key:os.environ.get(key) for key in ('CUDA_HOME','SSD_CUDA_ARCH','LD_PRELOAD',
        'NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','CUBLAS_WORKSPACE_CONFIG',
        'CUDA_LAUNCH_BLOCKING','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTORCH_CUDA_ALLOC_CONF')}
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),packages=packages,
                environment=environment)


def model_fingerprint(path):
    path = Path(path).resolve()
    files = {str(p.relative_to(path)): dict(size=p.stat().st_size, sha256=file_digest(p))
             for p in sorted(path.rglob('*')) if p.is_file() and
             p.suffix in ('.json', '.safetensors', '.bin', '.model', '.txt', '.tiktoken', '.jinja')}
    if not any(name.endswith(('.safetensors', '.bin')) for name in files):
        raise ValueError(f'No model weights found: {path}')
    return dict(path=str(path), sha256=digest(files), files=files)


def validate_models(models):
    # Recheck full content once per campaign invocation, never per request/block.
    for model in models.values():
        path = Path(model['path'])
        actual_names = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and
                        p.suffix in ('.json', '.safetensors', '.bin', '.model', '.txt', '.tiktoken', '.jinja')}
        if actual_names != set(model['files']):
            raise ValueError(f'Model file inventory changed: {path}')
        for name, info in model['files'].items():
            p = path / name
            if p.stat().st_size != info['size'] or file_digest(p) != info['sha256']:
                raise ValueError(f'Model/tokenizer changed: {p}')


def seal_manifest(value):
    return dict(value, manifest_sha256=digest(value))


def validate_manifest(manifest):
    body = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if digest(body) != manifest.get('manifest_sha256'):
        raise ValueError('Manifest hash mismatch')
    if manifest['spec'] != SPEC:
        raise ValueError('Campaign protocol differs from the declared Evaluation 1 protocol')
    seen = set()
    for split, per_family in [('tune', 8), ('heldout', 25), ('warmup', 1)]:
        rows = manifest['splits'][split]
        for family in FAMILIES:
            if sum(r['family'] == family for r in rows) != per_family:
                raise ValueError(f'Wrong {split}/{family} sample size')
        for row in rows:
            if row['id'] in seen or row['id'] in EXCLUDED:
                raise ValueError('Duplicate, overlapping, or previously investigated prompt')
            seen.add(row['id'])
            ids = row['token_ids']
            if not ids or len(ids) > SPEC['max_prompt_tokens'] or digest(ids) != row['token_ids_sha256']:
                raise ValueError('Invalid or modified prompt tokens')


def make_manifest(tokenizer, models, execution, software, hardware, loader):
    splits = dict(tune=[], heldout=[], warmup=[])
    for family in FAMILIES:
        tune = loader([family], 100000, split='tune', seed=0, tokenizer=tokenizer)
        heldout = loader([family], 100000, split='heldout', seed=0, tokenizer=tokenizer)
        tune = [r for r in tune if r['id'] not in EXCLUDED]
        heldout = [r for r in heldout if r['id'] not in EXCLUDED]
        if len(tune) < 9 or len(heldout) < 25:
            raise ValueError(f'Insufficient {family} samples')
        for split, rows in [('tune', tune[:8]), ('warmup', tune[8:9]), ('heldout', heldout[:25])]:
            for row in rows:
                if len(row['token_ids']) > 2048:
                    raise ValueError(f'Complete prompt exceeds 2048 tokens: {row["id"]}')
                splits[split].append(dict(row, token_ids_sha256=digest(row['token_ids'])))
    result = seal_manifest(dict(schema_version=1, spec=SPEC, models=models, execution=execution,
        software=software, hardware=hardware, excluded_ids=sorted(EXCLUDED), splits=splits,
        tokenizer=dict(type=type(tokenizer).__name__, chat_template_sha256=digest(tokenizer.chat_template))))
    validate_manifest(result)
    return result


def configurations():
    return ([dict(mode='ar', k=None, f=None)] +
            [dict(mode='sd', k=k, f=None) for k in (4, 6, 8)] +
            [dict(mode='ssd', k=k, f=f) for k in (4, 7, 10) for f in (1, 3)])


def config_id(config):
    return config['mode'] + (f'_k{config["k"]}' if config['k'] is not None else '') + (
        f'_f{config["f"]}' if config['f'] is not None else '')


def ordered_prompts(manifest, split, seed):
    rows = list(manifest['splits'][split])
    random.Random(seed).shuffle(rows)
    return rows


def make_job(manifest, phase, config, repeat, protocol, selection_sha=None):
    split = 'tune' if phase == 'tune' else 'heldout'
    seed = 1000 + repeat
    body = dict(phase=phase, config=config, repeat=repeat, protocol=protocol, order_seed=seed,
                manifest_sha256=manifest['manifest_sha256'], execution_sha256=manifest['execution']['sha256'],
                selection_sha256=selection_sha,
                prompt_ids=[r['id'] for r in ordered_prompts(manifest, split, seed)])
    body['id'] = f'{phase}/{protocol}/r{repeat}/{config_id(config)}'
    return dict(body, job_sha256=digest(body))


def tuning_jobs(manifest):
    configs = configurations()
    random.Random(0).shuffle(configs)
    return [make_job(manifest, 'tune', c, repeat, 'fixed')
            for repeat, order in enumerate((configs, list(reversed(configs)))) for c in order]


def heldout_jobs(manifest, selection):
    methods = list(itertools.permutations(('ar', 'sd', 'ssd')))
    random.Random(0).shuffle(methods)
    jobs = []
    for repeat in range(5):
        protocols = ('fixed', 'natural') if repeat % 2 == 0 else ('natural', 'fixed')
        for protocol in protocols:
            order = methods[repeat] if protocol == 'fixed' else tuple(reversed(methods[repeat]))
            for mode in order:
                jobs.append(make_job(manifest, 'heldout', selection['winners'][mode], repeat, protocol,
                                     selection['selection_sha256']))
    return jobs


def worker_settings(job):
    expected = dict(mode=job['config']['mode'], profile_capture=False, eager=False,
                    disable_prefix_cache=True, gpu_mem_util=0.7, draft_gpu_mem_util=0.7,
                    backup='jit', output_len=256, max_model_len=4096, max_prompt_tokens=2048,
                    warmup=3,order_seed=job['order_seed'],ignore_eos=job['protocol']=='fixed')
    if job['config']['mode']!='ar':
        expected['k']=job['config']['k']
    if job['config']['mode']=='ssd':
        expected['f']=job['config']['f']
    return expected


def validate_block(result, job, manifest):
    if result.get('job') != job or result.get('manifest_sha256') != manifest['manifest_sha256']:
        raise ValueError('Result provenance mismatch')
    args = result.get('args', {})
    expected=worker_settings(job)
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(f'Result changed execution setting: {key}')
    requests = result.get('requests', [])
    if [r['id'] for r in requests] != job['prompt_ids']:
        raise ValueError('Missing, extra, reordered, or duplicated requests')
    known = {r['id']: r for r in manifest['splits']['tune' if job['phase']=='tune' else 'heldout']}
    for row in requests:
        expected_row = known[row['id']]
        ids = row.get('output_token_ids', [])
        if row['family'] != expected_row['family'] or row['prompt_tokens'] != len(expected_row['token_ids']):
            raise ValueError('Request manifest mismatch')
        if row.get('input_token_ids_sha256') != expected_row['token_ids_sha256']:
            raise ValueError('Input token hash differs')
        if not 1 <= len(ids) <= 256 or len(ids) != row.get('output_tokens') or any(type(i) is not int for i in ids):
            raise ValueError('Invalid output tokens')
        if job['protocol']=='fixed' and len(ids) != 256:
            raise ValueError('Fixed-length request did not emit 256 tokens')
        if not math.isfinite(row['latency_s']) or row['latency_s'] <= 0:
            raise ValueError('Invalid request latency')
        if row.get('first_delivery_s') is None or not 0 <= row['first_delivery_s'] <= row['latency_s']:
            raise ValueError('Invalid first-delivery latency')
        accepted,verified=row.get('accepted_draft_tokens'),row.get('verified_draft_tokens')
        if type(accepted) is not int or type(verified) is not int or not 0 <= accepted <= verified:
            raise ValueError('Invalid draft acceptance counts')
    if result.get('hardware_allocation') != allocation(manifest, job['config']['mode']):
        raise ValueError('Hardware allocation differs')
    if result.get('hardware') != manifest['hardware']:
        raise ValueError('Hardware topology differs')
    uuids=[g['uuid'] for g in manifest['hardware']['gpus']]
    expected_visible=','.join(uuids if job['config']['mode']=='ssd' else uuids[:1])
    if (result.get('actual_dtype')!='torch.bfloat16' or result.get('prefix_cache_enabled') is not False or
            result.get('cuda_visible_devices')!=expected_visible or result.get('software')!=manifest['software']):
        raise ValueError('Actual worker configuration differs from the declared protocol')


def allocation(manifest, mode):
    gpus = manifest['hardware']['gpus']
    roles = [dict(role='target', physical_index=0, logical_index=0, uuid=gpus[0]['uuid'])]
    if mode != 'ar':
        index = 1 if mode == 'ssd' else 0
        roles.append(dict(role='draft', physical_index=index, logical_index=index, uuid=gpus[index]['uuid']))
    return dict(gpu_count=2 if mode=='ssd' else 1, roles=roles)


def gpu_inventory():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version',
                                    '--format=csv,noheader,nounits'], text=True,timeout=10)
    rows = []
    for line in text.strip().splitlines():
        index, uuid, name, driver = [x.strip() for x in line.split(',')]
        rows.append(dict(index=int(index), uuid=uuid, name=name, driver=driver))
    rows.sort(key=lambda r:r['index'])
    if len(rows) < 2 or any('L40S' not in r['name'] for r in rows[:2]):
        raise ValueError('The campaign requires physical GPUs 0 and 1 to be L40S')
    return dict(gpus=rows[:2], topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True,timeout=10))


def compute_processes(uuids):
    text = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
                                    '--format=csv,noheader,nounits'],text=True,timeout=10)
    rows=[]
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        uuid,pid,name=[x.strip() for x in line.split(',',2)]
        if uuid in uuids:
            rows.append(dict(uuid=uuid,pid=int(pid),name=name))
    return rows


def external_processes(processes, own_group=None):
    external=[]
    for row in processes:
        try:
            if own_group is not None and os.getpgid(row['pid']) == own_group:
                continue
        except ProcessLookupError:
            continue
        external.append(row)
    return external


def first_difference(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),
                min(len(a),len(b)) if len(a)!=len(b) else None)
