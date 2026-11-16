"""Summarize attention substitution and the slow original-AR numerical reference."""
import json
from pathlib import Path

from summarize_correctness import first_difference


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/'results/correctness/attention_backends'
    saved={r['id']:r for r in json.loads((root/'results/nsys/ar_run.json').read_text())['requests']}
    controls=[]
    for backend in ['math','flash']:
        path=root/f'results/correctness/attention_{backend}_controls.json'
        if not path.exists():
            continue
        for request in json.loads(path.read_text())['requests']:
            a=request['single']
            b=request['schedule_replay']
            controls.append(dict(backend=backend,id=request['id'],
                output_index=request['divergence_zero_based'],
                single_argmax=a['argmax'],schedule_argmax=b['argmax'],
                decisions_agree=a['argmax']==b['argmax'],
                single_top=a['top'][:3],schedule_top=b['top'][:3],
                same_token_history=True))
    pairs=[]
    for backend in ['math','flash']:
        paths=[folder/f'{backend}_{mode}.json' for mode in ['ar','sd']]
        if not all(p.exists() for p in paths):
            continue
        ar,sd=[{r['id']:r for r in json.loads(p.read_text())['requests']} for p in paths]
        for rid in sorted(ar.keys() & sd.keys()):
            a,b=ar[rid]['output_token_ids'],sd[rid]['output_token_ids']
            pairs.append(dict(backend=backend,id=rid,ar_length=len(a),sd_length=len(b),
                exact_match=a==b,first_difference=first_difference(a,b),
                ar_first_difference_from_original=first_difference(saved[rid]['output_token_ids'][:len(a)],a),
                sd_first_difference_from_original=first_difference(saved[rid]['output_token_ids'][:len(b)],b)))
    references=[]
    path=folder/'serial_reference_sd.json'
    if path.exists():
        for r in json.loads(path.read_text())['requests']:
            actual=r['output_token_ids']
            expected=saved[r['id']]['output_token_ids'][:len(actual)]
            references.append(dict(id=r['id'],output_length=len(actual),
                exact_match_to_original_ar=actual==expected,
                first_difference=first_difference(expected,actual)))
    summary=dict(diagnostic_only=True,production_default_changed=False,
                 controlled_histories=controls,free_generation_pairs=pairs,
                 serial_reference_vs_original_ar=references)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
