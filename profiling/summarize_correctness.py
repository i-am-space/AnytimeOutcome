"""Join diagnostic captures by request ID and retain only matching-history witnesses."""
import json
from pathlib import Path


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)) if len(a) != len(b) else None)


def decision(row):
    # Older captures store top-k only. torch.argmax resolves ties by first index.
    best = max(t['logit'] for t in row['top'])
    return dict(argmax=row.get('argmax', min(t['token_id'] for t in row['top'] if t['logit'] == best)),
                top=row['top'][:3], **{k: row[k] for k in ['argmax_agrees', 'max_abs_logit_delta'] if k in row})


def main():
    root = Path(__file__).resolve().parents[1]
    saved = {mode: {r['id']: r for r in json.loads((root / 'results/nsys' / name).read_text())['requests']}
             for mode, name in [('ar', 'ar_run.json'), ('sd', 'sd_k6_run.json'), ('ssd', 'ssd_k7_f3_run.json')]}
    captures = {p.stem: json.loads(p.read_text()) for p in (root / 'results/correctness').glob('*.json')
                if p.stem != 'summary'}
    captures = {name: data for name, data in captures.items() if 'mode' in data}
    results = []
    for rid, reference in saved['ar'].items():
        ar = reference['output_token_ids']
        sd = saved['sd'][rid]['output_token_ids']
        ssd = saved['ssd'][rid]['output_token_ids']
        index = first_difference(ar, sd)
        item = dict(id=rid, first_divergence_zero_based=index,
                    saved_sd_equals_ssd=sd == ssd, ar_token_id=ar[index], speculative_token_id=sd[index], captures={})
        for name, capture in captures.items():
            request = next((r for r in capture['requests'] if r['id'] == rid), None)
            if request is None:
                continue
            length = len(request['output_token_ids'])
            info = dict(reproduces_saved_output=request['reproduces_saved_output'], output_length=length,
                        first_mismatch_with_ar=first_difference(ar[:length], request['output_token_ids']), witnesses=[])
            for forward in capture['forwards']:
                if forward['request_id'] != rid or forward['prefill']:
                    continue
                start = forward['input_positions'][0] - len(request['prompt_token_ids'])
                # A verifier also computes logits after rejected draft tokens. Those
                # rows do not describe the shared committed history and must be excluded.
                if not (0 <= start < index and
                        forward['input_token_ids'][:index-start] == ar[start:index]):
                    continue
                row = next((r for r in forward['rows'] if r['output_index'] == index), None)
                if row is None:
                    continue
                witness = dict(input_start_output_index=start, same_committed_history=True,
                               actual_forward=decision(row))
                for field in ['same_prefix_sequential_replay', 'same_hidden_single_row_head']:
                    replay = next((r for r in forward.get(field, []) if r['output_index'] == index), None)
                    if replay:
                        witness[field] = decision(replay)
                info['witnesses'].append(witness)
            item['captures'][name] = info
        results.append(item)
    path = root / 'results/correctness/summary.json'
    path.write_text(json.dumps(dict(diagnostic_only=True, requests=results), indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
