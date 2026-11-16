"""Capture greedy divergences and replay verification rows from identical prefix KV.

Diagnostic only: CPU copies and replay make these runs unsuitable for timing.
Run after sourcing profiling/env.sh, using python -O (as for run_engine.py).
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_engine  # applies the existing Python -O / Dynamo compatibility patch
import torch
from ssd import LLM, SamplingParams
from ssd.engine.model_runner import ModelRunner
from ssd.layers.attention import Attention
from ssd.utils.context import get_context, set_context
from transformers import AutoTokenizer
from workloads import get_prompts
from attention_controls import (install_attention_control, install_rowwise_linear_control,
                                install_tokenwise_pointwise_control)


def top(logits, tokenizer):
    values, ids = logits.float().topk(8)
    return [dict(token_id=i, text=tokenizer.decode([i]), logit=v)
            for i, v in zip(ids.tolist(), values.tolist())]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['ar', 'sd', 'ssd'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--eager', action='store_true')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--output-len', type=int, default=256)
    ap.add_argument('--attention', choices=['portable', 'sdpa_flash', 'sdpa_math', 'portable_serial'], default='portable')
    ap.add_argument('--rowwise-linear', action='store_true')
    ap.add_argument('--tokenwise-pointwise', action='store_true')
    ap.add_argument('--progress', action='store_true')
    ap.add_argument('--families', nargs='+', choices=['chat', 'code', 'doc'], default=['chat', 'code', 'doc'])
    args = ap.parse_args()
    if (args.attention != 'portable' or args.rowwise_linear) and not args.eager:
        ap.error('Diagnostic SDPA controls require --eager')
    install_attention_control(args.attention)
    if args.tokenwise_pointwise and not args.rowwise_linear:
        ap.error('--tokenwise-pointwise requires --rowwise-linear')
    if args.rowwise_linear:
        reference_state = install_rowwise_linear_control()
    baseline_dir = Path(__file__).resolve().parents[1] / 'results/nsys'
    saved = {m: json.loads((baseline_dir / f'{name}_run.json').read_text())
             for m, name in [('ar', 'ar'), ('sd', 'sd_k6'), ('ssd', 'ssd_k7_f3')]}
    indexed = {m: {r['id']: r for r in d['requests']} for m, d in saved.items()}
    target, draft = os.environ['SSD_TARGET_MODEL'], os.environ['SSD_DRAFT_MODEL']
    tokenizer = AutoTokenizer.from_pretrained(target)
    prompts = get_prompts(args.families, 1, tokenizer=tokenizer, max_prompt_tokens=2048)
    state = {'request': None, 'prompt_len': 0, 'divergence': None}
    records = []
    original = ModelRunner.run_model

    @torch.inference_mode()
    def capture(runner, input_ids, positions, is_prefill, last_only=True, **kwargs):
        logits = original(runner, input_ids, positions, is_prefill, last_only, **kwargs)
        if runner.is_draft or state['request'] is None:
            return logits
        # Detach from any reused graph storage before diagnostic replay.
        returned_logits = logits.clone()
        logits = returned_logits.reshape(-1, returned_logits.shape[-1])
        pos = positions.tolist()
        out_indices = ([state['prompt_len'] - 1] if is_prefill else pos)
        out_indices = [p + 1 - state['prompt_len'] for p in out_indices]
        rows = []
        for row, index in enumerate(out_indices):
            if abs(index - state['divergence']) <= 8:
                rows.append(dict(output_index=index, argmax=int(logits[row].argmax()),
                                 top=top(logits[row], tokenizer)))
        entry = dict(request_id=state['request'], prefill=is_prefill, input_positions=pos,
                     input_token_ids=input_ids.tolist(), rows=rows)
        if rows and not is_prefill and 'head_input' in state:
            # Isolate final projection shape from upstream hidden-state differences.
            hidden = state['head_input'].reshape(-1, state['head_input'].shape[-1])
            entry['same_hidden_single_row_head'] = []
            for j, index in enumerate(out_indices):
                if abs(index - state['divergence']) <= 8:
                    head_logits = torch.nn.functional.linear(hidden[j:j+1], runner.model.lm_head.weight)[0]
                    entry['same_hidden_single_row_head'].append(dict(
                        output_index=index, argmax=int(head_logits.argmax()), top=top(head_logits, tokenizer),
                        max_abs_logit_delta=(head_logits.float()-logits[j].float()).abs().max().item()))
        if not is_prefill and not last_only and state['divergence'] in out_indices:
            context = get_context()
            slots = context.slot_mapping.long()
            # Only this suffix is overwritten; all prefix KV remains identical.
            caches = []
            for module in runner.model.modules():
                if isinstance(module, Attention):
                    for cache in (module.k_cache, module.v_cache):
                        flat = cache.view(-1, *cache.shape[2:])
                        caches.append((flat, flat[slots].clone()))
            sequential = []
            try:
                for j, p in enumerate(pos):
                    set_context(False, slot_mapping=context.slot_mapping[j:j+1],
                                context_lens=torch.tensor([p+1], dtype=torch.int32, device=runner.device),
                                block_tables=context.block_tables)
                    sequential.append(original(runner, input_ids[j:j+1], positions[j:j+1],
                                               False, True).clone())
            finally:
                for flat, backup in caches:
                    flat[slots] = backup
                set_context(**vars(context))
            seq_logits = torch.cat(sequential)
            entry['same_prefix_sequential_replay'] = [
                dict(output_index=index, argmax=int(seq_logits[j].argmax()),
                     top=top(seq_logits[j], tokenizer),
                     max_abs_logit_delta=(seq_logits[j].float()-logits[j].float()).abs().max().item(),
                     argmax_agrees=bool(seq_logits[j].argmax() == logits[j].argmax()))
                for j, index in enumerate(out_indices)]
        # Keep the whole target schedule so a separate, draft-free executor can
        # reproduce historical numerical effects in the KV cache.
        records.append(entry)
        return returned_logits

    ModelRunner.run_model = capture
    llm = LLM(target, num_gpus=2 if args.mode == 'ssd' else 1,
              speculate=args.mode != 'ar', draft=draft, speculate_k=7 if args.mode == 'ssd' else 6,
              draft_async=args.mode == 'ssd', async_fan_out=3, jit_speculate=args.mode == 'ssd',
              max_num_seqs=1, max_model_len=4096, enforce_eager=args.eager,
              gpu_memory_utilization=0.08, draft_gpu_memory_utilization=0.08)
    if args.tokenwise_pointwise:
        install_tokenwise_pointwise_control(llm.model_runner.model, reference_state)
    def remember_head_input(module, inputs):
        if state['request'] is not None:
            state['head_input'] = inputs[0]
    llm.model_runner.model.lm_head.register_forward_pre_hook(remember_head_input)
    sampling = SamplingParams(temperature=0, max_new_tokens=args.output_len, ignore_eos=True)
    # Same warmup workload as the saved baseline captures.
    for prompt in get_prompts(['chat'], args.warmup, split='tune', tokenizer=tokenizer):
        llm.generate([prompt['token_ids']], [sampling], use_tqdm=False)
    requests = []
    for prompt in prompts:
        ar = indexed['ar'][prompt['id']]['output_token_ids']
        sd = indexed['sd'][prompt['id']]['output_token_ids']
        divergence = next(i for i, (a, b) in enumerate(zip(ar, sd)) if a != b)
        state.update(request=prompt['id'], prompt_len=len(prompt['token_ids']), divergence=divergence)
        delivered = []
        last_progress = [0]
        def progress(seq_id, tokens):
            delivered.extend(tokens)
            if len(delivered)-last_progress[0] >= 32 or len(delivered) == args.output_len:
                mismatch = next((i for i,(a,b) in enumerate(zip(ar,delivered)) if a!=b),None)
                print(json.dumps(dict(progress=prompt['id'],tokens=len(delivered),
                                      first_difference_from_original_ar=mismatch)),flush=True)
                last_progress[0] = len(delivered)
        output, _ = llm.generate([prompt['token_ids']], [sampling], use_tqdm=False,
                                 stream_callback=progress if args.progress else None)
        ids = output[0]['token_ids']
        previous = indexed[args.mode][prompt['id']]['output_token_ids'][:args.output_len]
        request = dict(id=prompt['id'], prompt_token_ids=prompt['token_ids'],
                       saved_first_divergence=divergence,
                       ar_token=dict(id=ar[divergence], text=tokenizer.decode([ar[divergence]])),
                       speculative_token=dict(id=sd[divergence], text=tokenizer.decode([sd[divergence]])),
                       output_token_ids=ids, reproduces_saved_output=ids == previous,
                       first_mismatch_with_ar=next((i for i,(a,b) in enumerate(zip(ar,ids)) if a!=b),
                                                  min(len(ar),len(ids)) if len(ar[:args.output_len])!=len(ids) else None))
        requests.append(request)
        print(json.dumps({k:v for k,v in request.items() if k not in ('output_token_ids','prompt_token_ids')}), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(mode=args.mode, eager=args.eager, attention=args.attention,
            rowwise_linear=args.rowwise_linear,
            tokenwise_pointwise=args.tokenwise_pointwise,
            warmup=args.warmup, output_len=args.output_len, families=args.families, target=target, draft=draft,
            diagnostic_only=True, torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
            requests=requests, forwards=records), indent=2))
    state['request'] = None
    llm.exit(hard=True)


if __name__ == '__main__':
    main()
