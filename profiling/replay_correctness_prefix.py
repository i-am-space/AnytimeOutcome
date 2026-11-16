"""Teacher-force shared AR histories, then vary forward width from identical KV.

No draft, verifier, scheduler rollback, or cross-GPU communication participates.
All forwards are eager, using the same BF16 weights and portable attention backend.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_engine
import torch
from ssd import LLM, SamplingParams
from ssd.engine.sequence import Sequence
from ssd.layers.attention import Attention
from ssd.utils.context import get_context, set_context
from investigate_correctness import top
from workloads import get_prompts
from attention_controls import install_attention_control


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', help='Captured target schedule to replay without the SSD executor')
    parser.add_argument('--family', choices=['chat', 'code', 'doc'])
    parser.add_argument('--attention', choices=['portable', 'sdpa_flash', 'sdpa_math'], default='portable')
    parser.add_argument('--out')
    args = parser.parse_args()
    install_attention_control(args.attention)
    schedule = json.loads(Path(args.schedule).read_text()) if args.schedule else None
    root = Path(__file__).resolve().parents[1]
    ar = {r['id']: r for r in json.loads((root/'results/nsys/ar_run.json').read_text())['requests']}
    sd = {r['id']: r for r in json.loads((root/'results/nsys/sd_k6_run.json').read_text())['requests']}
    llm = LLM(os.environ['SSD_TARGET_MODEL'], num_gpus=1, max_num_seqs=1,
              max_model_len=4096, enforce_eager=True, gpu_memory_utilization=0.08)
    runner = llm.model_runner
    modules = [m for m in runner.model.modules() if isinstance(m, Attention)]
    hidden = {}
    def remember(module, inputs):
        hidden['value'] = inputs[0].clone()
    runner.model.lm_head.register_forward_pre_hook(remember)
    prompts = get_prompts([args.family] if args.family else ['chat','code','doc'], 1,
                          tokenizer=llm.tokenizer, max_prompt_tokens=2048)
    results = []
    for prompt in prompts:
        ids = ar[prompt['id']]['output_token_ids']
        spec = sd[prompt['id']]['output_token_ids']
        d = next(i for i,(a,b) in enumerate(zip(ids,spec)) if a != b)
        seq = Sequence(prompt['token_ids'], SamplingParams(temperature=0))
        seq.block_table = list(range((len(seq)+d+8+255)//256))
        first = runner.run([seq], True)[0]
        seq.num_cached_tokens = len(seq)
        predictions = [first]
        for i in range(d):
            seq.append_token(ids[i])
            predicted, logits = runner.run([seq], False, draft_return_logits=True)
            seq.num_cached_tokens += 1
            predictions.append(predicted[0])
        single = logits.reshape(-1, logits.shape[-1])[0].clone()
        single_hidden = hidden['value'].reshape(-1, hidden['value'].shape[-1])[0:1].clone()
        item = dict(id=prompt['id'], divergence_zero_based=d,
                    teacher_forced_predictions_equal_saved_ar=predictions == ids[:d+1],
                    first_teacher_forced_difference=next((i for i,(a,b) in enumerate(zip(predictions,ids)) if a!=b),None),
                    single=dict(argmax=int(single.argmax()), top=top(single,llm.tokenizer)), widths={})
        p = len(seq)-1
        bt = torch.tensor([seq.block_table],dtype=torch.int32,device=runner.device)
        max_positions = torch.arange(p,p+8,device=runner.device)
        slots = bt[0,max_positions//256].long()*256+max_positions%256
        caches = []
        ar_prefix_caches = []
        for module in modules:
            for cache in (module.k_cache,module.v_cache):
                flat=cache.view(-1,*cache.shape[2:])
                caches.append((flat,flat[slots].clone()))
                if schedule:
                    ar_prefix_caches.append(flat[:p+1].clone())
        old_flags=[m.speculate for m in modules]
        try:
            for module in modules:
                module.speculate=True
            for width in [1,7,8]:
                # Identical old prefix for every width; future rows are causally masked.
                for flat,backup in caches:
                    flat[slots]=backup
                values=[ids[d-1]]+ids[d:d+width-1]
                positions=max_positions[:width]
                set_context(False,cu_seqlens_q=torch.tensor([0,width],dtype=torch.int32,device=runner.device),
                            max_seqlen_q=width,slot_mapping=slots[:width].int(),
                            context_lens=torch.tensor([p+width],dtype=torch.int32,device=runner.device),block_tables=bt)
                multi=runner.run_model(torch.tensor(values,device=runner.device),positions,False,False)
                row=multi.reshape(-1,multi.shape[-1])[0]
                multi_hidden=hidden['value'].reshape(-1,hidden['value'].shape[-1])[0:1]
                row_head=torch.nn.functional.linear(multi_hidden,runner.model.lm_head.weight)[0]
                # Replicate the single hidden row across M to isolate LM-head shape.
                expanded_head=torch.nn.functional.linear(single_hidden.expand(width,-1).contiguous(),
                                                         runner.model.lm_head.weight)[0]
                item['widths'][str(width)]=dict(argmax=int(row.argmax()),top=top(row,llm.tokenizer),
                    max_abs_logit_delta_vs_single=(row.float()-single.float()).abs().max().item(),
                    max_abs_hidden_delta_vs_single=(multi_hidden.float()-single_hidden.float()).abs().max().item(),
                    multi_hidden_single_row_head=dict(argmax=int(row_head.argmax()),top=top(row_head,llm.tokenizer)),
                    single_hidden_multi_row_head=dict(argmax=int(expanded_head.argmax()),top=top(expanded_head,llm.tokenizer)))
        finally:
            for flat,backup in caches:
                flat[slots]=backup
            for module,flag in zip(modules,old_flags):
                module.speculate=flag
        if schedule:
            # Reconstruct SSD's target execution using only explicit logical
            # positions and a fresh contiguous KV mapping. No branch manager,
            # acceptance routine, draft model, or NCCL calls participate.
            fresh = Sequence(prompt['token_ids'], SamplingParams(temperature=0))
            fresh.block_table = seq.block_table
            runner.run([fresh], True)
            try:
                for module in modules:
                    module.speculate=True
                for forward in schedule['forwards']:
                    if forward['request_id'] != prompt['id'] or forward['prefill']:
                        continue
                    positions = torch.tensor(forward['input_positions'], device=runner.device)
                    values = torch.tensor(forward['input_token_ids'], device=runner.device)
                    width = len(forward['input_positions'])
                    forward_slots = bt[0,positions//256].long()*256+positions%256
                    set_context(False,cu_seqlens_q=torch.tensor([0,width],dtype=torch.int32,device=runner.device),
                        max_seqlen_q=width,slot_mapping=forward_slots.int(),
                        context_lens=torch.tensor([int(positions[-1])+1],dtype=torch.int32,device=runner.device),
                        block_tables=bt)
                    logits=runner.run_model(values,positions,False,False)
                    start=forward['input_positions'][0]-len(prompt['token_ids'])
                    row_index=d-start-1
                    if not (0 <= start < d and 0 <= row_index < width and
                            forward['input_token_ids'][:d-start] == ids[start:d]):
                        continue
                    observed=next((r for r in forward['rows'] if r['output_index']==d),None)
                    if observed is None:
                        continue
                    row=logits.reshape(-1,logits.shape[-1])[row_index]
                    same_top_values=all(float(row[t['token_id']]) == t['logit'] for t in observed['top'])
                    kv_deltas=[]
                    for layer_index, (module, old_k, old_v) in enumerate(zip(
                            modules,ar_prefix_caches[::2],ar_prefix_caches[1::2])):
                        for kind,cache,old in [('key',module.k_cache,old_k),('value',module.v_cache,old_v)]:
                            current=cache.view(-1,*cache.shape[2:])[:p+1]
                            delta=(current.float()-old.float()).abs()
                            kv_deltas.append(dict(layer=layer_index,kind=kind,max_abs=float(delta.max()),
                                                  mean_abs=float(delta.mean())))
                    item['schedule_replay']=dict(argmax=int(row.argmax()),top=top(row,llm.tokenizer),
                        captured_top8_logits_reproduced_exactly=same_top_values,
                        kv_prefix_comparison_to_ar=kv_deltas,
                        conclusion='Logical target schedule replay; no speculative engine state management')
                    break
            finally:
                for module,flag in zip(modules,old_flags):
                    module.speculate=flag
            if 'schedule_replay' not in item:
                raise RuntimeError('No matching-history divergence found in captured schedule')
        results.append(item)
        path=Path(args.out) if args.out else root/'results/correctness'/('schedule_controls.json' if schedule else 'prefix_controls.json')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(diagnostic_only=True,eager=True,attention=args.attention,
                                       torch_version=torch.__version__,
                                       requests=results),indent=2)+'\n')
        print(json.dumps(item),flush=True)
    llm.exit(hard=True)


if __name__=='__main__':
    main()
