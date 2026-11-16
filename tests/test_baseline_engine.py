"""Request-timing boundaries and opt-in prefix-cache control."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/ssd'))
import run_engine
from ssd.engine.block_manager import BlockManager
from ssd.engine.sequence import Sequence


@pytest.mark.parametrize('draft',[False,True])
@pytest.mark.parametrize('enabled',[False,True])
def test_prompt_cache_lookup_is_opt_in_override(monkeypatch,draft,enabled):
    monkeypatch.setattr(Sequence,'block_size',256,raising=False)
    manager=BlockManager(6,256,is_draft=draft,enable_prefix_cache=enabled)
    first=Sequence([7]*256+[8])
    manager.allocate(first)
    manager.deallocate(first)
    second=Sequence([7]*256+[8])
    manager.allocate(second)
    cached=second.num_draft_cached_tokens if draft else second.num_cached_tokens
    assert cached==(256 if enabled else 0)
    manager.deallocate(second)
    assert not manager.used_block_ids


def test_prefix_cache_default_is_preserved():
    assert BlockManager(2,256).enable_prefix_cache is True


def test_scheduler_propagates_cache_control_to_target_and_draft(monkeypatch):
    from types import SimpleNamespace
    import ssd.engine.scheduler as scheduler
    monkeypatch.setattr(scheduler.AutoTokenizer,'from_pretrained',lambda *args,**kwargs:None)
    config=SimpleNamespace(max_num_seqs=1,fan_out_list=None,fan_out_list_miss=None,
        draft_async=False,max_num_batched_tokens=4096,max_model_len=4096,eos=0,speculate=True,
        async_fan_out=3,speculate_k=6,kvcache_block_size=256,verbose=False,
        num_kvcache_blocks=10,model='unused',enable_prefix_cache=False)
    queue=scheduler.Scheduler(config,config)
    assert not queue.block_manager.enable_prefix_cache
    assert not queue.draft_block_manager.enable_prefix_cache


def test_parser_defaults_and_natural_eos(monkeypatch,tmp_path):
    monkeypatch.setattr(sys,'argv',['run_engine.py','--mode','ar','--out',str(tmp_path/'result.json')])
    args=run_engine.parse()
    assert not args.profile_capture and not args.disable_prefix_cache and args.ignore_eos
    monkeypatch.setattr(sys,'argv',sys.argv+['--no-ignore_eos','--profile-capture'])
    args=run_engine.parse()
    assert not args.ignore_eos and args.profile_capture


@pytest.mark.parametrize('capture',[False,True])
def test_warmup_outside_clock_and_profiler_and_callback_delivery(monkeypatch,capture):
    events=[]
    counter=iter([10.0,11.0,14.0])
    def clock():
        events.append('clock')
        return next(counter)
    class LLM:
        config=SimpleNamespace(speculate_k=4)
        def generate(self,inputs,sampling,use_tqdm,stream_callback):
            assert stream_callback is not None and not use_tqdm
            events.append('warmup' if inputs==[[99]] else 'generate')
            stream_callback(0,[1])
            return [dict(token_ids=[1,2])],dict(target_step_times=[0.1,0.2],
                accepted_suffix_lens_with_recovery=[1,3,5],cache_hits=[],prefill_total_time=0.1,decode_total_time=0.2)
    monkeypatch.setattr(run_engine.torch.cuda.profiler,'start',lambda:events.append('profile-start'))
    monkeypatch.setattr(run_engine.torch.cuda.profiler,'stop',lambda:events.append('profile-stop'))
    monkeypatch.setattr(run_engine.torch.cuda.nvtx,'range_push',lambda name:events.append('push'))
    monkeypatch.setattr(run_engine.torch.cuda.nvtx,'range_pop',lambda:events.append('pop'))
    item=dict(id='x',family='chat',category='test',token_ids=[5])
    warm=dict(item,token_ids=[99])
    rows=run_engine.execute_requests(LLM(),[item],[warm],None,
        synchronize=lambda:events.append('sync'),clock=clock,profile_capture=capture)
    assert rows[0]['latency_s']==4 and rows[0]['first_delivery_s']==1
    assert rows[0]['accepted_draft_tokens']==6 and rows[0]['verified_draft_tokens']==12
    assert events.index('warmup')<events.index('clock')<events.index('generate')
    assert events[events.index('clock')-1] in ('sync','push')
    assert ('profile-start' in events)==capture
    if capture:
        assert events.index('warmup')<events.index('profile-start')<events.index('clock')
    else:
        assert 'push' not in events
