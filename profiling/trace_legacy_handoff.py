"""Diagnostic-only hooks for the original checkout; never edits baseline files.

Each process writes its own JSONL. CUDA completion timestamps are aligned to a
bracketed monotonic host clock once, with calibration uncertainty preserved.
These hooks are never installed for performance workers.
"""
import json
import os
from pathlib import Path
import time


def install():
    import torch
    from ssd.engine.draft_runner import DraftRunner
    from ssd.engine.speculator_async import SpeculatorAsync
    from ssd.engine.prefix_publication import PrefixCheckpoint
    from ssd.engine.handoff_state import classify_trace  # loaded explicitly by worker for legacy checkout
    folder=Path(os.environ['AO_HANDOFF_TRACE_DIR']);folder.mkdir(parents=True,exist_ok=True)
    def emit(row):
        with (folder/f'{os.getpid()}.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
    original_decode=DraftRunner._decode_tree
    original_step=DraftRunner._decode_tree_step
    original_drain=DraftRunner._drain_published_tail
    original_request=SpeculatorAsync._speculation_request
    original_notify=SpeculatorAsync._notify
    original_poll=PrefixCheckpoint.before_step
    original_service=DraftRunner._service_spec_request
    active={}
    def flush(r):
        for row in getattr(r,'_handoff_diagnostic',[]):
            if row.get('emitted'):continue
            events=row['events']
            if not events or not events[max(events)].query():continue
            value={k:v for k,v in row.items() if k not in ['events','emitted']}
            for name,depth in [('prefix_ready_ns',r.config.early_publish_tokens),('full_ready_ns',r.config.speculate_k)]:
                value[name]=r._handoff_anchor+int(r._handoff_origin.elapsed_time(events[depth])*1e6) if depth in events else None
            value['clock_uncertainty_ns']=r._handoff_uncertainty
            emit(dict(kind='draft_completion',**value));row['emitted']=True
    def decode(r,payload):
        flush(r)
        if not hasattr(r,'_handoff_origin'):
            r._handoff_origin=torch.cuda.Event(enable_timing=True)
            torch.cuda.current_stream().synchronize()
            before=time.monotonic_ns();r._handoff_origin.record();r._handoff_origin.synchronize();after=time.monotonic_ns()
            r._handoff_anchor=(before+after)//2;r._handoff_uncertainty=(after-before)//2;r._handoff_diagnostic=[]
        row=dict(generation=r.cache_generation+1,producer_epoch=r.served_epoch,events={})
        active['row']=row;r._handoff_diagnostic.append(row)
        return original_decode(r,payload)
    def step(r,depth,*args,**kwargs):
        value=original_step(r,depth,*args,**kwargs)
        event=torch.cuda.Event(enable_timing=True);event.record();active['row']['events'][depth+1]=event
        return value
    def poll(ctl,submitted,pending):
        requested=pending() if callable(pending) else pending
        if requested and 'request_noticed_ns' not in active.get('row',{}):active['row']['request_noticed_ns']=time.monotonic_ns()
        return original_poll(ctl,submitted,pending)
    def drain(r):
        value=original_drain(r);flush(r);return value
    def service(r):
        if active.get('row'):
            active['row'].setdefault('request_noticed_ns',time.monotonic_ns())
        return original_service(r)
    def notify(r):
        posted=time.monotonic_ns();value=original_notify(r)
        emit(dict(kind='target_post',epoch=r.request_epoch.value if r.request_epoch else None,request_posted_ns=posted));return value
    def request(r,*a,**kw):
        value=original_request(r,*a,**kw)
        row=r.metrics['publication_records'][-1] if r.metrics and r.metrics['publication_records'] else {}
        emit(dict(kind='target_delivery',generation=row.get('generation'),epoch=row.get('epoch'),delivery_ns=time.monotonic_ns()))
        return value
    DraftRunner._decode_tree=decode;DraftRunner._decode_tree_step=step;DraftRunner._drain_published_tail=drain
    DraftRunner._service_spec_request=service
    PrefixCheckpoint.before_step=poll;SpeculatorAsync._notify=notify;SpeculatorAsync._speculation_request=request


def collect(folder):
    from ssd.engine.handoff_state import classify_trace
    rows=[]
    for p in Path(folder).glob('*.jsonl'):rows.extend(json.loads(line) for line in p.read_text().splitlines())
    posted={r['epoch']:r['request_posted_ns'] for r in rows if r['kind']=='target_post'}
    delivery={r['generation']:r for r in rows if r['kind']=='target_delivery'}
    result=[]
    for row in rows:
        if row['kind']!='draft_completion':continue
        target=delivery.get(row['generation'],{})
        row['delivery_ns']=target.get('delivery_ns');row['request_posted_ns']=posted.get(target.get('epoch'))
        row['classification']=classify_trace(row);result.append(row)
    return result
