"""Versioned greedy transport and independent host service, separate from legacy SSD.

Only the producer thread calls a model or writes KV/graph buffers. The service
thread receives requests, reads completed token columns, and sends IDs. A miss
is delegated to the producer. Every generation is joined before any KV reuse.
"""
import queue
import threading
import time

import torch
import torch.distributed as dist

from ssd.engine.draft_runner import DraftRunner
from ssd.engine.handoff_state import LiveTree, Registry, Work, classify_trace
from ssd.engine.prefix_publication import record_stream_tree
from ssd.utils.context import set_context, reset_context
from ssd.utils.async_helpers.nccl_pack import recv_int64
from ssd.utils.async_helpers.async_spec_helpers import make_glue_decode_input_ids


class GreedyDraftRunner(DraftRunner):
    def __init__(self, cfg, rank=0, init_q=None, request_epoch=None, publication_feedback=None,
                 request_clock=None, trace_queue=None):
        self.request_clock, self.trace_queue = request_clock, trace_queue
        self.registry = Registry()
        self.work_queue = queue.Queue()
        self.pending = threading.Event()
        self._live = None
        self._trace_rows = []
        self._origin = None
        self._clock_anchor = self._clock_uncertainty = 0
        super().__init__(cfg, rank, init_q, request_epoch, publication_feedback)

    def _init_prealloc_buffers(self):
        # No vocabulary-sized history/snapshot allocation in the IDs protocol.
        K, M = self.config.speculate_k, self.config.MQ_LEN
        d = self.device
        self._compute_stream = torch.cuda.current_stream()
        self._handoff_stream = torch.cuda.Stream(device=d, priority=-1)
        self._handoff_stream.wait_stream(self._compute_stream)
        self._score_item_bytes=torch.empty((),dtype=self.hf_config.torch_dtype).element_size()
        self._step_pos_offsets = torch.arange(K, device=d)[:, None] * M
        self._step_rope_offsets = torch.arange(K, device=d)[:, None]
        self._fan_idx_hit = torch.arange(K+1, device=d).repeat_interleave(self.config.fan_out_t)
        self._fan_idx_miss = torch.arange(K+1, device=d).repeat_interleave(self.config.fan_out_t_miss)
        self._arange_mq = torch.arange(M, device=d)
        self._arange_kp1 = torch.arange(K+1, device=d)
        self._arange_2kp1 = torch.arange(2*K+1, device=d)

    def _notify_observed(self, last):
        # Shared host clock: no CPU/GPU timestamp subtraction without calibration.
        while True:
            with self.request_epoch.get_lock():
                epoch = self.request_epoch.value
                if epoch > last:
                    return epoch, self.request_clock.value, time.monotonic_ns()
            time.sleep(.00005)

    def _receive(self, last):
        epoch, posted, noticed = self._notify_observed(last)
        command = torch.empty(1, dtype=torch.int64, device=self.device)
        dist.recv(command, src=0, group=self.async_pg)
        w = Work(int(command.item()), epoch, posted, noticed)
        if w.command == 0:
            meta = torch.empty(3, dtype=torch.int64, device=self.device)
            dist.recv(meta, src=0, group=self.async_pg)
            B, K, F = meta.tolist()
            if (B, K, F) != (1, self.config.speculate_k, self.config.async_fan_out):
                raise RuntimeError('Greedy protocol shape/version mismatch')
            fused = recv_int64(self.async_pg, src=0, total_length=5+self.config.max_blocks)
            w.packet = dict(keys=fused[:3].view(1, 3), num_tokens=fused[3:4],
                dbt=fused[4:4+self.config.max_blocks].view(1, -1).to(torch.int32),
                temperatures=fused[-1:].to(torch.int32).view(torch.float32))
            if w.packet['temperatures'].item() != 0:
                raise ValueError('Stochastic request on IDs-only transport')
        return w

    def _calibrate(self):
        if self.config.handoff_trace:
            self._origin = torch.cuda.Event(enable_timing=True)
            self._compute_stream.synchronize()
            before = time.monotonic_ns()
            self._origin.record(self._compute_stream)
            self._origin.synchronize()
            after = time.monotonic_ns()
            self._clock_anchor = (before+after)//2
            self._clock_uncertainty = (after-before)//2

    def _retire(self):
        tree = self._live
        if tree is None:
            return
        # This is called only after service work.done or for an unclaimed drain.
        started=time.perf_counter()
        tree.retire(consumer_done=True)
        if self.publication_feedback is not None:
            with self.publication_feedback.get_lock():
                self.publication_feedback[11] += time.perf_counter()-started
        if self.config.handoff_trace:
            row = dict(tree.trace, submitted_depth=tree.submitted,
                       clock_uncertainty_ns=self._clock_uncertainty)
            for name, depth in [('prefix_ready_ns', self.config.early_publish_tokens or self.config.speculate_k),
                                ('full_ready_ns', self.config.speculate_k),
                                ('submitted_tail_ready_ns', tree.submitted)]:
                row[name] = (self._clock_anchor + int(self._origin.elapsed_time(tree.events[depth])*1e6)
                             if 0 < depth <= tree.submitted else None)
            for name,event in getattr(tree,'trace_events',{}).items():
                row[name]=self._clock_anchor+int(self._origin.elapsed_time(event)*1e6)
            row['classification'] = classify_trace(row)
            self._trace_rows.append(row)
        self._live = None

    @torch.inference_mode()
    def _tree_step(self, depth, current, ropes, slots, lengths, payload):
        set_context(is_prefill=False, slot_mapping=slots[depth],
                    context_lens=lengths[depth].to(torch.int32), block_tables=payload['block_tables'])
        try:
            logits = self.run_model(current, ropes[depth], is_prefill=False, last_only=False,
                                    tree_decode_step=depth, cache_hits=payload['cache_hits'])
            return logits.reshape(-1, self.hf_config.vocab_size).argmax(-1)
        finally:
            reset_context()

    @torch.inference_mode()
    def _produce(self, partial, glue):
        with torch.cuda.stream(self._compute_stream):
            payload = self._build_tree_batch(partial, glue)
            B, K, F, N = payload['metadata_ints']
            _, ropes, lengths, slots = self._compute_step_positions_and_slot_maps(
                payload['positions'], payload['rope_positions'], payload['block_tables'], B, K, F, N,
                self.config.MQ_LEN)
            tokens = torch.zeros((N, K), dtype=torch.int64, device=self.device)
            fan = self._fan_idx_hit if payload['cache_hits_list'][0] else self._fan_idx_miss
            keys = torch.stack((payload['seq_ids_expanded'], fan, payload['rec_flat']), dim=1)
            expected = payload['base_num_tokens'][0]+fan+1
            checkpoints=bool(self.config.early_publish_tokens or self.config.handoff_trace)
            events = [None]+[torch.cuda.Event(enable_timing=self.config.handoff_trace)
                            if checkpoints or depth==K else None for depth in range(1,K+1)]
            self.cache_generation += 1
            tree = LiveTree(self.cache_generation, self.served_epoch, K, payload, tokens, keys, expected, events)
            self._live = tree
            self.registry.install(tree)
            current = payload['input_ids']
            try:
                for depth in range(K):
                    # No checkpoints at all for complete IDs-only SSD (P=0).
                    if self.config.early_publish_tokens:
                        pending = self.request_epoch.value > self.served_epoch
                        if pending and depth >= self.config.early_publish_tokens:
                            break
                        while depth-tree.completed() >= self.config.publication_max_inflight:
                            tree.events[depth-self.config.publication_max_inflight+1].synchronize()
                        if self.request_epoch.value > self.served_epoch and depth >= self.config.early_publish_tokens:
                            break
                    current = self._tree_step(depth, current, ropes, slots, lengths, payload)
                    tokens[:, depth].copy_(current)
                    if events[depth+1] is not None:events[depth+1].record(self._compute_stream)
                    tree.submit(depth+1)
            finally:
                tree.finish_submission()

    @torch.inference_mode()
    def _cached(self, w):
        tree = self.registry.get()
        if tree is None:
            return None
        if tree.epoch >= w.epoch:
            raise RuntimeError('Future/stale epoch in live-tree registry')
        minimum = self.config.early_publish_tokens or self.config.speculate_k
        ready = tree.ready(minimum)
        length = min(ready, self.config.speculate_k)
        # Restrict keys to actual parent positions, including short previous replies.
        eligible = tree.payload.get('valid_length', self.config.speculate_k)
        match = (tree.keys == w.packet['keys']).all(-1)
        match &= (tree.keys[:, 1] <= eligible) & (tree.expected == w.packet['num_tokens'][0])
        if not bool(match.any().item()):
            return None
        index = int(match.to(torch.int64).argmax().item())
        full_ready = tree.submitted == tree.maximum and tree.events[tree.maximum].query()
        bypass = self.config.handoff_bypass_ready and full_ready
        selected_ns=time.monotonic_ns()
        if self.config.handoff_trace:
            tree.trace_events={name:torch.cuda.Event(enable_timing=True) for name in ('copy_started_ns','copy_finished_ns')}
            tree.trace_events['copy_started_ns'].record(self._handoff_stream)
        if bypass:
            length = tree.maximum
            out = tree.tokens[index:index+1]  # immutable view; producer joins response before reuse
        else:
            out = torch.zeros((1, self.config.speculate_k), dtype=torch.int64, device=self.device)
            out[:, :length].copy_(tree.tokens[index:index+1, :length])
        if self.config.handoff_trace:tree.trace_events['copy_finished_ns'].record(self._handoff_stream)
        if self.publication_feedback is not None:
            with self.publication_feedback.get_lock():
                self.publication_feedback[9] += int(not tree.events[tree.submitted].query())
        record_stream_tree((tree.tokens, tree.keys, tree.expected, tree.payload), self._handoff_stream)
        tree.trace.update(request_posted_ns=w.posted_ns, request_noticed_ns=w.noticed_ns,
            snapshot_selected_ns=selected_ns, selected_length=length, full_ready_bypass=bypass)
        return out, torch.ones(1, dtype=torch.int64, device=self.device), length, tree

    @torch.inference_mode()
    def _jit_ids(self, packet):
        self._retire()
        K = self.config.speculate_k
        out = torch.empty((1, K), dtype=torch.int64, device=self.device)
        current = packet['keys'][:, -1]
        positions = packet['num_tokens']-1
        lengths = packet['num_tokens']
        for depth in range(K):
            block_idx = positions//self.block_size
            slot = packet['dbt'][0, block_idx]*self.block_size+positions%self.block_size
            set_context(is_prefill=False, slot_mapping=slot, context_lens=lengths.to(torch.int32),
                        block_tables=packet['dbt'], is_jit=True)
            try:
                logits = self.run_model(current, positions, is_prefill=False, last_only=True)
                current = logits.argmax(-1)
            finally:
                reset_context()
            out[:, depth].copy_(current)
            positions, lengths = positions+1, lengths+1
        done = torch.cuda.Event()
        done.record(self._compute_stream)
        return out, torch.zeros(1, dtype=torch.int64, device=self.device), K, done

    def _send(self, w, reply):
        out, hits, length, owner = reply
        if not isinstance(owner, LiveTree):
            self._handoff_stream.wait_event(owner)
        glue = make_glue_decode_input_ids(out, w.packet['keys'][:, 2])
        header = torch.tensor([length, w.epoch, self.cache_generation, 0, 0],
                              dtype=torch.int64, device=self.device)
        fused = torch.cat((hits.reshape(-1), out.reshape(-1), header))
        if isinstance(owner,LiveTree):owner.trace['response_send_started_ns']=time.monotonic_ns()
        dist.send(fused, dst=0, group=self.async_pg)
        done = torch.cuda.Event()
        done.record(self._handoff_stream)
        done.synchronize()
        if self.publication_feedback is not None:
            with self.publication_feedback.get_lock():
                self.publication_feedback[12] += int(isinstance(owner,LiveTree) and owner.trace.get('full_ready_bypass',False))
                self.publication_feedback[13] += fused.numel()*fused.element_size()
                self.publication_feedback[14] += self.config.speculate_k*self.hf_config.vocab_size*self._score_item_bytes
                self.publication_feedback[15] += int(not isinstance(owner,LiveTree))
        if isinstance(owner, LiveTree):
            owner.trace['delivery_ns'] = time.monotonic_ns()
            if self.publication_feedback is not None:
                with self.publication_feedback.get_lock():
                    self.publication_feedback[8] += int(length < owner.maximum)
                    self.publication_feedback[10] += int(not owner.events[owner.submitted].query())
        partial = dict(valid_length=length, num_tokens=w.packet['num_tokens'],
            seq_ids=w.packet['keys'][:, 0], temperatures=w.packet['temperatures'],
            dbt=w.packet['dbt'], cache_hits=hits, returned_tokens=out,
            target_recovery_activations=None, previous_activations=None,
            extend_counts=None, extend_eagle_acts=None, extend_token_ids=None)
        record_stream_tree((out, partial, glue), self._compute_stream)
        w.partial, w.glue = partial, glue
        self.registry.begin()  # next command must wait for the next generation, never reread this one

    @torch.inference_mode()
    def _service_loop(self):
        torch.cuda.set_device(self.rank)
        last = 0
        try:
            while True:
                with torch.cuda.stream(self._handoff_stream):
                    w = self._receive(last)
                    last = w.epoch
                    self.pending.set()
                    self.work_queue.put(w)
                    if w.command == 0:
                        reply = self._cached(w)
                        w.miss = reply is None
                        w.route.set()
                        if w.miss:
                            w.wait(w.model_done)
                            reply = w.reply
                        self._send(w, reply)
                        w.done.set()
                    else:
                        w.route.set()
                        w.wait(w.model_done)
                        w.done.set()
                        if w.command == 2:
                            return
        except BaseException as exc:
            self.registry.error = exc
            with self.registry.condition:
                self.registry.condition.notify_all()
            if 'w' in locals():
                w.error = exc
                w.route.set();w.done.set();w.model_done.set()
            failed = Work(-1, last, 0, time.monotonic_ns(), error=exc)
            failed.route.set()
            self.work_queue.put(failed)

    def _control(self, w):
        self._retire()
        self.registry.clear()
        if w.command == 1:
            self.draft_async_prefill()
            torch.cuda.reset_peak_memory_stats()
            self._calibrate()
        elif w.command == 3:
            self._compute_stream.synchronize()
            if self.trace_queue is not None:
                self.trace_queue.put(self._trace_rows)
                self._trace_rows = []
            dist.send(torch.tensor([w.epoch, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                                  dtype=torch.int64, device=self.device), dst=0, group=self.async_pg)
        elif w.command != 2:
            raise RuntimeError('Unknown greedy command')

    @torch.inference_mode()
    def draft_loop(self):
        self._draft_step_times = []
        if self.config.handoff_service == 'threaded':
            service = threading.Thread(target=self._service_loop, name='ssd-greedy-service', daemon=True)
            service.start()
            try:
                while True:
                    w = self.work_queue.get(timeout=1800)
                    w.wait(w.route)
                    if w.command == 0:
                        if w.miss:
                            with torch.cuda.stream(self._compute_stream):
                                w.reply = self._jit_ids(w.packet)
                            w.model_done.set()
                        w.wait(w.done)
                        self._retire()
                        self.served_epoch = w.epoch
                        self.pending.clear()
                        self._produce(w.partial, w.glue)
                    else:
                        with torch.cuda.stream(self._compute_stream):
                            self._control(w)
                        self.served_epoch = w.epoch
                        w.model_done.set()
                        w.wait(w.done)
                        self.pending.clear()
                        if w.command == 2:
                            break
            except BaseException as exc:
                self.registry.error = exc
                with self.registry.condition:self.registry.condition.notify_all()
                if 'w' in locals():
                    w.error = exc;w.model_done.set();w.done.set()
                raise
            service.join(timeout=5)
            if service.is_alive():raise RuntimeError('Handoff service did not shut down')
        else:
            last = 0
            while True:
                with torch.cuda.stream(self._handoff_stream):
                    w = self._receive(last)
                last = w.epoch
                if w.command == 0:
                    with torch.cuda.stream(self._handoff_stream):reply = self._cached(w)
                    if reply is None:
                        with torch.cuda.stream(self._compute_stream):reply = self._jit_ids(w.packet)
                    with torch.cuda.stream(self._handoff_stream):self._send(w, reply)
                    self._retire()
                    self.served_epoch = w.epoch
                    self._produce(w.partial, w.glue)
                else:
                    with torch.cuda.stream(self._compute_stream):self._control(w)
                    self.served_epoch = w.epoch
                    if w.command == 2:break
        self.exit()
