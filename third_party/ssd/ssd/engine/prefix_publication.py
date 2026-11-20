"""Completed-prefix publication; tail completion is deferred until KV reuse."""
from ssd.engine.publication_policy import PublicationCheckpoint
import torch


class PrefixSnapshot:
    """Reusable independent snapshot, retired after its response consumer.

    Eligible parent positions form a contiguous prefix of fan-out rows. Avoid
    boolean indexing/nonzero and device allocation during publication.
    """
    def __init__(self, maximum, fan_hit, fan_miss, vocab, dtype, device):
        if sum(fan_hit) != sum(fan_miss):
            raise ValueError('Snapshot fan-out capacities differ')
        self.counts = [[sum(fan[:p+1]) for p in range(maximum+1)]
                       for fan in (fan_miss, fan_hit)]
        rows = sum(fan_hit)
        self.lengths = torch.empty(rows, device=device, dtype=torch.int64)
        self.keys = torch.empty((rows, 3), device=device, dtype=torch.int64)
        self.expected = torch.empty(rows, device=device, dtype=torch.int64)
        self.tokens = torch.empty((rows, maximum), device=device, dtype=torch.int64)
        self.logits = torch.empty((rows, maximum, vocab), device=device, dtype=dtype)

    def copy(self, payload, tokens, logits, length, parent_positions, valid_lengths=None):
        if len(payload['cache_hits_list']) != 1:
            raise RuntimeError('Prefix snapshots require batch one')
        parent_length = payload['valid_length']
        if not 1 <= length <= self.tokens.shape[1] or not 1 <= parent_length <= self.tokens.shape[1]:
            raise RuntimeError('Invalid snapshot prefix or parent length')
        count = self.counts[int(bool(payload['cache_hits_list'][0]))][parent_length]
        self.keys[:count, 0].copy_(payload['seq_ids_expanded'][:count])
        self.keys[:count, 1].copy_(parent_positions[:count])
        self.keys[:count, 2].copy_(payload['rec_flat'][:count])
        torch.add(parent_positions[:count], payload['base_num_tokens'][0], out=self.expected[:count])
        self.expected[:count].add_(1)
        if valid_lengths is not None:
            torch.clamp(valid_lengths[:count], max=length, out=self.lengths[:count])
        # Later columns may still be changing on the producer stream.
        self.tokens[:count, :length].copy_(tokens[:count, :length])
        self.logits[:count, :length].copy_(logits[:count, :length])
        return (self.keys[:count], self.expected[:count],
                self.tokens[:count, :length], self.logits[:count, :length])


class PrefixCheckpoint(PublicationCheckpoint):
    def begin(self, stream):
        super().begin(stream)
        self.published = None
        self.launched = 0

    def submitted(self, depth):
        super().submitted(depth)
        self.launched = depth

    def poll(self, submitted):
        while self.completed < submitted:
            self.queries += 1
            if not self.events[self.completed + 1].query():
                break
            self.completed += 1

    def before_step(self, submitted, pending):
        # Accept a callable so a request arriving during backpressure is seen
        # before another step is launched.
        requested = pending if callable(pending) else lambda: pending
        self.poll(submitted)
        if requested() and submitted >= self.minimum:
            if self.completed < self.minimum:
                self.wait(self.minimum)  # Never wait for the later submitted tail.
                self.poll(submitted)
            self.published = self.completed
            self.decisions += 1
            return True
        if submitted - self.completed >= self.window:
            self.wait(submitted - self.window + 1)
            if requested() and self.completed >= self.minimum:
                self.published = self.completed
                self.decisions += 1
                return True
        return False

    def final_depth(self, pending):
        if self.published is None:
            self.poll(self.launched)
            self.published = (self.completed if pending and self.completed >= self.minimum
                              else self.launched)
        return self.published


def record_stream_tree(value, stream):
    """Retain allocator ownership for tensors consumed on another stream."""
    import torch
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            value.record_stream(stream)
    elif isinstance(value, dict):
        for item in value.values(): record_stream_tree(item, stream)
    elif isinstance(value, (tuple, list)):
        for item in value: record_stream_tree(item, stream)
