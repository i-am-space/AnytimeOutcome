"""Host ownership protocol. No CUDA or model operations are performed here."""
from dataclasses import dataclass, field
import threading
import time


@dataclass
class Work:
    command: int
    epoch: int
    posted_ns: int
    noticed_ns: int
    packet: dict = field(default_factory=dict)
    route: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    model_done: threading.Event = field(default_factory=threading.Event)
    miss: bool = False
    error: BaseException | None = None
    partial: dict | None = None
    glue: object = None
    reply: object = None

    def wait(self, event, timeout=120):
        if not event.wait(timeout):
            raise TimeoutError('Handoff owner did not complete work')
        if self.error is not None:
            raise RuntimeError('Independent handoff service failed') from self.error


class LiveTree:
    """A producer owns graph/KV buffers until its consumer and GPU tail finish.

    Submitted events are immutable and unique to this generation. A consumer
    may read only columns <= a completed event; the producer writes later ones.
    """
    def __init__(self, generation, epoch, maximum, payload, tokens, keys, expected, events):
        self.generation, self.epoch, self.maximum = generation, epoch, maximum
        self.payload, self.tokens, self.keys, self.expected = payload, tokens, keys, expected
        self.events = events
        self.submitted = 0
        self.stopped = False
        self.retired = False
        self.claimed = False
        self.condition = threading.Condition()
        self.trace = dict(generation=generation, producer_epoch=epoch)

    def submit(self, depth):
        with self.condition:
            if self.stopped or depth != self.submitted + 1 or depth > self.maximum:
                raise RuntimeError('Invalid producer submission')
            self.submitted = depth
            self.condition.notify_all()

    def finish_submission(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()

    def completed(self):
        # Events become visible only after record() and submit() under the lock.
        with self.condition:
            count = self.submitted
        completed = 0
        for depth in range(1, count + 1):
            if self.events[depth] is None or not self.events[depth].query():
                break
            completed = depth
        return completed

    def ready(self, minimum, timeout=120):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.submitted < minimum:
                if self.stopped or self.retired:
                    raise RuntimeError('Producer stopped before a publishable prefix')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('No publishable prefix')
                self.condition.wait(remaining)
            if self.claimed or self.retired:
                raise RuntimeError('Stale or already consumed tree')
            self.claimed = True
        self.events[minimum].synchronize()
        return max(minimum, self.completed())

    def retire(self, consumer_done):
        if not consumer_done:
            raise RuntimeError('Cannot reuse storage before its consumer completes')
        with self.condition:
            if not self.stopped:
                raise RuntimeError('Cannot retire while producer may submit more work')
            depth = self.submitted
        if depth:
            self.events[depth].synchronize()
        with self.condition:
            self.retired = True


class Registry:
    def __init__(self):
        self.condition = threading.Condition()
        self.live = None
        self.building = False
        self.error = None

    def begin(self):
        with self.condition:
            self.live = None
            self.building = True
            self.condition.notify_all()

    def install(self, tree):
        with self.condition:
            if self.live is not None:
                raise RuntimeError('Previous generation still advertised')
            self.live = tree
            self.building = False
            self.condition.notify_all()

    def get(self, timeout=120):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.building and self.live is None and self.error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Producer did not advertise a generation')
                self.condition.wait(remaining)
            if self.error:
                raise RuntimeError('Producer failed') from self.error
            return self.live

    def clear(self):
        with self.condition:
            self.live = None
            self.building = False
            self.condition.notify_all()


def classify_trace(row):
    """CPU/GPU clock alignment has bounded calibration uncertainty."""
    if row.get('prefix_ready_ns') is None or row.get('full_ready_ns') is None:
        return 'no_full_tail_measurement'
    p, full = row['prefix_ready_ns'], row['full_ready_ns']
    request, noticed, delivery = (row.get(k) for k in ('request_posted_ns','request_noticed_ns','delivery_ns'))
    epsilon = row.get('clock_uncertainty_ns', 0)
    if delivery is not None and delivery + epsilon < full:
        return 'delivered_before_full'
    if request is not None and request - epsilon >= full:
        return 'request_after_full'
    if noticed is None:return 'request_not_observed_at_checkpoint'
    if noticed - epsilon >= full:
        return 'request_observed_after_full'
    if delivery is None:return 'no_delivery_measurement'
    if delivery-epsilon < full:return 'ambiguous_at_clock_resolution'
    if p + epsilon < full:
        return 'handoff_finished_after_full'
    return 'ambiguous_at_clock_resolution'
