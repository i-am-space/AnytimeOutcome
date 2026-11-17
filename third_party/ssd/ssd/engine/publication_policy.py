"""Request-driven publication with bounded GPU submission and a causal heuristic.

The adaptive rule minimizes estimated (verification + extra drafting) time per
committed token. It is a local heuristic, not a prediction of full-request speedup.
Only observations from the current request are used. No target arithmetic changes.
"""
import math


def choose_depth(ready, maximum, verify_s, step_s, survival, min_gain=0.05):
    if not 1 <= ready <= maximum:
        raise ValueError('Invalid ready depth')
    if (not all(math.isfinite(x) for x in (verify_s, step_s, survival)) or
            verify_s <= 0 or step_s <= 0 or not 0 <= survival <= 1):
        return ready
    def cost(depth):
        expected_tokens = 1 + sum(survival ** i for i in range(1, depth + 1))
        return (verify_s + (depth - ready) * step_s) / expected_tokens
    best = min(range(ready, maximum + 1), key=lambda depth: (cost(depth), depth))
    return best if cost(best) < cost(ready) * (1 - min_gain) else ready


def observe_verification(feedback, elapsed_s, accepted, valid):
    """Censored geometric acceptance: untested tail tokens are not rejections."""
    if feedback is None:
        return
    if not 0 <= accepted <= valid or valid < 1 or elapsed_s <= 0:
        raise ValueError('Invalid verification observation')
    with feedback.get_lock():
        alpha = 1.0 if feedback[3] == 0 else .2
        feedback[0] += alpha * (elapsed_s - feedback[0])
        feedback[1] += alpha * (accepted - feedback[1])
        feedback[2] += alpha * (float(accepted < valid) - feedback[2])
        feedback[3] += 1


class PublicationCheckpoint:
    """At most window submitted steps are unfinished. Never expose live writes.

    Events are distinct per depth. A pending request drains the bounded tail
    before selecting a prefix; no communication/computation overlap is claimed.
    GPU-event timing includes gaps between steps and is a conservative estimate.
    """
    def __init__(self, minimum, maximum, window, adaptive, event_factory, feedback=None):
        self.minimum, self.maximum, self.window = minimum, maximum, window
        self.adaptive, self.feedback = adaptive, feedback
        self.events = [event_factory(enable_timing=adaptive) for _ in range(maximum + 1)]

    def begin(self, stream):
        self.stream = stream
        self.completed = 0
        self.goal = None
        self.queries = self.waits = self.decisions = self.extensions = 0
        if self.adaptive:
            self.events[0].record(stream)

    def submitted(self, depth):
        self.events[depth].record(self.stream)

    def wait(self, depth):
        self.events[depth].synchronize()
        self.waits += 1
        self.completed = max(self.completed, depth)

    def before_step(self, submitted, pending):
        while self.completed < submitted:
            self.queries += 1
            if not self.events[self.completed + 1].query():
                break
            self.completed += 1
        if pending and submitted >= self.minimum:
            # Drain only the bounded submitted tail. KV pages cannot be reused
            # until it completes; all these tokens may safely be offered.
            if self.completed < submitted:
                self.wait(submitted)
            if self.goal is None:
                self.goal = submitted
                self.decisions += 1
                if self.adaptive and self.feedback is not None:
                    with self.feedback.get_lock():
                        verify_s, successes, failures, observations = self.feedback[:4]
                    if observations >= 2:
                        step_s = self.events[0].elapsed_time(self.events[submitted]) / (1000 * submitted)
                        # Weak beta prior prevents a single perfect round from
                        # predicting guaranteed acceptance at every future depth.
                        survival = (successes + .5) / (successes + failures + 1)
                        self.goal = choose_depth(submitted, self.maximum, verify_s, step_s, survival)
                self.extensions += int(self.goal > submitted)
            if submitted >= self.goal:
                return True
        if submitted - self.completed >= self.window:
            self.wait(submitted - self.window + 1)
        return False

    def finish(self):
        if self.feedback is not None:
            with self.feedback.get_lock():
                for index, value in enumerate((self.queries, self.waits, self.decisions, self.extensions), 4):
                    self.feedback[index] += value
