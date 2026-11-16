"""Greedy acceptance must match a scalar oracle, independently of draft/cache policy."""
import itertools
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party/ssd'))
from ssd.utils.verify import verify


@pytest.mark.parametrize('k', [1, 6, 7])
@pytest.mark.parametrize('cache_hit,jit', list(itertools.product([False, True], repeat=2)))
def test_every_rejection_position_and_bonus(k, cache_hit, jit):
    # Each batch row rejects at a different position; the last accepts all K.
    predictions = torch.arange(k + 1).remainder(4).expand(k + 1, -1)
    logits = torch.full((k + 1, k + 1, 5), -10.0)
    logits.scatter_(2, predictions.unsqueeze(-1), 10.0)
    drafts = predictions[:, :k].clone()
    for rejection in range(k):
        drafts[rejection, rejection] = (drafts[rejection, rejection] + 1) % 5
    starts = torch.full((k + 1, 1), 4)
    speculations = torch.cat([starts, drafts], dim=1)
    suffixes, recovery = verify(logits, torch.zeros(k+1, k, 5), speculations,
        torch.zeros(k+1), torch.zeros(k+1),
        cache_hits=torch.full((k+1,), cache_hit), jit_speculate=jit)
    for row in range(k + 1):
        assert suffixes[row] == [4] + drafts[row, :row].tolist()
        assert recovery[row] == predictions[row, row].item()


def test_exact_tie_uses_same_argmax_rule_as_target():
    logits = torch.tensor([[[0., 2., 2.], [1., 1., 0.]]])
    suffixes, recovery = verify(logits, torch.zeros(1, 1, 3),
        torch.tensor([[2, 1]]), torch.zeros(1), torch.zeros(1))
    assert suffixes == [[2, 1]]
    assert recovery == [0]
