from pathlib import Path
from types import SimpleNamespace
import multiprocessing as mp
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'profiling'))
import run_adaptive_publication as campaign
from ssd.engine.llm_engine import LLMEngine


def test_feedback_does_not_leak_from_warmups_or_prior_requests():
    model = LLMEngine.__new__(LLMEngine)
    model.publication_feedback = mp.get_context('spawn').Array('d', [5.] * 8)
    model.config = SimpleNamespace(max_steps=None, draft_async=False)
    model.create_inference_step = lambda cfg: None
    model.is_finished = lambda: True
    result, _ = model.generate([], [], use_tqdm=False, stream_callback=lambda *args: None)
    assert result == []
    assert model.publication_feedback[:] == [0.] * 8


def test_grid_has_strong_complete_control_and_separates_implementation_effects():
    cfgs = campaign.configurations()
    assert len(cfgs) == 8
    assert {c['k'] for c in cfgs if c['label'].startswith('complete')} == {7, 10}
    for k in (7, 10):
        assert {'bounded', 'adaptive'} <= {c['policy'] for c in cfgs if c['k'] == k}
    assert any(c['label'] == 'uniform_k10' and c['p'] == 4 for c in cfgs)


def test_mismatch_stops_successor_blocks_and_saves_first_divergence(monkeypatch, tmp_path):
    ar = dict(id='ar', protocol='fixed', round=0, config=dict(mode='ar', label='ar'))
    new = dict(id='adaptive', protocol='fixed', round=0, config=dict(mode='ssd', label='adaptive'))
    def result(folder, job, d, q):
        return dict(job=job, requests=[dict(id='prompt', input_token_ids_sha256='same',
                    output_token_ids=[1, 2 if job == ar else 3])])
    monkeypatch.setattr(campaign.base, 'block_result', result)
    with pytest.raises(ValueError, match='Token mismatch'):
        campaign.verify_completed_block(tmp_path, new, dict(jobs=[ar, new]), {})
    failures = campaign.read_json(tmp_path/'adaptive_failures.json')
    assert len(failures) == 1 and not failures[0]['exact']
    assert failures[0]['first_difference'] == 1


def test_nondefault_policy_rejected_when_publication_is_disabled(tmp_path):
    from ssd.config import Config
    with pytest.raises(ValueError, match='requires early publication'):
        Config(model=str(tmp_path), publication_policy='adaptive')
