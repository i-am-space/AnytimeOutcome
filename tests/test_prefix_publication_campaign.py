from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'profiling'))
import run_prefix_publication as gate
from baseline_common import file_digest


def receipt(tmp_path, shared, foreign):
    output = tmp_path/'result.json'; output.write_text('{}')
    value = dict(definition_sha256='definition', returncode=0, samples=4, error=None,
        output_sha256=file_digest(output), shared_correctness=shared,
        performance_evidence=False, external_processes=foreign)
    return output, value


def test_shared_jobs_can_count_only_in_explicit_correctness_mode(tmp_path):
    output, value = receipt(tmp_path, True, [dict(pid=123)])
    gate.validate_receipt(value, dict(allow_shared_correctness=True), dict(sha256='definition'), output)
    with pytest.raises(ValueError, match='contaminated'):
        gate.validate_receipt(value, dict(allow_shared_correctness=False), dict(sha256='definition'), output)
    value['shared_correctness'] = False
    with pytest.raises(ValueError, match='contaminated'):
        gate.validate_receipt(value, dict(allow_shared_correctness=False), dict(sha256='definition'), output)


def test_neither_shared_nor_dedicated_diagnostics_can_claim_performance_evidence(tmp_path):
    output, value = receipt(tmp_path, False, [])
    value['performance_evidence'] = True
    with pytest.raises(ValueError):
        gate.validate_receipt(value, dict(allow_shared_correctness=False), dict(sha256='definition'), output)


def test_incomplete_and_changed_results_cannot_count(tmp_path):
    output, value = receipt(tmp_path, False, [])
    for field, invalid in [('samples', 0), ('returncode', -15), ('error', 'interrupted'),
                           ('definition_sha256', 'other'), ('output_sha256', 'changed')]:
        changed = dict(value, **{field: invalid})
        with pytest.raises(ValueError):
            gate.validate_receipt(changed, dict(allow_shared_correctness=False), dict(sha256='definition'), output)


def test_grid_compares_strong_complete_baseline_and_separates_drain_from_overlap():
    cfgs = gate.configurations()
    assert len(cfgs) == 9
    for k in (7, 10):
        assert {c['label'] for c in cfgs if c['k'] == k} >= {f'{p}_k{k}' for p in ('complete', 'uniform', 'bounded', 'overlap')}
    stress = gate.configurations(True)
    assert len(stress) == 2 and {c['label'] for c in stress} == {'ar', 'overlap_k10'}


def test_importing_gate_does_not_install_artificial_delay_or_change_previous_runtime():
    from ssd.engine.draft_runner import DraftRunner
    assert DraftRunner._decode_tree_step.__name__ == '_decode_tree_step'
    assert gate.base.code_files.__module__ == 'run_overnight_extensions'
