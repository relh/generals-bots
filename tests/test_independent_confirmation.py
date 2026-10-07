"""Confirmation is unavailable without sealed positive development evidence."""
import json
import pytest
from integrations import independent_confirmation as confirmation


def test_pending_confirmation_fails_before_output(tmp_path):
    with pytest.raises(ValueError, match='awaits'):
        confirmation.evaluate(tmp_path, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_empty_binding_set_is_rejected(tmp_path, monkeypatch):
    plan = confirmation.read(confirmation.PLAN_PATH)
    plan.update(status='sealed after positive independent development audit', bindings={})
    path = tmp_path / 'plan.json'; path.write_text(json.dumps(plan))
    monkeypatch.setattr(confirmation, 'PLAN_PATH', path)
    with pytest.raises(ValueError, match='bindings'):
        confirmation.plan_for(tmp_path)


def test_unbound_candidate_is_rejected(tmp_path, monkeypatch):
    plan = confirmation.read(confirmation.PLAN_PATH)
    plan['status'] = 'sealed after positive independent development audit'
    for name in plan['bindings']:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}')
        plan['bindings'][name] = confirmation.digest(path)
    path = tmp_path / 'plan.json'; path.write_text(json.dumps(plan))
    monkeypatch.setattr(confirmation, 'PLAN_PATH', path)
    with pytest.raises(ValueError, match='candidate identity'):
        confirmation.plan_for(tmp_path)


def test_comparison_gate_requires_both_positive_and_strata_guard():
    report = dict(initial_state_cluster_ci95=[.01, .05], by_opponent_and_seat={
        'opponent': {'0': dict(games=100, paired_signed_score_delta=.01)}})
    assert confirmation.selected([report, report])
    assert not confirmation.selected([report])
    bad = dict(report, initial_state_cluster_ci95=[0, .05])
    assert not confirmation.selected([report, bad])
    bad = dict(report, by_opponent_and_seat={'opponent': {'0': dict(games=100, paired_signed_score_delta=-.11)}})
    assert not confirmation.selected([report, bad])
