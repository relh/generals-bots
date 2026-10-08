"""Shared trial configuration belongs to each explicit experiment instance."""
import pytest
from integrations.policy_trial import Trial, write


def test_trial_plan_is_required_and_instance_local(tmp_path):
    write(tmp_path/'assets/cold/asset.json', dict(sampler={}))
    first = Trial(tmp_path, tmp_path/'first', plan=dict(evaluation_seed=17001101))
    second = Trial(tmp_path, tmp_path/'second', plan=dict(evaluation_seed=18001101))
    assert first.plan['evaluation_seed'] == 17001101
    assert second.plan['evaluation_seed'] == 18001101
    with pytest.raises(TypeError):
        Trial(tmp_path, tmp_path/'missing-plan')
