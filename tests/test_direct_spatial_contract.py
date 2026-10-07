"""The current native hook must keep structured forward/backward algebra aligned."""

import hashlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from integrations import direct_spatial_optimization as adapter
from integrations.spatial_action_sampling import acting_logits


@pytest.mark.parametrize('channels', [11, 12])
def test_native_graph_rejects_noncanonical_observation_before_parameter_loading(channels):
    populations = [SimpleNamespace(name=name, n=count) for name, count in [
        ('input', channels * 441), ('SiLU', 441 * 32), ('ContextSiLU', 441 * 32),
        ('GlobalSiLU', 32), ('Output', 3530),
    ]]
    policy = SimpleNamespace(buffers=SimpleNamespace(fn=SimpleNamespace(
        spec=SimpleNamespace(pooled=SimpleNamespace(populations=populations)))))
    with pytest.raises(ValueError, match='sixteen public planes'):
        adapter.DirectSpatial(policy)


def test_selected_native_hook_matches_independent_action_and_value_vjp(tmp_path, monkeypatch):
    from integrations import spatial_optimizer_layout

    bridge = tmp_path / 'native.py'
    bridge.write_text('# Isolated bridge boundary for the current hook.\n')
    monkeypatch.setattr(adapter, 'BRIDGE_SHA256', hashlib.sha256(bridge.read_bytes()).hexdigest())
    monkeypatch.setattr(adapter, 'verify_configuration', lambda configuration: None)
    monkeypatch.setattr(adapter, 'DirectSpatial', lambda policy: SimpleNamespace(
        observation_size=7056,
        forward=lambda parameters, observations: parameters.reshape(1, 1, 3530),
        gradient=lambda parameters, observations, cotangents: cotangents.reshape(parameters.shape),
    ))
    monkeypatch.setattr(spatial_optimizer_layout, 'logical_optimizer_shapes',
                        lambda model, buffers, **options: ([(3530,)], []))
    for key, value in {
        'METTA_DIRECT_SPATIAL_ROLLOUT': '1', 'METTA_SPATIAL_OPTIMIZER_LAYOUT': 'logical',
        'METTA_SPATIAL_MUON_CONTEXT_MATRIX': '1', 'METTA_SPATIAL_MUON_DENSE_ORIENTATION': 'canonical',
        'METTA_SPATIAL_POLICY_TEMPERATURE': '.05', 'METTA_SPATIAL_SPLIT_TEMPERATURE': '.15',
        'METTA_SPATIAL_FULL_ACTION_TEMPERATURE': '1',
        'METTA_SPATIAL_ROUTE_HALF_WEIGHT': '0', 'METTA_SPATIAL_NEUTRAL_ROUTE_BIAS': '0',
        'METTA_SPATIAL_WEAK_OWNED_ROUTE_PENALTY': '0', 'METTA_SPATIAL_DOOMED_ATTACK_ROUTE_PENALTY': '0',
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('METTA_SPATIAL_EARLY_ROUTE_TEMPERATURE', raising=False)
    monkeypatch.delenv('METTA_SPATIAL_EARLY_ROUTE_TURNS', raising=False)

    class Native:
        def __init__(self, configuration):
            self.buffers = SimpleNamespace()
            self.updates = 0
            self.active_objectives = set()
            self.teacher_phase = SimpleNamespace(ppo_coefficient=.7)

        def _forward_arrays(self, *args):
            raise AssertionError('Retired generic forward must not execute')

        def backward_device_arrays(self, *args):
            raise AssertionError('Retired generic backward must not execute')

    adapter.install(SimpleNamespace(__file__=str(bridge), NativeFabricPolicy=Native))
    policy = Native('{}')
    parameters = jnp.linspace(-.3, .2, 3530)
    observations = jnp.zeros((1, 1, 7056))
    outputs, _, tape = policy._forward_arrays(parameters, None, observations, jnp.zeros((1, 1)), 1, 1, True)
    action_cot = jnp.linspace(-.1, .3, 3529).reshape(1, 1, 3529)
    value_cot = jnp.array([[.4]])
    expected, vjp = jax.vjp(lambda p: acting_logits(p.reshape(1, 1, 3530), .05, .15, jnp), parameters)
    np.testing.assert_allclose(outputs, expected, rtol=3e-7, atol=1e-6)
    np.testing.assert_array_equal(outputs[..., -1], parameters[-1].reshape(1, 1))
    gradient = policy.backward_device_arrays(tape, action_cot, value_cot)
    reference = vjp(jnp.concatenate((action_cot, value_cot[..., None]), axis=-1) * .7)[0]
    np.testing.assert_allclose(gradient, reference, rtol=2e-5, atol=2e-6)
    with pytest.raises(TypeError, match='canonical forward tape'):
        policy.backward_device_arrays(object(), action_cot, value_cot)
