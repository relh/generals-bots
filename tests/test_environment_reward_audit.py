"""Exercise the exact audit wrapper that aborted scaled-reward job35851."""

from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from integrations import environment_reward_audit as audit

WEIGHTS = dict(
    reward_scale=1.0,
    shaping_weight=0.25,
    army_shaping_weight=0.5,
    land_shaping_weight=0.3,
    castle_shaping_weight=0.0,
    land_gain_reward_weight=0.0,
)


def exercise_scaled_population_audit(scale):
    # Win-only terminal intervals before scaling: loss/draw[-.2,.2], win[.8,1.2].
    # Include both endpoints, both seats, and a nonterminal with positive reward.
    rewards = jnp.array([0.8, 1.2, -0.2, 0.2, 0.8, 0.0], jnp.float32) * scale
    terminals = jnp.array([1, 1, 1, 1, 0, 1], jnp.float32)
    result = (jnp.zeros((6, 1)), jnp.ones((6, 3529)), rewards, terminals, False)

    class Population:
        _reward_options = WEIGHTS | {"reward_scale": scale}
        _terminal_reward_mode = "win_only"
        _population_labels = np.array([0, 0, 1, 1, 1, 0])
        _population_checksums = ()
        _population_script_names = ("first", "second")
        spec = SimpleNamespace(agents=6, action_sizes=(3529,))
        sides = jnp.array([0, 1, 0, 1, 0, 0], jnp.int32)

        def step_device(self, actions):
            return result

    with (
        patch.dict("os.environ", METTA_AUDIT_POPULATION_WINS="1", METTA_AUDIT_SPATIAL_SPLITS="0"),
        patch.object(audit.atexit, "register"),
    ):
        audit.install(Population)
        env = Population()
        actual = env.step_device(jnp.zeros((6, 1)))
    assert actual is result
    np.testing.assert_array_equal(env._population_audit_counts, [[2, 1, 1, 1], [1, 1, 0, 0]])
    assert env._population_win_threshold == 0.5 * scale
    assert int(env._reward_audit_counts[2]) == 5
    if scale == 0.5:
        assert int(env._reward_audit_counts[5]) == 0
    return env


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.02])
def test_scaled_wrapper_counts_wins_without_changing_transitions(scale):
    exercise_scaled_population_audit(scale)


@pytest.mark.parametrize(
    "change",
    [
        dict(reward_scale=0.0),
        dict(reward_scale=float("nan")),
        dict(shaping_weight=1.0),
        dict(land_gain_reward_weight=0.1),
    ],
)
def test_ambiguous_or_invalid_reward_contract_remains_rejected(change):
    with pytest.raises(ValueError, match="bounded win-only"):
        audit.population_win_threshold("win_only", WEIGHTS | change)
