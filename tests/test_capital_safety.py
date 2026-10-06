import numpy as np
import pytest
import jax
import jax.numpy as jnp

from generals.core import coworld_game as engine
from integrations.capital_safety import safe_actions, constrain_logits
from integrations.generate_defense_curriculum import legal_actions
from integrations.puffer_codec import encode_coworld_directional_observation, decode_action
from integrations.spatial_policy_bundle import structured_action_probabilities


def position(seat=0, time=100, capital_army=3, reserve_army=7, attacker_army=10,
             countercapture=False):
    armies = np.zeros((21, 21), np.int32)
    own = np.zeros((2, 21, 21), bool)
    for player, cell, count in ((seat, (8, 8), capital_army), (seat, (8, 7), reserve_army),
                                (1 - seat, (7, 8), attacker_army), (1 - seat, (1, 1), 5)):
        own[player, *cell] = True
        armies[cell] = count
    if countercapture:
        own[seat, 1, 2], armies[1, 2] = True, 30
    mountains = np.ones((21, 21), bool)
    mountains[:18, :18] = False
    generals = np.zeros_like(mountains)
    generals[8, 8] = generals[1, 1] = True
    positions = np.asarray(((8, 8), (1, 1)) if seat == 0 else ((1, 1), (8, 8)))
    state = engine.GameState(armies, own, ~own.any(axis=0) & ~mountains, generals,
                             np.zeros_like(mountains), mountains, ~mountains, positions,
                             np.arange(2), np.zeros(2, bool), np.int32(time), np.int32(-1), np.int32(0))
    return jax.tree.map(jnp.asarray, state)


@jax.jit
def losses(state, own, others, seat):
    def ours(action):
        def theirs(other):
            commands = jnp.stack((action, other))
            commands = jnp.where(seat == 0, commands, commands[::-1])
            return engine.step(state, commands, general_trade=False)[0].eliminated[seat]
        return jax.vmap(theirs)(others)
    return jax.vmap(ours)(own)


@pytest.mark.parametrize("seat,time", [(0, 100), (0, 101), (1, 100), (1, 101)])
def test_tied_reinforcement_survives_every_legal_opponent_move(seat, time):
    state = position(seat, time)
    obs, _, _ = legal_actions(state, seat)
    values, legal = encode_coworld_directional_observation(obs)
    selected = np.asarray(safe_actions(np.asarray(values), np)) & np.asarray(legal)
    assert selected.sum() == 1  # Full joins 6+3; half joins only 3+3 and loses.
    ours = jnp.stack([decode_action(int(i), 21) for i in np.flatnonzero(selected)])
    _, _, others = legal_actions(state, 1 - seat)
    assert not np.asarray(losses(state, ours, others, jnp.int32(seat))).any()


def test_countercapture_preserved_and_numpy_jax_probability_gradient_parity():
    state = position(countercapture=True, reserve_army=20)
    values, legal = encode_coworld_directional_observation(engine.get_observation(state, 0))
    allowed = safe_actions(np.asarray(values), np)
    np.testing.assert_array_equal(allowed, np.asarray(safe_actions(values, jnp)))
    capture = 2 * 441 + 1 * 21 + 2
    assert allowed[capture] and allowed[capture + 1764]
    selected = np.flatnonzero(allowed & np.asarray(legal))
    _, _, others = legal_actions(state, 1)
    assert not np.asarray(losses(state, jnp.stack([decode_action(int(i), 21) for i in selected]),
                                 others, jnp.int32(0))).any()
    raw = np.linspace(-1, 1, 3530, dtype=np.float32)
    actual = structured_action_probabilities(raw, legal, 1, 1, observations=values, capital_safety=True)
    from integrations.spatial_action_sampling import acting_logits
    logits = constrain_logits(acting_logits(jnp.asarray(raw), 1, 1, jnp)[:3529], values, jnp)
    expected = jax.nn.softmax(jnp.where(legal, logits, -jnp.inf))
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    target = selected[0]
    gradient = jax.grad(lambda x: -jax.nn.log_softmax(constrain_logits(x, values, jnp))[target])(jnp.asarray(raw[:3529]))
    independent = jax.grad(lambda x: -jax.nn.log_softmax(jnp.where(allowed, x, -jnp.inf))[target])(jnp.asarray(raw[:3529]))
    np.testing.assert_allclose(gradient, independent, atol=1e-7)
    assert np.isfinite(gradient).all() and np.all(np.asarray(gradient)[~allowed] == 0)
    extreme = np.where(allowed, -1e20, 1e20).astype(np.float32)
    guarded = jax.nn.softmax(constrain_logits(jnp.asarray(extreme), values, jnp))
    assert np.all(np.asarray(guarded)[~allowed] == 0)


def test_healthy_lost_and_unbounded_states_leave_distribution_unchanged():
    for state in (position(capital_army=20), position(reserve_army=2), position(attacker_army=20000)):
        values, legal = encode_coworld_directional_observation(engine.get_observation(state, 0))
        raw = np.linspace(-2, 2, 3530, dtype=np.float32)
        original = structured_action_probabilities(raw, legal, 1, 1, observations=values)
        guarded = structured_action_probabilities(raw, legal, 1, 1, observations=values, capital_safety=True)
        np.testing.assert_array_equal(original, guarded)
    integers = np.arange(16385, dtype=np.float32)
    encoded = np.log1p(integers) / 8
    np.testing.assert_array_equal(np.floor(np.expm1(encoded * 8) + .5), integers)
    np.testing.assert_array_equal(np.asarray(jnp.floor(jnp.expm1(jnp.asarray(encoded) * 8) + .5)), integers)
