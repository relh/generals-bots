"""Semantic checks for the reusable PPO foundation."""
import pytest

pytest.importorskip("equinox")
pytest.importorskip("optax")
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from generals.core import game
from generals.modifiers.build_castles import build_cost_grid
from generals.training.checkpoint import load_checkpoint, save_checkpoint
from generals.training.network import SpatialPolicy, action_mask, build_costs, decode_action, encode_action, features
from generals.training.ppo import Batch, compute_gae, update
from generals.training.rewards import shaped_reward


def observation(shape=(4, 4)):
    grid = jnp.zeros(shape, jnp.int32).at[0, 0].set(1).at[-1, -1].set(2)
    state = game.create_initial_state(grid)
    return state, game.get_observation(state, 0)


def test_terminal_reward_comes_from_winner_after_general_becomes_castle():
    state, _ = observation()
    state = state._replace(armies=state.armies.at[3, 2].set(20),
                           ownership=state.ownership.at[0, 3, 2].set(True),
                           ownership_neutral=state.ownership_neutral.at[3, 2].set(False))
    previous = game.get_observation(state, 0)
    final, info = game.step(state, jnp.array([[0, 3, 2, 3, 0], [1, 0, 0, 0, 0]]))
    assert bool(info.is_done)
    assert not bool(final.generals[3, 3])
    for side, expected in ((0, 1), (1, -1)):
        prior = game.get_observation(state, side)
        reward, components = shaped_reward(prior, game.get_observation(final, side), info.winner,
                                            side, info.is_done, weight=0)
        assert float(reward) == expected
        assert float(components[0]) == expected


def test_gae_bootstraps_timeout_but_stops_trace_at_reset():
    # First transition times out; second is an unrelated terminal episode.
    values = jnp.array([[2.0], [100.0]])
    result = compute_gae(jnp.array([[1.0], [3.0]]), values, jnp.array([[10.0], [900.0]]),
                         jnp.array([[False], [True]]), jnp.array([[True], [False]]), gamma=0.9, lam=1)
    np.testing.assert_allclose(result[:, 0], [8, -97])


def test_discounted_shaping_telescopes_through_terminal():
    state, first = observation()
    richer = first._replace(owned_army_count=jnp.array(200), owned_land_count=jnp.array(8))
    _, parts1 = shaped_reward(first, richer, -1, 0, False, gamma=0.9, weight=0.2)
    _, parts2 = shaped_reward(richer, richer, 0, 0, True, gamma=0.9, weight=0.2)
    assert float(parts2[3]) == 0
    np.testing.assert_allclose(parts1[1] + 0.9 * parts2[1], -0.2 * parts1[2], atol=1e-7)


def test_task_time_limit_draw_has_zero_outcome_and_zero_bootstrap():
    _, obs = observation()
    obs = obs._replace(owned_army_count=jnp.array(100))
    reward, components = shaped_reward(obs, obs, -1, 0, True, weight=0.2)
    assert float(components[0]) == 0
    assert float(components[3]) == 0
    advantages = compute_gae(reward[None, None], jnp.array([[2.0]]), jnp.array([[100.0]]),
                             jnp.array([[True]]), jnp.array([[True]]))
    np.testing.assert_allclose(advantages, [[float(reward) - 2]], atol=1e-6)


def test_canonical_pass_rectangles_and_build_actions():
    _, obs = observation((4, 6))
    shape = obs.armies.shape
    indices = jnp.arange(9 * 24 + 1)
    decoded = jax.vmap(lambda i: decode_action(i, shape))(indices)
    restored = jax.vmap(lambda a: encode_action(a, shape))(decoded)
    np.testing.assert_array_equal(restored, indices)
    assert int((decoded[:, 0] == 1).sum()) == 1
    assert int((decoded[:, 0] == 2).sum()) == 24
    mask = action_mask(obs)
    assert mask.shape == (217,)
    assert int(mask.sum()) == 1  # all own initial stacks are one


def test_build_mask_matches_engine_price_without_hidden_state():
    state, _ = observation((8, 8))
    state = state._replace(ownership=state.ownership.at[0, 1, 0].set(True),
                           armies=state.armies.at[1, 0].set(47))
    obs = game.get_observation(state, 0)
    np.testing.assert_array_equal(build_costs(obs), build_cost_grid(state, 0))
    mask = action_mask(obs, True)
    assert bool(mask[8 * 64 + 8])
    poor = obs._replace(armies=obs.armies.at[1, 0].set(46))
    assert not bool(action_mask(poor, True)[8 * 64 + 8])


def test_same_network_runs_different_board_shapes():
    network = SpatialPolicy(jax.random.PRNGKey(0), width=8)
    for shape in ((4, 4), (8, 6), (18, 21)):
        _, obs = observation(shape)
        logits, value = network(features(obs), action_mask(obs))
        assert logits.shape == (9 * shape[0] * shape[1] + 1,)
        assert bool(jnp.isfinite(value))
        np.testing.assert_array_equal(network.act(obs, jax.random.PRNGKey(1)), [1, 0, 0, 0, 0])


def test_checkpoint_restores_exact_next_optimizer_update(tmp_path):
    network = SpatialPolicy(jax.random.PRNGKey(0), width=8)
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(0.001))
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    state, obs = observation()
    inputs, mask = features(obs), action_mask(obs)
    batch = Batch(inputs[None], mask[None], jnp.array([144]), jnp.array([0.0]), jnp.array([1.0]), jnp.array([1.0]))
    network, opt_state, _ = update(network, opt_state, batch, optimizer)
    key = jax.random.PRNGKey(9)
    snapshot = dict(network=network, optimizer=opt_state, rng=key, states=state,
                    pool=jax.tree.map(lambda x: x[None], state), config={"seed": 9}, iteration=1)
    path = tmp_path / "checkpoint.pkl"
    save_checkpoint(path, snapshot)
    restored = load_checkpoint(path)
    expected = update(network, opt_state, batch, optimizer)
    actual = update(restored["network"], restored["optimizer"], batch, optimizer)
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(jax.random.split(restored["rng"]), jax.random.split(key))


def test_full_training_resume_matches_uninterrupted_run(tmp_path):
    from dataclasses import replace
    from generals.training.train import Config, run

    config = Config(num_envs=2, steps=4, iterations=2, pool_size=2, board_size=6,
                    opponents="random", width=4, minibatch_size=8, epochs=1,
                    save_every=1, truncation=5, build_castles=True)
    uninterrupted = run(config, tmp_path / "whole")
    run(replace(config, iterations=1), tmp_path / "split")
    resumed = run(config, tmp_path / "split", tmp_path / "split" / "checkpoint.pkl")
    for a, b in zip(jax.tree.leaves(uninterrupted), jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(a, b)


def test_actual_mutual_deathtouch_rollout_counts_terminal_draw():
    from generals import GeneralsEnv
    from generals.training.train import Config, make_rollout, outcome_counts

    class FixedPolicy(eqx.Module):
        def __call__(self, inputs, mask):
            # Full UP from (1, 2): channel0 * 6 + position5.
            logits = jnp.full(mask.shape, -100.0).at[5].set(100.0)
            return jnp.where(mask, logits, -1e9), jnp.array(0.0)

    class FixedOpponent:
        def act(self, obs, key):
            return jnp.array([0, 1, 0, 0, 0])

    state = game.create_initial_state(jnp.array([[1, 0, 2], [0, 0, 0]], jnp.int32))
    state = state._replace(armies=state.armies.at[1, 2].set(10).at[1, 0].set(10),
                          ownership=state.ownership.at[0, 1, 2].set(True).at[1, 1, 0].set(True),
                          ownership_neutral=state.ownership_neutral.at[1, 2].set(False).at[1, 0].set(False))
    states = jax.tree.map(lambda x: x[None], state)
    env = GeneralsEnv(grid_dims=(2, 3), pool_size=1, deathtouch_turn=0)
    config = Config(num_envs=1, steps=1)
    _, data = make_rollout(env, [FixedOpponent()], config)(
        FixedPolicy(), states, states, jnp.array([0]), jnp.array([0]), jnp.zeros(1), jax.random.PRNGKey(0))
    assert bool(data["terminated"][0, 0])
    assert not bool(data["truncated"][0, 0])
    assert bool(data["bootstrap_terminal"][0, 0])
    assert float(data["outcome"][0, 0]) == 0
    assert outcome_counts(data["terminated"], data["truncated"], data["outcome"]) == dict(
        completed_episodes=1, wins=0, losses=0, draws=1, timeouts=0, terminal_draws=1)


def test_sentinel_training_opponent_uses_environment_horizon():
    from generals import GeneralsEnv
    from generals.training.train import Config, opponents_for

    opponent = opponents_for(Config(opponents="sentinel"),
                             GeneralsEnv(grid_dims=(8, 8), truncation=500, build_castles=True))[0]
    assert opponent.max_turns == 500
    assert opponent.build_castles
