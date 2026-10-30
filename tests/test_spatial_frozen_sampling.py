"""Frozen training opponents honor the same action mode as their serving bundle."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from integrations.spatial_policy_bundle import structured_action_probabilities
from integrations.spatial_frozen_sampling import frozen_action_indices


@pytest.mark.parametrize("workers", [1, 4, 8])
def test_single_frozen_match_accepts_population_worker_metadata(tmp_path, monkeypatch, workers):
    pytest.importorskip("metta_training")
    from integrations import spatial_selfplay as module

    (tmp_path / "build.json").write_text(json.dumps({
        "config": {"python_environment": {"options": {}}},
    }))
    monkeypatch.setattr(module, "SpatialPlayerPolicy", lambda bundle: SimpleNamespace())
    reached = []

    class BaseReached(Exception):
        pass

    # Explicit base signature rejects population-only keywords, as the real
    # Generals adapter did when a four-worker checkpoint entered its self-match.
    def base_init(self, *, context, parallel_games, balance_opponent_sides, shaping_gamma):
        reached.append((context, parallel_games, balance_opponent_sides, shaping_gamma))
        raise BaseReached

    monkeypatch.setattr(module.BatchedGeneralsSelfPlayPufferEnvironment, "__init__", base_init)
    context = object()
    with pytest.raises(BaseReached):
        module.SpatialFrozenOpponentPufferEnvironment(
            frozen_bundle=tmp_path, context=context, parallel_games=16,
            balance_opponent_sides=True, shaping_gamma=.999,
            opponent_weights=[8, 6, 6], scripted_opponents=["classic_siege_padded"],
            classic_siege_workers=workers,
        )
    assert reached == [(context, 16, True, .999)]


def test_frozen_sampled_opponent_matches_serving_distribution():
    count = 20_000
    outputs = np.zeros((3530,), np.float32)
    outputs[0], outputs[1764], outputs[3528] = .15, 0, .1
    legal = np.zeros((3529,), bool)
    legal[[0, 1764, 3528]] = True
    expected = structured_action_probabilities(outputs, legal, .05, .15)
    policy = SimpleNamespace(action_mode="structured_sample", move_temperature=.05,
                             split_temperature=.15)
    keys = jax.random.split(jax.random.PRNGKey(123), count)
    choose = jax.jit(lambda k: frozen_action_indices(
        policy, jnp.broadcast_to(outputs, (count, 3530)),
        jnp.broadcast_to(legal, (count, 3529)), k
    ))
    sampled = np.asarray(choose(keys))
    assert np.isin(sampled, [0, 1764, 3528]).all()
    np.testing.assert_array_equal(sampled, np.asarray(choose(keys)))
    for action in (0, 1764, 3528):
        assert abs(np.mean(sampled == action) - expected[action]) < .02


def test_frozen_half_weight_matches_serving_route_distribution():
    count = 12_000
    outputs = np.zeros(3530, np.float32)
    outputs[[0, 1, 1764, 1765]] = [0, -.2, -.3, -.1]
    legal = np.zeros(3529, bool)
    legal[[0, 1, 1764, 1765]] = True
    expected = structured_action_probabilities(outputs, legal, .05, .15, route_half_weight=.25)
    policy = SimpleNamespace(action_mode="structured_sample", move_temperature=.05,
                             split_temperature=.15, route_half_weight=.25)
    keys = jax.random.split(jax.random.PRNGKey(291), count)
    sampled = np.asarray(jax.jit(lambda k: frozen_action_indices(
        policy, jnp.broadcast_to(outputs, (count, 3530)),
        jnp.broadcast_to(legal, (count, 3529)), k,
    ))(keys))
    for action in (0, 1, 1764, 1765):
        assert abs(np.mean(sampled == action) - expected[action]) < .02


def test_frozen_opening_schedule_matches_serving_temperature_by_public_turn():
    count = 8000
    outputs = np.zeros(3530, np.float32)
    outputs[0], outputs[1] = .1, 0
    legal = np.zeros(3529, bool)
    legal[[0, 1]] = True
    observations = np.zeros((2 * count, 16 * 441), np.float32)
    observations[:count, 11 * 441] = 99 / 2000
    observations[count:, 11 * 441] = 100 / 2000
    policy = SimpleNamespace(action_mode="structured_sample", move_temperature=.05,
                             split_temperature=.15, early_route_temperature=.1, early_route_turns=100)
    keys = jax.random.split(jax.random.PRNGKey(118), 2 * count)
    sampled = np.asarray(jax.jit(lambda k, o: frozen_action_indices(
        policy, jnp.broadcast_to(outputs, (2 * count, 3530)),
        jnp.broadcast_to(legal, (2 * count, 3529)), k, o,
    ))(keys, jnp.asarray(observations)))
    expected_early = structured_action_probabilities(outputs, legal, .1, .15)[0]
    expected_late = structured_action_probabilities(outputs, legal, .05, .15)[0]
    assert abs(np.mean(sampled[:count] == 0) - expected_early) < .02
    assert abs(np.mean(sampled[count:] == 0) - expected_late) < .02
    with pytest.raises(ValueError, match="public observations"):
        frozen_action_indices(policy, jnp.asarray(outputs[None]), jnp.asarray(legal[None]), keys[:1])


def test_frozen_argmax_opponent_keeps_legacy_selection():
    outputs = np.zeros((2, 3530), np.float32)
    outputs[:, 0], outputs[:, 1764], outputs[:, 3528] = .15, 0, .1
    legal = np.zeros((2, 3529), bool)
    legal[:, [1764, 3528]] = True
    chosen = frozen_action_indices(SimpleNamespace(action_mode="argmax"),
                                   jnp.asarray(outputs), jnp.asarray(legal),
                                   jax.random.split(jax.random.PRNGKey(7), 2))
    np.testing.assert_array_equal(np.asarray(chosen), [3528, 3528])


def test_frozen_neutral_route_bonus_matches_serving_distribution():
    count = 10_000
    source = 10 * 21 + 10
    up, right = source, 3 * 441 + source
    outputs = np.zeros(3530, np.float32)
    legal = np.zeros(3529, bool)
    legal[[up, right]] = True
    public = np.zeros(16 * 441, np.float32)
    public[6 * 441 + source + 1] = 1
    expected = structured_action_probabilities(outputs, legal, .05, .15,
                                               observations=public, neutral_route_bias=3.0)
    policy = SimpleNamespace(action_mode="structured_sample", move_temperature=.05,
                             split_temperature=.15, neutral_route_bias=3.0)
    keys = jax.random.split(jax.random.PRNGKey(109), count)
    sampled = np.asarray(jax.jit(lambda k: frozen_action_indices(
        policy, jnp.broadcast_to(outputs, (count, 3530)),
        jnp.broadcast_to(legal, (count, 3529)), k,
        jnp.broadcast_to(public, (count, public.size)),
    ))(keys))
    assert abs(np.mean(sampled == up) - expected[up]) < .02
    with pytest.raises(ValueError, match="public observations"):
        frozen_action_indices(policy, jnp.asarray(outputs[None]), jnp.asarray(legal[None]),
                              keys[:1])


def test_frozen_weak_owned_route_penalty_matches_serving_distribution():
    count = 10_000
    source = 10 * 21 + 10
    up, right = source, 3 * 441 + source
    outputs = np.zeros(3530, np.float32)
    legal = np.zeros(3529, bool)
    legal[[up, right]] = True
    public = np.zeros(16 * 441, np.float32)
    public[source] = np.log1p(4) / 8
    public[4 * 441 + source] = 1
    public[4 * 441 + source - 21] = 1
    expected = structured_action_probabilities(outputs, legal, .05, .15,
                                               observations=public, weak_owned_route_penalty=4.0)
    policy = SimpleNamespace(action_mode="structured_sample", move_temperature=.05,
                             split_temperature=.15, neutral_route_bias=0.0,
                             weak_owned_route_penalty=4.0)
    keys = jax.random.split(jax.random.PRNGKey(110), count)
    sampled = np.asarray(jax.jit(lambda k: frozen_action_indices(
        policy, jnp.broadcast_to(outputs, (count, 3530)),
        jnp.broadcast_to(legal, (count, 3529)), k,
        jnp.broadcast_to(public, (count, public.size)),
    ))(keys))
    assert abs(np.mean(sampled == up) - expected[up]) < .02


def test_frozen_doomed_attack_route_penalty_matches_serving_distribution():
    count = 10_000
    source = 10 * 21 + 10
    up, right = source, 3 * 441 + source
    outputs = np.zeros(3530, np.float32)
    legal = np.zeros(3529, bool)
    legal[[up, right]] = True
    public = np.zeros(16 * 441, np.float32)
    public[source] = np.log1p(5) / 8
    public[source - 21] = np.log1p(4) / 8
    public[5 * 441 + source - 21] = 1
    expected = structured_action_probabilities(outputs, legal, .05, .15,
                                               observations=public, doomed_attack_route_penalty=4.0)
    policy = SimpleNamespace(action_mode="structured_sample", move_temperature=.05,
                             split_temperature=.15, neutral_route_bias=0.0,
                             doomed_attack_route_penalty=4.0)
    keys = jax.random.split(jax.random.PRNGKey(111), count)
    sampled = np.asarray(jax.jit(lambda k: frozen_action_indices(
        policy, jnp.broadcast_to(outputs, (count, 3530)),
        jnp.broadcast_to(legal, (count, 3529)), k,
        jnp.broadcast_to(public, (count, public.size)),
    ))(keys))
    assert abs(np.mean(sampled == up) - expected[up]) < .02
