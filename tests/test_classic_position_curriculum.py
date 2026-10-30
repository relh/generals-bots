"""Curriculum data identity, live-state checks, reset behavior, and held-out isolation."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from integrations.classic_position_curriculum import load_positions, mix_initial_positions


def archive(tmp_path, **changes):
    own = np.zeros((2, 2, 21, 21), bool)
    own[:, 0, 1, 1] = True
    own[:, 1, 19, 19] = True
    generals = own.any(axis=1)
    fields = dict(
        armies=generals.astype(np.int32) * 20,
        ownership=own,
        ownership_neutral=~own.any(axis=1),
        generals=generals,
        castles=np.zeros_like(generals),
        mountains=np.zeros_like(generals),
        passable=np.ones_like(generals),
        general_positions=np.array([[[1, 1], [19, 19]]] * 2, np.int32),
        teams=np.array([[0, 1]] * 2, np.int32),
        eliminated=np.zeros((2, 2), bool),
        time=np.array([100, 150], np.int32),
        winner=np.full(2, -1, np.int32),
        pool_idx=np.array([7, 11], np.int32),
    )
    fields.update(changes)
    path = tmp_path / "positions.npz"
    np.savez(path, **fields)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_checksum_and_terminal_rejection(tmp_path):
    path, digest = archive(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        load_positions(path, "0" * 64)
    positions = load_positions(path, digest)
    np.testing.assert_array_equal(positions.pool_idx, [0, 0])
    for changes in (
        {"winner": np.array([0, -1], np.int32)},
        {"time": np.array([100, 2000], np.int32)},
        {"time": np.array([100, 150], np.int64)},
    ):
        path, digest = archive(tmp_path, **changes)
        with pytest.raises(ValueError):
            load_positions(path, digest)


def test_mixed_reset_is_deterministic_and_keeps_fresh_maps(tmp_path):
    positions = load_positions(*archive(tmp_path))
    pool = positions._replace(time=jnp.zeros(2, jnp.int32))

    def fresh(pool, key):
        index = jax.random.randint(key, (), 0, 2)
        return jax.tree.map(lambda x: x[index], pool)

    initial = jax.jit(jax.vmap(mix_initial_positions(fresh, positions, 0.25), in_axes=(None, 0)))
    keys = jax.random.split(jax.random.PRNGKey(83), 64)
    first, second = initial(pool, keys), initial(pool, keys)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    times = np.asarray(first.time)
    assert np.isin(times, [0, 100, 150]).all()
    assert 4 < (times > 0).sum() < 40


def test_selfplay_recycles_into_valid_public_curriculum_observations(tmp_path):
    pytest.importorskip("metta_training", reason="Optional private training adapter; exercised with pinned framework")
    from metta_training.environment import EnvironmentContext

    from integrations.metta_puffer import BatchedGeneralsSelfPlayPufferEnvironment

    path, digest = archive(tmp_path)
    env = BatchedGeneralsSelfPlayPufferEnvironment(
        context=EnvironmentContext(seed=1, index=0, mode="train", output=tmp_path),
        parallel_games=4,
        require_gpu=False,
        coworld_classic=True,
        coworld_pool_size=16,
        balance_opponent_sides=True,
        coworld_position_pool=str(path),
        coworld_position_pool_sha256=digest,
        coworld_position_probability=1.0,
        directional_features=True,
        compact_features=True,
        lean_features=True,
        public_scalar_features=True,
        terminal_reward_mode="win_only",
        shaping_weight=0.0,
    )
    try:
        env.reset_device("curriculum-reset-proof")
        assert np.isin(np.asarray(env.states.time), [100, 150]).all()
        env.states = env.states._replace(time=jnp.full(4, 1999, jnp.int32))
        values, masks, rewards, done, _ = env.step_device(jnp.full((8, 1), 3528, jnp.int32))
        assert np.asarray(done).all() and not np.asarray(rewards).any()
        assert np.isin(np.asarray(env.states.time), [100, 150]).all()
        expected_values, expected_masks = env._observe_both(env.states)
        np.testing.assert_array_equal(values, np.asarray(expected_values).reshape(8, -1))
        np.testing.assert_array_equal(masks, np.asarray(expected_masks).reshape(8, -1))
    finally:
        env.close()


def test_population_evaluation_disables_training_start_pool(tmp_path):
    pytest.importorskip("metta_training", reason="Optional private training adapter; exercised with pinned framework")
    from integrations import evaluate_spatial_population as evaluation

    build = tmp_path / "build.json"
    build.write_text(
        json.dumps(
            dict(
                config=dict(
                    python_environment=dict(
                        factory="integrations.spatial_selfplay:SpatialPopulationOpponentPufferEnvironment",
                        options=dict(
                            coworld_classic=True,
                            terminal_reward_mode="win_only",
                            coworld_position_probability=0.25,
                            coworld_position_pool="private-training-only.npz",
                        ),
                    )
                )
            )
        )
    )

    class Captured(Exception):
        pass

    def construct(**options):
        assert options["coworld_position_probability"] == 0
        raise Captured

    argv = [
        "evaluate",
        "--bundle",
        "unused",
        "--population-build",
        str(build),
        "--output",
        str(tmp_path / "eval"),
        "--seed",
        "3",
        "--sample-seed",
        "4",
    ]
    with (
        patch("sys.argv", argv),
        patch.object(evaluation.jax, "devices", return_value=[SimpleNamespace(platform="gpu")]),
        patch.object(evaluation, "SpatialPlayerPolicy", return_value=SimpleNamespace(action_mode="structured_sample")),
        patch.object(evaluation, "SpatialPopulationOpponentPufferEnvironment", side_effect=construct),
    ):
        with pytest.raises(Captured):
            evaluation.main()


@pytest.mark.parametrize("height,width", [(18, 21), (21, 18), (19, 20)])
def test_hosted_padding_preserves_classic_transition_and_valid_archive(tmp_path, height, width):
    from generals.core import coworld_game, game
    from integrations.classic_position_curriculum import pad_position

    grid = jnp.zeros((height, width), jnp.int32).at[1, 1].set(1).at[height - 2, width - 2].set(2)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies * 20, time=jnp.int32(100))
    padded = pad_position(state)
    path = tmp_path / "padded.npz"
    np.savez(path, **{k: np.asarray(v)[None] for k, v in padded._asdict().items()})
    load_positions(path, hashlib.sha256(path.read_bytes()).hexdigest())
    actions = jnp.array([[0, 1, 1, 3, 0], [0, height - 2, width - 2, 2, 0]], jnp.int32)
    advance = jax.jit(lambda s: coworld_game.step(s, actions, general_trade=False))
    normal, normal_info = advance(state)
    expanded, expanded_info = advance(jax.tree.map(jnp.asarray, padded))
    grids = {"armies", "ownership", "ownership_neutral", "generals", "castles", "mountains", "passable"}
    for key in state._fields:
        actual = np.asarray(getattr(expanded, key))
        if key in grids:
            actual = actual[..., :height, :width]
        np.testing.assert_array_equal(actual, np.asarray(getattr(normal, key)))
    for key in ("army", "land", "winner", "is_done"):
        np.testing.assert_array_equal(getattr(normal_info, key), getattr(expanded_info, key))
