from types import SimpleNamespace

import numpy as np
import pytest

from competition.agents.expander_python.agent import Agent
from integrations.classic_siege_native import ClassicSiegeBatch, compile_library


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    return ClassicSiegeBatch(compile_library(tmp_path_factory.mktemp("siege") / "opponent.so"))


def observation(grid, h, w, turn):
    return SimpleNamespace(H=h, W=w, turn=turn, type_grid=grid[0, :h, :w].tolist(),
                           owner_grid=grid[1, :h, :w].tolist(), army_grid=grid[2, :h, :w].tolist())


def test_public_scenarios_match_python_and_do_not_mutate_inputs(native):
    # Small visible battles test city staging, BFS ordering, siege memory,
    # late reinforcement and both rectangular dimensions. No hidden state.
    rng = np.random.default_rng(739)
    grids = np.zeros((48, 3, 21, 21), np.int32)
    dims = np.array([[18 + i % 4, 18 + (i // 4) % 4] for i in range(48)], np.int32)
    turns = np.array([i * 39 for i in range(48)], np.int32)
    memories = native.initial_memory(48)
    expected = []
    expected_memory = []
    for i, (h, w) in enumerate(dims):
        grid = grids[i]
        grid[0].fill(2)
        grid[0, :h, :w] = 1
        grid[1, 2:9, 2:9] = 1
        grid[2, 2:9, 2:9] = rng.integers(1, 40, size=(7, 7))
        grid[0, 3, 3] = 4
        grid[0, 7, 9] = 3
        grid[2, 7, 9] = 45
        if i % 3 == 0:
            grid[0, 8, 10] = 4
            grid[1, 8, 10] = 2
            grid[2, 8, 10] = 50
        if i % 3 == 1:
            # Hidden remembered crown is an obstacle but remains the BFS root.
            grid[0, 8, 10] = 5
            memories[i, 1] = 8 * 21 + 10
        agent = Agent(i % 2, int(h), int(w))
        if memories[i, 1] >= 0:
            agent.enemy_general = (8, 10)
        expected.append(agent.act(observation(grid, int(h), int(w), int(turns[i]))))
        expected_memory.append([-1 if p is None else p[0] * 21 + p[1]
                                for p in (agent.city, agent.enemy_general, agent.spearhead)])
    saved = [x.copy() for x in (dims, turns, grids, memories)]
    actions, after = native(dims, turns, grids, memories)
    np.testing.assert_array_equal(actions, expected)
    np.testing.assert_array_equal(after, expected_memory)
    for original, copy in zip((dims, turns, grids, memories), saved):
        np.testing.assert_array_equal(original, copy)


def test_reset_clears_memory_even_at_nonzero_turn(native):
    grids = np.zeros((2, 3, 21, 21), np.int32)
    grids[:, 0] = 1
    dims = np.full((2, 2), 21, np.int32)
    turns = np.array([900, 1500], np.int32)
    memory = native.initial_memory(2)
    actions, after = native(dims, turns, grids, memory)
    np.testing.assert_array_equal(actions, [[1, 0, 0, 0, 0]] * 2)
    np.testing.assert_array_equal(after, memory)


@pytest.mark.parametrize("bad", [-2, 441, 20 * 21])
def test_invalid_memory_rejected_before_out_of_bounds(native, bad):
    dims = np.array([[18, 18]], np.int32)
    memory = np.array([[bad, -1, -1]], np.int32)
    with pytest.raises(ValueError, match="code 3"):
        native(dims, np.array([1], np.int32), np.zeros((1, 3, 21, 21), np.int32), memory)


def test_malformed_arrays_are_rejected(native):
    with pytest.raises(ValueError, match="shapes differ"):
        native(np.array([[21, 21]], np.int32), np.array([0], np.int32),
               np.zeros((1, 3, 20, 20), np.int32), native.initial_memory(1))
    with pytest.raises(ValueError, match="int32"):
        native(np.array([[21, 21]], np.int64), np.array([0], np.int32),
               np.zeros((1, 3, 21, 21), np.int32), native.initial_memory(1))


def test_compiled_callback_and_episode_memory_reset(native):
    import jax
    import jax.numpy as jnp
    from generals.core import game
    from integrations.classic_siege_native import padded_device_actions, reset_completed_memory

    grid = jnp.zeros((21, 21), jnp.int32).at[3, 3].set(1).at[17, 17].set(2)
    state = game.create_initial_state(grid)
    # Same public position on two rows; explicitly carry memory through JIT.
    obs = jax.vmap(lambda side: game.get_observation(state, side))(jnp.array([0, 1]))
    memory = jnp.asarray(native.initial_memory(2))
    choose = jax.jit(lambda o, m: padded_device_actions(native, o, m))
    actions, after = choose(obs, memory)
    np.testing.assert_array_equal(actions, [3528, 3528])
    np.testing.assert_array_equal(after, memory)
    dirty = jnp.array([[3, 5, 9], [4, 6, 10]], jnp.int32)
    cleaned = jax.jit(reset_completed_memory)(dirty, jnp.array([True, False]))
    np.testing.assert_array_equal(cleaned, [[-1, -1, -1], [4, 6, 10]])
