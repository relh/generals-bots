import numpy as np
import pytest

from integrations.classic_siege_native import ClassicSiegeBatch, compile_library


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return compile_library(tmp_path_factory.mktemp("siege") / "opponent.so")


@pytest.fixture(scope="module", params=[1, 2, 4, 8])
def native(library, request):
    return ClassicSiegeBatch(library, workers=request.param)


@pytest.mark.parametrize("workers", [0, -1, 9, True, 1.5, "4"])
def test_workers_are_explicit_and_bounded(workers):
    with pytest.raises(ValueError, match="workers"):
        ClassicSiegeBatch("unused.so", workers=workers)


def test_public_tactical_choices_and_immutable_inputs(native):
    grids = np.zeros((3, 3, 21, 21), np.int32)
    grids[:, 0] = 2  # padded cells are impassable
    grids[:, 0, :18, :19] = 1
    grids[:, 0, 3, 3] = 4
    grids[:, 1, 3, 3] = 1
    grids[:, 2, 3, 3] = [20, 20, 3]
    grids[:, 1, 3, 4] = 2
    grids[:, 2, 3, 4] = [2, 8, 10]
    grids[1, 0, 3, 4] = 4  # a capital must receive the full winning stack
    grids[2, 1, 3, 2] = 1
    grids[2, 2, 3, 2] = 18  # gather toward stronger enemy contact before turn 800
    for row, col in ((2, 2), (4, 2), (3, 1), (2, 3), (4, 3)):
        grids[2, 0, row, col] = 2
    dims = np.full((3, 2), [18, 19], np.int32)
    turns = np.full(3, 100, np.int32)
    memory = native.initial_memory(3)
    saved = [x.copy() for x in (dims, turns, grids, memory)]
    actions, after = native(dims, turns, grids, memory)
    np.testing.assert_array_equal(actions, [[0, 3, 3, 3, 1],
                                           [0, 3, 3, 3, 0],
                                           [0, 3, 2, 3, 0]])
    np.testing.assert_array_equal(after[1, 1], 3 * 21 + 4)
    for original, copy in zip(saved, (dims, turns, grids, memory)):
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


@pytest.mark.parametrize("count", [0, 1, 13])
def test_parallel_batch_boundaries_preserve_serial_results(native, library, count):
    dims = np.full((count, 2), 21, np.int32)
    turns = np.full(count, 1000, np.int32)
    grids = np.zeros((count, 3, 21, 21), np.int32)
    grids[:, 0] = 1
    grids[:, 1, 3:9, 3:9] = 1
    grids[:, 2, 3:9, 3:9] = 20
    memory = native.initial_memory(count)
    expected = ClassicSiegeBatch(library)(dims, turns, grids, memory)
    actual = native(dims, turns, grids, memory)
    for before, after in zip(expected, actual):
        np.testing.assert_array_equal(before, after)


def test_rejected_later_chunk_finishes_without_mutating_inputs(native):
    dims = np.full((13, 2), 21, np.int32)
    turns = np.zeros(13, np.int32)
    grids = np.zeros((13, 3, 21, 21), np.int32)
    memory = native.initial_memory(13)
    memory[-1, -1] = 441
    before = memory.copy()
    with pytest.raises(ValueError, match="code 3"):
        native(dims, turns, grids, memory)
    np.testing.assert_array_equal(memory, before)


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
