"""Bounded GPU probe for the Coworld Classic map generator."""

import time

import jax
import jax.numpy as jnp
import numpy as np

from generals.core.grid import generate_grid
from generals import GeneralsEnv


def main():
    if not jax.devices("cuda"):
        raise RuntimeError("Map generator probe requires a CUDA device")
    for index, dims in enumerate(((18, 20), (21, 21))):
        started = time.monotonic()
        grid = generate_grid(
            jax.random.PRNGKey(index + 1),
            grid_dims=(21, 21), playable_dims=jnp.asarray(dims), pad_to=21,
            mountain_density_range=(0.24, 0.26), min_generals_distance=17,
        )
        board = np.asarray(grid)
        height, width = dims
        assert np.all(board[height:] == -2)
        assert np.all(board[:, width:] == -2)
        assert (board == 1).sum() == (board == 2).sum() == 1
        castles = board[board > 2]
        assert 9 <= len(castles) <= 11
        assert np.all((40 <= castles) & (castles <= 50))
        print(f"dims={dims} seconds={time.monotonic() - started:.3f} device={grid.device}", flush=True)

    for pool_size in (16, 64):
        env = GeneralsEnv(
            min_grid_size=18, max_grid_size=21, pad_to=21, truncation=1200,
            mountain_density_range=(0.24, 0.26), min_generals_distance=17,
            dynamic_pool=True, pool_size=pool_size,
        )
        started = time.monotonic()
        pool, _ = env.reset(jax.random.PRNGKey(pool_size))
        jax.block_until_ready(pool.armies)
        print(f"pool={pool_size} seconds={time.monotonic() - started:.3f} device={pool.armies.device}", flush=True)


if __name__ == "__main__":
    main()
