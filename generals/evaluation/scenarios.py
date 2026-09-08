"""Small diagnostic suites and actual competition-board generation."""

from functools import partial

import jax
import numpy as np

from generals.core.env import GeneralsEnv
from generals.core.grid import generate_grid
from generals.modifiers.build_castles import strip_neutral_castles

from .arena import Rules


def make_suite(name, seed, count):
    if name in ("open4", "terrain4"):
        from examples._experimental.ppo.evaluate import make_boards

        boards = make_boards("open" if name == "open4" else "terrain", seed, count)
        if name == "open4" and count < len(boards):
            indices = np.random.default_rng(seed).choice(len(boards), count, replace=False)
            boards = [boards[i] for i in indices]
        return boards, Rules()
    competition = name == "competition"
    if competition:
        env = GeneralsEnv(mode="competition")
        rules = Rules(env.truncation, env.build_castles, env.deathtouch_turn)
        dims = [
            (
                int(jax.random.randint(jax.random.PRNGKey(seed + i), (), 18, 22)),
                int(jax.random.randint(jax.random.fold_in(jax.random.PRNGKey(seed + i), 1), (), 18, 22)),
            )
            for i in range(count)
        ]
    else:
        size = {"classic8": 8, "classic12": 12}[name]
        env = GeneralsEnv(
            grid_dims=(size, size),
            min_generals_distance=size - 2,
            num_castles_range=(2, 5),
            castle_val_range=(20, 41),
            truncation=800,
        )
        rules = Rules(env.truncation)
        dims = [(size, size)] * count
    boards = []

    # Compile the generator once per shape. Actual rectangles are not padded
    # for play; batching groups equal shapes in the CLI.
    @partial(jax.jit, static_argnames=("h", "w"))
    def board(key, h, w):
        grid = generate_grid(
            key,
            grid_dims=(h, w),
            pad_to=max(h, w),
            mountain_density_range=env.mountain_density_range,
            num_castles_range=env.num_castles_range,
            min_generals_distance=env.min_generals_distance,
            castle_val_range=env.castle_val_range,
        )[:h, :w]
        return strip_neutral_castles(grid) if competition else grid

    for i, (h, w) in enumerate(dims):
        key = jax.random.split(jax.random.PRNGKey(seed + i))[1]
        boards.append(np.asarray(board(key, h, w)))
    return boards, rules


def paired_cases(boards, seed, repeats=1):
    rng = np.random.default_rng(seed)
    cases = []
    for board_id, grid in enumerate(boards):
        for repeat in range(repeats):
            action_seed = int(rng.integers(0, 2**31))
            for swapped in (False, True):
                board = np.where(grid == 1, 2, np.where(grid == 2, 1, grid)) if swapped else grid
                for seat in (0, 1):
                    cases.append(
                        dict(
                            board_id=board_id,
                            repeat=repeat,
                            swapped=int(swapped),
                            seat=seat,
                            action_seed=action_seed,
                            grid=board,
                        )
                    )
    return cases
