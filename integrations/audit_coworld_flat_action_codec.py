"""Compare flat and factorized public Classic action masks on generated games."""

import jax
import numpy as np

from generals import GeneralsEnv
from generals.core import game
from integrations.puffer_codec import decode_action, encode_coworld_directional_observation


def main():
    env = GeneralsEnv(
        min_grid_size=18, max_grid_size=21, pad_to=21, truncation=1200,
        mountain_density_range=(0.24, 0.26), min_generals_distance=17,
        num_castles_range=(9, 11), castle_val_range=(40, 51),
        build_castles=False, deathtouch_turn=None, pool_size=16, dynamic_pool=True,
    )
    pool, _ = env.reset(jax.random.PRNGKey(733))
    cells = 21 * 21
    checked = 0
    for index in range(16):
        state = jax.tree.map(lambda field: field[index], pool)
        for side in (0, 1):
            obs = game.get_observation(state, side)
            old_values, old_mask = encode_coworld_directional_observation(obs)
            new_values, flat_mask = encode_coworld_directional_observation(obs, factorized_actions=False)
            np.testing.assert_array_equal(np.asarray(new_values), np.asarray(old_values))
            old = np.asarray(old_mask, dtype=bool)
            flat = np.asarray(flat_mask, dtype=bool)
            assert old.shape == (4 * cells + 3,) and flat.shape == (8 * cells + 1,)
            np.testing.assert_array_equal(flat[:4 * cells], old[:4 * cells])
            np.testing.assert_array_equal(flat[4 * cells:8 * cells], old[:4 * cells])
            assert old[4 * cells:].all() and flat[8 * cells]
            legal = np.flatnonzero(old[:4 * cells])
            for move in legal[:32]:
                for split in (0, 1):
                    flat_action = np.asarray(decode_action(move + split * 4 * cells, 21))
                    old_action = np.asarray(decode_action(move, 21, split))
                    np.testing.assert_array_equal(flat_action, old_action)
                    assert flat[move + split * 4 * cells]
            np.testing.assert_array_equal(
                np.asarray(decode_action(8 * cells, 21)),
                np.asarray(decode_action(4 * cells, 21, 0)),
            )
            checked += 1
    print(f"CLASSIC_FLAT_MASK_AND_DECODER_PARITY states={checked} cells={cells}", flush=True)


if __name__ == "__main__":
    main()
