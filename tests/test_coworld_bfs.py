"""The early-stop route search preserves the distance field on 21×21 maps."""

import jax
import jax.numpy as jnp
import numpy as np

from generals.agents.hunter_agent import _bfs


def test_early_stop_bfs_matches_full_relaxation():
    size = 21
    unreachable = jnp.int32(size * size + 5)

    def reference(passable, sources):
        def relax(_, distances):
            neighbor = jnp.minimum(
                jnp.minimum(
                    jnp.roll(distances, 1, 0).at[0].set(unreachable),
                    jnp.roll(distances, -1, 0).at[-1].set(unreachable),
                ),
                jnp.minimum(
                    jnp.roll(distances, 1, 1).at[:, 0].set(unreachable),
                    jnp.roll(distances, -1, 1).at[:, -1].set(unreachable),
                ),
            )
            return jnp.where(sources, 0, jnp.where(passable, jnp.minimum(distances, neighbor + 1), unreachable))

        return jax.lax.fori_loop(0, size * size, relax, jnp.where(sources, 0, unreachable))

    key = jax.random.PRNGKey(917)
    for index in range(8):
        key, terrain_key, source_key = jax.random.split(key, 3)
        passable = jax.random.uniform(terrain_key, (size, size)) > 0.26
        sources = jax.random.uniform(source_key, (size, size)) > (0.99 if index % 2 else 0.94)
        np.testing.assert_array_equal(np.asarray(_bfs(passable, sources)), np.asarray(reference(passable, sources)))
