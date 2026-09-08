"""Sentinel's language-neutral competition protocol adapter.

Default rules are competition. Set SENTINEL_MODE=classic for ordinary matches.
Only protocol actions are written to stdout.
"""
import os
import sys

import jax
import jax.numpy as jnp

from generals.agents.sentinel_agent import SentinelAgent
from generals.core.observation import Observation


def read_observation(stream, height, width):
    header = stream.readline()
    if not header:
        return None
    turn, my_land, my_army, opp_land, opp_army = map(int, header.split())
    grids = []
    for _ in range(3):
        rows = []
        for _ in range(height):
            line = stream.readline()
            if not line:
                raise EOFError('incomplete observation frame')
            row = list(map(int, line.split()))
            if len(row) != width:
                raise ValueError('observation row has incorrect width')
            rows.append(row)
        grids.append(jnp.asarray(rows, dtype=jnp.int32))
    types, owners, armies = grids
    visible = (types != 0) & (types != 5)
    return Observation(
        armies=armies, generals=types == 4, castles=types == 3, mountains=types == 2,
        owned_cells=owners == 1, opponent_cells=owners == 2,
        neutral_cells=(owners == 0) & visible & (types != 2), fog_cells=types == 0,
        structures_in_fog=types == 5, owned_land_count=jnp.int32(my_land),
        owned_army_count=jnp.int32(my_army), opponent_land_count=jnp.int32(opp_land),
        opponent_army_count=jnp.int32(opp_army), timestep=jnp.int32(turn),
        allied_cells=jnp.zeros((height, width), dtype=bool),
        allied_land_count=jnp.int32(0), allied_army_count=jnp.int32(0),
    )


def main():
    handshake = sys.stdin.readline()
    if not handshake:
        return
    player, height, width = map(int, handshake.split())
    mode = os.environ.get('SENTINEL_MODE', 'competition')
    if mode not in ('classic', 'competition'):
        raise ValueError(f'unsupported SENTINEL_MODE={mode!r}')
    agent = SentinelAgent(build_castles=mode == 'competition',
                          deathtouch_turn=800 if mode == 'competition' else None,
                          max_turns=1200 if mode == 'competition' else 800)
    key = jax.random.PRNGKey(player)
    while (obs := read_observation(sys.stdin, height, width)) is not None:
        key, action_key = jax.random.split(key)
        action = agent.act(obs, action_key)
        print(' '.join(str(int(x)) for x in action), flush=True)


if __name__ == '__main__':
    main()
