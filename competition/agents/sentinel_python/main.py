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
                raise EOFError("incomplete observation frame")
            row = list(map(int, line.split()))
            if len(row) != width:
                raise ValueError("observation row has incorrect width")
            rows.append(row)
        grids.append(jnp.asarray(rows, dtype=jnp.int32))
    types, owners, armies = grids
    visible = (types != 0) & (types != 5)
    return Observation(
        armies=armies,
        generals=types == 4,
        castles=types == 3,
        mountains=types == 2,
        owned_cells=owners == 1,
        opponent_cells=owners == 2,
        neutral_cells=(owners == 0) & visible & (types != 2),
        fog_cells=types == 0,
        structures_in_fog=types == 5,
        owned_land_count=jnp.int32(my_land),
        owned_army_count=jnp.int32(my_army),
        opponent_land_count=jnp.int32(opp_land),
        opponent_army_count=jnp.int32(opp_army),
        timestep=jnp.int32(turn),
        allied_cells=jnp.zeros((height, width), dtype=bool),
        allied_land_count=jnp.int32(0),
        allied_army_count=jnp.int32(0),
    )


def make_agent():
    mode = os.environ.get("SENTINEL_MODE", "competition")
    if mode not in ("classic", "competition"):
        raise ValueError(f"unsupported SENTINEL_MODE={mode!r}")
    variant = os.environ.get("SENTINEL_VARIANT", "v2")
    agent_type = SentinelAgent
    options = {}
    if variant in ("v3", "v3-memory", "v3-defense", "v3-disabled"):
        from generals.agents.sentinel_v3_agent import SentinelV3Agent

        agent_type = SentinelV3Agent
        options = dict(
            remember_threats=variant in ("v3", "v3-memory"),
            sustained_defense=variant in ("v3", "v3-defense"),
        )
    elif variant in ("v4", "v4-adjacent"):
        from generals.agents.sentinel_v4_agent import SentinelV4Agent

        agent_type = SentinelV4Agent
        options = dict(build_threat_horizon=2 if variant == "v4" else 1)
    elif variant in ("v5", "v5-disabled"):
        from generals.agents.sentinel_v5_agent import SentinelV5Agent

        agent_type = SentinelV5Agent
        options = dict(intercept_threats=variant == "v5")
    elif variant in ("v6", "v6-disabled"):
        from generals.agents.sentinel_v6_agent import SentinelV6Agent

        agent_type = SentinelV6Agent
        options = dict(commit_defense=variant == "v6")
    elif variant in ("v7", "v7-disabled"):
        from generals.agents.sentinel_v7_agent import SentinelV7Agent

        agent_type = SentinelV7Agent
        options = dict(concentrate_armies=variant == "v7")
    elif variant in ("v8", "v8-cheap", "v8-direct", "v8-disabled", "v8-no-concentration"):
        from generals.agents.sentinel_v8_agent import SentinelV8Agent

        agent_type = SentinelV8Agent
        options = dict(
            concentrate_armies=variant != "v8-no-concentration",
            cheapest_collection=variant in ("v8", "v8-cheap", "v8-no-concentration"),
            direct_deployment=variant in ("v8", "v8-direct", "v8-no-concentration"),
        )
    elif variant in ("v9", "v9-disabled"):
        from generals.agents.sentinel_v9_agent import SentinelV9Agent

        agent_type = SentinelV9Agent
        options = dict(remember_enemy_general=variant == "v9")
    elif variant != "v2":
        raise ValueError(f"unsupported SENTINEL_VARIANT={variant!r}")
    return agent_type(
        build_castles=mode == "competition",
        deathtouch_turn=800 if mode == "competition" else None,
        max_turns=1200 if mode == "competition" else 800,
        **options,
    )


def main():
    handshake = sys.stdin.readline()
    if not handshake:
        return
    player, height, width = map(int, handshake.split())
    agent = make_agent()
    stateful = hasattr(agent, "initial_memory")
    memory = agent.initial_memory((height, width)) if stateful else None
    key = jax.random.PRNGKey(player)
    while (obs := read_observation(sys.stdin, height, width)) is not None:
        key, action_key = jax.random.split(key)
        if stateful:
            action, memory, _ = agent.step(obs, action_key, memory)
        else:
            action = agent.act(obs, action_key)
        print(" ".join(str(int(x)) for x in action), flush=True)


if __name__ == "__main__":
    main()
