"""Batched matches with exact rule modifiers and observation-only agent inputs."""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from generals.core import game
from generals.core.action import DIRECTIONS
from generals.modifiers import build_castles, deathtouch


@dataclass(frozen=True)
class Rules:
    max_turns: int = 500
    build_castles: bool = False
    deathtouch_turn: int | None = None


class MatchResult(NamedTuple):
    state: game.GameState
    finished: jax.Array
    # Per player: passes, splits, build attempts, successful builds,
    # invalid physical move attempts (before move-order interactions),
    # malformed commands rejected before execution.
    counters: jax.Array


def valid_commands(actions, shape, rules):
    h, w = shape
    kind, row, col, direction, split = actions.T
    integral = jnp.all(jnp.isfinite(actions) & (actions == actions.astype(jnp.int32)), axis=1)
    source_ok = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    move_ok = (kind == 0) & source_ok & (direction >= 0) & (direction < 4) & ((split == 0) | (split == 1))
    build_ok = (kind == 2) & rules.build_castles & source_ok
    return integral & ((kind == 1) | move_ok | build_ok)


def sanitize_actions(actions, shape, rules):
    if actions.shape != (2, 5):
        raise ValueError("two-player actions must have shape (2, 5)")
    valid = valid_commands(actions, shape, rules)
    return jnp.where(valid[:, None], actions, jnp.array([1, 0, 0, 0, 0])).astype(jnp.int32)


@partial(jax.jit, static_argnames=("rules",))
def transition(state, actions, rules):
    actions = sanitize_actions(actions, state.armies.shape, rules)
    if rules.build_castles:
        state, actions = build_castles.apply_build_actions(state, actions)
    if rules.deathtouch_turn is not None:
        return deathtouch.step(state, actions, rules.deathtouch_turn)
    return game.step(state, actions)


def action_counters(before, after, actions, rules=Rules()):
    h, w = before.armies.shape
    players = jnp.arange(2)
    kind, r, c, direction, split = actions.T
    rs, cs = jnp.clip(r.astype(jnp.int32), 0, h - 1), jnp.clip(c.astype(jnp.int32), 0, w - 1)
    ds = jnp.clip(direction.astype(jnp.int32), 0, 3)
    dest = jnp.stack([r, c], axis=-1) + DIRECTIONS[ds]
    dr, dc = dest[:, 0], dest[:, 1]
    valid = (
        (r >= 0)
        & (r < h)
        & (c >= 0)
        & (c < w)
        & (dr >= 0)
        & (dr < h)
        & (dc >= 0)
        & (dc < w)
        & (direction >= 0)
        & (direction < 4)
        & before.ownership[players, rs, cs]
        & (before.armies[rs, cs] > 1)
        & before.passable[jnp.clip(dr.astype(jnp.int32), 0, h - 1), jnp.clip(dc.astype(jnp.int32), 0, w - 1)]
    )
    command_ok = valid_commands(actions, (h, w), rules)
    built = (
        command_ok
        & (kind == 2)
        & before.ownership[players, rs, cs]
        & ~before.generals[rs, cs]
        & ~before.castles[rs, cs]
        & after.castles[rs, cs]
    )
    return jnp.stack(
        [kind == 1, (kind == 0) & (split == 1), kind == 2, built, (kind == 0) & ~(valid & command_ok), ~command_ok],
        axis=-1,
    ).astype(jnp.int32)


def make_runner(candidate, opponent, rules=Rules(), *, from_states=False):
    """Each callable receives only its own Observation and PRNG key.

    `seats` gives the candidate's player ID. Completed games, including mutual
    deathtouch draws, freeze at the first terminal state. Timeouts are draws.
    """

    @jax.jit
    def run(grids, keys, seats):
        states = grids if from_states else jax.vmap(game.create_initial_state)(grids)
        finished = states.winner >= 0
        counters = jnp.zeros((len(seats), 2, 6), dtype=jnp.int32)

        def cond(carry):
            states, _, finished, _ = carry
            return jnp.any(~finished & (states.time < rules.max_turns))

        def body(carry):
            states, keys, finished, counters = carry
            split = jax.vmap(lambda key: jax.random.split(key, 3))(keys)
            ours = jax.vmap(candidate)(jax.vmap(game.get_observation)(states, seats), split[:, 1])
            theirs = jax.vmap(opponent)(jax.vmap(game.get_observation)(states, 1 - seats), split[:, 2])
            actions = jnp.stack(
                [jnp.where(seats[:, None] == 0, ours, theirs), jnp.where(seats[:, None] == 0, theirs, ours)], axis=1
            )
            new_states, info = jax.vmap(lambda s, a: transition(s, a, rules))(states, actions)
            active = ~finished & (states.time < rules.max_turns)
            increment = jax.vmap(lambda s, n, a: action_counters(s, n, a, rules))(states, new_states, actions)
            counters = counters + jnp.where(active[:, None, None], increment, 0)
            states = jax.tree.map(
                lambda new, old: jnp.where(active.reshape((-1,) + (1,) * (new.ndim - 1)), new, old), new_states, states
            )
            finished = finished | (active & info.is_done)
            return states, split[:, 0], finished, counters

        states, _, finished, counters = jax.lax.while_loop(cond, body, (states, keys, finished, counters))
        return MatchResult(states, finished, counters)

    return run
