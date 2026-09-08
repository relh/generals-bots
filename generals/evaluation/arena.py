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


class MemoryMatchResult(NamedTuple):
    """Optional final policy state, indexed by candidate/opponent then game."""

    state: game.GameState
    finished: jax.Array
    counters: jax.Array
    candidate_memory: object
    opponent_memory: object
    keys: jax.Array


def initial_memory(policy, shape):
    """Stateless callables use an empty pytree; stateful policies implement both methods."""
    has_initial = callable(getattr(policy, "initial_memory", None))
    has_step = callable(getattr(policy, "step", None))
    if has_initial != has_step:
        raise TypeError("A stateful policy must provide both initial_memory(shape) and step(obs, key, memory)")
    return policy.initial_memory(shape) if has_initial else ()


def policy_step(policy, observation, key, memory):
    """Only the player's fog observation, explicit key and own memory cross this boundary."""
    if callable(getattr(policy, "step", None)):
        return policy.step(observation, key, memory)
    return policy(observation, key), memory, {}


def select_active(active, new, old):
    return jax.tree.map(lambda n, o: jnp.where(active.reshape((-1,) + (1,) * (n.ndim - 1)), n, o), new, old)


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


def make_runner(candidate, opponent, rules=Rules(), *, from_states=False, with_memory=False):
    """Each policy receives only its own Observation, PRNG key and optional memory.

    `seats` gives the candidate's player ID. Completed games, including mutual
    deathtouch draws, freeze at the first terminal state. Timeouts are draws.
    Memory is initialized independently for each game and policy on every call.
    ``with_memory=True`` additionally exposes final memories and per-game keys;
    the default return type remains MatchResult for existing callers.
    When supplying existing states, pass their ``initial_finished`` flags to
    ``run``: GameState alone does not encode terminal deathtouch draws.
    """

    @jax.jit
    def run(grids, keys, seats, initial_finished=None):
        states = grids if from_states else jax.vmap(game.create_initial_state)(grids)
        finished = states.winner >= 0
        if initial_finished is not None:
            finished = finished | jnp.asarray(initial_finished, dtype=jnp.bool_)
        counters = jnp.zeros((len(seats), 2, 6), dtype=jnp.int32)
        shape = states.armies.shape[1:]
        memories = tuple(
            jax.tree.map(
                lambda leaf: jnp.broadcast_to(jnp.asarray(leaf), (len(seats),) + jnp.shape(leaf)),
                initial_memory(policy, shape),
            )
            for policy in (candidate, opponent)
        )

        def cond(carry):
            states, _, finished, _, _ = carry
            return jnp.any(~finished & (states.time < rules.max_turns))

        def body(carry):
            states, keys, finished, counters, memories = carry
            split = jax.vmap(lambda key: jax.random.split(key, 3))(keys)
            ours, own_memory, _ = jax.vmap(lambda obs, key, memory: policy_step(candidate, obs, key, memory))(
                jax.vmap(game.get_observation)(states, seats), split[:, 1], memories[0]
            )
            theirs, enemy_memory, _ = jax.vmap(lambda obs, key, memory: policy_step(opponent, obs, key, memory))(
                jax.vmap(game.get_observation)(states, 1 - seats), split[:, 2], memories[1]
            )
            actions = jnp.stack(
                [jnp.where(seats[:, None] == 0, ours, theirs), jnp.where(seats[:, None] == 0, theirs, ours)], axis=1
            )
            new_states, info = jax.vmap(lambda s, a: transition(s, a, rules))(states, actions)
            active = ~finished & (states.time < rules.max_turns)
            increment = jax.vmap(lambda s, n, a: action_counters(s, n, a, rules))(states, new_states, actions)
            counters = counters + jnp.where(active[:, None, None], increment, 0)
            states = select_active(active, new_states, states)
            memories = select_active(active, (own_memory, enemy_memory), memories)
            keys = select_active(active, split[:, 0], keys)
            finished = finished | (active & info.is_done)
            return states, keys, finished, counters, memories

        states, keys, finished, counters, memories = jax.lax.while_loop(
            cond, body, (states, keys, finished, counters, memories)
        )
        if with_memory:
            return MemoryMatchResult(states, finished, counters, memories[0], memories[1], keys)
        return MatchResult(states, finished, counters)

    return run
