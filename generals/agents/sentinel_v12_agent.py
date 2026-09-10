"""Experimental public-route guard over the frozen V10 commander.

Validate the actual issued action against current visible enemy/home routes.
An interception's additive donor estimate can miss a cheaper route opened by
vacating another garrison. This guard recomputes the whole public cost field;
it does not certify future routes, enemy replies, merges or unseen threats.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .sentinel_agent import _build_prices, _distance
from .sentinel_v3_agent import _campaign_decision
from .sentinel_v5_agent import _HORIZON, _home_growth
from .sentinel_v6_agent import _destination
from .sentinel_v10_agent import SentinelV10Agent


def _project_action(obs, action, prices):
    """Our observable transfer/combat/build, without an intervening enemy action."""
    a = obs.armies
    width = a.shape[1]
    source = action[1] * width + action[2]
    destination = _destination(action, width, a.shape)
    flat = a.reshape(-1)
    moving = jnp.maximum(jnp.where(action[4] == 1, flat[source] // 2, flat[source] - 1), 0)
    moving = jnp.where(action[0] == 0, moving, 0)
    spent = jnp.where(action[0] == 2, prices.reshape(-1)[source], 0)
    mine = obs.owned_cells.reshape(-1)
    capture = (action[0] == 0) & ~mine[destination] & (moving > flat[destination])
    # PASS/BUILD coordinates and ignored direction/split cannot cause combat.
    target_army = jnp.where(mine[destination], flat[destination] + moving,
                            jnp.abs(flat[destination] - moving))
    post = flat.at[source].add(-moving - spent)
    post = post.at[destination].set(jnp.where(action[0] == 0, target_army, post[destination]))
    owned = mine.at[destination].set(mine[destination] | capture)
    enemies = obs.opponent_cells.reshape(-1).at[destination].set(
        obs.opponent_cells.reshape(-1)[destination] & ~capture,
    )
    return post.reshape(a.shape), owned.reshape(a.shape), enemies.reshape(a.shape)


def _deficits(obs, armies, owned, enemies, terrain, home, distance, touch):
    costs = 1 + jnp.where(owned | obs.neutral_cells, armies, 0)
    resistance = _distance(terrain, home, costs)
    resistance += jnp.where(touch, -jnp.sum(jnp.where(home, armies, 0)),
                            _home_growth(obs.timestep, distance))
    relevant = enemies & (distance <= _HORIZON)
    return jnp.where(relevant, jnp.maximum(armies - resistance, 0), 0)


def _winning(obs, action, deathtouch_turn):
    width = obs.armies.shape[1]
    source = action[1] * width + action[2]
    destination = _destination(action, width, obs.armies.shape)
    a = obs.armies.reshape(-1)
    moving = jnp.where(action[4] == 1, a[source] // 2, a[source] - 1)
    touch = obs.timestep >= deathtouch_turn if deathtouch_turn is not None else jnp.array(False)
    return ((action[0] == 0) & (moving > 0)
            & (obs.opponent_cells & obs.generals).reshape(-1)[destination]
            & ((moving > a[destination]) | touch))


class SentinelV12Agent:
    def __init__(self, id="Sentinel-v12", *, build_castles=False, deathtouch_turn=None,
                 max_turns=1200, guard_routes=True):
        self.id = id
        self.guard_routes = guard_routes
        self.deathtouch_turn = deathtouch_turn
        self.parent = SentinelV10Agent(build_castles=build_castles, deathtouch_turn=deathtouch_turn,
                                      max_turns=max_turns)
        self.campaign = self.parent.base.base.base.base.campaign

    def initial_memory(self, shape):
        return self.parent.initial_memory(shape)

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        action, returned, telemetry = self.parent.step(obs, key, memory)
        if not self.guard_routes:
            return action, returned, telemetry
        a, mine = obs.armies, obs.owned_cells
        home = mine & obs.generals
        terrain = ~(obs.mountains | obs.structures_in_fog)
        distance = _distance(terrain, home)
        touch = (obs.timestep + distance - 1 >= self.deathtouch_turn
                 if self.deathtouch_turn is not None else jnp.zeros_like(mine))
        active = jnp.any(home) & jnp.any(obs.opponent_cells & (distance <= _HORIZON))
        active &= ~_winning(obs, action, self.deathtouch_turn)

        def inspect(_):
            prices = _build_prices(mine & (obs.generals | obs.castles))
            before = _deficits(obs, a, mine, obs.opponent_cells, terrain, home, distance, touch)

            def evaluate(proposal):
                armies, owned, enemies = _project_action(obs, proposal, prices)
                after = _deficits(obs, armies, owned, enemies, terrain, home, distance, touch)
                return jnp.all(after <= before), jnp.max(after)

            parent_safe, parent_deficit = evaluate(action)

            def replace(_):
                campaign, campaign_telemetry, scores = _campaign_decision(self.campaign, obs, key)
                imminent = campaign_telemetry["adjacent_threat"]
                home_army = campaign_telemetry["general_army"]
                pass_unsafe = (imminent > home_army) | (campaign_telemetry["deathtouch_active"] & (imminent > 0))
                pass_score = jnp.where(pass_unsafe, -2000 - jnp.maximum(imminent - home_army, 0) * 20, 0)
                flat = scores.reshape(-1)
                # Four highest-scoring moves plus the best moves from four
                # other sources avoid spending every probe on one large army.
                top_scores, top = jax.lax.top_k(flat, 4)
                by_source = jnp.max(scores.reshape(-1, 8), axis=1)
                source_ids = jnp.arange(a.size)
                unused = ~jnp.any(source_ids[:, None] == (top // 8)[None], axis=1)
                _, diverse = jax.lax.top_k(jnp.where(unused, by_source, -1e9), 4)
                diverse = diverse * 8 + jnp.argmax(scores.reshape(-1, 8), axis=1)[diverse]
                indices = jnp.concatenate((top, diverse))
                ranked_scores = jnp.concatenate((top_scores, flat[diverse]))
                order = jnp.argsort(-ranked_scores, stable=True)
                indices, ranked_scores = indices[order], ranked_scores[order]
                cells = indices // 8
                moves = jnp.stack((jnp.zeros_like(cells), cells // a.shape[1], cells % a.shape[1],
                                   (indices // 2) % 4, indices % 2), axis=1)
                candidates = jnp.concatenate((campaign[None], moves), axis=0)
                # Canonical PASS is a known unchanged projection and remains
                # available even if the bounded productive search is exhausted.
                fallback = jnp.array([1, 0, 0, 0, 0], jnp.int32)
                initial = (jnp.int32(0), jnp.array(False), fallback, jnp.int32(-1),
                           jnp.int32(0), jnp.max(before))

                def condition(state):
                    i, found, *_ = state
                    return (i < 9) & ~found

                def body(state):
                    i, found, selected, rank, probes, deficit = state
                    proposal = candidates[i]
                    # Equal full/half transfers from army2 are one candidate.
                    source = proposal[1] * a.shape[1] + proposal[2]
                    destination = _destination(proposal, a.shape[1], a.shape)
                    amount = jnp.where(proposal[4] == 1, a.reshape(-1)[source] // 2,
                                       a.reshape(-1)[source] - 1)
                    sources = candidates[:, 1] * a.shape[1] + candidates[:, 2]
                    destinations = jax.vmap(lambda x: _destination(x, a.shape[1], a.shape))(candidates)
                    amounts = jnp.where(candidates[:, 4] == 1, a.reshape(-1)[sources] // 2,
                                        a.reshape(-1)[sources] - 1)
                    same = ((candidates[:, 0] == proposal[0]) & (sources == source)
                            & ((proposal[0] != 0) | ((destinations == destination) & (amounts == amount))))
                    duplicate = jnp.any((jnp.arange(9) < i) & same)
                    eligible = (i == 0) | (ranked_scores[jnp.maximum(i - 1, 0)] > pass_score)
                    candidate_is_parent = jnp.all(proposal == action)
                    attempt = eligible & ~duplicate & ~candidate_is_parent
                    safe, after = jax.lax.cond(attempt, lambda _: evaluate(proposal),
                                              lambda _: (jnp.array(False), deficit), operand=None)
                    safe |= attempt & _winning(obs, proposal, self.deathtouch_turn)
                    return (i + 1, found | safe, jnp.where(safe, proposal, selected),
                            jnp.where(safe, i, rank), probes + attempt.astype(jnp.int32),
                            jnp.where(safe, after, deficit))

                _, found, chosen, rank, probes, after = jax.lax.while_loop(condition, body, initial)
                exhausted = ~found
                return chosen, jnp.array(True), rank, probes, exhausted, jnp.max(before), parent_deficit, after

            return jax.lax.cond(parent_safe,
                lambda _: (action, jnp.array(False), jnp.int32(-1), jnp.int32(0), jnp.array(False),
                           jnp.max(before), parent_deficit, parent_deficit), replace, operand=None)

        chosen, rejected, rank, probes, exhausted, before, parent_after, after = jax.lax.cond(
            active, inspect,
            lambda _: (action, jnp.array(False), jnp.int32(-1), jnp.int32(0), jnp.array(False),
                       jnp.float32(0), jnp.float32(0), jnp.float32(0)), operand=None,
        )
        changed = jnp.any(chosen != action)
        time = obs.timestep.astype(jnp.int32)
        blank = self.initial_memory(a.shape).base
        blank = blank._replace(last_turn=time, defense=blank.defense._replace(last_turn=time))
        returned = returned._replace(base=jax.tree.map(
            lambda old, empty: jnp.where(changed, empty, old), returned.base, blank,
        ))
        return chosen, returned, telemetry | dict(
            route_guard_active=active, route_guard_parent_rejected=rejected,
            route_guard_issued=changed, route_guard_parent_action_issued=~changed,
            route_guard_candidate_rank=rank, route_guard_probes=probes,
            route_guard_search_exhausted=exhausted, route_guard_pass=chosen[0] == 1,
            route_guard_deficit_before=before, route_guard_parent_deficit_after=parent_after,
            route_guard_deficit_after=after, building=chosen[0] == 2,
        )
