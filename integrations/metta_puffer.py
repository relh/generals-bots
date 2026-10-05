"""Device-resident Classic games with one public sixteen-plane flat action codec."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext, EnvironmentSpec

from generals import GeneralsEnv
from generals.core import coworld_game, game
from integrations.classic_contract import CLASSIC_MAP_OPTIONS, verify_engine
from integrations.puffer_codec import decode_action, encode_coworld_directional_observation


class BatchedGeneralsSelfPlayPufferEnvironment:
    """Advance both seats together; population wrappers choose the opposing actions."""

    def __init__(
        self, *, context: EnvironmentContext, parallel_games=8192,
        coworld_pool_size=8192, horizon=2000, require_gpu=True,
        balance_opponent_sides=True, shaping_weight=0.25, shaping_gamma=0.999,
        reward_scale=0.5, army_shaping_weight=0.5, land_shaping_weight=0.3,
        terminal_reward_mode="win_only", coworld_position_pool=None,
        coworld_position_pool_sha256=None, coworld_position_probability=0.0,
    ):
        verify_engine()
        if context.mode != "train":
            raise ValueError("Classic device environments require an explicit train context")
        if isinstance(parallel_games, bool) or parallel_games < 2 or parallel_games % 2:
            raise ValueError("Classic games require positive balanced seat pairs")
        if balance_opponent_sides is not True:
            raise ValueError("Classic population games require balanced seats")
        if coworld_pool_size < 16 or coworld_pool_size % 16:
            raise ValueError("Classic map pool must contain the sixteen board dimensions evenly")
        if require_gpu and jax.devices()[0].platform != "gpu":
            raise RuntimeError("Classic training requires a CUDA JAX device")
        if not 1 <= horizon <= 2000 or (require_gpu and horizon != 2000):
            raise ValueError("GPU Classic games require the official 2000-turn cap")
        if terminal_reward_mode not in ("win_only", "signed"):
            raise ValueError("Unknown Classic terminal reward objective")
        coefficients = (shaping_weight, army_shaping_weight, land_shaping_weight)
        if any(not np.isfinite(x) or x < 0 for x in coefficients):
            raise ValueError("Shaping coefficients must be finite and nonnegative")
        if not np.isfinite(shaping_gamma) or not 0 < shaping_gamma <= 1:
            raise ValueError("Shaping discount must be finite and in (0, 1]")
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("Reward scale must be finite and positive")
        probability = coworld_position_probability
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Position curriculum probability must be in [0, 1]")
        env = GeneralsEnv(**dict(CLASSIC_MAP_OPTIONS, truncation=horizon, pool_size=coworld_pool_size))

        def initial_state(pool, key):
            index = jax.random.randint(key, (), 0, env.pool_size)
            return jax.tree.map(lambda field: field[index], pool)

        if probability:
            if not coworld_position_pool or not coworld_position_pool_sha256:
                raise ValueError("Position curriculum requires its explicit archive and checksum")
            from integrations.classic_position_curriculum import load_positions, mix_initial_positions

            positions = load_positions(coworld_position_pool, coworld_position_pool_sha256)
            initial_state = mix_initial_positions(initial_state, positions, probability)
        self.base = SimpleNamespace(env=env, pool=None, position_probability=probability)
        self.parallel_games, self.horizon, self.turn = parallel_games, horizon, 0
        self.spec = EnvironmentSpec(observation_size=7056, action_sizes=[3529], agents=2 * parallel_games)
        self._self_sides = jnp.arange(2, dtype=jnp.int32)
        self._init_states = jax.jit(jax.vmap(initial_state, in_axes=(None, 0)))
        self._observe_both = jax.jit(jax.vmap(lambda state: jax.vmap(
            lambda side: encode_coworld_directional_observation(game.get_observation(state, side))
        )(self._self_sides)))
        self._reward_options = dict(shaping_weight=shaping_weight, shaping_gamma=shaping_gamma,
                                   reward_scale=reward_scale, army_shaping_weight=army_shaping_weight,
                                   land_shaping_weight=land_shaping_weight)
        self._terminal_reward_mode = terminal_reward_mode

        def margin(ours, theirs):
            return (ours - theirs) / (ours + theirs + 1)

        def potential(info, side):
            return (army_shaping_weight * margin(info.army[side], info.army[1 - side])
                    + land_shaping_weight * margin(info.land[side], info.land[1 - side]))

        def advance_one(state, pool, indices, splits, key):
            next_key, reset_key = jax.random.split(key)
            actions = jax.vmap(lambda index: decode_action(index, 21))(indices)
            previous_info = game.get_info(state)
            timestep, next_state = env.step(state, actions, pool)
            done = timestep.terminated | timestep.truncated

            def side_reward(side):
                outcome = jnp.where(timestep.terminated, timestep.reward[side], 0.0)
                terminal = (jnp.where(timestep.terminated & (timestep.reward[side] > 0), 1.0, 0.0)
                            if terminal_reward_mode == "win_only" else outcome)
                shaped = shaping_weight * (shaping_gamma * potential(timestep.info, side) * ~done
                                           - potential(previous_info, side))
                return (terminal + shaped) * reward_scale

            rewards = jax.vmap(side_reward)(self._self_sides)

            def recycle(_):
                fresh = initial_state(pool, reset_key)
                return fresh, jax.random.fold_in(next_key, 1), coworld_game.get_observations(fresh)

            next_state, next_key, observations = jax.lax.cond(
                done, recycle, lambda _: (next_state, next_key, timestep.observation), operand=None
            )
            values, masks = jax.vmap(encode_coworld_directional_observation)(observations)
            return next_state, next_key, values, masks, rewards, done

        self._advance_self_states = jax.jit(jax.vmap(advance_one, in_axes=(0, None, 0, 0, 0)))

    def _reset_states(self, seed):
        numeric = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:4], "little")
        self._pool_seed, self._pool_generation = jax.random.PRNGKey(numeric ^ 0xC0A17D), 0
        self.base.pool, _ = self.base.env.reset(self._pool_seed)
        self.turn = 0
        self.sides = jnp.asarray((numeric + np.arange(self.parallel_games, dtype=np.int64)) % 2, jnp.int32)
        state_keys = jax.random.split(jax.random.PRNGKey(numeric), self.parallel_games)
        self.keys = jax.random.split(jax.random.PRNGKey(numeric ^ 0xA5A5A5A5), self.parallel_games)
        self.states = self._init_states(self.base.pool, state_keys)
        if self.base.position_probability:
            print("INITIAL_POSITION_MIX " + json.dumps(dict(
                games=self.parallel_games, midgame=int(np.count_nonzero(np.asarray(self.states.time) > 0)),
                probability=self.base.position_probability,
            )), flush=True)

    def reset_device(self, seed):
        self._reset_states(seed)
        values, masks = self._observe_both(self.states)
        return values.reshape((self.spec.agents, -1)), masks.reshape((self.spec.agents, -1)).astype(jnp.uint8)

    def step_device(self, actions):
        if actions.shape != (self.spec.agents, 1):
            raise ValueError("Classic flat actions must have one index per agent")
        paired = actions.reshape((self.parallel_games, 2)).astype(jnp.int32)
        self.states, self.keys, values, masks, rewards, done = self._advance_self_states(
            self.states, self.base.pool, paired, jnp.zeros_like(paired), self.keys
        )
        self.turn += 1
        if self.turn >= self.horizon:
            self.turn, self._pool_generation = 0, self._pool_generation + 1
            self.base.pool, _ = self.base.env.reset(jax.random.fold_in(self._pool_seed, self._pool_generation))
        return (values.reshape((self.spec.agents, -1)), masks.reshape((self.spec.agents, -1)).astype(jnp.uint8),
                rewards.reshape((self.spec.agents,)).astype(jnp.float32), jnp.repeat(done, 2).astype(jnp.float32), False)

    def close(self):
        pass
