"""One-seat Generals environment for Metta's native PufferLib trainer.

The policy sees the same fogged observation and legal moves as a live player.
The other seat uses the repository's Expander agent. An episode ends on a win,
loss, or the explicit finite game horizon.
"""

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext, EnvironmentSpec, NumericObservation, NumericTransition

from generals import GeneralsEnv
from generals.agents import ExpanderAgent
from generals.core import game
from generals.core.action import compute_valid_move_mask_obs


def _encode(obs):
    planes = obs.as_tensor().astype(jnp.float32)
    for channel in (0, 9, 10, 11, 12):
        planes = planes.at[channel].set(jnp.log1p(jnp.maximum(planes[channel], 0)) / 8.0)
    planes = planes.at[13].set(jnp.minimum(planes[13] / 1200.0, 2.0))
    moves = compute_valid_move_mask_obs(obs).transpose(2, 0, 1).reshape(-1)
    mask = jnp.concatenate((moves, moves, jnp.ones((1,), dtype=bool)))
    return planes.reshape(-1), mask


def _decode(index, size):
    cells = size * size
    channel, position = index // cells, index % cells
    move = jnp.array([0, position // size, position % size, channel % 4, channel // 4], dtype=jnp.int32)
    return jnp.where(index == 8 * cells, jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32), move)


class GeneralsPufferEnvironment:
    def __init__(self, *, context: EnvironmentContext, board_size: int = 10, horizon: int = 300):
        self.size = board_size
        self.env = GeneralsEnv(
            grid_dims=(board_size, board_size),
            truncation=horizon,
            pool_size=8,
            mountain_density_range=(0.18, 0.26),
            num_castles_range=(2, 5),
        )
        self.spec = EnvironmentSpec(observation_size=14 * board_size * board_size, action_sizes=[8 * board_size**2 + 1])
        self.pool, _ = self.env.reset(jax.random.PRNGKey(context.seed + context.index))
        self._init_state = jax.jit(self.env.init_state)
        self._observe = jax.jit(lambda state, side: _encode(game.get_observation(state, side)))
        opponent = ExpanderAgent()
        env = self.env

        @jax.jit
        def advance(state, pool, side, index, key):
            opponent_key, next_key = jax.random.split(key)
            enemy = opponent.act(game.get_observation(state, 1 - side), opponent_key)
            ours = _decode(index, board_size)
            actions = jnp.where(side == 0, jnp.stack((ours, enemy)), jnp.stack((enemy, ours)))
            previous = game.get_observation(state, side)
            timestep, next_state = env.step(state, actions, pool)
            final = game.get_observation(timestep.last_state, side)
            old_army = (previous.owned_army_count - previous.opponent_army_count) / (
                previous.owned_army_count + previous.opponent_army_count + 1
            )
            new_army = (final.owned_army_count - final.opponent_army_count) / (
                final.owned_army_count + final.opponent_army_count + 1
            )
            old_land = (previous.owned_land_count - previous.opponent_land_count) / (
                previous.owned_land_count + previous.opponent_land_count + 1
            )
            new_land = (final.owned_land_count - final.opponent_land_count) / (
                final.owned_land_count + final.opponent_land_count + 1
            )
            done = timestep.terminated | timestep.truncated
            outcome = jnp.where(timestep.terminated, timestep.reward[side], 0.0)
            reward = outcome + 0.2 * (
                0.99 * (0.5 * new_army + 0.3 * new_land) * ~done - (0.5 * old_army + 0.3 * old_land)
            )
            values, mask = _encode(final)
            return next_state, next_key, values, mask, reward, done, outcome

        self._advance = advance

    def _observation(self, values, mask):
        return NumericObservation(values=[np.asarray(values).tolist()], action_masks=[np.asarray(mask).tolist()])

    def reset(self, seed: str) -> NumericObservation:
        numeric_seed = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:4], "little")
        self.state = self._init_state(jax.random.PRNGKey(numeric_seed))
        self.key = jax.random.PRNGKey(numeric_seed ^ 0xA5A5A5A5)
        self.side = jnp.int32(numeric_seed % 2)
        values, mask = self._observe(self.state, self.side)
        return self._observation(values, mask)

    def step(self, actions: list[list[int]]) -> NumericTransition:
        self.state, self.key, values, mask, reward, done, outcome = self._advance(
            self.state, self.pool, self.side, actions[0][0], self.key
        )
        final = bool(done)
        score = float(outcome)
        return NumericTransition(
            observation=self._observation(values, mask),
            rewards=[float(reward)],
            terminated=[final],
            episode_done=final,
            score=score,
            perf=(score + 1) / 2,
        )

    def close(self) -> None:
        pass
