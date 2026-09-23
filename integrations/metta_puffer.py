"""One-seat Generals environment for Metta's native PufferLib trainer.

The policy sees the same fogged observation and legal moves as a live player.
The other seat uses a repository scripted agent. An episode ends on a win,
loss, or the explicit finite game horizon.
"""

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import (
    EnvironmentContext,
    EnvironmentSpec,
    NumericObservation,
    NumericTransition,
    TeacherTargets,
)

from generals import GeneralsEnv
from generals.agents import ExpanderAgent, HunterAgent, RandomAgent
from generals.agents.harvester_agent import HarvesterAgent
from generals.core import game
from integrations.puffer_codec import decode_action, encode_observation


class GeneralsPufferEnvironment:
    def __init__(
        self,
        *,
        context: EnvironmentContext,
        board_size: int = 10,
        horizon: int = 300,
        opponent: str = "expander",
        shaping_weight: float = 0.2,
        teacher: str | None = None,
        imitation_weight: float = 0.0,
        supervise_teacher: bool = False,
    ):
        if imitation_weight < 0 or ((imitation_weight or supervise_teacher) and teacher is None):
            raise ValueError("Imitation reward or supervision requires a teacher")
        self.size = board_size
        self.supervise_teacher = supervise_teacher
        self.training = context.mode == "train"
        self.env = GeneralsEnv(
            grid_dims=(board_size, board_size),
            truncation=horizon,
            pool_size=8,
            mountain_density_range=(0.18, 0.26),
            num_castles_range=(2, 5),
        )
        self.spec = EnvironmentSpec(
            observation_size=14 * board_size * board_size,
            action_sizes=[8 * board_size**2 + 1],
            teacher=supervise_teacher,
        )
        self.pool, _ = self.env.reset(jax.random.PRNGKey(context.seed + context.index))
        self._init_state = jax.jit(self.env.init_state)
        self._observe = jax.jit(lambda state, side: encode_observation(game.get_observation(state, side)))
        opponent_types = {
            "expander": ExpanderAgent,
            "hunter": HunterAgent,
            "random": RandomAgent,
            "harvester": HarvesterAgent,
        }
        teacher_agent = opponent_types[teacher]() if teacher is not None else None
        if supervise_teacher:
            self._teacher = jax.jit(lambda state, side, key: teacher_agent.act(game.get_observation(state, side), key))
        opponent_agents = (
            (RandomAgent(), ExpanderAgent(), HunterAgent()) if opponent == "mixed" else (opponent_types[opponent](),)
        )
        self.num_opponents = len(opponent_agents)
        opponent_branches = tuple(lambda args, agent=agent: agent.act(*args) for agent in opponent_agents)
        env = self.env

        @jax.jit
        def advance(state, pool, side, opponent_id, index, key):
            opponent_key, next_key = jax.random.split(key)
            enemy = jax.lax.switch(
                opponent_id, opponent_branches, (game.get_observation(state, 1 - side), opponent_key)
            )
            ours = decode_action(index, board_size)
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
            reward = outcome + shaping_weight * (
                0.99 * (0.5 * new_army + 0.3 * new_land) * ~done - (0.5 * old_army + 0.3 * old_land)
            )
            if teacher_agent is not None:
                suggested = teacher_agent.act(previous, jax.random.fold_in(opponent_key, 37))
                reward = reward + imitation_weight * jnp.all(ours == suggested) * (suggested[0] == 0)
            values, mask = encode_observation(final)
            return next_state, next_key, values, mask, reward, done, outcome

        self._advance = advance

    def _observation(self, values, mask):
        legal = np.asarray(mask, dtype=bool)
        teachers = []
        if self.supervise_teacher:
            probabilities = np.zeros(self.spec.action_sizes[0], dtype=np.float32)
            weight = 0.0
            if self.training:
                opponent_key, _ = jax.random.split(self.key)
                action = np.asarray(self._teacher(self.state, self.side, jax.random.fold_in(opponent_key, 37)))
                cells = self.size**2
                index = (
                    8 * cells if action[0] else (action[4] * 4 + action[3]) * cells + action[1] * self.size + action[2]
                )
                if legal[index]:
                    probabilities[index] = 1.0
                    weight = 1.0
            teachers = [TeacherTargets(probabilities=probabilities.tolist(), weights=[weight])]
        return NumericObservation(
            values=[np.asarray(values).tolist()], action_masks=[legal.tolist()], teachers=teachers
        )

    def reset(self, seed: str) -> NumericObservation:
        numeric_seed = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:4], "little")
        self.state = self._init_state(jax.random.PRNGKey(numeric_seed))
        self.key = jax.random.PRNGKey(numeric_seed ^ 0xA5A5A5A5)
        self.side = jnp.int32(numeric_seed % 2)
        self.opponent_id = jnp.int32(numeric_seed % self.num_opponents)
        values, mask = self._observe(self.state, self.side)
        return self._observation(values, mask)

    def step(self, actions: list[list[int]]) -> NumericTransition:
        self.state, self.key, values, mask, reward, done, outcome = self._advance(
            self.state, self.pool, self.side, self.opponent_id, actions[0][0], self.key
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
