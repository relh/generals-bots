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
from generals.agents.sentinel_agent import SentinelAgent
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
        factorized_actions: bool = False,
        classic_maps: bool = False,
    ):
        if imitation_weight < 0 or ((imitation_weight or supervise_teacher) and teacher is None):
            raise ValueError("Imitation reward or supervision requires a teacher")
        self.size = board_size
        self.supervise_teacher = supervise_teacher
        self.factorized_actions = factorized_actions
        self.training = context.mode == "train"
        map_options = (
            {"min_generals_distance": board_size - 2, "castle_val_range": (20, 41)} if classic_maps else {}
        )
        self.env = GeneralsEnv(
            grid_dims=(board_size, board_size),
            truncation=horizon,
            pool_size=8,
            mountain_density_range=(0.18, 0.26),
            num_castles_range=(2, 5),
            **map_options,
        )
        self.spec = EnvironmentSpec(
            observation_size=14 * board_size * board_size,
            action_sizes=[4 * board_size**2 + 1, 2] if factorized_actions else [8 * board_size**2 + 1],
            teacher=supervise_teacher,
        )
        self.pool, _ = self.env.reset(jax.random.PRNGKey(context.seed + context.index))
        self._init_state = jax.jit(self.env.init_state)
        self._observe = jax.jit(
            lambda state, side: encode_observation(
                game.get_observation(state, side), factorized_actions=factorized_actions
            )
        )
        opponent_types = {
            "expander": ExpanderAgent,
            "hunter": HunterAgent,
            "random": RandomAgent,
            "harvester": HarvesterAgent,
            "sentinel": SentinelAgent,
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
        def advance(state, pool, side, opponent_id, index, split, key):
            opponent_key, next_key = jax.random.split(key)
            enemy = jax.lax.switch(
                opponent_id, opponent_branches, (game.get_observation(state, 1 - side), opponent_key)
            )
            ours = decode_action(index, board_size, split if factorized_actions else None)
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
            values, mask = encode_observation(final, factorized_actions=factorized_actions)
            return next_state, next_key, values, mask, reward, done, outcome

        self._advance = advance

    def _observation(self, values, mask):
        legal = np.asarray(mask, dtype=bool)
        teachers = []
        if self.supervise_teacher:
            probabilities = np.zeros(sum(self.spec.action_sizes), dtype=np.float32)
            weights = [0.0] * len(self.spec.action_sizes)
            if self.training:
                opponent_key, _ = jax.random.split(self.key)
                action = np.asarray(self._teacher(self.state, self.side, jax.random.fold_in(opponent_key, 37)))
                cells = self.size**2
                index = (
                    (4 if self.factorized_actions else 8) * cells
                    if action[0]
                    else ((action[3] if self.factorized_actions else action[4] * 4 + action[3]) * cells)
                    + action[1] * self.size
                    + action[2]
                )
                if legal[index]:
                    probabilities[index] = 1.0
                    weights[0] = 1.0
                    if self.factorized_actions and not action[0]:
                        probabilities[self.spec.action_sizes[0] + action[4]] = 1.0
                        weights[1] = 1.0
            teachers = [TeacherTargets(probabilities=probabilities.tolist(), weights=weights)]
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
            self.state,
            self.pool,
            self.side,
            self.opponent_id,
            actions[0][0],
            actions[0][1] if self.factorized_actions else 0,
            self.key,
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


class BatchedGeneralsPufferEnvironment:
    """Run independent Generals games in one JAX device step."""

    def __init__(
        self, *, context: EnvironmentContext, parallel_games: int = 16, require_gpu: bool = True, **options
    ):
        if parallel_games < 1:
            raise ValueError("parallel_games must be positive")
        if require_gpu and not jax.devices("cuda"):
            raise RuntimeError("Batched Generals training requires a CUDA JAX device")
        self.base = GeneralsPufferEnvironment(context=context, **options)
        self.parallel_games = parallel_games
        self.spec = self.base.spec.model_copy(update={"agents": parallel_games})
        self.horizon = self.base.env.truncation
        self.turn = 0
        self.finished = np.zeros(parallel_games, dtype=bool)
        self.outcomes = np.zeros(parallel_games, dtype=np.float32)
        self.completed = np.zeros(parallel_games, dtype=np.int32)
        self._init_states = jax.jit(jax.vmap(self.base._init_state))
        self._observe_states = jax.jit(jax.vmap(self.base._observe))
        if self.base.supervise_teacher:
            self._teacher_actions = jax.jit(jax.vmap(self.base._teacher))

        def advance_one(state, pool, side, opponent_id, index, split, key, alive):
            def active(_):
                next_state, next_key, values, mask, reward, done, outcome = self.base._advance(
                    state, pool, side, opponent_id, index, split, key
                )
                if self.base.training:
                    def recycle(_):
                        reset_key, following_key = jax.random.split(next_key)
                        reset_state = self.base._init_state(reset_key)
                        reset_values, reset_mask = self.base._observe(reset_state, side)
                        return reset_state, following_key, reset_values, reset_mask

                    next_state, next_key, values, mask = jax.lax.cond(
                        done, recycle, lambda _: (next_state, next_key, values, mask), operand=None
                    )
                return next_state, next_key, values, mask, reward, done, outcome

            def inactive(_):
                values, mask = self.base._observe(state, side)
                return state, key, values, mask, jnp.float32(0), jnp.bool_(False), jnp.float32(0)

            return jax.lax.cond(alive, active, inactive, operand=None)

        self._advance_states = jax.jit(jax.vmap(advance_one, in_axes=(0, None, 0, 0, 0, 0, 0, 0)))

    def _observation(self, values, masks):
        public_values = np.asarray(values).copy()
        public_values[self.finished] = 0
        legal = np.asarray(masks, dtype=bool).copy()
        legal[self.finished] = False
        legal[self.finished, self.spec.action_sizes[0] - 1] = True
        if self.base.factorized_actions:
            legal[self.finished, self.spec.action_sizes[0] :] = True

        teachers = []
        if self.base.supervise_teacher:
            actions = None
            if self.base.training:
                teacher_keys = jax.vmap(lambda key: jax.random.fold_in(jax.random.split(key)[0], 37))(self.keys)
                actions = np.asarray(self._teacher_actions(self.states, self.sides, teacher_keys))
            for game_index in range(self.parallel_games):
                probabilities = np.zeros(sum(self.spec.action_sizes), dtype=np.float32)
                weights = [0.0] * len(self.spec.action_sizes)
                if actions is not None and not self.finished[game_index]:
                    action = actions[game_index]
                    cells = self.base.size**2
                    index = (
                        (4 if self.base.factorized_actions else 8) * cells
                        if action[0]
                        else ((action[3] if self.base.factorized_actions else action[4] * 4 + action[3]) * cells)
                        + action[1] * self.base.size
                        + action[2]
                    )
                    if legal[game_index, index]:
                        probabilities[index] = 1.0
                        weights[0] = 1.0
                        if self.base.factorized_actions and not action[0]:
                            probabilities[self.spec.action_sizes[0] + action[4]] = 1.0
                            weights[1] = 1.0
                teachers.append(TeacherTargets(probabilities=probabilities.tolist(), weights=weights))
        return NumericObservation(
            values=public_values.tolist(), action_masks=legal.tolist(), teachers=teachers
        )

    def reset(self, seed: str) -> NumericObservation:
        numeric_seed = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:4], "little")
        self.turn = 0
        self.finished[:] = False
        self.outcomes[:] = 0
        self.completed[:] = 0
        indices = np.arange(self.parallel_games, dtype=np.int64)
        self.sides = jnp.asarray((numeric_seed + indices) % 2, dtype=jnp.int32)
        self.opponent_ids = jnp.asarray((numeric_seed + indices) % self.base.num_opponents, dtype=jnp.int32)
        state_keys = jax.random.split(jax.random.PRNGKey(numeric_seed), self.parallel_games)
        self.keys = jax.random.split(jax.random.PRNGKey(numeric_seed ^ 0xA5A5A5A5), self.parallel_games)
        self.states = self._init_states(state_keys)
        values, masks = self._observe_states(self.states, self.sides)
        return self._observation(values, masks)

    def step(self, actions: list[list[int]]) -> NumericTransition:
        if len(actions) != self.parallel_games:
            raise ValueError("Expected one action per parallel game")
        indices = jnp.asarray([action[0] for action in actions], dtype=jnp.int32)
        splits = jnp.asarray(
            [action[1] if self.base.factorized_actions else 0 for action in actions], dtype=jnp.int32
        )
        was_finished = self.finished.copy()
        self.states, self.keys, values, masks, rewards, done, outcomes = self._advance_states(
            self.states,
            self.base.pool,
            self.sides,
            self.opponent_ids,
            indices,
            splits,
            self.keys,
            jnp.asarray(~was_finished),
        )
        newly_finished = np.asarray(done, dtype=bool)
        if self.base.training:
            self.completed += newly_finished
        else:
            self.finished |= newly_finished
        self.outcomes += np.asarray(outcomes, dtype=np.float32)
        self.turn += 1
        episode_done = self.turn >= self.horizon or (not self.base.training and bool(self.finished.all()))
        if episode_done and self.base.training:
            games = int(self.completed.sum() + np.count_nonzero(~newly_finished))
            score = float(self.outcomes.sum() / games)
        else:
            score = float(self.outcomes.mean()) if episode_done else 0.0
        return NumericTransition(
            observation=self._observation(values, masks),
            rewards=np.asarray(rewards).tolist(),
            terminated=[True] * self.parallel_games if episode_done else newly_finished.tolist(),
            episode_done=episode_done,
            score=score,
            perf=(score + 1) / 2,
        )

    def close(self) -> None:
        self.base.close()
