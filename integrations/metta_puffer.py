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
from generals.agents.harvester_agent import ExpanderHarvesterAgent, HarvesterAgent, SprintHarvesterAgent
from generals.agents.sentinel_agent import SentinelAgent
from generals.core import game
from integrations.puffer_codec import (
    decode_action, encode_coworld_directional_observation, encode_coworld_lean_observation,
    encode_coworld_hinted_observation, encode_coworld_observation,
    encode_coworld_packed_directional_observation, encode_observation,
    hinted_replay_indices,
)


def _castle_control_margin(state: game.GameState, side: jnp.ndarray) -> jnp.ndarray:
    owned = jnp.sum(state.castles & state.ownership[side])
    opposing = jnp.sum(state.castles & state.ownership[1 - side])
    return (owned - opposing) / (jnp.sum(state.castles) + 1)


class GeneralsPufferEnvironment:
    def __init__(
        self,
        *,
        context: EnvironmentContext,
        board_size: int = 10,
        horizon: int = 300,
        opponent: str = "expander",
        shaping_weight: float = 0.2,
        shaping_gamma: float = 0.99,
        army_shaping_weight: float = 0.5,
        land_shaping_weight: float = 0.3,
        castle_shaping_weight: float = 0.0,
        teacher: str | None = None,
        imitation_weight: float = 0.0,
        supervise_teacher: bool = False,
        sparse_teacher: bool = False,
        factorized_actions: bool = False,
        classic_maps: bool = False,
        coworld_classic: bool = False,
        coworld_small_map_curriculum: bool = False,
        coworld_tiny_map_curriculum: bool = False,
        coworld_pool_size: int = 256,
        compact_features: bool = False,
        lean_features: bool = False,
        directional_features: bool = False,
        packed_directional_features: bool = False,
        hint_features: bool = False,
        prior_hint_features: bool = False,
        sprint_hint_features: bool = False,
        expander_hint_features: bool = False,
        context_hint_features: bool = False,
        packed_context_hint_features: bool = False,
        neighbor_threat_hint_features: bool = False,
        general_distance_hint_features: bool = False,
        teacher_rollouts: bool = False,
        goal_features: bool = False,
    ):
        if min(shaping_weight, army_shaping_weight, land_shaping_weight, castle_shaping_weight) < 0:
            raise ValueError("Shaping weights must be nonnegative")
        if not 0 < shaping_gamma <= 1:
            raise ValueError("Shaping discount must be in (0, 1]")
        if imitation_weight < 0 or ((imitation_weight or supervise_teacher or sparse_teacher) and teacher is None):
            raise ValueError("Imitation reward or supervision requires a teacher")
        if sparse_teacher and (supervise_teacher or not factorized_actions):
            raise ValueError("Sparse teacher requires factorized actions without dense supervision")
        if coworld_classic and classic_maps:
            raise ValueError("Choose one map distribution")
        if (coworld_small_map_curriculum or coworld_tiny_map_curriculum) and not coworld_classic:
            raise ValueError("Map curriculum requires Coworld Classic padding")
        if coworld_small_map_curriculum and coworld_tiny_map_curriculum:
            raise ValueError("Choose one map curriculum")
        if coworld_classic and (coworld_pool_size < 16 or coworld_pool_size % 16):
            raise ValueError("Coworld map pool must contain the 16 board sizes evenly")
        if compact_features and (not coworld_classic or not factorized_actions or goal_features):
            raise ValueError("Compact observations require Coworld Classic and factorized actions")
        if lean_features and (not compact_features or goal_features):
            raise ValueError("Lean observations require compact Coworld Classic features")
        if directional_features and (not lean_features or goal_features):
            raise ValueError("Directional observations require lean Coworld Classic features")
        if packed_directional_features and (not lean_features or directional_features or goal_features):
            raise ValueError("Packed directional observations require lean Coworld Classic features")
        if hint_features and (not lean_features or directional_features or packed_directional_features or goal_features):
            raise ValueError("Hinted observations require lean Coworld Classic features")
        if prior_hint_features and not hint_features:
            raise ValueError("Signed prior hints require hinted observations")
        if sprint_hint_features and not prior_hint_features:
            raise ValueError("Sprint hints require signed prior hints")
        if expander_hint_features and (not prior_hint_features or sprint_hint_features):
            raise ValueError("Expander hints require signed prior hints without sprint hints")
        if context_hint_features and not hint_features:
            raise ValueError("Context hints require hinted observations")
        if packed_context_hint_features and (not hint_features or context_hint_features):
            raise ValueError("Packed context hints require hinted observations without full context")
        if neighbor_threat_hint_features and (
            not expander_hint_features or context_hint_features or packed_context_hint_features
        ):
            raise ValueError("Neighbor threat hints require signed Expander hints without other context")
        if general_distance_hint_features and (
            not expander_hint_features or context_hint_features or packed_context_hint_features
            or neighbor_threat_hint_features
        ):
            raise ValueError("General distance hints require signed Expander hints without other context")
        expected_teacher = "expander_harvester" if expander_hint_features else "sprinter" if sprint_hint_features else "harvester"
        if prior_hint_features and teacher != expected_teacher:
            raise ValueError("Signed hint replay labels must match the scripted teacher")
        if teacher_rollouts and (teacher is None or not (sparse_teacher or supervise_teacher)):
            raise ValueError("Teacher rollouts require supervised teacher actions")
        if coworld_classic:
            board_size = 21
            horizon = 300 if coworld_tiny_map_curriculum else 600 if coworld_small_map_curriculum else 1200
        self.size = board_size
        self.supervise_teacher = supervise_teacher
        self.sparse_teacher = sparse_teacher
        self.factorized_actions = factorized_actions
        self.goal_features = goal_features
        self.compact_features = compact_features
        self.lean_features = lean_features
        self.directional_features = directional_features
        self.packed_directional_features = packed_directional_features
        self.hint_features = hint_features
        self.prior_hint_features = prior_hint_features
        self.sprint_hint_features = sprint_hint_features
        self.expander_hint_features = expander_hint_features
        self.context_hint_features = context_hint_features
        self.packed_context_hint_features = packed_context_hint_features
        self.neighbor_threat_hint_features = neighbor_threat_hint_features
        self.general_distance_hint_features = general_distance_hint_features
        self.teacher_rollouts = teacher_rollouts and context.mode == "train"
        self.coworld_classic = coworld_classic
        self.training = context.mode == "train"
        if coworld_classic:
            self.env = GeneralsEnv(
                min_grid_size=6 if coworld_tiny_map_curriculum else 10 if coworld_small_map_curriculum else 18,
                max_grid_size=8 if coworld_tiny_map_curriculum else 12 if coworld_small_map_curriculum else 21,
                pad_to=21, truncation=horizon,
                mountain_density_range=(0.18, 0.22) if coworld_tiny_map_curriculum else (0.24, 0.26),
                min_generals_distance=4 if coworld_tiny_map_curriculum else 8 if coworld_small_map_curriculum else 17,
                num_castles_range=(0, 3) if coworld_tiny_map_curriculum else (2, 5) if coworld_small_map_curriculum else (9, 11),
                castle_val_range=(10, 21) if coworld_tiny_map_curriculum else (20, 41) if coworld_small_map_curriculum else (40, 51),
                build_castles=False, deathtouch_turn=None,
                pool_size=coworld_pool_size, dynamic_pool=True,
            )
        else:
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
            observation_size=(14 if context_hint_features else 10 if packed_context_hint_features or neighbor_threat_hint_features or general_distance_hint_features else 11 if directional_features else 8 if lean_features else 14 if compact_features else 21 if goal_features else 14)
            * board_size * board_size,
            action_sizes=[4 * board_size**2 + 1, 2] if factorized_actions else [8 * board_size**2 + 1],
            teacher=supervise_teacher,
            replay_metadata_size=2 if sparse_teacher else 0,
        )
        self.pool = None if coworld_classic else self.env.reset(jax.random.PRNGKey(context.seed + context.index))[0]
        def initial_state(pool, key):
            if coworld_classic:
                index = jax.random.randint(key, (), 0, self.env.pool_size)
                return jax.tree.map(lambda field: field[index], pool)
            return self.env.init_state(key)

        self._initial_state = jax.jit(initial_state)
        self._encode = (
            (lambda obs: encode_coworld_hinted_observation(
                obs, signed_flags=prior_hint_features, sprint_hint=sprint_hint_features,
                expander_hint=expander_hint_features, context_features=context_hint_features,
                packed_context_features=packed_context_hint_features,
                neighbor_threat_features=neighbor_threat_hint_features,
                general_distance_features=general_distance_hint_features,
            ))
            if hint_features
            else encode_coworld_packed_directional_observation
            if packed_directional_features
            else encode_coworld_directional_observation
            if directional_features
            else encode_coworld_lean_observation
            if lean_features
            else encode_coworld_observation
            if compact_features
            else lambda obs: encode_observation(
                obs, factorized_actions=factorized_actions, goal_features=goal_features
            )
        )
        self._observe = jax.jit(
            lambda state, side: self._encode(game.get_observation(state, side))
        )
        opponent_types = {
            "expander": ExpanderAgent,
            "hunter": HunterAgent,
            "random": RandomAgent,
            "harvester": HarvesterAgent,
            "sprinter": SprintHarvesterAgent,
            "expander_harvester": ExpanderHarvesterAgent,
            "sentinel": SentinelAgent,
        }
        teacher_agent = opponent_types[teacher]() if teacher is not None else None
        if supervise_teacher or sparse_teacher:
            self._teacher = jax.jit(lambda state, side, key: teacher_agent.act(game.get_observation(state, side), key))
        opponent_agents = (
            (RandomAgent(), ExpanderAgent(), HunterAgent()) if opponent == "mixed"
            else (ExpanderHarvesterAgent(), ExpanderHarvesterAgent(), ExpanderHarvesterAgent(), SentinelAgent())
            if opponent == "strong_mixed" else (opponent_types[opponent](),)
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
            def margin(ours, theirs):
                return (ours - theirs) / (ours + theirs + 1)

            old_potential = jnp.float32(0)
            new_potential = jnp.float32(0)
            if army_shaping_weight:
                old_potential += army_shaping_weight * margin(
                    previous.owned_army_count, previous.opponent_army_count
                )
                new_potential += army_shaping_weight * margin(
                    final.owned_army_count, final.opponent_army_count
                )
            if land_shaping_weight:
                old_potential += land_shaping_weight * margin(
                    previous.owned_land_count, previous.opponent_land_count
                )
                new_potential += land_shaping_weight * margin(
                    final.owned_land_count, final.opponent_land_count
                )
            if castle_shaping_weight:
                old_potential += castle_shaping_weight * _castle_control_margin(state, side)
                new_potential += castle_shaping_weight * _castle_control_margin(timestep.last_state, side)
            done = timestep.terminated | timestep.truncated
            outcome = jnp.where(timestep.terminated, timestep.reward[side], 0.0)
            reward = outcome + shaping_weight * (
                shaping_gamma * new_potential * ~done - old_potential
            )
            if teacher_agent is not None and imitation_weight:
                suggested = teacher_agent.act(previous, jax.random.fold_in(opponent_key, 37))
                reward = reward + imitation_weight * jnp.all(ours == suggested) * (suggested[0] == 0)
            values, mask = self._encode(final)
            return next_state, next_key, values, mask, reward, done, outcome

        self._advance = advance

    def _observation(self, values, mask):
        legal = np.asarray(mask, dtype=bool)
        teachers = []
        metadata = []
        if self.sparse_teacher:
            indices = [-1.0, -1.0]
            if self.training:
                opponent_key, _ = jax.random.split(self.key)
                action = np.asarray(self._teacher(self.state, self.side, jax.random.fold_in(opponent_key, 37)))
                cells = self.size**2
                index = 4 * cells if action[0] else action[3] * cells + action[1] * self.size + action[2]
                if legal[index]:
                    indices[0] = float(index)
                    if not action[0]:
                        indices[1] = float(action[4])
            metadata = [indices]
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
            values=[np.asarray(values).tolist()], action_masks=[legal.tolist()],
            teachers=teachers, replay_metadata=metadata,
        )

    def reset(self, seed: str) -> NumericObservation:
        numeric_seed = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:4], "little")
        if self.coworld_classic:
            self.pool, _ = self.env.reset(jax.random.PRNGKey(numeric_seed ^ 0xC0A17D))
        self.state = self._initial_state(self.pool, jax.random.PRNGKey(numeric_seed))
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
        self, *, context: EnvironmentContext, parallel_games: int = 16, require_gpu: bool = True,
        sentinel_teacher_fraction: float = 0.0, sentinel_teacher_interval: int = 1,
        sentinel_teacher_only: bool = False, **options
    ):
        if parallel_games < 1:
            raise ValueError("parallel_games must be positive")
        if not 0 <= sentinel_teacher_fraction <= 1:
            raise ValueError("Sentinel teacher fraction must be in [0, 1]")
        if sentinel_teacher_interval < 1:
            raise ValueError("Sentinel teacher interval must be positive")
        if require_gpu and not jax.devices("cuda"):
            raise RuntimeError("Batched Generals training requires a CUDA JAX device")
        self.base = GeneralsPufferEnvironment(context=context, **options)
        self.parallel_games = parallel_games
        self.sentinel_teacher_games = round(parallel_games * sentinel_teacher_fraction) if self.base.training else 0
        self.sentinel_teacher_interval = sentinel_teacher_interval
        self.sentinel_teacher_only = sentinel_teacher_only
        if self.base.training and sentinel_teacher_only and not self.sentinel_teacher_games:
            raise ValueError("Sentinel-only labels require a nonzero Sentinel teacher fraction")
        if self.sentinel_teacher_games and not (
            self.base.sparse_teacher and self.base.prior_hint_features and not self.base.teacher_rollouts
        ):
            raise ValueError("Sentinel label mix requires sparse signed-hint labels and policy rollouts")
        self.spec = self.base.spec.model_copy(update={"agents": parallel_games})
        self.horizon = self.base.env.truncation
        self.turn = 0
        self.finished = np.zeros(parallel_games, dtype=bool)
        self.outcomes = np.zeros(parallel_games, dtype=np.float32)
        self.completed = np.zeros(parallel_games, dtype=np.int32)
        self._init_states = jax.jit(jax.vmap(self.base._initial_state, in_axes=(None, 0)))
        self._observe_states = jax.jit(jax.vmap(self.base._observe))
        if self.base.supervise_teacher or self.base.sparse_teacher:
            self._teacher_actions = jax.jit(jax.vmap(self.base._teacher))
        if self.sentinel_teacher_games:
            sentinel = SentinelAgent()
            self._sentinel_actions = jax.jit(jax.vmap(
                lambda state, side, key: sentinel.act(game.get_observation(state, side), key)
            ))

        def advance_one(state, pool, side, opponent_id, index, split, key, cached_teacher_action, alive):
            def active(_):
                executed_index, executed_split = index, split
                if self.base.teacher_rollouts:
                    action = cached_teacher_action
                    executed_index = jnp.where(
                        action[0] == 1, 4 * self.base.size**2,
                        action[3] * self.base.size**2 + action[1] * self.base.size + action[2],
                    )
                    executed_split = action[4]
                next_state, next_key, values, mask, reward, done, outcome = self.base._advance(
                    state, pool, side, opponent_id, executed_index, executed_split, key
                )
                if self.base.training:
                    def recycle(_):
                        reset_key, following_key = jax.random.split(next_key)
                        reset_state = self.base._initial_state(pool, reset_key)
                        reset_values, reset_mask = self.base._observe(reset_state, side)
                        return reset_state, following_key, reset_values, reset_mask

                    next_state, next_key, values, mask = jax.lax.cond(
                        done, recycle, lambda _: (next_state, next_key, values, mask), operand=None
                    )
                return next_state, next_key, values, mask, reward, done, outcome

            def inactive(_):
                values, mask = self.base._observe(state, side)
                return state, key, values, mask, jnp.float32(0), jnp.bool_(False), jnp.float32(0)

            next_state, next_key, values, mask, reward, done, outcome = jax.lax.cond(
                alive, active, inactive, operand=None
            )
            teacher_action = None
            if self.base.teacher_rollouts or (
                self.base.training and (self.base.supervise_teacher or self.base.sparse_teacher)
                and not self.base.prior_hint_features
            ):
                teacher_key = jax.random.fold_in(jax.random.split(next_key)[0], 37)
                teacher_action = self.base._teacher(next_state, side, teacher_key)
            return next_state, next_key, values, mask, reward, done, outcome, teacher_action

        self._advance_states = jax.jit(jax.vmap(advance_one, in_axes=(0, None, 0, 0, 0, 0, 0, 0, 0)))

    def _sentinel_labels(self):
        if not self.sentinel_teacher_games or self.turn % self.sentinel_teacher_interval:
            return None
        count = self.sentinel_teacher_games
        keys = jax.vmap(lambda key: jax.random.fold_in(jax.random.split(key)[0], 37))(self.keys[:count])
        states = jax.tree.map(lambda value: value[:count] if value is not None else None, self.states)
        return np.asarray(self._sentinel_actions(states, self.sides[:count], keys))

    def _observation(self, values, masks, teacher_actions=None, sentinel_actions=None):
        if self.base.training and not self.base.sparse_teacher and not self.base.supervise_teacher:
            # Training never marks a game as finished: it recycles terminal
            # states inside the JAX step. Both arrays are fresh device outputs.
            # Avoid copying them and revalidating every cell on every step.
            return NumericObservation.model_construct(
                values=np.asarray(values), action_masks=np.asarray(masks, dtype=bool),
                replay_metadata=[], teachers=[], assignments=[],
            )
        public_values = np.asarray(values).copy()
        public_values[self.finished] = 0
        legal = np.asarray(masks, dtype=bool).copy()
        legal[self.finished] = False
        legal[self.finished, self.spec.action_sizes[0] - 1] = True
        if self.base.factorized_actions:
            legal[self.finished, self.spec.action_sizes[0] :] = True

        probabilities = None
        weights = None
        metadata = None
        if self.base.sparse_teacher:
            metadata = np.full((self.parallel_games, 2), -1, dtype=np.float32)
            if self.base.training:
                if self.base.prior_hint_features:
                    if not self.sentinel_teacher_only:
                        labels = hinted_replay_indices(
                            public_values, self.base.size,
                            14 if self.base.context_hint_features
                            else 10 if self.base.packed_context_hint_features or self.base.neighbor_threat_hint_features or self.base.general_distance_hint_features
                            else 8,
                        )
                        rows = np.arange(self.parallel_games)
                        labeled = (~self.finished) & legal[rows, labels[:, 0]]
                        metadata[labeled, 0] = labels[labeled, 0]
                        metadata[labeled, 1] = labels[labeled, 1]
                    if sentinel_actions is not None:
                        count = self.sentinel_teacher_games
                        cells = self.base.size**2
                        actions = sentinel_actions
                        move_index = actions[:, 3] * cells + actions[:, 1] * self.base.size + actions[:, 2]
                        index = np.where(actions[:, 0] == 1, self.spec.action_sizes[0] - 1, move_index)
                        chosen = np.arange(count)
                        valid = (~self.finished[:count]) & legal[chosen, index]
                        metadata[:count] = -1
                        metadata[chosen[valid], 0] = index[valid]
                        moving = valid & (actions[:, 0] == 0)
                        metadata[chosen[moving], 1] = actions[moving, 4]
                else:
                    if teacher_actions is None:
                        teacher_keys = jax.vmap(lambda key: jax.random.fold_in(jax.random.split(key)[0], 37))(self.keys)
                        teacher_actions = self._teacher_actions(self.states, self.sides, teacher_keys)
                    actions = np.asarray(teacher_actions)
                    cells = self.base.size**2
                    move_index = actions[:, 3] * cells + actions[:, 1] * self.base.size + actions[:, 2]
                    index = np.where(actions[:, 0] == 1, self.spec.action_sizes[0] - 1, move_index)
                    rows = np.arange(self.parallel_games)
                    labeled = (~self.finished) & legal[rows, index]
                    metadata[labeled, 0] = index[labeled]
                    moving = labeled & (actions[:, 0] == 0)
                    metadata[moving, 1] = actions[moving, 4]
        if self.base.supervise_teacher:
            probabilities = np.zeros((self.parallel_games, sum(self.spec.action_sizes)), dtype=np.float32)
            weights = np.zeros((self.parallel_games, len(self.spec.action_sizes)), dtype=np.float32)
            if self.base.training:
                if teacher_actions is None:
                    teacher_keys = jax.vmap(lambda key: jax.random.fold_in(jax.random.split(key)[0], 37))(self.keys)
                    teacher_actions = self._teacher_actions(self.states, self.sides, teacher_keys)
                actions = np.asarray(teacher_actions)
                cells = self.base.size**2
                direction = actions[:, 3] if self.base.factorized_actions else actions[:, 4] * 4 + actions[:, 3]
                move_index = direction * cells + actions[:, 1] * self.base.size + actions[:, 2]
                index = np.where(actions[:, 0] == 1, self.spec.action_sizes[0] - 1, move_index)
                rows = np.arange(self.parallel_games)
                labeled = (~self.finished) & legal[rows, index]
                probabilities[rows[labeled], index[labeled]] = 1
                weights[labeled, 0] = 1
                if self.base.factorized_actions:
                    moving = labeled & (actions[:, 0] == 0)
                    probabilities[rows[moving], self.spec.action_sizes[0] + actions[moving, 4]] = 1
                    weights[moving, 1] = 1
        observation = NumericObservation.from_arrays(public_values, legal, probabilities, weights)
        return observation.model_copy(update={"replay_metadata": metadata.tolist()}) if metadata is not None else observation

    def reset(self, seed: str) -> NumericObservation:
        numeric_seed = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:4], "little")
        if self.base.coworld_classic:
            self.base.pool, _ = self.base.env.reset(jax.random.PRNGKey(numeric_seed ^ 0xC0A17D))
        self.turn = 0
        self.finished[:] = False
        self.outcomes[:] = 0
        self.completed[:] = 0
        indices = np.arange(self.parallel_games, dtype=np.int64)
        self.sides = jnp.asarray((numeric_seed + indices) % 2, dtype=jnp.int32)
        self.opponent_ids = jnp.asarray((numeric_seed + indices) % self.base.num_opponents, dtype=jnp.int32)
        state_keys = jax.random.split(jax.random.PRNGKey(numeric_seed), self.parallel_games)
        self.keys = jax.random.split(jax.random.PRNGKey(numeric_seed ^ 0xA5A5A5A5), self.parallel_games)
        self.states = self._init_states(self.base.pool, state_keys)
        values, masks = self._observe_states(self.states, self.sides)
        self.cached_teacher_actions = None
        if self.base.teacher_rollouts:
            teacher_keys = jax.vmap(lambda key: jax.random.fold_in(jax.random.split(key)[0], 37))(self.keys)
            self.cached_teacher_actions = self._teacher_actions(self.states, self.sides, teacher_keys)
        return self._observation(values, masks, self.cached_teacher_actions, self._sentinel_labels())

    def step(self, actions: list[list[int]]) -> NumericTransition:
        if len(actions) != self.parallel_games:
            raise ValueError("Expected one action per parallel game")
        action_array = jnp.asarray(np.asarray(actions, dtype=np.int32))
        indices = action_array[:, 0]
        splits = action_array[:, 1] if self.base.factorized_actions else jnp.zeros_like(indices)
        was_finished = self.finished.copy()
        cached_teacher_actions = (
            self.cached_teacher_actions if self.base.teacher_rollouts
            else jnp.zeros((self.parallel_games, 5), dtype=jnp.int32)
        )
        self.states, self.keys, values, masks, rewards, done, outcomes, teacher_actions = self._advance_states(
            self.states,
            self.base.pool,
            self.sides,
            self.opponent_ids,
            indices,
            splits,
            self.keys,
            cached_teacher_actions,
            jnp.asarray(~was_finished),
        )
        if self.base.teacher_rollouts:
            self.cached_teacher_actions = teacher_actions
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
            observation=self._observation(values, masks, teacher_actions, self._sentinel_labels()),
            rewards=np.asarray(rewards).tolist(),
            terminated=[True] * self.parallel_games if episode_done else newly_finished.tolist(),
            episode_done=episode_done,
            score=score,
            perf=(score + 1) / 2,
        )

    def close(self) -> None:
        self.base.close()
