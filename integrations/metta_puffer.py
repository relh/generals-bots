"""One-seat Generals environment for Metta's native PufferLib trainer.

The policy sees the same fogged observation and legal moves as a live player.
The other seat uses a repository scripted agent. An episode ends on a win,
loss, or the explicit finite game horizon.
"""

import hashlib
import json
from pathlib import Path

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
    calibrate_hint_features,
    decode_action, encode_coworld_directional_observation, encode_coworld_lean_observation,
    encode_coworld_hinted_observation, encode_coworld_observation,
    encode_coworld_packed_directional_observation, encode_observation,
    hinted_replay_indices, hinted_teacher_action_device,
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
        deduplicate_opponent_branches: bool = False,
        shaping_weight: float = 0.2,
        shaping_gamma: float = 0.99,
        reward_scale: float = 1.0,
        army_shaping_weight: float = 0.5,
        land_shaping_weight: float = 0.3,
        castle_shaping_weight: float = 0.0,
        land_gain_reward_weight: float = 0.0,
        teacher: str | None = None,
        imitation_weight: float = 0.0,
        supervise_teacher: bool = False,
        sparse_teacher: bool = False,
        factorized_actions: bool = False,
        mask_pass_when_moves_exist: bool = False,
        classic_maps: bool = False,
        coworld_classic: bool = False,
        coworld_small_map_curriculum: bool = False,
        coworld_tiny_map_curriculum: bool = False,
        coworld_pool_size: int = 256,
        compact_features: bool = False,
        lean_features: bool = False,
        directional_features: bool = False,
        directional_time_features: bool = False,
        packed_directional_features: bool = False,
        hint_features: bool = False,
        prior_hint_features: bool = False,
        sprint_hint_features: bool = False,
        expander_hint_features: bool = False,
        context_hint_features: bool = False,
        move_hint_scale: float = 1.0,
        split_hint_scale: float = 1.0,
        packed_context_hint_features: bool = False,
        neighbor_threat_hint_features: bool = False,
        general_distance_hint_features: bool = False,
        teacher_rollouts: bool = False,
        goal_features: bool = False,
    ):
        if min(shaping_weight, army_shaping_weight, land_shaping_weight, castle_shaping_weight) < 0:
            raise ValueError("Shaping weights must be nonnegative")
        if not np.isfinite(land_gain_reward_weight) or land_gain_reward_weight < 0:
            raise ValueError("Land gain reward weight must be finite and nonnegative")
        if not 0 < shaping_gamma <= 1:
            raise ValueError("Shaping discount must be in (0, 1]")
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("Reward scale must be finite and positive")
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
        if compact_features and (not coworld_classic or (not factorized_actions and not directional_features) or goal_features):
            raise ValueError("Compact observations require Coworld Classic and factorized actions")
        if lean_features and (not compact_features or goal_features):
            raise ValueError("Lean observations require compact Coworld Classic features")
        if directional_features and (not lean_features or goal_features):
            raise ValueError("Directional observations require lean Coworld Classic features")
        if directional_time_features and not directional_features:
            raise ValueError("Directional time features require directional observations")
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
        if not (0 < move_hint_scale <= 1 and 0 < split_hint_scale <= 1):
            raise ValueError("Hint confidence scales must be in (0, 1]")
        if (move_hint_scale != 1 or split_hint_scale != 1) and not prior_hint_features:
            raise ValueError("Hint calibration requires signed prior features")
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
            # The hosted Classic 1v1 competition variant caps games at 2000 turns.
            horizon = 300 if coworld_tiny_map_curriculum else 600 if coworld_small_map_curriculum else 2000
        self.size = board_size
        self.supervise_teacher = supervise_teacher
        self.sparse_teacher = sparse_teacher
        self.factorized_actions = factorized_actions
        self.mask_pass_when_moves_exist = mask_pass_when_moves_exist
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
            observation_size=(14 if context_hint_features else 10 if packed_context_hint_features or neighbor_threat_hint_features or general_distance_hint_features else 12 if directional_time_features else 11 if directional_features else 8 if lean_features else 14 if compact_features else 21 if goal_features else 14)
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
            else (lambda obs: encode_coworld_directional_observation(
                obs, factorized_actions=factorized_actions, include_timestep=directional_time_features,
            ))
            if directional_features
            else encode_coworld_lean_observation
            if lean_features
            else encode_coworld_observation
            if compact_features
            else lambda obs: encode_observation(
                obs, factorized_actions=factorized_actions, goal_features=goal_features
            )
        )
        if move_hint_scale != 1 or split_hint_scale != 1:
            uncalibrated_encode = self._encode

            def calibrated_encode(obs):
                values, mask = uncalibrated_encode(obs)
                return calibrate_hint_features(
                    values,
                    board_size,
                    move_hint_scale=move_hint_scale,
                    split_hint_scale=split_hint_scale,
                ), mask

            self._encode = calibrated_encode
        if mask_pass_when_moves_exist:
            encode = self._encode
            pass_index = (4 if factorized_actions else 8) * board_size**2

            def encode_without_optional_pass(obs):
                values, mask = encode(obs)
                mask = mask.at[pass_index].set(~jnp.any(mask[:pass_index]))
                return values, mask

            self._encode = encode_without_optional_pass
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
        if deduplicate_opponent_branches and opponent != "strong_mixed":
            raise ValueError("Deduplicated opponent branches require strong_mixed")
        branch_agents = (opponent_agents[0], opponent_agents[-1]) if deduplicate_opponent_branches else opponent_agents
        opponent_branches = tuple(lambda args, agent=agent: agent.act(*args) for agent in branch_agents)
        env = self.env

        @jax.jit
        def advance(state, pool, side, opponent_id, index, split, key):
            opponent_key, next_key = jax.random.split(key)
            selector = jnp.where(opponent_id == 3, 1, 0) if deduplicate_opponent_branches else opponent_id
            enemy = jax.lax.switch(
                selector, opponent_branches, (game.get_observation(state, 1 - side), opponent_key)
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
            if land_gain_reward_weight:
                reward += land_gain_reward_weight * (
                    jnp.float32(final.owned_land_count) - jnp.float32(previous.owned_land_count)
                )
            if teacher_agent is not None and imitation_weight:
                suggested = teacher_agent.act(previous, jax.random.fold_in(opponent_key, 37))
                reward = reward + imitation_weight * jnp.all(ours == suggested) * (suggested[0] == 0)
            values, mask = self._encode(final)
            return next_state, next_key, values, mask, reward * reward_scale, done, outcome

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
        sentinel_teacher_only: bool = False, group_device_opponents: bool = False,
        balance_opponent_sides: bool = False, audit_native_actions: bool = False, **options
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
        if group_device_opponents and parallel_games % self.base.num_opponents:
            raise ValueError("Grouped device opponents require complete interleaved opponent groups")
        if balance_opponent_sides and parallel_games % (2 * self.base.num_opponents):
            raise ValueError("Balanced opponent sides require complete pairs for every opponent")
        if balance_opponent_sides and group_device_opponents:
            raise ValueError("Balanced opponent sides are incompatible with interleaved opponent groups")
        self.balance_opponent_sides = balance_opponent_sides
        self.parallel_games = parallel_games
        self.audit_native_actions = audit_native_actions
        if audit_native_actions:
            self._audit_output = context.output / f"native-action-audit-{context.index}.jsonl"
            self._audit_output.parent.mkdir(parents=True, exist_ok=True)
            self._audit_steps = 0
            sizes = jnp.asarray(self.base.spec.action_sizes)
            offsets = jnp.concatenate((jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(sizes)[:-1]))

            @jax.jit
            def validate_actions(actions, masks):
                encoded = jnp.isfinite(actions) & (actions == jnp.floor(actions))
                bounded = (actions >= 0) & (actions < sizes)
                indices = jnp.clip(jnp.nan_to_num(actions), 0, sizes - 1).astype(jnp.int32)
                selected = masks[jnp.arange(parallel_games)[:, None], indices + offsets] != 0
                return jnp.stack((jnp.sum(~encoded), jnp.sum(~bounded), jnp.sum(~selected)))

            self._validate_native_actions = validate_actions
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
            if self.base.training and self.base.supervise_teacher and self.base.prior_hint_features and not self.base.teacher_rollouts:
                teacher_action = hinted_teacher_action_device(values, self.base.size)
            elif self.base.teacher_rollouts or (
                self.base.training and (
                    self.base.supervise_teacher
                    or (self.base.sparse_teacher and not self.base.prior_hint_features)
                )
            ):
                teacher_key = jax.random.fold_in(jax.random.split(next_key)[0], 37)
                teacher_action = self.base._teacher(next_state, side, teacher_key)
            return next_state, next_key, values, mask, reward, done, outcome, teacher_action

        self._advance_states = jax.jit(jax.vmap(advance_one, in_axes=(0, None, 0, 0, 0, 0, 0, 0, 0)))

        def advance_device_one(state, pool, side, opponent_id, index, split, key):
            next_state, next_key, values, mask, reward, done, _ = self.base._advance(
                state, pool, side, opponent_id, index, split, key
            )

            def recycle(_):
                reset_key, following_key = jax.random.split(next_key)
                reset_state = self.base._initial_state(pool, reset_key)
                reset_values, reset_mask = self.base._observe(reset_state, side)
                return reset_state, following_key, reset_values, reset_mask

            next_state, next_key, values, mask = jax.lax.cond(
                done, recycle, lambda _: (next_state, next_key, values, mask), operand=None
            )
            teacher_action = None
            if self.base.supervise_teacher:
                if self.base.prior_hint_features:
                    teacher_action = hinted_teacher_action_device(values, self.base.size)
                else:
                    teacher_key = jax.random.fold_in(jax.random.split(next_key)[0], 37)
                    teacher_action = self.base._teacher(next_state, side, teacher_key)
            return next_state, next_key, values, mask, reward, done, teacher_action

        def advance_device(states, pool, sides, opponent_ids, indices, splits, keys):
            if group_device_opponents:
                # Reset assigns (seed + lane) % num_opponents, so each strided
                # group has a uniform opponent. Keep that switch argument
                # unbatched: the other opponent branches need not execute.
                count = self.base.num_opponents
                groups = tuple(
                    jax.vmap(advance_device_one, in_axes=(0, None, 0, None, 0, 0, 0))(
                        jax.tree.map(lambda field: field[offset::count], states),
                        pool, sides[offset::count], opponent_ids[offset],
                        indices[offset::count], splits[offset::count], keys[offset::count],
                    )
                    for offset in range(count)
                )
                advanced = jax.tree.map(
                    lambda *parts: jnp.stack(parts, axis=1).reshape((parallel_games, *parts[0].shape[1:])),
                    *groups,
                )
            else:
                advanced = jax.vmap(
                    advance_device_one, in_axes=(0, None, 0, 0, 0, 0, 0)
                )(states, pool, sides, opponent_ids, indices, splits, keys)
            next_states, next_keys, values, masks, rewards, done, teacher_actions = advanced
            return (
                next_states, next_keys, self._device_transport(values, masks, teacher_actions),
                masks, rewards, done,
            )

        self._advance_device_states = jax.jit(advance_device)

    def _sentinel_labels(self):
        if not self.sentinel_teacher_games or self.turn % self.sentinel_teacher_interval:
            return None
        count = self.sentinel_teacher_games
        keys = jax.vmap(lambda key: jax.random.fold_in(jax.random.split(key)[0], 37))(self.keys[:count])
        states = jax.tree.map(lambda value: value[:count] if value is not None else None, self.states)
        return np.asarray(self._sentinel_actions(states, self.sides[:count], keys))

    def _observation(self, values, masks, teacher_actions=None, sentinel_actions=None):
        # Queue both device-to-host copies before waiting for either one.
        # The dense Classic observation and action mask otherwise serialize
        # two large transfers on every environment step.
        values.copy_to_host_async()
        masks.copy_to_host_async()
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

    def _reset_states(self, seed: str):
        numeric_seed = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:4], "little")
        if self.base.coworld_classic:
            self._pool_seed = jax.random.PRNGKey(numeric_seed ^ 0xC0A17D)
            self._pool_generation = 0
            self.base.pool, _ = self.base.env.reset(self._pool_seed)
        self.turn = 0
        self.finished[:] = False
        self.outcomes[:] = 0
        self.completed[:] = 0
        indices = np.arange(self.parallel_games, dtype=np.int64)
        self.sides = jnp.asarray((numeric_seed + indices) % 2, dtype=jnp.int32)
        opponent_indices = indices // 2 if self.balance_opponent_sides else indices
        self.opponent_ids = jnp.asarray((numeric_seed + opponent_indices) % self.base.num_opponents, dtype=jnp.int32)
        state_keys = jax.random.split(jax.random.PRNGKey(numeric_seed), self.parallel_games)
        self.keys = jax.random.split(jax.random.PRNGKey(numeric_seed ^ 0xA5A5A5A5), self.parallel_games)
        self.states = self._init_states(self.base.pool, state_keys)
        values, masks = self._observe_states(self.states, self.sides)
        self.cached_teacher_actions = None
        if self.base.teacher_rollouts:
            teacher_keys = jax.vmap(lambda key: jax.random.fold_in(jax.random.split(key)[0], 37))(self.keys)
            self.cached_teacher_actions = self._teacher_actions(self.states, self.sides, teacher_keys)
        return values, masks

    def reset(self, seed: str) -> NumericObservation:
        values, masks = self._reset_states(seed)
        return self._observation(values, masks, self.cached_teacher_actions, self._sentinel_labels())

    def _device_transport(self, values, masks, teacher_actions):
        if not self.base.supervise_teacher:
            return values
        cells = self.base.size**2
        direction = teacher_actions[:, 3] if self.base.factorized_actions else teacher_actions[:, 4] * 4 + teacher_actions[:, 3]
        move = direction * cells + teacher_actions[:, 1] * self.base.size + teacher_actions[:, 2]
        index = jnp.where(teacher_actions[:, 0] == 1, self.spec.action_sizes[0] - 1, move)
        legal = jnp.take_along_axis(masks, index[:, None], axis=1)[:, 0]
        probabilities = jnp.where(legal[:, None], jax.nn.one_hot(index, self.spec.action_sizes[0]), 0)
        weights = [legal.astype(jnp.float32)]
        if self.base.factorized_actions:
            moving = legal & (teacher_actions[:, 0] == 0)
            splits = jnp.where(moving[:, None], jax.nn.one_hot(teacher_actions[:, 4], 2), 0)
            probabilities = jnp.concatenate((probabilities, splits), axis=1)
            weights.append(moving.astype(jnp.float32))
        return jnp.concatenate((
            values, probabilities.astype(jnp.float32), masks.astype(jnp.float32),
            jnp.stack(weights, axis=1), jnp.zeros((values.shape[0], 2), dtype=jnp.float32),
        ), axis=1)

    def reset_device(self, seed: str):
        if not self.base.training or self.base.teacher_rollouts or self.base.sparse_teacher:
            raise ValueError("Device-resident Classic training requires policy rollouts without replay metadata")
        values, masks = self._reset_states(seed)
        if self.audit_native_actions:
            self._audit_masks = masks
        if self.base.supervise_teacher:
            if self.base.prior_hint_features:
                teacher_actions = jax.vmap(lambda row: hinted_teacher_action_device(row, self.base.size))(values)
            else:
                teacher_keys = jax.vmap(lambda key: jax.random.fold_in(jax.random.split(key)[0], 37))(self.keys)
                teacher_actions = self._teacher_actions(self.states, self.sides, teacher_keys)
            values = self._device_transport(values, masks, teacher_actions)
        return values, masks.astype(jnp.uint8)

    def step_device(self, actions):
        if not self.base.training or self.base.teacher_rollouts or self.base.sparse_teacher:
            raise ValueError("Device-resident Classic training requires policy rollouts without replay metadata")
        if self.audit_native_actions:
            expected_shape = (self.parallel_games, len(self.spec.action_sizes))
            if actions.shape != expected_shape:
                raise ValueError(f"Native action shape {actions.shape}; expected {expected_shape}")
            failures = np.asarray(self._validate_native_actions(actions, self._audit_masks))
            self._audit_steps += 1
            if failures.any() or self._audit_steps % 32 == 0:
                record = {
                    "device_steps": self._audit_steps,
                    "agent_actions": self._audit_steps * self.parallel_games,
                    "head_decisions": self._audit_steps * self.parallel_games * len(self.spec.action_sizes),
                    "invalid_encoding": int(failures[0]), "out_of_bounds": int(failures[1]),
                    "masked_actions": int(failures[2]),
                    "scope": "Synchronized native sampled actions against preceding observation masks",
                }
                with self._audit_output.open("a") as output:
                    output.write(json.dumps(record) + "\n")
            if failures.any():
                raise ValueError(f"Native action audit failed: {record}")
        indices = actions[:, 0].astype(jnp.int32)
        splits = actions[:, 1].astype(jnp.int32) if self.base.factorized_actions else jnp.zeros_like(indices)
        self.states, self.keys, values, masks, rewards, done = self._advance_device_states(
            self.states, self.base.pool, self.sides, self.opponent_ids,
            indices, splits, self.keys,
        )
        if self.audit_native_actions:
            self._audit_masks = masks
        self.turn += 1
        if self.turn >= self.horizon:
            self.turn = 0
            if self.base.coworld_classic:
                self._pool_generation += 1
                self.base.pool, _ = self.base.env.reset(
                    jax.random.fold_in(self._pool_seed, self._pool_generation)
                )
        # Individual games already recycle on their own terminal flags. A map
        # pool refresh must not terminate unfinished games or reset their state.
        return values, masks.astype(jnp.uint8), rewards.astype(jnp.float32), done.astype(jnp.float32), False

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
        # Start the large transfers while the small reward and terminal
        # arrays are being read below.
        values.copy_to_host_async()
        masks.copy_to_host_async()
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


class BatchedGeneralsSelfPlayPufferEnvironment(BatchedGeneralsPufferEnvironment):
    """Train the same device policy in both seats of each Classic game."""

    def __init__(self, *, context: EnvironmentContext, parallel_games: int = 16, **options):
        if context.mode != "train":
            raise ValueError("Self-play is a training environment; use the one-seat adapter for evaluation")
        if options.get("teacher_rollouts") or options.get("sparse_teacher"):
            raise ValueError("Self-play requires policy actions in both seats without replay targets")
        if options.get("audit_native_actions"):
            raise ValueError("The one-seat native-action audit does not apply to self-play")
        self._reward_options = {
            name: float(options.get(name, default))
            for name, default in (
                ("shaping_weight", 0.2), ("shaping_gamma", 0.99), ("reward_scale", 1.0),
                ("army_shaping_weight", 0.5), ("land_shaping_weight", 0.3),
                ("castle_shaping_weight", 0.0), ("land_gain_reward_weight", 0.0),
            )
        }
        super().__init__(context=context, parallel_games=parallel_games, **options)
        self.spec = self.base.spec.model_copy(update={"agents": 2 * parallel_games})
        if self.base.supervise_teacher:
            if self.base.prior_hint_features:
                self._self_transport = jax.jit(lambda values, masks, states: self._device_transport(
                    values, masks,
                    jax.vmap(lambda row: hinted_teacher_action_device(row, self.base.size))(values),
                ))
            else:
                def teacher_both(states):
                    return jax.vmap(lambda state: jax.vmap(
                        lambda side: self.base._teacher(state, side, jax.random.PRNGKey(0))
                    )(self._self_sides))(states).reshape((self.spec.agents, 5))

                self._self_transport = jax.jit(lambda values, masks, states: self._device_transport(
                    values, masks, teacher_both(states),
                ))
        self._self_sides = jnp.arange(2, dtype=jnp.int32)
        self._observe_both = jax.jit(jax.vmap(
            lambda state: jax.vmap(lambda side: self.base._observe(state, side))(self._self_sides)
        ))
        env = self.base.env
        encode = self.base._encode
        initial_state = self.base._initial_state
        size = self.base.size
        weights = self._reward_options

        def margin(ours, theirs):
            return (ours - theirs) / (ours + theirs + 1)

        def potential(observation, state, side):
            value = weights["army_shaping_weight"] * margin(
                observation.owned_army_count, observation.opponent_army_count
            )
            value += weights["land_shaping_weight"] * margin(
                observation.owned_land_count, observation.opponent_land_count
            )
            if weights["castle_shaping_weight"]:
                value += weights["castle_shaping_weight"] * _castle_control_margin(state, side)
            return value

        def advance_one(state, pool, indices, splits, key):
            next_key, reset_key = jax.random.split(key)
            actions = jax.vmap(lambda index, split: decode_action(
                index, size, split if self.base.factorized_actions else None
            ))(indices, splits)
            previous = jax.vmap(lambda side: game.get_observation(state, side))(self._self_sides)
            timestep, next_state = env.step(state, actions, pool)
            final = jax.vmap(lambda side: game.get_observation(timestep.last_state, side))(self._self_sides)
            done = timestep.terminated | timestep.truncated

            def side_reward(side, old, new):
                outcome = jnp.where(timestep.terminated, timestep.reward[side], 0.0)
                shaped = weights["shaping_weight"] * (
                    weights["shaping_gamma"] * potential(new, timestep.last_state, side) * ~done
                    - potential(old, state, side)
                )
                land_gain = weights["land_gain_reward_weight"] * (
                    jnp.float32(new.owned_land_count) - jnp.float32(old.owned_land_count)
                )
                return (outcome + shaped + land_gain) * weights["reward_scale"]

            rewards = jax.vmap(side_reward)(self._self_sides, previous, final)

            def recycle(_):
                fresh = initial_state(pool, reset_key)
                return fresh, jax.random.fold_in(next_key, 1)

            next_state, next_key = jax.lax.cond(
                done, recycle, lambda _: (next_state, next_key), operand=None
            )
            values, masks = jax.vmap(lambda side: encode(game.get_observation(next_state, side)))(
                self._self_sides
            )
            return next_state, next_key, values, masks, rewards, done

        self._advance_self_states = jax.jit(jax.vmap(advance_one, in_axes=(0, None, 0, 0, 0)))

    def reset_device(self, seed: str):
        self._reset_states(seed)
        values, masks = self._observe_both(self.states)
        values = values.reshape((self.spec.agents, -1))
        masks = masks.reshape((self.spec.agents, -1))
        if self.base.supervise_teacher:
            values = self._self_transport(values, masks, self.states)
        return values, masks.astype(jnp.uint8)

    def step_device(self, actions):
        expected = (self.spec.agents, len(self.spec.action_sizes))
        if actions.shape != expected:
            raise ValueError(f"Self-play action shape {actions.shape}; expected {expected}")
        paired = actions.reshape((self.parallel_games, 2, len(self.spec.action_sizes))).astype(jnp.int32)
        splits = paired[:, :, 1] if self.base.factorized_actions else jnp.zeros_like(paired[:, :, 0])
        self.states, self.keys, values, masks, rewards, done = self._advance_self_states(
            self.states, self.base.pool, paired[:, :, 0], splits, self.keys
        )
        self.turn += 1
        if self.turn >= self.horizon:
            self.turn = 0
            if self.base.coworld_classic:
                self._pool_generation += 1
                self.base.pool, _ = self.base.env.reset(
                    jax.random.fold_in(self._pool_seed, self._pool_generation)
                )
        values = values.reshape((self.spec.agents, -1))
        masks = masks.reshape((self.spec.agents, -1))
        if self.base.supervise_teacher:
            values = self._self_transport(values, masks, self.states)
        return (
            values,
            masks.astype(jnp.uint8),
            rewards.reshape((self.spec.agents,)).astype(jnp.float32),
            jnp.repeat(done, 2).astype(jnp.float32),
            False,
        )

    def reset(self, seed: str):
        raise NotImplementedError("Self-play uses device-resident training only")

    def step(self, actions):
        raise NotImplementedError("Self-play uses device-resident training only")


class BatchedGeneralsFrozenOpponentPufferEnvironment(BatchedGeneralsSelfPlayPufferEnvironment):
    """Expose one learner seat per game against a frozen actor and optional scripted mix."""

    def __init__(self, *, frozen_build: str, frozen_checkpoint: str, frozen_sha256: str,
                 context: EnvironmentContext, parallel_games: int = 16,
                 scripted_hint_fraction: float = 0.0, frozen_codec: str = "same",
                 frozen_legacy_fabric: str | None = None, **options):
        from metta_training.inference import FrozenPolicy
        from metta_training.model_config import FrozenPolicyConfig
        from metta_training.native_fabric import compile_policy

        factorized = bool(options.get("factorized_actions", False))
        if frozen_codec not in ("same", "hinted_gen0"):
            raise ValueError("Unknown frozen opponent codec")
        if frozen_codec == "hinted_gen0" and (
            factorized or not options.get("directional_features", False)
            or options.get("hint_features", False)
            or frozen_sha256 != "e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf"
        ):
            raise ValueError("Generation-0 opponent requires a pinned flat hint-free learner")
        if not factorized and scripted_hint_fraction:
            raise ValueError("Scripted hint mix requires factorized actions")

        super().__init__(context=context, parallel_games=parallel_games, **options)
        self._frozen_codec = frozen_codec
        self.spec = self.spec.model_copy(update={"agents": parallel_games})
        if scripted_hint_fraction not in (0.0, 0.5) or (scripted_hint_fraction and parallel_games % 4):
            raise ValueError("Scripted hint mix requires half of games in balanced four-game groups")
        self._rows = jnp.arange(parallel_games)
        self._frozen_rows = jnp.arange(parallel_games) if not scripted_hint_fraction else jnp.asarray(
            [row for row in range(parallel_games) if row % 4 < 2], jnp.int32
        )
        self._scripted_rows = jnp.asarray(
            [row for row in range(parallel_games) if row % 4 >= 2], jnp.int32
        ) if scripted_hint_fraction else None
        frozen_slots = parallel_games if not scripted_hint_fraction else parallel_games // 2
        move_scale = options.get("move_hint_scale", 1.0)
        split_scale = options.get("split_hint_scale", 1.0)
        if factorized and (move_scale != split_scale or move_scale not in (0.25, 1.0)):
            raise ValueError("Frozen generation-0 opponent requires the audited hint codecs")
        frozen_hint_gain = 1.0 / move_scale if factorized else 1.0
        frozen_config = FrozenPolicyConfig(
            build=Path(frozen_build), checkpoint=Path(frozen_checkpoint),
            sha256=frozen_sha256, device="cuda:0",
        )
        if frozen_codec == "hinted_gen0":
            import importlib.util
            import sys

            legacy_path = Path(frozen_legacy_fabric) if frozen_legacy_fabric else None
            if legacy_path is None or hashlib.sha256(legacy_path.read_bytes()).hexdigest() != (
                "04d317499de676eb74a91deb2e8b52528c97831d5895a0e1f80b991426238992"
            ):
                raise ValueError("Generation-0 opponent requires its pinned Fabric implementation")
            module_name = "integrations.generals_fabric"
            current_module = sys.modules.get(module_name)
            spec = importlib.util.spec_from_file_location(module_name, legacy_path)
            if spec is None or spec.loader is None:
                raise ValueError("Cannot load pinned generation-0 Fabric implementation")
            legacy_module = importlib.util.module_from_spec(spec)
            try:
                sys.modules[module_name] = legacy_module
                spec.loader.exec_module(legacy_module)
                frozen = FrozenPolicy(frozen_config)
            finally:
                if current_module is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = current_module
        else:
            frozen = FrozenPolicy(frozen_config)
        model = frozen.policy
        expected_frozen_model = (
            "a5a48d16d5c44f057f8c8323b6c531ccce6f06de26a0a47cefe4d68f527de2dd"
            if frozen_codec == "hinted_gen0" else
            "a5a48d16d5c44f057f8c8323b6c531ccce6f06de26a0a47cefe4d68f527de2dd"
            if factorized else
            "c0046141f74f771e8eba6b5296f04913f8736eae6803dab717b90a49fe8b161d"
        )
        if json.loads(Path(frozen_build).read_text())["model_sha256"] != expected_frozen_model:
            raise ValueError("Frozen opponent graph does not match the audited action codec")
        if frozen_codec == "hinted_gen0":
            if model.observation_size != 14 * self.base.size**2 or model.action_sizes != [4 * self.base.size**2 + 1, 2]:
                raise ValueError("Generation-0 opponent has the wrong observation or action layout")
            self._frozen_observe = jax.jit(jax.vmap(
                lambda state, side: encode_coworld_hinted_observation(
                    game.get_observation(state, side),
                    signed_flags=True, expander_hint=True, context_features=True,
                )
            ))
        elif model.observation_size != self.base.spec.observation_size or model.action_sizes != self.base.spec.action_sizes:
            raise ValueError("Frozen opponent model and game codec differ")
        self._frozen_model = model
        self._frozen_function = compile_policy(
            model.graph, inputs=model.inputs, outputs=model.outputs, slots=frozen_slots,
            output_order=model.output_order, standard=model.standard_compile,
        )
        parameters = jnp.asarray(np.frombuffer(frozen.parameters, np.float32).copy())
        state = jnp.zeros((frozen_slots, model.state_words), jnp.float32)
        self._frozen_sigma = model.buffers.unpack_device(parameters, state)

        @jax.jit
        def frozen_actions(values, masks):
            board_values = values[:, :model.observation_size]
            if factorized:
                planes = board_values.reshape(frozen_slots, 14, self.base.size**2)
                board_values = jnp.concatenate((planes[:, :2], planes[:, 2:8] * frozen_hint_gain,
                                                planes[:, 8:]), axis=1).reshape(frozen_slots, -1)
            board = board_values.reshape(frozen_slots, 1, model.observation_size)
            board = board.transpose(1, 2, 0)[..., None]
            board = jnp.pad(board, ((0, 0), (0, model.graph_input_size - model.observation_size),
                                    (0, 0), (0, 0)))
            _, prediction = self._frozen_function.forward(
                self._frozen_sigma, {"observations": board},
                reset=jnp.zeros((1, frozen_slots), bool),
            )
            logits = model.model_predictions_device(self._frozen_function, prediction)[:, 0]
            moves = model.action_sizes[0]
            move = jnp.argmax(jnp.where(masks[:, :moves], logits[:, :moves], -jnp.inf), axis=1)
            if frozen_codec == "hinted_gen0":
                split = jnp.argmax(jnp.where(masks[:, moves:], logits[:, moves:moves + 2], -jnp.inf), axis=1)
                cells = self.base.size**2
                return jnp.where(move == 4 * cells, 8 * cells, move + 4 * cells * split)[:, None].astype(jnp.int32)
            if not factorized:
                return move[:, None].astype(jnp.int32)
            split = jnp.argmax(jnp.where(masks[:, moves:], logits[:, moves:moves + 2], -jnp.inf), axis=1)
            return jnp.stack((move, split), axis=1).astype(jnp.int32)

        self._frozen_actions = frozen_actions
        if scripted_hint_fraction:
            @jax.jit
            def scripted_actions(values):
                action = jax.vmap(lambda row: hinted_teacher_action_device(row, self.base.size))(values)
                cells = self.base.size**2
                index = jnp.where(
                    action[:, 0] == 1, 4 * cells,
                    action[:, 3] * cells + action[:, 1] * self.base.size + action[:, 2],
                )
                return jnp.stack((index, action[:, 4]), axis=1).astype(jnp.int32)
            self._scripted_actions = scripted_actions

    def reset_device(self, seed: str):
        self._reset_states(seed)
        values, masks = self._observe_both(self.states)
        self._cached_values, self._cached_masks = values, masks
        learner_values = values[self._rows, self.sides]
        learner_masks = masks[self._rows, self.sides]
        if self.base.supervise_teacher:
            learner_values = self._self_transport(learner_values, learner_masks, self.states)
        return (
            learner_values,
            learner_masks.astype(jnp.uint8),
        )

    def step_device(self, actions):
        expected = (self.parallel_games, len(self.spec.action_sizes))
        if actions.shape != expected:
            raise ValueError(f"Learner action shape {actions.shape}; expected {expected}")
        values, masks = self._cached_values, self._cached_masks
        frozen_sides = 1 - self.sides
        if self._frozen_codec == "hinted_gen0":
            frozen_states = jax.tree.map(lambda field: field[self._frozen_rows], self.states)
            frozen_values, frozen_masks = self._frozen_observe(
                frozen_states, frozen_sides[self._frozen_rows]
            )
        else:
            frozen_values = values[self._frozen_rows, frozen_sides[self._frozen_rows]]
            frozen_masks = masks[self._frozen_rows, frozen_sides[self._frozen_rows]]
        frozen = self._frozen_actions(frozen_values, frozen_masks)
        opponent = jnp.zeros((self.parallel_games, len(self.spec.action_sizes)), jnp.int32)
        opponent = opponent.at[self._frozen_rows].set(frozen)
        if self._scripted_rows is not None:
            scripted = self._scripted_actions(
                values[self._scripted_rows, frozen_sides[self._scripted_rows]]
            )
            opponent = opponent.at[self._scripted_rows].set(scripted)
        paired = jnp.zeros((self.parallel_games, 2, len(self.spec.action_sizes)), jnp.int32)
        paired = paired.at[self._rows, self.sides].set(actions.astype(jnp.int32))
        paired = paired.at[self._rows, frozen_sides].set(opponent)
        self.states, self.keys, values, masks, rewards, done = self._advance_self_states(
            self.states, self.base.pool, paired[:, :, 0], paired[:, :, 1], self.keys
        )
        self._cached_values, self._cached_masks = values, masks
        self.turn += 1
        if self.turn >= self.horizon:
            self.turn = 0
            if self.base.coworld_classic:
                self._pool_generation += 1
                self.base.pool, _ = self.base.env.reset(
                    jax.random.fold_in(self._pool_seed, self._pool_generation)
                )
        learner_values = values[self._rows, self.sides]
        learner_masks = masks[self._rows, self.sides]
        if self.base.supervise_teacher:
            learner_values = self._self_transport(learner_values, learner_masks, self.states)
        return (
            learner_values,
            learner_masks.astype(jnp.uint8),
            rewards[self._rows, self.sides].astype(jnp.float32),
            done.astype(jnp.float32),
            False,
        )
