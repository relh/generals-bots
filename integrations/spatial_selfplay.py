"""One learner per Classic game against an immutable portable spatial actor."""

import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from integrations.metta_puffer import BatchedGeneralsSelfPlayPufferEnvironment
from integrations.spatial_policy_bundle import SpatialPlayerPolicy


class SpatialFrozenOpponentPufferEnvironment(BatchedGeneralsSelfPlayPufferEnvironment):
    def __init__(self, *, frozen_bundle, context, parallel_games=4096, **options):
        if options.get("teacher") is not None or options.get("supervise_teacher") or options.get("teacher_rollouts"):
            raise ValueError("Spatial frozen opponents require teacher-free training")
        if not options.get("balance_opponent_sides") or parallel_games % 2:
            raise ValueError("Spatial frozen opponents require balanced seats")
        bundle = Path(frozen_bundle)
        frozen = SpatialPlayerPolicy(bundle)
        source = json.loads((bundle / "build.json").read_text())["config"]["python_environment"]["options"]
        flags = ("factorized_actions", "compact_features", "lean_features", "directional_features",
                 "packed_directional_features", "hint_features", "prior_hint_features", "sprint_hint_features",
                 "expander_hint_features", "context_hint_features", "coworld_classic")
        if any(bool(source.get(k)) != bool(options.get(k)) for k in flags):
            raise ValueError("Frozen opponent and learner public codecs differ")
        super().__init__(context=context, parallel_games=parallel_games, **options)
        if (self.spec.observation_size, tuple(self.spec.action_sizes)) != (4851, (3529,)):
            raise ValueError("Spatial frozen opponents require the public flat codec")
        self.spec = self.spec.model_copy(update={"agents": parallel_games})
        self._rows = jnp.arange(parallel_games)
        self._frozen = frozen

        @jax.jit
        def opposing_actions(values, masks):
            with jax.default_matmul_precision("highest"):
                outputs = frozen._forward(values, jnp)
            return jnp.argmax(jnp.where(masks, outputs[:, :3529], -jnp.inf), axis=1).astype(jnp.int32)

        self._opposing_actions = opposing_actions
        if os.environ.get("METTA_AUDIT_DEVICE_REWARDS") == "1":
            # A built environment can be imported by its source loader before
            # the startup import hook sees it. Install on the actual class.
            from integrations.environment_reward_audit import install

            install(type(self))

    def reset_device(self, seed):
        self._reset_states(seed)
        values, masks = self._observe_both(self.states)
        self._cached_values, self._cached_masks = values, masks
        return values[self._rows, self.sides], masks[self._rows, self.sides].astype(jnp.uint8)

    def step_device(self, actions):
        if actions.shape != (self.parallel_games, 1):
            raise ValueError("Spatial learner requires one flat action per game")
        values, masks = self._cached_values, self._cached_masks
        opposing_sides = 1 - self.sides
        opposing = self._opposing_indices(values, masks)
        paired = jnp.zeros((self.parallel_games, 2), jnp.int32)
        paired = paired.at[self._rows, self.sides].set(actions[:, 0].astype(jnp.int32))
        paired = paired.at[self._rows, opposing_sides].set(opposing)
        self.states, self.keys, values, masks, rewards, done = self._advance_self_states(
            self.states, self.base.pool, paired, jnp.zeros_like(paired), self.keys
        )
        self._cached_values, self._cached_masks = values, masks
        self.turn += 1
        if self.turn >= self.horizon:
            self.turn = 0
            self._pool_generation += 1
            self.base.pool, _ = self.base.env.reset(jax.random.fold_in(self._pool_seed, self._pool_generation))
        return (values[self._rows, self.sides], masks[self._rows, self.sides].astype(jnp.uint8),
                rewards[self._rows, self.sides].astype(jnp.float32), done.astype(jnp.float32), False)

    def _opposing_indices(self, values, masks):
        opposing_sides = 1 - self.sides
        return self._opposing_actions(values[self._rows, opposing_sides], masks[self._rows, opposing_sides])


class SpatialMixedFrozenOpponentPufferEnvironment(SpatialFrozenOpponentPufferEnvironment):
    """Half frozen-policy games, one quarter Expander and one quarter Sentinel.

    Adjacent rows share an opponent type and learn opposite player sides.
    Scripted agents receive their own public observation and supply only the
    opposing action. They never supply learner actions or training targets.
    """

    def __init__(self, *, context, parallel_games=4096, **options):
        if parallel_games % 8:
            raise ValueError("Mixed spatial opponents require complete eight-game seat groups")
        super().__init__(context=context, parallel_games=parallel_games, **options)
        from generals.agents.harvester_agent import ExpanderHarvesterAgent
        from generals.agents.sentinel_agent import SentinelAgent
        from generals.core import game

        self._mix_labels = np.tile(np.asarray((0, 0, 0, 0, 1, 1, 2, 2)), parallel_games // 8)
        self._mix_rows = [
            jnp.asarray(np.flatnonzero(self._mix_labels == label), jnp.int32) for label in range(3)
        ]
        self._scripted_actions = []
        for agent in (ExpanderHarvesterAgent(), SentinelAgent()):
            def indices(states, sides, keys, agent=agent):
                actions = jax.vmap(lambda state, side, key: agent.act(game.get_observation(state, side), key))(
                    states, sides, keys
                )
                moves = (actions[:, 4] * 4 + actions[:, 3]) * 441 + actions[:, 1] * 21 + actions[:, 2]
                return jnp.where(actions[:, 0] != 0, 3528, moves).astype(jnp.int32)

            self._scripted_actions.append(jax.jit(indices))

        def opposing_indices(states, sides, keys, values, masks):
            opponents = jnp.zeros(self.parallel_games, jnp.int32)
            rows = self._mix_rows[0]
            opponents = opponents.at[rows].set(
                self._opposing_actions(values[rows, sides[rows]], masks[rows, sides[rows]])
            )
            for rows, act in zip(self._mix_rows[1:], self._scripted_actions, strict=True):
                selected = jax.tree_util.tree_map(lambda leaf: leaf[rows], states)
                selected_keys = jax.vmap(lambda key: jax.random.fold_in(key, 833))(keys[rows])
                opponents = opponents.at[rows].set(act(selected, sides[rows], selected_keys))
            return opponents

        self._mixed_opposing_indices = jax.jit(opposing_indices)
        self._mix_output = context.output / "spatial-opponent-mix.json"
        self._mix_checkpoint_sha256 = hashlib.sha256(
            Path(options["frozen_bundle"]).joinpath("policy.bin").read_bytes()
        ).hexdigest()

    def reset_device(self, seed):
        result = super().reset_device(seed)
        sides = np.asarray(self.sides)
        counts = {
            label: {str(side): int(np.count_nonzero((self._mix_labels == index) & (sides == side))) for side in (0, 1)}
            for index, label in enumerate(("frozen", "expander_harvester", "sentinel"))
        }
        if any(counts[label]["0"] != counts[label]["1"] or counts[label]["0"] == 0 for label in counts):
            raise ValueError("Mixed spatial opponent seats are not balanced")
        record = dict(
            counts=counts, frozen_checkpoint_sha256=self._mix_checkpoint_sha256, seed=seed,
            episode_limit=self.horizon,
            scope="Opponent actions only; no teacher targets or learner action overrides",
        )
        self._mix_output.parent.mkdir(parents=True, exist_ok=True)
        self._mix_output.write_text(json.dumps(record, indent=2) + "\n")
        print("SPATIAL_OPPONENT_MIX " + json.dumps(record), flush=True)
        return result

    def _opposing_indices(self, values, masks):
        return self._mixed_opposing_indices(
            self.states, 1 - self.sides, self.keys, values, masks
        )
