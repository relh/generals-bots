"""One learner per Classic game against an immutable portable spatial actor."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp

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
        opposing = self._opposing_actions(values[self._rows, opposing_sides], masks[self._rows, opposing_sides])
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
