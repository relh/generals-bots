"""Device-resident Classic training against a verified frozen native generation."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from integrations.metta_puffer import BatchedGeneralsSelfPlayPufferEnvironment
from integrations.native_puffer_policy import NativePufferPolicy


class NativeFrozenOpponentPufferEnvironment(BatchedGeneralsSelfPlayPufferEnvironment):
    """One learner row per game, balanced seats, frozen recurrent opponent."""

    def __init__(self, *, frozen_build, frozen_training, frozen_checkpoint,
                 frozen_sha256, context, parallel_games=4096, **options):
        if options.get("teacher") is not None or options.get("supervise_teacher"):
            raise ValueError("Frozen native generations require teacher-free learner actions")
        # Require the existing public flat codec, including its exact feature flags.
        record = json.loads(Path(frozen_build).read_text())
        source_options = record["config"]["python_environment"]["options"]
        codec_flags = (
            "factorized_actions", "compact_features", "lean_features",
            "directional_features", "packed_directional_features", "hint_features",
            "prior_hint_features", "sprint_hint_features", "expander_hint_features",
            "context_hint_features", "coworld_classic",
        )
        if any(bool(source_options.get(k)) != bool(options.get(k)) for k in codec_flags):
            raise ValueError("Frozen opponent and learner public codecs differ")
        if not options.get("balance_opponent_sides") or parallel_games % 2:
            raise ValueError("Native generations require an even, balanced game batch")
        frozen = NativePufferPolicy(
            Path(frozen_build), Path(frozen_training), Path(frozen_checkpoint), frozen_sha256
        )
        if (frozen.observation_size, frozen.action_sizes) != (4851, (3529,)):
            raise ValueError("Native frozen self-play requires the public flat contract")
        if frozen.hint_prior is not None:
            raise ValueError("Native frozen self-play requires learned policy outputs")
        super().__init__(context=context, parallel_games=parallel_games, **options)
        if (self.spec.observation_size, tuple(self.spec.action_sizes)) != (4851, (3529,)):
            raise ValueError("Learner public dimensions differ from the frozen actor")
        self.spec = self.spec.model_copy(update={"agents": parallel_games})
        self._rows = jnp.arange(parallel_games)
        self._frozen = frozen
        self._frozen_state = frozen.initial_state(parallel_games)

    def reset_device(self, seed):
        self._reset_states(seed)
        self._frozen_state = self._frozen.initial_state(self.parallel_games)
        values, masks = self._observe_both(self.states)
        self._cached_values, self._cached_masks = values, masks
        return values[self._rows, self.sides], masks[self._rows, self.sides].astype(jnp.uint8)

    def step_device(self, actions):
        if actions.shape != (self.parallel_games, 1):
            raise ValueError("Native learner actions must have one flat head per game")
        values, masks = self._cached_values, self._cached_masks
        opposing_sides = 1 - self.sides
        opposing_actions, next_carry = self._frozen.actions(
            values[self._rows, opposing_sides], masks[self._rows, opposing_sides],
            self._frozen_state,
        )
        paired = jnp.zeros((self.parallel_games, 2), jnp.int32)
        paired = paired.at[self._rows, self.sides].set(actions[:, 0].astype(jnp.int32))
        paired = paired.at[self._rows, opposing_sides].set(opposing_actions[:, 0])
        self.states, self.keys, values, masks, rewards, done = self._advance_self_states(
            self.states, self.base.pool, paired, jnp.zeros_like(paired), self.keys
        )
        # The returned public views already belong to recycled states on done rows.
        self._frozen_state = jnp.where(done[None, :, None], 0, next_carry)
        self._cached_values, self._cached_masks = values, masks
        self.turn += 1
        if self.turn >= self.horizon:
            self.turn = 0
            self._pool_generation += 1
            self.base.pool, _ = self.base.env.reset(
                jax.random.fold_in(self._pool_seed, self._pool_generation)
            )
        return (
            values[self._rows, self.sides], masks[self._rows, self.sides].astype(jnp.uint8),
            rewards[self._rows, self.sides].astype(jnp.float32), done.astype(jnp.float32), False,
        )
