"""One learner per Classic game against an immutable portable spatial actor."""

import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from integrations.metta_puffer import BatchedGeneralsSelfPlayPufferEnvironment
from integrations.spatial_action_sampling import (acting_logits, public_doomed_attack_route_penalty,
                                                  public_neutral_route_bonus, public_weak_owned_route_penalty)
from integrations.spatial_policy_bundle import SpatialPlayerPolicy


def frozen_action_indices(policy, outputs, masks, keys, observations=None):
    """Select frozen opponent actions using the bundle's serving contract."""
    if policy.action_mode == "structured_sample":
        logits = acting_logits(outputs, policy.move_temperature, policy.split_temperature, jnp)[:, :3529]
        if getattr(policy, "neutral_route_bias", 0.0):
            if observations is None:
                raise ValueError("Neutral route bias requires frozen public observations")
            logits += public_neutral_route_bonus(observations, policy.neutral_route_bias, jnp)
        if getattr(policy, "weak_owned_route_penalty", 0.0):
            if observations is None:
                raise ValueError("Weak owned route penalty requires frozen public observations")
            logits += public_weak_owned_route_penalty(observations, policy.weak_owned_route_penalty, jnp)
        if getattr(policy, "doomed_attack_route_penalty", 0.0):
            if observations is None:
                raise ValueError("Doomed attack route penalty requires frozen public observations")
            logits += public_doomed_attack_route_penalty(observations, policy.doomed_attack_route_penalty, jnp)
        legal_logits = jnp.where(masks, logits, -jnp.inf)
        random_keys = jax.vmap(lambda key: jax.random.fold_in(key, 834))(keys)
        return jax.vmap(jax.random.categorical)(random_keys, legal_logits).astype(jnp.int32)
    return jnp.argmax(jnp.where(masks, outputs[:, :3529], -jnp.inf), axis=1).astype(jnp.int32)


class SpatialFrozenOpponentPufferEnvironment(BatchedGeneralsSelfPlayPufferEnvironment):
    def __init__(self, *, frozen_bundle, context, parallel_games=4096, **options):
        # Evaluation can load a population training manifest while instantiating
        # this single-opponent wrapper. Pool scheduling is irrelevant here.
        options.pop("opponent_weights", None)
        options.pop("scripted_opponents", None)
        if options.get("teacher") is not None or options.get("supervise_teacher") or options.get("teacher_rollouts"):
            raise ValueError("Spatial frozen opponents require teacher-free training")
        if not options.get("balance_opponent_sides") or parallel_games % 2:
            raise ValueError("Spatial frozen opponents require balanced seats")
        bundle = Path(frozen_bundle)
        frozen = SpatialPlayerPolicy(bundle)
        source = json.loads((bundle / "build.json").read_text())["config"]["python_environment"]["options"]
        flags = ("factorized_actions", "compact_features", "lean_features", "directional_features",
                 "directional_time_features", "packed_directional_features", "hint_features", "prior_hint_features", "sprint_hint_features",
                 "expander_hint_features", "context_hint_features", "coworld_classic")
        if any(bool(source.get(k)) != bool(options.get(k)) for k in flags):
            raise ValueError("Frozen opponent and learner public codecs differ")
        super().__init__(context=context, parallel_games=parallel_games, **options)
        if self.spec.observation_size not in (4851, 5292, 7056) or tuple(self.spec.action_sizes) != (3529,):
            raise ValueError("Spatial frozen opponents require the public flat codec")
        if frozen.observation_size != self.spec.observation_size and not (
            frozen.channels == 11 and options.get("public_scalar_features")
        ):
            raise ValueError("Frozen opponent must use the same codec or the preserved eleven-channel prefix")
        if frozen.channels == 16 and options.get("public_scalar_ablation") and not frozen.public_scalar_ablation:
            raise ValueError("A full-scalar frozen opponent requires unabbreviated scalar observations")
        self.spec = self.spec.model_copy(update={"agents": parallel_games})
        self._rows = jnp.arange(parallel_games)
        self._frozen = frozen
        self._public_scalar_features = bool(options.get("public_scalar_features"))
        self._public_scalar_ablation = bool(options.get("public_scalar_ablation"))

        @jax.jit
        def opposing_actions(values, masks, keys):
            with jax.default_matmul_precision("highest"):
                outputs = frozen._forward(values[:, :frozen.observation_size], jnp)
            return frozen_action_indices(frozen, outputs, masks, keys, values)

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
        return self._opposing_actions(
            values[self._rows, opposing_sides], masks[self._rows, opposing_sides], self.keys[self._rows]
        )


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
                self._opposing_actions(values[rows, sides[rows]], masks[rows, sides[rows]], keys[rows])
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
            coworld_classic_rules=self.base.env.coworld_classic_rules,
            observation_size=self.spec.observation_size,
            public_scalar_features=self._public_scalar_features,
            public_scalar_ablation=self._public_scalar_ablation,
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


class SpatialPopulationOpponentPufferEnvironment(SpatialFrozenOpponentPufferEnvironment):
    """Balanced Classic games against frozen policies and selected scripts."""

    def __init__(self, *, frozen_bundles, context, parallel_games=4096,
                 opponent_weights=None,
                 scripted_opponents=("expander_harvester", "sentinel"), **options):
        bundles = tuple(map(Path, frozen_bundles))
        script_names = tuple(scripted_opponents)
        available_scripts = ("expander_harvester", "sentinel", "sentinel_v5")
        if (not script_names or len(set(script_names)) != len(script_names)
                or any(name not in available_scripts for name in script_names)):
            raise ValueError("Population scripts must be distinct known opponent names")
        if len(bundles) < 2 or bundles[0] != Path(options.get("frozen_bundle", "")):
            raise ValueError("Population requires at least two bundles, starting with frozen_bundle")
        if parallel_games < 2 * (len(bundles) + len(script_names)) or parallel_games % 2:
            raise ValueError("Population needs a pair of games per opponent")
        super().__init__(context=context, parallel_games=parallel_games, **options)
        if not self.base.coworld_classic or not self.base.env.coworld_classic_rules:
            raise ValueError("Population opponents require official Coworld Classic rules")
        from generals.agents.harvester_agent import ExpanderHarvesterAgent
        from generals.agents.sentinel_agent import SentinelAgent
        from generals.agents.sentinel_v5_agent import SentinelV5Agent
        from generals.core import game

        frozen = (self._frozen,) + tuple(SpatialPlayerPolicy(path) for path in bundles[1:])
        flags = ("factorized_actions", "compact_features", "lean_features", "directional_features",
                 "directional_time_features", "packed_directional_features", "hint_features",
                 "prior_hint_features", "sprint_hint_features", "expander_hint_features",
                 "context_hint_features", "coworld_classic")
        for path, policy in zip(bundles[1:], frozen[1:], strict=True):
            source = json.loads((path / "build.json").read_text())["config"]["python_environment"]["options"]
            if any(bool(source.get(key)) != bool(options.get(key)) for key in flags):
                raise ValueError("Population frozen policy codec differs: " + str(path))
            if policy.observation_size != self.spec.observation_size and not (
                policy.channels == 11 and self._public_scalar_features
            ):
                raise ValueError("Population frozen policy observation size differs: " + str(path))
            if policy.channels == 16 and self._public_scalar_ablation and not policy.public_scalar_ablation:
                raise ValueError("Population frozen policy needs visible scalar features")

        checksums = tuple(hashlib.sha256((path / "policy.bin").read_bytes()).hexdigest() for path in bundles)
        if len(set(checksums)) != len(checksums):
            raise ValueError("Population frozen policies must be distinct")
        count = len(frozen) + len(script_names)
        weights = tuple(opponent_weights) if opponent_weights is not None else (1,) * count
        if len(weights) != count or any(not isinstance(weight, int) or weight <= 0 for weight in weights):
            raise ValueError("Population opponent weights must be positive integers, one per opponent")
        if parallel_games // 2 < sum(weights):
            raise ValueError("Population needs at least one full weighted opponent cycle")
        weighted_labels = np.repeat(np.arange(count, dtype=np.int32), weights)
        labels = np.repeat(np.resize(weighted_labels, parallel_games // 2), 2)
        self._population_labels = labels
        self._population_weights = weights
        self._population_rows = tuple(
            jnp.asarray(np.flatnonzero(labels == label), jnp.int32) for label in range(count)
        )
        self._population_checksums = checksums
        self._population_action_selection = tuple(
            ({"mode": policy.action_mode, "move_temperature": policy.move_temperature,
              "split_temperature": policy.split_temperature,
              "neutral_route_bias": policy.neutral_route_bias,
              **({"weak_owned_route_penalty": policy.weak_owned_route_penalty}
                 if policy.weak_owned_route_penalty else {}),
              **({"doomed_attack_route_penalty": policy.doomed_attack_route_penalty}
                 if policy.doomed_attack_route_penalty else {})}
             if policy.action_mode == "structured_sample" else {"mode": "argmax"})
            for policy in frozen
        )

        def scripted_indices(states, sides, keys, agent):
            actions = jax.vmap(lambda state, side, key: agent.act(game.get_observation(state, side), key))(
                states, sides, keys
            )
            moves = (actions[:, 4] * 4 + actions[:, 3]) * 441 + actions[:, 1] * 21 + actions[:, 2]
            return jnp.where(actions[:, 0] != 0, 3528, moves).astype(jnp.int32)

        script_factories = {"expander_harvester": ExpanderHarvesterAgent,
                            "sentinel": SentinelAgent, "sentinel_v5": SentinelV5Agent}
        scripts = tuple(script_factories[name]() for name in script_names)
        self._population_script_names = script_names

        def opposing_indices(states, sides, keys, values, masks):
            result = jnp.zeros(parallel_games, jnp.int32)
            for rows, policy in zip(self._population_rows[:len(frozen)], frozen, strict=True):
                with jax.default_matmul_precision("highest"):
                    logits = policy._forward(values[rows, sides[rows], :policy.observation_size], jnp)
                chosen = frozen_action_indices(policy, logits, masks[rows, sides[rows]], keys[rows],
                                               values[rows, sides[rows]])
                result = result.at[rows].set(chosen)
            for rows, agent in zip(self._population_rows[len(frozen):], scripts, strict=True):
                selected = jax.tree_util.tree_map(lambda leaf: leaf[rows], states)
                selected_keys = jax.vmap(lambda key: jax.random.fold_in(key, 833))(keys[rows])
                result = result.at[rows].set(scripted_indices(selected, sides[rows], selected_keys, agent))
            return result

        self._population_opposing_indices = jax.jit(opposing_indices)
        self._population_output = context.output / "spatial-opponent-population.json"

    def reset_device(self, seed):
        result = super().reset_device(seed)
        sides = np.asarray(self.sides)
        names = tuple("frozen_" + digest[:12] for digest in self._population_checksums) + self._population_script_names
        counts = {
            name: {str(side): int(np.count_nonzero((self._population_labels == index) & (sides == side)))
                   for side in (0, 1)}
            for index, name in enumerate(names)
        }
        if any(item["0"] != item["1"] or item["0"] == 0 for item in counts.values()):
            raise ValueError("Population opponent seats are not balanced")
        record = dict(counts=counts, opponent_weights=self._population_weights,
                      frozen_policy_sha256=self._population_checksums,
                      frozen_action_selection=self._population_action_selection, seed=seed,
                      episode_limit=self.horizon, coworld_classic_rules=self.base.env.coworld_classic_rules,
                      observation_size=self.spec.observation_size,
                      scope="Opponent actions only; no teacher targets or learner action overrides")
        self._population_output.parent.mkdir(parents=True, exist_ok=True)
        self._population_output.write_text(json.dumps(record, indent=2) + "\n")
        print("SPATIAL_OPPONENT_POPULATION " + json.dumps(record), flush=True)
        return result

    def _opposing_indices(self, values, masks):
        return self._population_opposing_indices(
            self.states, 1 - self.sides, self.keys, values, masks
        )
