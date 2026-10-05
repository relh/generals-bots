"""One learner per Classic game against an immutable portable spatial actor."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from integrations.metta_puffer import BatchedGeneralsSelfPlayPufferEnvironment
from integrations.spatial_frozen_sampling import frozen_action_indices
from integrations.spatial_policy_bundle import SpatialPlayerPolicy


class SpatialFrozenOpponentPufferEnvironment(BatchedGeneralsSelfPlayPufferEnvironment):
    def __init__(self, *, frozen_bundle, context, parallel_games=4096, **options):
        # Evaluation can load a population training manifest while instantiating
        # this single-opponent wrapper. Pool scheduling is irrelevant here.
        options.pop("opponent_weights", None)
        options.pop("scripted_opponents", None)
        options.pop("classic_siege_workers", None)
        if not options.get("balance_opponent_sides") or parallel_games % 2:
            raise ValueError("Spatial frozen opponents require balanced seats")
        bundle = Path(frozen_bundle)
        frozen = SpatialPlayerPolicy(bundle)
        if frozen.channels != 16:
            raise ValueError("Frozen opponents require the current sixteen-plane public codec")
        super().__init__(context=context, parallel_games=parallel_games, **options)
        self.spec = self.spec.model_copy(update={"agents": parallel_games})
        self._rows = jnp.arange(parallel_games)
        self._frozen = frozen
        self._public_scalar_features = True
        self._public_scalar_ablation = False

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
        self._reset_completed_opponents(done)
        self.turn += 1
        if self.turn >= self.horizon:
            self.turn = 0
            self._pool_generation += 1
            self.base.pool, _ = self.base.env.reset(jax.random.fold_in(self._pool_seed, self._pool_generation))
        return (values[self._rows, self.sides], masks[self._rows, self.sides].astype(jnp.uint8),
                rewards[self._rows, self.sides].astype(jnp.float32), done.astype(jnp.float32), False)

    def _reset_completed_opponents(self, done):
        """Stateless opponents need no episode cleanup."""

    def _opposing_indices(self, values, masks):
        opposing_sides = 1 - self.sides
        return self._opposing_actions(
            values[self._rows, opposing_sides], masks[self._rows, opposing_sides], self.keys[self._rows]
        )


class SpatialPopulationOpponentPufferEnvironment(SpatialFrozenOpponentPufferEnvironment):
    """Balanced Classic games against frozen policies and selected scripts."""

    def __init__(self, *, frozen_bundles, context, parallel_games=4096,
                 opponent_weights=None, classic_siege_workers=1,
                 scripted_opponents=("expander_harvester", "sentinel"), **options):
        bundles = tuple(map(Path, frozen_bundles))
        script_names = tuple(scripted_opponents)
        available_scripts = ("expander_harvester", "sentinel", "classic_siege_padded")
        if (not script_names or len(set(script_names)) != len(script_names)
                or any(name not in available_scripts for name in script_names)):
            raise ValueError("Population scripts must be distinct known opponent names")
        if len(bundles) < 2 or bundles[0] != Path(options.get("frozen_bundle", "")):
            raise ValueError("Population requires at least two bundles, starting with frozen_bundle")
        if parallel_games < 2 * (len(bundles) + len(script_names)) or parallel_games % 2:
            raise ValueError("Population needs a pair of games per opponent")
        super().__init__(context=context, parallel_games=parallel_games, **options)
        if not self.base.env.coworld_classic_rules:
            raise ValueError("Population opponents require official Coworld Classic rules")
        from generals.agents.harvester_agent import ExpanderHarvesterAgent
        from generals.agents.sentinel_agent import SentinelAgent
        from generals.core import game

        frozen = (self._frozen,) + tuple(SpatialPlayerPolicy(path) for path in bundles[1:])
        if any(policy.channels != 16 for policy in frozen):
            raise ValueError("Population policies require the current sixteen-plane public codec")

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
              **({"log_gap_scale": policy.log_gap_scale} if policy.log_gap_scale else {}),
              **({"full_action_temperature": policy.full_action_temperature}
                 if policy.full_action_temperature != 1 else {}),
              **({"route_half_weight": policy.route_half_weight}
                 if policy.route_half_weight else {}),
              **({"early_route_temperature": policy.early_route_temperature,
                  "early_route_turns": policy.early_route_turns}
                 if policy.early_route_temperature is not None else {}),
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
                            "sentinel": SentinelAgent}
        scripts = tuple(script_factories[name]() if name in script_factories else None for name in script_names)
        self._population_script_names = script_names
        self._siege_rows = None
        self._native_opponent_contract = None
        native_siege = None
        if "classic_siege_padded" in script_names:
            from integrations.classic_siege_native import SOURCE, ClassicSiegeBatch, compile_library, padded_device_actions

            context.output.mkdir(parents=True, exist_ok=True)
            self._siege_build_directory = tempfile.TemporaryDirectory(prefix="native-siege-", dir=context.output)
            native_siege = ClassicSiegeBatch(
                compile_library(Path(self._siege_build_directory.name) / "opponent.so"),
                workers=classic_siege_workers,
            )
            self._siege_rows = self._population_rows[len(frozen) + script_names.index("classic_siege_padded")]
            self._native_opponent_contract = dict(
                name="classic_siege_padded", source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                workers=native_siege.workers,
                observation="Public wire grids padded to 21x21", frontier_tie_break="row-major",
                memory="Explicit arrays reset on every completed episode, including curriculum resets",
                execution="One pure batched CPU callback; end-to-end GPU throughput requires qualification",
            )
        self._siege_memory = jnp.full((0 if self._siege_rows is None else len(self._siege_rows), 3), -1, jnp.int32)

        def opposing_indices(states, sides, keys, values, masks, siege_memory):
            result = jnp.zeros(parallel_games, jnp.int32)
            for rows, policy in zip(self._population_rows[:len(frozen)], frozen, strict=True):
                with jax.default_matmul_precision("highest"):
                    logits = policy._forward(values[rows, sides[rows], :policy.observation_size], jnp)
                chosen = frozen_action_indices(policy, logits, masks[rows, sides[rows]], keys[rows],
                                               values[rows, sides[rows]])
                result = result.at[rows].set(chosen)
            for rows, agent in zip(self._population_rows[len(frozen):], scripts, strict=True):
                selected = jax.tree_util.tree_map(lambda leaf: leaf[rows], states)
                if agent is None:
                    observations = jax.vmap(game.get_observation)(selected, sides[rows])
                    chosen, siege_memory = padded_device_actions(native_siege, observations, siege_memory)
                    result = result.at[rows].set(chosen)
                else:
                    selected_keys = jax.vmap(lambda key: jax.random.fold_in(key, 833))(keys[rows])
                    result = result.at[rows].set(scripted_indices(selected, sides[rows], selected_keys, agent))
            return result, siege_memory

        self._population_opposing_indices = jax.jit(opposing_indices)
        self._population_output = context.output / "spatial-opponent-population.json"

    def reset_device(self, seed):
        result = super().reset_device(seed)
        self._siege_memory = jnp.full_like(self._siege_memory, -1)
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
        if self._native_opponent_contract is not None:
            record["native_opponent"] = self._native_opponent_contract
        self._population_output.parent.mkdir(parents=True, exist_ok=True)
        self._population_output.write_text(json.dumps(record, indent=2) + "\n")
        print("SPATIAL_OPPONENT_POPULATION " + json.dumps(record), flush=True)
        return result

    def _opposing_indices(self, values, masks):
        result, self._siege_memory = self._population_opposing_indices(
            self.states, 1 - self.sides, self.keys, values, masks, self._siege_memory
        )
        return result

    def _reset_completed_opponents(self, done):
        if self._siege_rows is not None:
            from integrations.classic_siege_native import reset_completed_memory

            self._siege_memory = reset_completed_memory(self._siege_memory, done[self._siege_rows])
