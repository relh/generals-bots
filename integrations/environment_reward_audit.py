"""Count actual device rewards and episode ends without changing transitions."""

import atexit
import functools
import json
import importlib.abc
import importlib.machinery
import os
import sys

import jax
import jax.numpy as jnp


@jax.jit
def accumulate(counts, rewards, terminals):
    ended = terminals != 0
    clipped = jnp.isfinite(rewards) & (jnp.abs(rewards) > 1)
    delta = jnp.stack((
        jnp.sum(rewards > 0), jnp.sum(rewards < 0),
        jnp.sum(ended), jnp.sum(ended & (rewards == 0)),
        jnp.sum(~jnp.isfinite(rewards)),
        jnp.sum(clipped), jnp.sum(ended & clipped),
    )).astype(jnp.int32)
    return counts + delta


@jax.jit
def accumulate_flat_splits(counts, actions):
    """Count sampled flat full, half, and pass actions on the device."""
    indices = actions[:, 0].astype(jnp.int32)
    delta = jnp.stack((
        jnp.sum(indices < 1764),
        jnp.sum((indices >= 1764) & (indices < 3528)),
        jnp.sum(indices == 3528),
    )).astype(jnp.uint32)
    return counts + delta


@jax.jit
def accumulate_population_outcomes(counts, rewards, terminals, labels, sides):
    """Count finished games and unambiguous win-only rewards by opponent/seat."""
    group = labels * 2 + sides
    ended = terminals != 0
    wins = ended & (rewards > 0.5)
    delta = jnp.stack((
        jnp.bincount(group, weights=ended.astype(jnp.int32), length=counts.shape[1]),
        jnp.bincount(group, weights=wins.astype(jnp.int32), length=counts.shape[1]),
    ))
    return counts + delta.astype(jnp.int32)


def install(environment_class):
    if getattr(environment_class.step_device, "_generals_reward_audit", False):
        return
    if environment_class.__dict__.get("_reward_audit_installed", False):
        raise RuntimeError("Reward audit already installed")
    original = environment_class.step_device
    audit_flat_splits = os.environ.get("METTA_AUDIT_SPATIAL_SPLITS") == "1"
    audit_population = os.environ.get("METTA_AUDIT_POPULATION_WINS") == "1"

    @functools.wraps(original)
    def step(self, actions):
        result = original(self, actions)
        if not hasattr(self, "_reward_audit_counts"):
            self._reward_audit_counts = jnp.zeros(7, jnp.int32)
            self._reward_audit_ticks = 0
            if audit_flat_splits:
                if tuple(self.spec.action_sizes) != (3529,):
                    raise ValueError("Spatial split audit requires one 3529-action flat head")
                self._split_audit_counts = jnp.zeros(3, jnp.uint32)
            if audit_population:
                weights = self._reward_options
                maximum_terminal_shaping = weights["shaping_weight"] * sum(
                    weights.get(name, 0.0) for name in (
                        "army_shaping_weight", "land_shaping_weight",
                        "castle_shaping_weight", "frontier_shaping_weight",
                    )
                )
                if (self._terminal_reward_mode != "win_only"
                        or weights.get("land_gain_reward_weight", 0.0) != 0
                        or weights["reward_scale"] != 1
                        or maximum_terminal_shaping >= 0.5
                        or not hasattr(self, "_population_labels")):
                    raise ValueError("Population win audit requires bounded win-only rewards")
                self._population_audit_labels = jnp.asarray(self._population_labels, jnp.int32)
                self._population_audit_names = tuple(
                    "frozen_" + digest[:12] for digest in self._population_checksums
                ) + self._population_script_names
                self._population_audit_counts = jnp.zeros((2, 2 * len(self._population_audit_names)), jnp.int32)

            def report():
                counts = list(map(int, jax.device_get(self._reward_audit_counts)))
                record = dict(
                    ticks=self._reward_audit_ticks,
                    agent_steps=self._reward_audit_ticks * self.spec.agents,
                    positive_rewards=counts[0], negative_rewards=counts[1],
                    terminal_agents=counts[2], zero_reward_terminal_agents=counts[3],
                    nonfinite_rewards=counts[4],
                    native_clipped_rewards=counts[5], native_clipped_terminal_rewards=counts[6],
                )
                if audit_flat_splits:
                    full, half, passing = map(int, jax.device_get(self._split_audit_counts))
                    record.update(full_actions=full, half_actions=half, pass_actions=passing)
                    if full + half + passing != record["agent_steps"]:
                        raise ValueError("Spatial split audit did not count every action")
                if audit_population:
                    finished, wins = ([int(value) for value in row]
                                      for row in jax.device_get(self._population_audit_counts))
                    if sum(finished) != record["terminal_agents"] or any(
                        won > ended for won, ended in zip(wins, finished, strict=True)
                    ):
                        raise ValueError("Population terminal audit disagrees with reward audit")
                    record["population_outcomes"] = {
                        name: {
                            str(side): {"finished": finished[2 * index + side],
                                        "wins": wins[2 * index + side]}
                            for side in (0, 1)
                        }
                        for index, name in enumerate(self._population_audit_names)
                    }
                print("DEVICE_REWARD_AUDIT " + json.dumps(record), flush=True)

            self._reward_audit_report = report
            atexit.register(report)
        self._reward_audit_counts = accumulate(self._reward_audit_counts, result[2], result[3])
        if audit_flat_splits:
            self._split_audit_counts = accumulate_flat_splits(self._split_audit_counts, actions)
        if audit_population:
            self._population_audit_counts = accumulate_population_outcomes(
                self._population_audit_counts, result[2], result[3],
                self._population_audit_labels, self.sides,
            )
        self._reward_audit_ticks += 1
        if self._reward_audit_ticks % 512 == 0:
            self._reward_audit_report()
        return result

    step._generals_reward_audit = True
    environment_class.step_device = step
    environment_class._reward_audit_installed = True


def activate():
    """Install in each interpreter, including the native trainer's interpreter."""
    names = {
        "integrations.metta_puffer": "BatchedGeneralsSelfPlayPufferEnvironment",
        "integrations.spatial_selfplay": "SpatialFrozenOpponentPufferEnvironment",
    }
    for name, class_name in names.items():
        if name in sys.modules:
            cls = getattr(sys.modules[name], class_name)
            if not cls.__dict__.get("_reward_audit_installed", False):
                install(cls)
    if any(getattr(finder, "_generals_reward_audit_hook", False) for finder in sys.meta_path):
        return

    class Loader(importlib.abc.Loader):
        def __init__(self, original, class_name):
            self.original = original
            self.class_name = class_name

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            install(getattr(module, self.class_name))

    class Finder(importlib.abc.MetaPathFinder):
        _generals_reward_audit_hook = True

        def find_spec(self, fullname, path=None, target=None):
            if fullname not in names:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                raise ImportError("Generals environment missing")
            spec.loader = Loader(spec.loader, names[fullname])
            return spec

    sys.meta_path.insert(0, Finder())
