"""Count actual device rewards and episode ends without changing transitions."""

import atexit
import functools
import json
import importlib.abc
import importlib.machinery
import sys

import jax
import jax.numpy as jnp


@jax.jit
def accumulate(counts, rewards, terminals):
    ended = terminals != 0
    delta = jnp.stack((
        jnp.sum(rewards > 0), jnp.sum(rewards < 0),
        jnp.sum(ended), jnp.sum(ended & (rewards == 0)),
        jnp.sum(~jnp.isfinite(rewards)),
    )).astype(jnp.int32)
    return counts + delta


def install(environment_class):
    if getattr(environment_class, "_reward_audit_installed", False):
        raise RuntimeError("Reward audit already installed")
    original = environment_class.step_device

    @functools.wraps(original)
    def step(self, actions):
        result = original(self, actions)
        if not hasattr(self, "_reward_audit_counts"):
            self._reward_audit_counts = jnp.zeros(5, jnp.int32)
            self._reward_audit_ticks = 0

            def report():
                counts = list(map(int, jax.device_get(self._reward_audit_counts)))
                print("DEVICE_REWARD_AUDIT " + json.dumps(dict(
                    ticks=self._reward_audit_ticks,
                    agent_steps=self._reward_audit_ticks * self.spec.agents,
                    positive_rewards=counts[0], negative_rewards=counts[1],
                    terminal_agents=counts[2], zero_reward_terminal_agents=counts[3],
                    nonfinite_rewards=counts[4],
                )), flush=True)

            self._reward_audit_report = report
            atexit.register(report)
        self._reward_audit_counts = accumulate(self._reward_audit_counts, result[2], result[3])
        self._reward_audit_ticks += 1
        if self._reward_audit_ticks % 512 == 0:
            self._reward_audit_report()
        return result

    environment_class.step_device = step
    environment_class._reward_audit_installed = True


def activate():
    """Install in each interpreter, including the native trainer's interpreter."""
    name = "integrations.metta_puffer"
    if name in sys.modules:
        cls = sys.modules[name].BatchedGeneralsSelfPlayPufferEnvironment
        if not getattr(cls, "_reward_audit_installed", False):
            install(cls)
        return

    class Loader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            install(module.BatchedGeneralsSelfPlayPufferEnvironment)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != name:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                raise ImportError("Generals environment missing")
            spec.loader = Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, Finder())
