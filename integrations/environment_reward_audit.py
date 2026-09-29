"""Count actual device rewards and episode ends without changing transitions."""

import atexit
import functools
import json

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
