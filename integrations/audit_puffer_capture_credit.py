"""One-step capture diagnostic; results cannot establish Classic arena strength.

Uses the real game step, directional observation, full/half/pass codec and legal
mask. Each row resets after one decision on a tiny tactical position embedded
in a 21x21 board. Rewards use the production potential-only terminal formula.
No teacher, forced action, or gradient modification is supplied.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from generals.core import game
from integrations.puffer_codec import decode_action, encode_coworld_directional_observation
from metta_training.environment import EnvironmentSpec


class CaptureCreditEnvironment:
    def __init__(self, context, parallel_games=4096, shaping_gamma=0.999):
        if shaping_gamma != 0.999:
            raise ValueError("Capture diagnostic must match the production PPO discount")
        self.spec = EnvironmentSpec(observation_size=4851, action_sizes=[3529], agents=parallel_games)
        grid = jnp.full((21, 21), -2, jnp.int32).at[10, 10].set(1).at[9, 10].set(2)
        self.state = game.create_initial_state(grid)._replace(
            armies=jnp.zeros((21, 21), jnp.int32).at[10, 10].set(4).at[9, 10].set(2),
        )
        obs = game.get_observation(self.state, 0)
        values, masks = encode_coworld_directional_observation(obs, factorized_actions=False)
        self.values = jnp.broadcast_to(values, (parallel_games, 4851))
        self.masks = jnp.broadcast_to(masks.astype(jnp.uint8), (parallel_games, 3529))
        self.done = jnp.ones((parallel_games,), jnp.float32)
        self.old_potential = (
            0.5 * (obs.owned_army_count - obs.opponent_army_count)
            / (obs.owned_army_count + obs.opponent_army_count + 1)
            + 0.3 * (obs.owned_land_count - obs.opponent_land_count)
            / (obs.owned_land_count + obs.opponent_land_count + 1)
        )
        enemy_pass = jnp.array([1, 0, 0, 0, 0], jnp.int32)

        def transition(index):
            actions = jnp.stack((decode_action(index, 21), enemy_pass))
            state, info = game.step(self.state, actions)
            outcome = jnp.where(info.is_done, jnp.where(info.winner == 0, 1.0, -1.0), 0.0)
            # All rows terminate/truncate after this decision. Therefore the
            # gamma*next-potential term is zero, exactly as in production.
            reward = 0.5 * (outcome - self.old_potential)
            return reward.astype(jnp.float32), info.winner == 0

        self.transition = jax.jit(jax.vmap(transition))
        self.frame = 0
        self.window_wins = jnp.int32(0)
        self.output = Path(context.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.report = self.output / "capture-credit.jsonl"

    def self_check(self):
        legal = jnp.flatnonzero(self.masks[0], size=3)
        assert int(jnp.sum(self.masks[0])) == 3
        assert tuple(map(int, legal)) == (220, 1984, 3528)
        rewards, wins = self.transition(legal)
        assert tuple(map(bool, wins)) == (True, False, False)
        assert float(rewards[0]) > 0 and float(rewards[1]) < 0
        assert float(rewards[1]) == float(rewards[2])
        return {"legal_indices": list(map(int, legal)), "rewards": list(map(float, rewards))}

    def reset_device(self, seed):
        return self.values, self.masks

    def step_device(self, actions):
        rewards, wins = self.transition(actions[:, 0].astype(jnp.int32))
        self.window_wins += jnp.sum(wins, dtype=jnp.int32)
        self.frame += 1
        if self.frame % 32 == 0:
            record = {
                "scope": "one-step capture credit diagnostic, not arena evaluation",
                "device_frames": self.frame,
                "environment_steps": self.frame * self.spec.agents,
                "window_steps": 32 * self.spec.agents,
                "capture_rate": int(self.window_wins) / (32 * self.spec.agents),
            }
            with self.report.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print("CAPTURE_CREDIT " + json.dumps(record), flush=True)
            self.window_wins = jnp.int32(0)
        return self.values, self.masks, rewards, self.done, False

    def close(self):
        pass
