"""Real-game corridor diagnostic for multi-step native PPO credit.

Eight layouts balance the player sides and four directions. The opponent
passes. This fixture is not an arena evaluation or a publishable training run.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from generals.core import game
from integrations.puffer_codec import decode_action, encode_coworld_directional_observation
from metta_training.environment import EnvironmentSpec


class SequenceCreditEnvironment:
    def __init__(self, context, parallel_games=4096, shaping_gamma=0.999,
                 distance=4, turn_limit=16):
        if shaping_gamma != 0.999 or not 2 <= distance <= 8 or turn_limit < distance:
            raise ValueError("Invalid diagnostic geometry or production discount")
        if parallel_games % 8:
            raise ValueError("Balance eight direction-by-side layouts")
        self.spec = EnvironmentSpec(observation_size=4851, action_sizes=[3529], agents=parallel_games)
        self.sides = jnp.arange(parallel_games, dtype=jnp.int32) % 2
        self.directions = (jnp.arange(parallel_games, dtype=jnp.int32) // 2) % 4
        self.distance = distance
        self.turn_limit = turn_limit
        self.gamma = shaping_gamma

        def initial(side, direction):
            delta = game.DIRECTIONS[direction]
            grid = jnp.full((21, 21), -2, jnp.int32)
            for offset in range(distance + 1):
                position = jnp.array([10, 10]) + offset * delta
                grid = grid.at[position[0], position[1]].set(0)
            end = jnp.array([10, 10]) + distance * delta
            grid = grid.at[10, 10].set(side + 1).at[end[0], end[1]].set(2 - side)
            state = game.create_initial_state(grid)
            return state._replace(armies=state.armies.at[10, 10].set(distance + 8).at[end[0], end[1]].set(2))

        self.initial = jax.jit(jax.vmap(initial))(self.sides, self.directions)
        self.states = self.initial

        def encode(state, side):
            return encode_coworld_directional_observation(game.get_observation(state, side), factorized_actions=False)

        self.encode = jax.jit(jax.vmap(encode))

        def potential(state, side):
            obs = game.get_observation(state, side)
            return (0.5 * (obs.owned_army_count - obs.opponent_army_count)
                    / (obs.owned_army_count + obs.opponent_army_count + 1)
                    + 0.3 * (obs.owned_land_count - obs.opponent_land_count)
                    / (obs.owned_land_count + obs.opponent_land_count + 1))

        def advance(state, original, side, index):
            actions = jnp.tile(jnp.array([1, 0, 0, 0, 0], jnp.int32), (2, 1))
            actions = actions.at[side].set(decode_action(index, 21))
            next_state, info = game.step(state, actions)
            done = info.is_done | (next_state.time >= turn_limit)
            win = info.is_done & (info.winner == side)
            outcome = jnp.where(info.is_done, jnp.where(win, 1.0, -1.0), 0.0)
            reward = 0.5 * (outcome + shaping_gamma * potential(next_state, side) * ~done
                            - potential(state, side))
            # Production contract: reward/done belong to the transition while
            # returned observations and masks belong to the recycled state.
            recycled = jax.lax.cond(done, lambda: original, lambda: next_state)
            values, masks = encode(recycled, side)
            return recycled, values, masks.astype(jnp.uint8), reward.astype(jnp.float32), done, win

        self.advance = jax.jit(jax.vmap(advance))
        self.frame = 0
        self.window_wins = jnp.int32(0)
        self.window_episodes = jnp.int32(0)
        self.output = Path(context.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.report = self.output / "sequence-credit.jsonl"

    def reset_device(self, seed):
        self.states = self.initial
        values, masks = self.encode(self.states, self.sides)
        return values, masks.astype(jnp.uint8)

    def step_device(self, actions):
        self.states, values, masks, rewards, done, wins = self.advance(
            self.states, self.initial, self.sides, actions[:, 0].astype(jnp.int32))
        self.window_wins += jnp.sum(wins, dtype=jnp.int32)
        self.window_episodes += jnp.sum(done, dtype=jnp.int32)
        self.frame += 1
        if self.frame % 32 == 0:
            episodes = int(self.window_episodes)
            record = {
                "scope": "multi-step capture diagnostic, not arena evaluation",
                "device_frames": self.frame,
                "environment_steps": self.frame * self.spec.agents,
                "window_steps": 32 * self.spec.agents,
                "episodes": episodes,
                "wins": int(self.window_wins),
                "capture_rate": int(self.window_wins) / episodes if episodes else None,
                "distance": self.distance, "turn_limit": self.turn_limit,
            }
            with self.report.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print("SEQUENCE_CREDIT " + json.dumps(record), flush=True)
            self.window_wins = jnp.int32(0)
            self.window_episodes = jnp.int32(0)
        return values, masks, rewards, done.astype(jnp.float32), False

    def self_check(self):
        self.reset_device(0)
        for offset in range(self.distance):
            coordinates = jnp.array([10, 10]) + offset * game.DIRECTIONS[self.directions]
            indices = self.directions * 441 + coordinates[:, 0] * 21 + coordinates[:, 1]
            _, masks = self.encode(self.states, self.sides)
            assert bool(jnp.all(masks[jnp.arange(self.spec.agents), indices]))
            _, _, rewards, done, _ = self.step_device(indices[:, None])
            assert bool(jnp.all(jnp.isfinite(rewards)))
            assert bool(jnp.all(done == (offset == self.distance - 1)))
        assert int(self.window_wins) == self.spec.agents
        assert bool(jnp.all(self.states.time == 0))
        # Passing must truncate with the same reset-observation contract.
        for offset in range(self.turn_limit):
            _, _, _, done, _ = self.step_device(jnp.full((self.spec.agents, 1), 3528))
            assert bool(jnp.all(done == (offset == self.turn_limit - 1)))
        assert int(self.window_wins) == self.spec.agents
        self.reset_device(0)
        self.frame = 0
        self.window_wins = self.window_episodes = jnp.int32(0)
        return {"layouts": 8, "full_move_capture_steps": self.distance,
                "pass_truncation_steps": self.turn_limit}

    def close(self):
        pass
