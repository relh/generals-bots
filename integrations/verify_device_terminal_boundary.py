"""Reproduce batch-window terminal flags on valid fresh game states on GPU."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment

parser = argparse.ArgumentParser()
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--expect-artificial-terminals", type=int, choices=(0, 64), default=64)
args = parser.parse_args()
assert jax.devices()[0].platform == "gpu"
args.output.mkdir(parents=True, exist_ok=False)
options = json.loads(args.build.read_text())["config"]["python_environment"]["options"]
options.update(parallel_games=64)
env = BatchedGeneralsPufferEnvironment(
    context=EnvironmentContext(seed=1380, index=0, mode="train", output=args.output), **options
)
try:
    values, masks = env.reset_device("1380:boundary")
    planes = np.asarray(values).reshape(64, 14, 441)
    moves = planes[:, 4:8].reshape(64, 1764).argmax(axis=1)
    moves = np.where(planes[:, 3, 0] > 0, 1764, moves)
    actions = jnp.asarray(np.stack((moves, planes[:, 2, 0] > 0), axis=1), dtype=jnp.int32)
    assert np.asarray(masks)[np.arange(64), moves].all()
    # Fresh states are valid when an individual game has just recycled.
    env.turn = env.horizon - 1
    oracle = env._advance_device_states(
        env.states, env.base.pool, env.sides, env.opponent_ids,
        actions[:, 0], actions[:, 1], env.keys,
    )
    _, _, _, returned_terminals, episode_done = env.step_device(actions)
    actual = np.asarray(oracle[-1], bool)
    returned = np.asarray(returned_terminals, bool)
    assert not actual.any(), "The fresh games must still be unfinished"
    assert all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(jax.tree.leaves(oracle[0]), jax.tree.leaves(env.states), strict=True))
    result = dict(scope="GPU batch-window lifecycle reproduction; no skill or SPS claim",
                  games=64, actual_game_terminals=int(actual.sum()),
                  returned_terminals=int(returned.sum()),
                  artificial_terminals=int((returned & ~actual).sum()),
                  episode_done=bool(episode_done))
    assert result["artificial_terminals"] == args.expect_artificial_terminals
    assert bool(episode_done) == (args.expect_artificial_terminals > 0)
    if args.expect_artificial_terminals == 0:
        assert env._pool_generation == 1
        env.states = env.states._replace(time=jnp.full_like(env.states.time, env.horizon - 1))
        env.turn = 0
        passes = jnp.zeros((64, 2), dtype=jnp.int32).at[:, 0].set(1764)
        _, _, rewards, natural_terminals, natural_episode_done = env.step_device(passes)
        assert np.asarray(natural_terminals, bool).all()
        assert not natural_episode_done
        assert (np.asarray(env.states.time) == 0).all()
        assert np.isfinite(np.asarray(rewards)).all()
        result["natural_terminal_recycles"] = 64
    (args.output / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
finally:
    env.close()
