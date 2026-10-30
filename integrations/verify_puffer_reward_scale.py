"""Verify constant reward scaling on paired, frozen-policy GPU trajectories."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import jax
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment
from integrations.native_puffer_policy import NativePufferPolicy


def main():
    parser = argparse.ArgumentParser()
    for name in ("build", "training", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--seed", type=int, default=1373)
    parser.add_argument("--games", type=int, default=256)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "gpu"
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.build.read_text())
    options = manifest["config"]["python_environment"]["options"].copy()
    options.update(parallel_games=args.games, coworld_pool_size=256,
                   army_shaping_weight=.5, land_shaping_weight=.3, castle_shaping_weight=0.)
    assert options["shaping_weight"] == 1.0 and options["shaping_gamma"] == .999
    assert options["imitation_weight"] == 0.0 and not options["teacher_rollouts"]
    # |potential| <= .8: terminal |reward| <= 1.8, nonterminal <= 1.6.
    # Multiplying the complete reward by .5 preserves the objective and avoids the native clamp.
    bound = .5 * max(1 + .8, (1 + options["shaping_gamma"]) * .8)
    policy = NativePufferPolicy(args.build, args.training, args.checkpoint, args.sha256)
    state = policy.initial_state(args.games)
    environments = []
    for scale in (1., .5):
        output = args.output / f"scale-{scale:g}"
        output.mkdir()
        context = EnvironmentContext(seed=args.seed, index=0, mode="evaluate", output=output)
        environments.append(BatchedGeneralsPufferEnvironment(
            context=context, reward_scale=scale, **options))
    raw_clips = scaled_clips = active_steps = 0
    max_scaled = 0.
    try:
        observations = [env.reset(f"{args.seed}:0:0") for env in environments]
        np.testing.assert_array_equal(observations[0].values, observations[1].values)
        np.testing.assert_array_equal(observations[0].action_masks, observations[1].action_masks)
        for turn in range(environments[0].horizon):
            actions, state = policy.actions(
                np.asarray(observations[0].values), np.asarray(observations[0].action_masks), state)
            active = ~environments[0].finished.copy()
            transitions = [env.step(np.asarray(actions, dtype=np.int32)) for env in environments]
            raw, scaled = [np.asarray(t.rewards, dtype=np.float32)[active] for t in transitions]
            assert np.isfinite(raw).all() and np.isfinite(scaled).all()
            np.testing.assert_allclose(scaled, .5 * raw, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(transitions[0].terminated, transitions[1].terminated)
            np.testing.assert_array_equal(environments[0].outcomes, environments[1].outcomes)
            for first, second in zip(jax.tree.leaves(environments[0].states),
                                     jax.tree.leaves(environments[1].states), strict=True):
                np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
            active_steps += int(raw.size)
            raw_clips += int((np.abs(raw) > 1).sum())
            scaled_clips += int((np.abs(scaled) > 1).sum())
            max_scaled = max(max_scaled, float(np.abs(scaled).max()))
            assert max_scaled <= bound + 1e-6
            observations = [t.observation for t in transitions]
            np.testing.assert_array_equal(observations[0].values, observations[1].values)
            np.testing.assert_array_equal(observations[0].action_masks, observations[1].action_masks)
            if transitions[0].episode_done:
                assert transitions[1].episode_done and environments[0].finished.all()
                break
        assert environments[0].finished.all() and scaled_clips == 0 and raw_clips > 0
    finally:
        for env in environments:
            env.close()
    result = dict(scope="Paired frozen GPU trajectories; no training SPS or skill claim",
                  checkpoint_sha256=args.sha256, seed=args.seed, games=args.games, turns=turn + 1,
                  active_steps=active_steps, raw_clipped_steps=raw_clips, scaled_clipped_steps=scaled_clips,
                  maximum_absolute_scaled_reward=max_scaled, analytic_bound=bound,
                  identical_states_observations_masks_terminals_outcomes=True,
                  adapter_sha256=hashlib.sha256(
                      Path(inspect.getfile(BatchedGeneralsPufferEnvironment)).read_bytes()).hexdigest())
    (args.output / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
