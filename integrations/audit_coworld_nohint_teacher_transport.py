"""Verify no-hint self-play teacher targets are separate from actor observations."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import BatchedGeneralsSelfPlayPufferEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nohint-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "gpu"
    build = json.loads(args.nohint_build.read_text())
    assert build["config"]["python_environment"]["spec"]["observation_size"] == 8 * 21 * 21
    options = dict(build["config"]["python_environment"]["options"])
    assert not options["hint_features"] and not options["prior_hint_features"]
    options.update(
        teacher="expander_harvester",
        supervise_teacher=True,
        parallel_games=16,
        coworld_pool_size=16,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    environment = BatchedGeneralsSelfPlayPufferEnvironment(
        context=EnvironmentContext(seed=1411, index=0, mode="train", output=args.output),
        **options,
    )
    try:
        transported, masks = environment.reset_device("1411:0:0")
        raw, raw_masks = environment._observe_both(environment.states)
        raw = np.asarray(raw).reshape(32, -1)
        masks = np.asarray(masks)
        transported = np.asarray(transported)
        assert transported.shape == (32, 8 * 21 * 21 + 1767 + 1767 + 2 + 2)
        assert np.array_equal(transported[:, : 8 * 21 * 21], raw)
        assert np.array_equal(masks, np.asarray(raw_masks).reshape(32, -1))
        assert np.array_equal(transported[:, 8 * 21 * 21 + 1767 : -4], masks)
        actions = jax.vmap(
            lambda state: jax.vmap(lambda side: environment.base._teacher(state, side, jax.random.PRNGKey(0)))(
                environment._self_sides
            )
        )(environment.states)
        actions = np.asarray(actions).reshape(32, 5)
        cells = environment.base.size**2
        teacher_index = np.where(
            actions[:, 0] == 1,
            4 * cells,
            actions[:, 3] * cells + actions[:, 1] * environment.base.size + actions[:, 2],
        )
        probabilities = transported[:, 8 * cells : 8 * cells + 1765]
        weights = transported[:, -4:-2]
        assert np.array_equal(np.argmax(probabilities, axis=1), teacher_index)
        assert np.all(weights[:, 0] == 1)
        assert np.array_equal(weights[:, 1], (actions[:, 0] == 0).astype(np.float32))
        assert masks[np.arange(32), teacher_index].all()
        result = {
            "games": 16,
            "learner_seats": 32,
            "actor_observations_equal_public_nohint_codec": True,
            "teacher_targets_match_expander_harvester": True,
            "teacher_actions_legal": True,
        }
        (args.output / "teacher-transport.json").write_text(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
