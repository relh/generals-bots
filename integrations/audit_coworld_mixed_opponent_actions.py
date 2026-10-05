"""Check balanced seats and legal scripted actions in the frozen/scripted mix."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from metta_training.environment import EnvironmentContext

from generals.agents.harvester_agent import ExpanderHarvesterAgent
from generals.core import game
from integrations.metta_puffer import BatchedGeneralsFrozenOpponentPufferEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-build", type=Path, required=True)
    parser.add_argument("--frozen-build", type=Path, required=True)
    parser.add_argument("--frozen-checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "gpu"
    manifest = json.loads(args.learner_build.read_text())
    options = dict(manifest["config"]["python_environment"]["options"])
    options.update(
        parallel_games=16, coworld_pool_size=16, supervise_teacher=False,
        deduplicate_opponent_branches=False, scripted_hint_fraction=0.5,
        frozen_build=str(args.frozen_build), frozen_checkpoint=str(args.frozen_checkpoint),
        frozen_sha256=args.frozen_sha256,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    environment = BatchedGeneralsFrozenOpponentPufferEnvironment(
        context=EnvironmentContext(seed=1402, index=0, mode="train", output=args.output),
        **options,
    )
    try:
        learner_values, learner_masks = environment.reset_device("1402:0:0")
        assert learner_values.shape[0] == learner_masks.shape[0] == environment.spec.agents == 16
        frozen_rows = np.asarray(environment._frozen_rows)
        scripted_rows = np.asarray(environment._scripted_rows)
        sides = np.asarray(environment.sides)
        assert len(frozen_rows) == len(scripted_rows) == 8
        assert set(frozen_rows) | set(scripted_rows) == set(range(16))
        assert np.bincount(sides[frozen_rows], minlength=2).tolist() == [4, 4]
        assert np.bincount(sides[scripted_rows], minlength=2).tolist() == [4, 4]
        values, masks = environment._observe_both(environment.states)
        opposing_sides = 1 - environment.sides
        scripted_values = values[environment._scripted_rows, opposing_sides[environment._scripted_rows]]
        scripted_masks = masks[environment._scripted_rows, opposing_sides[environment._scripted_rows]]
        actions = np.asarray(environment._scripted_actions(scripted_values))
        legality = np.asarray(scripted_masks)[np.arange(8), actions[:, 0]]
        assert legality.all()
        agent = ExpanderHarvesterAgent()
        scripted_states = jax.tree.map(lambda leaf: leaf[environment._scripted_rows], environment.states)
        originals = jax.vmap(lambda state, side: agent.act(
            game.get_observation(state, side), jax.random.PRNGKey(0)
        ))(scripted_states,
           opposing_sides[environment._scripted_rows])
        originals = np.asarray(originals)
        cells = environment.base.size**2
        expected_moves = np.where(
            originals[:, 0] == 1, 4 * cells,
            originals[:, 3] * cells + originals[:, 1] * environment.base.size + originals[:, 2],
        )
        assert np.array_equal(actions[:, 0], expected_moves)
        assert np.array_equal(actions[:, 1], originals[:, 4])
        result = {
            "games": 16,
            "frozen_games": 8,
            "scripted_games": 8,
            "frozen_side_counts": [4, 4],
            "scripted_side_counts": [4, 4],
            "scripted_actions_legal": True,
            "scripted_actions_match_expander_harvester": True,
        }
        (args.output / "actions.json").write_text(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
