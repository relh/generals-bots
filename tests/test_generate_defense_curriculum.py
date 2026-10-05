import hashlib
import json

import jax.numpy as jnp
import numpy as np
import pytest

from generals.core import coworld_game as engine
from integrations.classic_position_curriculum import configure_positions, load_positions
from integrations.generate_defense_curriculum import _seed, audit_position, construct_position, generate


def source():
    grid = jnp.zeros((21, 21), jnp.int32).at[4, 4].set(1).at[18, 18].set(2)
    return engine.create_initial_state(grid)


@pytest.mark.parametrize("seat", [0, 1])
def test_credible_attack_has_legal_visible_defense_and_lost_control(seat):
    state, attack, witness = construct_position(source(), 0, 3, variant=7)
    lost, _, _ = construct_position(source(), 0, 3, variant=7, lost=True)
    if seat:
        state = state._replace(ownership=state.ownership[::-1], general_positions=state.general_positions[::-1])
        lost = lost._replace(ownership=lost.ownership[::-1], general_positions=lost.general_positions[::-1])
    obs, _, actions, survivors, audit = audit_position(state, attack, seat)
    assert audit["pass_loses_capital"] and audit["defendable"]
    assert np.asarray(obs.opponent_cells).any()
    matching = np.all(np.asarray(actions) == np.asarray(witness), axis=1)
    assert matching.any() and survivors[matching].all()
    assert not audit_position(lost, attack, seat)[4]["defendable"]


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError, match="distinct open neighbors"):
        construct_position(source(), 0, 0, variant=0)
    blocked = source()._replace(passable=source().passable.at[3, 4].set(False))
    with pytest.raises(ValueError, match="distinct open neighbors"):
        construct_position(blocked, 0, 3, variant=0)


def test_generated_archive_is_paired_independent_and_binds_teacher_free_reset(tmp_path):
    output = tmp_path / "generated"
    excluded = [_seed(7000101, 0)]
    manifest, report = generate(output, seed=7000101, pairs=1, excluded_seeds=excluded)
    assert manifest["count"] == 2 and manifest["teacher_labels_enabled"] is False
    assert manifest["provenance"][0]["map_seed"] not in excluded
    assert manifest["provenance"][0]["map_sha256"] == manifest["provenance"][1]["map_sha256"]
    states = load_positions(output / "positions.npz", manifest["positions_sha256"])
    np.testing.assert_array_equal(states.ownership[0], states.ownership[1][::-1])
    assert hashlib.sha256((output / "positions.npz").read_bytes()).hexdigest() == manifest["positions_sha256"]
    options = {"coworld_classic": True, "teacher": None}
    configure_positions(options, output / "manifest.json")
    assert options["coworld_position_probability"] == 0.25
    assert not any(row["competent_on_this_probe"] for row in report["teacher_admission"].values())
    assert json.loads((output / "manifest.json").read_text())["root_training_seed"] == 7000101
    with pytest.raises(ValueError, match="new directory"):
        generate(output, seed=7000101, pairs=1)
