"""Add castle-control shaping to the pinned v11 strong-opponent environment."""

import hashlib
import shutil
import sys
from pathlib import Path


EXPECTED_ENV = "459e10eab59671c160d11ae8f49a3ffa31e487adf3a606ac7fedc3748745a56d"
EXPECTED_TEST = "574cf9066664d61cae1595d3367e917d8b811bb4f67ed26fb10c16dbe0b0e687"


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("Expected one exact v11 environment block")
    return source.replace(old, new)


def main() -> None:
    source, test_source, destination = map(Path, sys.argv[1:])
    original = source / "integrations" / "metta_puffer.py"
    data = original.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_ENV:
        raise ValueError("Unexpected v11 strong-mix environment source")
    test = test_source / "tests" / "test_metta_puffer.py"
    if hashlib.sha256(test.read_bytes()).hexdigest() != EXPECTED_TEST:
        raise ValueError("Unexpected castle-control test source")
    text = data.decode()
    text = replace_once(
        text,
        "\n\nclass GeneralsPufferEnvironment:\n",
        "\n\ndef _castle_control_margin(state: game.GameState, side: jnp.ndarray) -> jnp.ndarray:\n"
        "    owned = jnp.sum(state.castles & state.ownership[side])\n"
        "    opposing = jnp.sum(state.castles & state.ownership[1 - side])\n"
        "    return (owned - opposing) / (jnp.sum(state.castles) + 1)\n"
        "\n\nclass GeneralsPufferEnvironment:\n",
    )
    text = replace_once(
        text,
        '        shaping_weight: float = 0.2,\n',
        '        shaping_weight: float = 0.2,\n        castle_shaping_weight: float = 0.0,\n',
    )
    text = replace_once(
        text,
        '    ):\n        if imitation_weight < 0 or ((imitation_weight or supervise_teacher or sparse_teacher) and teacher is None):\n',
        '    ):\n        if castle_shaping_weight < 0:\n'
        '            raise ValueError("Castle shaping weight must be nonnegative")\n'
        '        if imitation_weight < 0 or ((imitation_weight or supervise_teacher or sparse_teacher) and teacher is None):\n',
    )
    text = replace_once(
        text,
        '            done = timestep.terminated | timestep.truncated\n'
        '            outcome = jnp.where(timestep.terminated, timestep.reward[side], 0.0)\n'
        '            reward = outcome + shaping_weight * (\n'
        '                0.99 * (0.5 * new_army + 0.3 * new_land) * ~done - (0.5 * old_army + 0.3 * old_land)\n'
        '            )\n',
        '            old_castles = _castle_control_margin(state, side)\n'
        '            new_castles = _castle_control_margin(timestep.last_state, side)\n'
        '            done = timestep.terminated | timestep.truncated\n'
        '            outcome = jnp.where(timestep.terminated, timestep.reward[side], 0.0)\n'
        '            reward = outcome + shaping_weight * (\n'
        '                0.99 * (0.5 * new_army + 0.3 * new_land + castle_shaping_weight * new_castles) * ~done\n'
        '                - (0.5 * old_army + 0.3 * old_land + castle_shaping_weight * old_castles)\n'
        '            )\n',
    )
    shutil.copytree(source, destination)
    patched = destination / "integrations" / "metta_puffer.py"
    patched.write_text(text)
    (destination / "tests" / "test_metta_puffer_castle.py").write_bytes(test.read_bytes())
    print("source_sha256=" + EXPECTED_ENV)
    print("patched_sha256=" + hashlib.sha256(patched.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
