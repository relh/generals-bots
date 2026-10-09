"""Immutable midgame starts for teacher-free Classic reinforcement learning."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from generals.core.game import GameState

ENGINE_SHA256 = "f39e448a6b2822869d75cb07cce4cb43d589c4112fef04007ade951809d4a318"


def configure_positions(options, manifest_path):
    """Bind the bounded 25% training experiment to verified, versioned inputs."""
    from integrations.classic_contract import validate_environment_options

    projected = validate_environment_options(options)
    if projected != options:
        raise ValueError("Curriculum configuration requires current Classic runtime options")
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "classic-midgame-positions-v1"
        or manifest.get("engine_sha256") != ENGINE_SHA256
    ):
        raise ValueError("Position curriculum requires verified teacher-free Classic inputs")
    path = manifest_path.parent / "positions.npz"
    positions = load_positions(path, manifest["positions_sha256"])
    if len(positions.time) != manifest.get("count") or len(manifest.get("provenance", [])) != len(positions.time):
        raise ValueError("Position curriculum provenance count differs")
    configured_path = options.get("coworld_position_pool")
    if (
        isinstance(configured_path, str)
        and Path(configured_path).is_absolute()
        and Path(configured_path).resolve() == path.resolve()
    ):
        pool_path = configured_path
    else:
        pool_path = str(path.resolve())
    options.update(
        coworld_position_pool=pool_path,
        coworld_position_pool_sha256=manifest["positions_sha256"],
        coworld_position_probability=0.25,
    )
    return dict(
        positions=len(positions.time),
        probability=0.25,
        positions_sha256=manifest["positions_sha256"],
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        heldout_initial_positions="fresh maps only",
    )


def pad_position(state):
    """Use the training engine's bottom/right mountain padding for a hosted board."""
    height, width = state.armies.shape
    if not 18 <= height <= 21 or not 18 <= width <= 21:
        raise ValueError("Position is not an official Classic board shape")
    grids = {"armies", "ownership", "ownership_neutral", "generals", "castles", "mountains", "passable"}
    fields = {}
    for key, value in state._asdict().items():
        array = np.asarray(value)
        if key in grids:
            padding = [(0, 0)] * (array.ndim - 2) + [(0, 21 - height), (0, 21 - width)]
            array = np.pad(array, padding, constant_values=key == "mountains")
        fields[key] = array
    return GameState(**fields)


def load_positions(path, expected_sha256):
    path = Path(path)
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("Classic position archive exceeds the bounded input size")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Classic position archive checksum differs")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(GameState._fields):
            raise ValueError("Classic position archive fields differ")
        arrays = {key: archive[key] for key in GameState._fields}
    count = len(arrays["time"])
    if not 1 <= count <= 4096:
        raise ValueError("Classic position count must be between 1 and 4096")
    shapes = dict(
        ownership=(2, 21, 21), general_positions=(2, 2), teams=(2,), eliminated=(2,), time=(), winner=(), pool_idx=()
    )
    integers = {"armies", "general_positions", "teams", "time", "winner", "pool_idx"}
    for key, value in arrays.items():
        shape = (count, *shapes.get(key, (21, 21)))
        dtype = np.dtype("int32" if key in integers else "bool")
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"Classic position shape or dtype differs: {key}")
    owned = arrays["ownership"]
    generals = arrays["generals"]
    if (
        np.any(arrays["armies"] < 0)
        or np.any(arrays["time"] <= 0)
        or np.any(arrays["time"] >= 2000)
        or np.any(arrays["winner"] != -1)
        or arrays["eliminated"].any()
        or not np.all(arrays["teams"] == [0, 1])
        or np.any(owned.sum(axis=1) > 1)
        or not np.array_equal(arrays["ownership_neutral"], arrays["passable"] & ~owned.any(axis=1))
        or not np.array_equal(arrays["passable"], ~arrays["mountains"])
        or np.any(owned & ~arrays["passable"][:, None])
        or not np.all(generals.sum(axis=(1, 2)) == 2)
        or not np.all((owned & generals[:, None]).sum(axis=(2, 3)) == 1)
    ):
        raise ValueError("Classic positions must be valid, live two-player states")
    positions = arrays["general_positions"]
    if np.any(positions < 0) or np.any(positions >= 21):
        raise ValueError("Classic general coordinates are outside the board")
    for side in (0, 1):
        rows, cols = positions[:, side].T
        if not np.all(generals[np.arange(count), rows, cols] & owned[np.arange(count), side, rows, cols]):
            raise ValueError("Classic general coordinates disagree with ownership")
    # The original replay's map-pool index is not part of its game position.
    arrays["pool_idx"] = np.zeros(count, np.int32)
    return GameState(**{key: jnp.asarray(value) for key, value in arrays.items()})


def mix_initial_positions(initial_state, positions, probability):
    """Sample a position on reset; no stored action, target, or teacher is used."""
    if not np.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("Position curriculum probability must be in (0, 1]")
    probability = float(probability)

    def initial(pool, key):
        # Preserve the baseline map/key draw even when taking a curriculum start.
        fresh = initial_state(pool, key)
        choose = jax.random.bernoulli(jax.random.fold_in(key, 77101), probability)
        index = jax.random.randint(jax.random.fold_in(key, 77102), (), 0, len(positions.time))
        return jax.tree.map(lambda normal, saved: jnp.where(choose, saved[index], normal), fresh, positions)

    return initial
