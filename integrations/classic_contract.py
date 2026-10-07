"""Shared, dependency-free qualification contract for hosted Classic policies.

Training curricula may use other shapes, but cannot qualify as hosted Classic.
Validation inspects effective build/run configurations without importing JAX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

ENGINE_SHA256 = "f39e448a6b2822869d75cb07cce4cb43d589c4112fef04007ade951809d4a318"
ENGINE_COMMIT = "0fcb5a00226387670624d2f326f6d5ad61914584"
CLASSIC_MAP_OPTIONS = {
    "min_grid_size": 18,
    "max_grid_size": 21,
    "pad_to": 21,
    "truncation": 2000,
    "mountain_density_range": (0.24, 0.26),
    "min_generals_distance": 17,
    "num_castles_range": (9, 11),
    "castle_val_range": (40, 51),
    "build_castles": False,
    "deathtouch_turn": None,
    "coworld_classic_rules": True,
    "dynamic_pool": True,
}


CURRENT_ENVIRONMENT_FIELDS = {
    "parallel_games",
    "coworld_pool_size",
    "horizon",
    "require_gpu",
    "shaping_weight",
    "shaping_gamma",
    "reward_scale",
    "army_shaping_weight",
    "land_shaping_weight",
    "terminal_reward_mode",
    "monotone_force_potential",
    "balance_opponent_sides",
    "coworld_position_pool",
    "coworld_position_pool_sha256",
    "coworld_position_probability",
    "frozen_bundle",
    "frozen_bundles",
    "opponent_weights",
    "scripted_opponents",
    "classic_siege_workers",
}
def validate_environment_options(options: dict) -> dict:
    """Validate the single current runtime schema and copy mutable values."""
    if not isinstance(options, dict):
        raise ValueError("Environment options must be an object")
    unknown = set(options) - CURRENT_ENVIRONMENT_FIELDS
    if unknown:
        raise ValueError("Unsupported environment option fields: " + ", ".join(sorted(unknown)))
    return deepcopy(options)


def verify_engine(path: Path | None = None) -> str:
    path = path or Path(__file__).resolve().parents[1] / "generals/core/coworld_game.py"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != ENGINE_SHA256:
        raise ValueError(f"Classic engine source differs from pinned official source: {actual}")
    return actual


def validate_training_contract(build: dict, run: dict, *, engine_path: Path | None = None) -> dict:
    """Reject game, discount, and rollout mismatches before GPU work."""
    env = build["python_environment"]
    options = validate_environment_options(env["options"])
    overrides = run["overrides"]
    if options.get("horizon") != 2000:
        raise ValueError("Hosted qualification requires a 2000-turn game limit")
    gamma = overrides.get("train.gamma")
    if isinstance(gamma, bool) or not isinstance(gamma, (float, int)) or not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("Explicit finite learner train.gamma in (0, 1] is required")
    if options.get("shaping_gamma") != gamma:
        raise ValueError("Explicit environment shaping_gamma must equal learner train.gamma")
    names = ("parallel_games", "horizon", "minibatch")
    values = (options.get("parallel_games"), overrides.get("train.horizon"), overrides.get("train.minibatch_size"))
    geometry = dict(zip(names, values))
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("Rollout geometry must contain positive integers")
    if overrides.get("vec.total_agents") != values[0] or env["spec"].get("agents") != values[0]:
        raise ValueError("Environment, vector, and build agent counts differ")
    geometry["steps_per_epoch"] = values[0] * values[1]
    if geometry["steps_per_epoch"] % values[2]:
        raise ValueError("Minibatch must divide the complete rollout")
    if options.get("balance_opponent_sides") is not True:
        raise ValueError("Qualification requires balanced opponent sides")
    # Shape is a codec contract, not evidence of rule parity; engine is separately verified.
    fabric = build["fabric"]
    if fabric.get("observation_size") != env["spec"].get("observation_size") or fabric.get("action_sizes") != env[
        "spec"
    ].get("action_sizes"):
        raise ValueError("Policy and environment observation/action dimensions differ")
    return {
        "contract": "coworld-classic-v1",
        "engine_sha256": verify_engine(engine_path),
        "map_options": CLASSIC_MAP_OPTIONS,
        "shaping_gamma": options["shaping_gamma"],
        "learner_gamma": gamma,
        "training_geometry": geometry,
        "balance_opponent_sides": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_training_contract(json.loads(args.build.read_text()), json.loads(args.run.read_text()))
    data = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
