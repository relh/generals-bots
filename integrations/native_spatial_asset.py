"""Current native spatial weights and optional authentic learner state.

Assets carry model/ABI identity and opaque ancestor hashes, never synthetic RL
training records. Source cleanup copies bytes after a separate equivalence proof.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

from integrations.classic_contract import ENGINE_SHA256
from integrations.learner_checkpoint import LearnerCheckpoint

SCHEMA = "generals-native-spatial-asset-v1"
FIELDS = {
    "schema",
    "fabric",
    "factory_source_sha256",
    "model_sha256",
    "abi_sha256",
    "parameter_count",
    "policy_sha256",
    "learner_sha256",
    "sampler",
    "provenance",
    "training_seeds",
    "learner_configuration",
    "training_contract",
}
FACTORY = "integrations.generals_fabric:two_stage_tied_local_action_policy"
BRIDGE_SHA256 = "c1bed03201af5133badfe8c5fa1566efc3830acc73c68798fbf5b7f7d7e051c1"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def check_digest(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("Asset requires full SHA-256 identities")


def validate_fabric(fabric):
    if (
        fabric["factory"] != FACTORY
        or fabric["observation_size"] != 7056
        or fabric["action_sizes"] != [3529]
        or fabric["compiler"] != "standard"
    ):
        raise ValueError("Asset requires the current standard 16-plane flat spatial model")
    options = fabric["options"]
    expected = {
        "channels",
        "height",
        "width",
        "features_per_site",
        "global_features",
        "context_radius",
        "route_prior_strength",
        "source_army_prior_strength",
        "half_prior_scale",
        "full_split_prior_strength",
    }
    if (
        set(options) != expected
        or options["channels"] != 16
        or options["height"] != 21
        or options["width"] != 21
        or options["features_per_site"] != 32
        or options["global_features"] != 32
        or options["context_radius"] not in (1.01, 2.01)
    ):
        raise ValueError("Asset architecture options differ from the qualified current model")
    for key in ("route_prior_strength", "source_army_prior_strength", "full_split_prior_strength"):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Asset requires finite positive current priors")
    scale = options["half_prior_scale"]
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or not 0 <= scale <= 1
    ):
        raise ValueError("Invalid half-army prior scale")


def validate_sampler(sampler):
    required = {"mode", "move_temperature", "split_temperature"}
    optional = {
        "neutral_route_bias",
        "weak_owned_route_penalty",
        "doomed_attack_route_penalty",
        "early_route_temperature",
        "early_route_turns",
        "route_half_weight",
        "full_action_temperature",
        "log_gap_scale",
    }
    if (
        not isinstance(sampler, dict)
        or sampler.get("mode") != "structured_sample"
        or not required <= sampler.keys() <= required | optional
    ):
        raise ValueError("Asset requires the current explicit structured sampler")
    for key in ("move_temperature", "split_temperature", "full_action_temperature", "early_route_temperature"):
        if key in sampler:
            value = sampler[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid sampler temperature")
    for key in (
        "neutral_route_bias",
        "weak_owned_route_penalty",
        "doomed_attack_route_penalty",
        "route_half_weight",
        "log_gap_scale",
    ):
        if key in sampler:
            value = sampler[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid sampler route setting")
    if sampler.get("route_half_weight", 0) > 1:
        raise ValueError("Invalid route half weight")
    if ("early_route_temperature" in sampler) != ("early_route_turns" in sampler):
        raise ValueError("Early sampler schedule requires temperature and turns together")
    if "early_route_turns" in sampler and (
        type(sampler["early_route_turns"]) is not int or sampler["early_route_turns"] <= 0
    ):
        raise ValueError("Early sampler turns must be a positive integer")


def training_contract(options, overrides):
    """Effective current game/codec/reward objective, excluding distributions."""
    reward = {
        key: options[key]
        for key in (
            "terminal_reward_mode",
            "reward_scale",
            "shaping_weight",
            "shaping_gamma",
            "army_shaping_weight",
            "land_shaping_weight",
        )
    }
    contract = {
        "schema": "generals-classic-training-contract-v1",
        "engine_sha256": ENGINE_SHA256,
        "episode_horizon": options.get("horizon", 2000),
        "observation_size": 7056,
        "action_sizes": [3529],
        "codec": "classic-directional-scalars-v1",
        "reward": reward,
        "learner_gamma": overrides["train.gamma"],
    }
    validate_objective(contract)
    return contract


def validate_objective(contract):
    if (
        set(contract)
        != {
            "schema",
            "engine_sha256",
            "episode_horizon",
            "observation_size",
            "action_sizes",
            "codec",
            "reward",
            "learner_gamma",
        }
        or contract["schema"] != "generals-classic-training-contract-v1"
        or contract["engine_sha256"] != ENGINE_SHA256
        or contract["episode_horizon"] != 2000
        or contract["observation_size"] != 7056
        or contract["action_sizes"] != [3529]
        or contract["codec"] != "classic-directional-scalars-v1"
    ):
        raise ValueError("Learner asset requires the canonical Classic game and codec")
    reward = contract["reward"]
    numeric = {"reward_scale", "shaping_weight", "shaping_gamma", "army_shaping_weight", "land_shaping_weight"}
    if set(reward) != numeric | {"terminal_reward_mode"} or reward["terminal_reward_mode"] != "win_only":
        raise ValueError("Learner asset requires the current Classic win objective")
    for key in numeric:
        value = reward[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Learner asset reward settings must be finite and nonnegative")
    gamma = contract["learner_gamma"]
    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, (int, float))
        or not 0 <= gamma <= 1
        or reward["shaping_gamma"] != gamma
        or reward["reward_scale"] <= 0
    ):
        raise ValueError("Learner asset shaping and learner discounts must agree")


@dataclass(frozen=True)
class NativeSpatialAsset:
    metadata: dict
    policy: bytes
    learner: bytes | None

    def verify_target(self, *, factory_source_sha256, model_sha256, abi_sha256):
        for key, expected in [
            ("factory_source_sha256", factory_source_sha256),
            ("model_sha256", model_sha256),
            ("abi_sha256", abi_sha256),
        ]:
            check_digest(expected)
            if self.metadata[key] != expected:
                raise ValueError("Native asset target identity differs: " + key)


def load_asset(path: Path, *, manifest_sha256: str) -> NativeSpatialAsset:
    path = Path(path)
    check_digest(manifest_sha256)
    raw = path.read_bytes()
    if sha256(raw) != manifest_sha256:
        raise ValueError("Native asset manifest checksum differs")
    metadata = json.loads(raw)
    if set(metadata) != FIELDS or metadata["schema"] != SCHEMA:
        raise ValueError("Require the current native spatial asset schema")
    for key in ("factory_source_sha256", "model_sha256", "abi_sha256", "policy_sha256"):
        check_digest(metadata[key])
    validate_fabric(metadata["fabric"])
    validate_sampler(metadata["sampler"])
    provenance = metadata["provenance"]
    if provenance["operation"] not in ("source_cleanup", "supervised", "reinforcement_learning", "policy_ablation"):
        raise ValueError("Unknown native asset provenance operation")
    ancestors = provenance["ancestors"]
    if not isinstance(ancestors, dict) or not ancestors:
        raise ValueError("Native asset must retain hashed opaque ancestors")
    for label, value in ancestors.items():
        if not isinstance(label, str) or not label:
            raise ValueError("Ancestor names must be nonempty strings")
        check_digest(value)
    steps = provenance["reinforcement_learning_steps_added"]
    if type(steps) is not int or steps < 0 or (provenance["operation"] != "reinforcement_learning" and steps != 0):
        raise ValueError("Non-training provenance operations add zero RL steps")
    if provenance["operation"] == "source_cleanup":
        check_digest(provenance["abi_proof_sha256"])
    count = metadata["parameter_count"]
    expected_count = 570668 if metadata["fabric"]["options"]["context_radius"] == 1.01 else 578860
    if type(count) is not int or count != expected_count:
        raise ValueError("Native asset parameter count differs from its current model layout")
    policy = path.with_name("policy.bin").read_bytes()
    if (
        sha256(policy) != metadata["policy_sha256"]
        or len(policy) != count * 4
        or not all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", policy))
    ):
        raise ValueError("Native asset weights must match the complete finite float32 vector")
    seeds = metadata["training_seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Asset must retain explicit unique nonnegative training seeds")
    configuration, objective = metadata["learner_configuration"], metadata["training_contract"]
    learner = None
    if metadata["learner_sha256"] is not None:
        if (
            not isinstance(configuration, dict)
            or set(configuration) != {"seed", "overrides"}
            or type(configuration["seed"]) is not int
            or configuration["seed"] < 0
            or configuration["seed"] not in seeds
            or not isinstance(configuration["overrides"], dict)
            or not isinstance(objective, dict)
        ):
            raise ValueError("Learner asset requires explicit actual configuration and training contract")
        for key, value in configuration["overrides"].items():
            if (
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, (str, int, float))
                or (isinstance(value, float) and not math.isfinite(value))
            ):
                raise ValueError("Learner override fields must be explicit finite current settings")
        validate_objective(objective)
        if objective["learner_gamma"] != configuration["overrides"]["train.gamma"]:
            raise ValueError("Learner configuration and training contract discounts differ")
        check_digest(metadata["learner_sha256"])
        learner = path.with_name("policy.bin.learner").read_bytes()
        if sha256(learner) != metadata["learner_sha256"]:
            raise ValueError("Native asset learner checksum differs")
        LearnerCheckpoint.from_bytes(learner, count)
    elif configuration is not None or objective is not None:
        raise ValueError("Policy-only assets must declare no learner configuration or restore contract")
    return NativeSpatialAsset(metadata, policy, learner)


def write_asset(
    output: Path,
    *,
    fabric,
    factory_source_sha256,
    model_sha256,
    abi_sha256,
    policy: Path,
    sampler,
    provenance,
    training_seeds,
    learner: Path | None = None,
    learner_configuration=None,
    training_contract=None,
) -> Path:
    """Author a new immutable asset; original policy/optimizer files are untouched."""
    output = Path(output)
    metadata = dict(
        schema=SCHEMA,
        fabric=fabric,
        factory_source_sha256=factory_source_sha256,
        model_sha256=model_sha256,
        abi_sha256=abi_sha256,
        parameter_count=Path(policy).stat().st_size // 4,
        policy_sha256=sha256(Path(policy).read_bytes()),
        learner_sha256=sha256(Path(learner).read_bytes()) if learner else None,
        sampler=sampler,
        provenance=provenance,
        training_seeds=training_seeds,
        learner_configuration=learner_configuration,
        training_contract=training_contract,
    )
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(policy, output / "policy.bin")
    if learner:
        shutil.copyfile(learner, output / "policy.bin.learner")
    manifest = output / "asset.json"
    manifest.write_bytes(canonical_json(metadata) + b"\n")
    load_asset(manifest, manifest_sha256=sha256(manifest.read_bytes()))
    return manifest


def abi_descriptor(policy):
    """Source-independent complete flat layout and realized template stamp.

    Source cleanup additionally needs independent callback/logit/gradient proof;
    the target's actual factory/runtime fingerprint binds executable semantics.
    """
    import jax
    import numpy as np

    from integrations.direct_spatial_optimization import DirectSpatial
    from integrations.spatial_optimizer_layout import logical_optimizer_shapes

    def array(value):
        dtype = str(value.dtype) if hasattr(value, "dtype") else None
        if hasattr(value, "dtype") and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            value = jax.random.key_data(value)
        data = np.asarray(value)
        return {
            "shape": list(data.shape),
            "dtype": dtype or str(data.dtype),
            "sha256": sha256(np.ascontiguousarray(data).tobytes()),
        }

    buffers = policy.buffers
    model = DirectSpatial(policy)
    maps = {}
    for name in (
        "input_kernel",
        "context_kernel",
        "action_kernel",
        "local_weight",
        "local_bias",
        "context_weight",
        "context_bias",
        "global_weight",
        "global_bias",
        "global_kernel",
        "readout_kernel",
        "output_weight",
        "output_bias",
    ):
        maps[name] = array(getattr(model, name))
    for i, (source, weights) in enumerate(model.priors):
        maps[f"prior_source_{i}"] = array(source)
        maps[f"prior_weight_{i}"] = array(weights)
    parameters = [
        {
            "path": jax.tree_util.keystr(path),
            "shape": list(block.shape),
            "offset": block.offset,
            "size": block.size,
            "padding": block.padding,
        }
        for (path, _), block in zip(
            jax.tree_util.tree_flatten_with_path(buffers.fn.params(buffers.template))[0],
            buffers.parameters,
            strict=True,
        )
    ]
    template = {section: value for section, value in buffers.template.items() if section != "params"}
    leaves = [
        {"path": jax.tree_util.keystr(path), **array(value)}
        for path, value in jax.tree_util.tree_flatten_with_path(template)[0]
    ]
    optimizer_shapes, optimizer_blocks = logical_optimizer_shapes(model, buffers, context_matrix=True)
    return {
        "schema": "generals-native-spatial-abi-v1",
        "bridge_sha256": BRIDGE_SHA256,
        "observation_size": model.observation_size,
        "output_size": 3530,
        "parameter_words": buffers.parameter_words,
        "state_words": buffers.state_words,
        "inputs": policy.inputs,
        "outputs": list(policy.outputs),
        "output_order": list(policy.output_order),
        "parameters": parameters,
        "template": leaves,
        "maps": maps,
        "state_shapes": [list(shape) for shape in buffers.state_shapes],
        "state_dtypes": [str(dtype) for dtype in buffers.state_dtypes],
        "state_sizes": list(buffers.state_sizes),
        "optimizer_shapes": [list(shape) for shape in optimizer_shapes],
        "optimizer_blocks": optimizer_blocks,
    }


def abi_digest(policy):
    return sha256(canonical_json(abi_descriptor(policy)))
