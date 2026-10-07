"""Compare one exported Classic checkpoint's native and portable logits on hosted replays."""

# Fault capture must cover native library imports as well as inference.
# ruff: noqa: E402
import argparse
import faulthandler
import gzip
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

faulthandler.enable(all_threads=True)

import jax
import jax.numpy as jnp
import numpy as np

from generals.core import coworld_game
from integrations.softmax.engine import Match
from integrations.softmax.neural_codec import encode_wire_observation
from integrations.spatial_action_sampling import (
    acting_logits,
    public_doomed_attack_route_penalty,
    public_early_route_temperature,
    public_neutral_route_bonus,
    public_weak_owned_route_penalty,
)
from integrations.spatial_policy_bundle import SpatialPlayerPolicy, structured_action_probabilities

ENGINE_SHA256 = "f39e448a6b2822869d75cb07cce4cb43d589c4112fef04007ade951809d4a318"
FRAME_FIELDS = ("turn", "type_grid", "owner_grid", "army_grid", "army", "land")


def stage(name, **details):
    """Flush a boundary before asynchronous/native work can terminate Python."""
    print("PARITY_STAGE " + json.dumps(dict(stage=name, **details)), flush=True)


def verified_views(replay_root, game_indices, turns):
    manifest = json.loads((replay_root / "leader-replay-manifest.json").read_text())
    views, masks, labels = [], [], []
    for index in game_indices:
        item = manifest[index]
        blob = (replay_root / "leader-replays" / (item["episode_id"] + ".bin")).read_bytes()
        if hashlib.sha256(blob).hexdigest() != item["sha256"]:
            raise ValueError(f"Hosted replay checksum differs: {index}")
        replay = json.loads(gzip.decompress(blob))
        if replay["ruleset"] != "classic":
            raise ValueError(f"Hosted replay is not Classic: {index}")
        match = Match(replay["seed"])
        for turn_index, receipt in enumerate(replay["turns"]):
            if turn_index > max(turns):
                break
            frame = match.frame()
            if (
                not receipt["applied"]
                or receipt["turn"] != match.turn
                or any(frame[field] != replay["frames"][turn_index][field] for field in FRAME_FIELDS)
            ):
                raise ValueError(f"Hosted Classic replay diverged: game {index}, turn {turn_index}")
            if turn_index in turns:
                for side in (0, 1):
                    values, legal = encode_wire_observation(match.observation(side))
                    views.append(values)
                    masks.append(legal)
                    labels.append((index, turn_index, side))
            match.advance(receipt["actions"])
    if not views:
        raise ValueError("Selected replay turns yielded no public observations")
    return np.asarray(views, np.float32), np.asarray(masks, bool), labels


def rollout_probabilities(outputs, observations, legal, policy):
    """Apply the native rollout action transform, including public route adjustments."""
    values = jnp.asarray(observations[None, None, :])
    predictions = jnp.asarray(outputs[None, None, :])
    temperature = policy.move_temperature
    if policy.early_route_temperature is not None:
        temperature = public_early_route_temperature(
            values,
            temperature,
            policy.early_route_temperature,
            policy.early_route_turns,
            jnp,
        )
    acting = acting_logits(
        predictions, temperature, policy.split_temperature, jnp, route_half_weight=policy.route_half_weight
    )[..., :3529]
    if policy.neutral_route_bias:
        acting += public_neutral_route_bonus(values, policy.neutral_route_bias, jnp)
    if policy.weak_owned_route_penalty:
        acting += public_weak_owned_route_penalty(values, policy.weak_owned_route_penalty, jnp)
    if policy.doomed_attack_route_penalty:
        acting += public_doomed_attack_route_penalty(values, policy.doomed_attack_route_penalty, jnp)
    acting = acting / policy.full_action_temperature
    logits = jnp.where(jnp.asarray(legal[None, None, :]), acting, -jnp.inf)
    return np.asarray(jax.nn.softmax(logits)[0, 0])


def audit(bundle, replay_root, factory_source, game_indices, turns, batch_size):
    engine = Path(coworld_game.__file__)
    if hashlib.sha256(engine.read_bytes()).hexdigest() != ENGINE_SHA256:
        raise ValueError("Local Classic engine differs from the pinned hosted revision")
    if batch_size <= 0:
        raise ValueError("Native batch size must be positive")
    portable = SpatialPlayerPolicy(bundle)
    if portable.observation_size != 7056 or portable.action_mode != "structured_sample":
        raise ValueError("Expected a sampled public-scalar Classic checkpoint")
    source_sha = hashlib.sha256(factory_source.read_bytes()).hexdigest()
    if portable.asset.metadata["factory_source_sha256"] != source_sha:
        raise ValueError("Native model factory differs from the exported checkpoint")
    spec = importlib.util.spec_from_file_location("integrations.generals_fabric", factory_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from integrations.direct_spatial_optimization import install

    acting_environment = {
        "METTA_DIRECT_SPATIAL_ROLLOUT": "1",
        "METTA_SPATIAL_POLICY_TEMPERATURE": str(portable.move_temperature),
        "METTA_SPATIAL_SPLIT_TEMPERATURE": str(portable.split_temperature),
        "METTA_SPATIAL_ROUTE_HALF_WEIGHT": str(portable.route_half_weight),
        "METTA_SPATIAL_FULL_ACTION_TEMPERATURE": str(portable.full_action_temperature),
        "METTA_SPATIAL_NEUTRAL_ROUTE_BIAS": str(portable.neutral_route_bias),
        "METTA_SPATIAL_WEAK_OWNED_ROUTE_PENALTY": str(portable.weak_owned_route_penalty),
        "METTA_SPATIAL_DOOMED_ATTACK_ROUTE_PENALTY": str(portable.doomed_attack_route_penalty),
    }
    if portable.early_route_temperature is not None:
        acting_environment.update(
            METTA_SPATIAL_EARLY_ROUTE_TEMPERATURE=str(portable.early_route_temperature),
            METTA_SPATIAL_EARLY_ROUTE_TURNS=str(portable.early_route_turns),
        )
    os.environ.update(acting_environment)
    import metta_training.native_fabric as native_module

    if not getattr(native_module.NativeFabricPolicy, "_generals_direct_spatial", False):
        install(native_module)
    NativeFabricPolicy = native_module.NativeFabricPolicy

    device = jax.devices()[0]
    stage("layout_start", inference_backend=device.platform)
    # Match the native build: topology, template and flat gather layout are host
    # metadata. GPU execution begins with explicit checkpoint/input placement.
    with jax.default_device(jax.devices("cpu")[0]):
        native = NativeFabricPolicy(json.dumps(portable.asset.metadata["fabric"]))
    stage("layout_ready", parameter_words=native.buffers.parameter_words)
    from metta_training.model_config import FabricConfig
    from metta_training.native_build import fabric_fingerprint

    from integrations.native_spatial_asset import abi_digest

    portable.asset.verify_target(
        factory_source_sha256=source_sha,
        model_sha256=fabric_fingerprint(FabricConfig.model_validate(portable.asset.metadata["fabric"])),
        abi_sha256=abi_digest(native),
    )
    stage("identity_verified")
    weights = np.frombuffer(portable.asset.policy, "<f4")
    if weights.size != native.buffers.parameter_words or not np.isfinite(weights).all():
        raise ValueError("Native checkpoint parameter layout differs")
    views, masks, labels = verified_views(replay_root, game_indices, turns)
    if views.shape[1] != native.observation_size or masks.shape != (len(views), 3529):
        raise ValueError("Hosted public codec differs from native policy dimensions")
    native_outputs, native_acting, portable_outputs = [], [], []
    stage("input_placement_start", public_states=len(views))
    parameters = jax.device_put(weights, device)
    parameters.block_until_ready()
    stage("input_placement_ready")
    for start in range(0, len(views), batch_size):
        selected = views[start : start + batch_size]
        state = jax.device_put(
            np.repeat(native.buffers.pack_state(native.buffers.template), len(selected), axis=0), device
        )
        observations = jax.device_put(selected[:, None, :], device)
        terminals = jax.device_put(np.zeros((len(selected), 1), np.float32), device)
        stage("forward_start", batch_start=start, batch_size=len(selected))
        with jax.default_matmul_precision("highest"):
            raw_output = native.direct_spatial.forward(parameters, observations)
            acting_output, _, _ = native._forward_arrays(
                parameters,
                state,
                observations,
                terminals,
                len(selected),
                1,
                True,
            )
        raw_output.block_until_ready()
        acting_output.block_until_ready()
        stage("forward_ready", batch_start=start)
        native_outputs.append(np.asarray(raw_output)[:, 0])
        native_acting.append(np.asarray(acting_output)[:, 0])
        portable_outputs.append(portable.forward(selected))
    native_outputs = np.concatenate(native_outputs)
    native_acting = np.concatenate(native_acting)
    portable_outputs = np.concatenate(portable_outputs)
    if (
        native_outputs.shape != portable_outputs.shape
        or native_outputs.shape != (len(views), 3530)
        or native_acting.shape != native_outputs.shape
    ):
        raise ValueError("Native and serving logits have different shapes")
    maximum = float(np.max(np.abs(native_outputs - portable_outputs)))
    np.testing.assert_allclose(native_outputs, portable_outputs, rtol=2e-5, atol=2e-5)
    stage("probabilities_start")
    action_matches = 0
    probability_max = 0.0
    rollout_transform_max = 0.0
    for index, (observation, legal) in enumerate(zip(views, masks, strict=True)):
        move_temperature = portable.move_temperature
        if portable.early_route_temperature is not None:
            move_temperature = float(
                public_early_route_temperature(
                    observation[None, :],
                    move_temperature,
                    portable.early_route_temperature,
                    portable.early_route_turns,
                    np,
                )[0, 0]
            )
        kwargs = dict(
            observations=observation,
            neutral_route_bias=portable.neutral_route_bias,
            weak_owned_route_penalty=portable.weak_owned_route_penalty,
            doomed_attack_route_penalty=portable.doomed_attack_route_penalty,
            route_half_weight=portable.route_half_weight,
            full_action_temperature=portable.full_action_temperature,
        )
        direct_logits = native_acting[index, :3529]
        rollout_logits = np.where(legal, direct_logits, -np.inf)
        native_prob = np.exp(rollout_logits - np.max(rollout_logits))
        native_prob /= native_prob.sum()
        independent_prob = rollout_probabilities(native_outputs[index], observation, legal, portable)
        rollout_transform_max = max(rollout_transform_max, float(np.max(np.abs(native_prob - independent_prob))))
        serving_prob = structured_action_probabilities(
            portable_outputs[index], legal, move_temperature, portable.split_temperature, **kwargs
        )
        probability_max = max(probability_max, float(np.max(np.abs(native_prob - serving_prob))))
        action_matches += int(np.argmax(native_prob) == np.argmax(serving_prob))
    if action_matches != len(views) or probability_max > 1e-5 or rollout_transform_max > 1e-5:
        raise ValueError("Native and serving action distributions differ")
    stage("verified", matching_top_actions=action_matches)
    return dict(
        layout_backend="cpu",
        inference_backend=device.platform,
        checkpoint_sha256=portable.asset.metadata["policy_sha256"],
        factory_source_sha256=source_sha,
        engine_sha256=ENGINE_SHA256,
        serving_action_selection=portable.asset.metadata["sampler"],
        bundle_manifest_sha256=hashlib.sha256((bundle / "spatial-policy.json").read_bytes()).hexdigest(),
        hosted_games=len(game_indices),
        replay_turns=sorted({item[1] for item in labels}),
        public_states=len(views),
        native_batch=batch_size,
        max_logit_difference=maximum,
        max_action_probability_difference=probability_max,
        max_rollout_transform_difference=rollout_transform_max,
        matching_top_actions=action_matches,
        jax_backend=jax.default_backend(),
        scope=(
            "Exact checkpoint: native Fabric forward versus exported NumPy forward and legal sampled "
            "action probabilities on SHA-verified Classic hosted observations"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--factory-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, nargs="+", default=[0, 6, 7, 10])
    parser.add_argument("--turns", type=int, nargs="+", default=[0, 25, 99, 100, 150, 200])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not args.games or not args.turns or min(args.games) < 0 or min(args.turns) < 0:
        raise ValueError("Select nonnegative hosted replay and turn indices")
    result = audit(args.bundle, args.replay_root, args.factory_source, args.games, set(args.turns), args.batch_size)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
