"""Export a verified flat checkpoint and measure the hosted action path."""

import argparse
import hashlib
import json
import statistics
import time
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import jax
import platform

from metta_training.inference import FrozenPolicy
from metta_training.model_config import FrozenPolicyConfig
from metta_training.policy_bundle import export_frozen_policy, load_frozen_policy_bundle

from integrations.softmax.neural_codec import encode_wire_observation
from integrations.softmax.neural_player import select_action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--training", type=Path, help="Required for verified native export")
    parser.add_argument("--factory-source", type=Path,
                        help="Exact archived generals_fabric.py used to build this checkpoint")
    args = parser.parse_args()
    if args.native and (args.training is None or args.factory_source is not None):
        parser.error("Native export requires --training and does not use --factory-source")
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == args.sha256
    manifest = json.loads(args.build.read_text())
    if args.factory_source is not None:
        assert args.factory_source.name == "generals_fabric.py"
        assert manifest["config"]["fabric"]["factory"].startswith("integrations.generals_fabric:")
        spec = importlib.util.spec_from_file_location("integrations.generals_fabric", args.factory_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    if args.native:
        assert manifest["config"]["fabric"] is None
    else:
        assert manifest["config"]["fabric"]["factory"].endswith(":two_stage_tied_local_action_policy")
    assert manifest["config"]["python_environment"]["spec"]["action_sizes"] == [3529]
    assert manifest["config"]["python_environment"]["options"]["factorized_actions"] is False
    assert manifest["config"]["python_environment"]["options"]["directional_features"]
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = args.output / "bundle"
    if args.native:
        from integrations.native_policy_bundle import NativePlayerPolicy, export_bundle

        export_bundle(args.build, args.training, args.checkpoint, args.sha256, bundle)
        policy = NativePlayerPolicy(bundle)
    else:
        export_frozen_policy(FrozenPolicyConfig(
            build=args.build, checkpoint=args.checkpoint, sha256=args.sha256, device="cpu",
        ), bundle)
        policy = FrozenPolicy(load_frozen_policy_bundle(bundle))
    if args.factory_source is not None:
        (bundle / "model-source").mkdir()
        shutil.copyfile(args.factory_source, bundle / "model-source" / "generals_fabric.py")
    policy.reset("context-hosted-probe")
    options = manifest["config"]["python_environment"]["options"]
    codec = {
        "directional": True,
        "factorized_actions": False,
        "directional_time_features": options.get("directional_time_features", False),
    }
    messages = []
    for height, width in [(18, 21), (21, 18), (19, 20), (21, 21)]:
        kinds = [[1] * width for _ in range(height)]
        owners = [[0] * width for _ in range(height)]
        armies = [[0] * width for _ in range(height)]
        kinds[2][2], owners[2][2], armies[2][2] = 4, 1, 20
        owners[2][3], armies[2][3] = 1, 10
        kinds[height - 3][width - 3] = 4
        owners[height - 3][width - 3] = 2
        armies[height - 3][width - 3] = 20
        messages.append(dict(
            height=height, width=width, type_grid=kinds, owner_grid=owners,
            army_grid=armies, my_land=2, my_army=30, opp_land=1, opp_army=20, turn=50,
        ))
    for message in messages:
        select_action(policy, message, codec)
    durations = []
    for _ in range(8):
        for message in messages:
            start = time.perf_counter()
            action = select_action(policy, message, codec)
            durations.append(time.perf_counter() - start)
            _, mask = encode_wire_observation(message, **codec)
            index = 3528 if action[0] else action[4] * 1764 + action[3] * 441 + action[1] * 21 + action[2]
            assert bool(np.asarray(mask)[index]), action
            assert action[4] in (0, 1)
    result = dict(
        checkpoint_sha256=args.sha256, model_sha256=manifest["model_sha256"],
        measured_actions=len(durations), warmup_actions=len(messages),
        mean_reply_seconds=statistics.mean(durations), max_reply_seconds=max(durations),
        under_500ms=max(durations) < 0.5,
        scope="Synthetic public boards; hosted startup and match strength remain unproven",
        platform=platform.platform(), jax_version=jax.__version__,
        factory_source_sha256=(hashlib.sha256(args.factory_source.read_bytes()).hexdigest()
                               if args.factory_source is not None else None),
    )
    (args.output / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["under_500ms"]:
        raise SystemExit("Warm hosted action path exceeds the 500ms deadline")


if __name__ == "__main__":
    main()
