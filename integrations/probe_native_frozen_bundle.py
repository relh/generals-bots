"""Export a native checkpoint and verify its CPU action path."""

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
from websockets.asyncio.server import serve

from integrations.native_policy_bundle import NativePlayerPolicy, export_bundle
from integrations.softmax.neural_codec import encode_wire_observation
from integrations.softmax.neural_player import play, select_action
from integrations.softmax.protocol import VERSION


async def verify_wire(bundle, messages, codec):
    reply_times = []

    async def handler(ws):
        await ws.send(json.dumps(dict(type="hello", protocol_version=VERSION, ruleset="classic")))
        for turn, original in enumerate(messages):
            message = original | {"type": "observation", "turn": turn}
            start = time.perf_counter()
            await ws.send(json.dumps(message))
            reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=.5))
            reply_times.append(time.perf_counter() - start)
            assert reply["type"] == "action" and reply["turn"] == turn
            action = reply["action"]
            _, mask = encode_wire_observation(message, **codec)
            index = 1764 if action[0] else action[3] * 441 + action[1] * 21 + action[2]
            assert bool(np.asarray(mask)[index]) and action[4] in (0, 1)
        await ws.send(json.dumps(dict(type="final")))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        await asyncio.wait_for(play(f"ws://127.0.0.1:{port}", bundle), timeout=60)
    assert len(reply_times) == len(messages)
    return max(reply_times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == args.sha256
    manifest = json.loads(args.build.read_text())
    assert manifest["config"]["fabric"] is None
    assert manifest["config"]["python_environment"]["options"]["context_hint_features"]
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = args.output / "bundle"
    export_bundle(args.build, args.training, args.checkpoint, args.sha256, bundle)
    policy = NativePlayerPolicy(bundle)
    policy.reset("context-hosted-probe")
    options = manifest["config"]["python_environment"]["options"]
    codec = {
        "expander_context_prior_hinted": True,
        "move_hint_scale": options.get("move_hint_scale", 1.0),
        "split_hint_scale": options.get("split_hint_scale", 1.0),
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
            index = 1764 if action[0] else action[3] * 441 + action[1] * 21 + action[2]
            assert bool(np.asarray(mask)[index]), action
            assert action[4] in (0, 1)
    result = dict(
        checkpoint_sha256=args.sha256, model_sha256=manifest["model_sha256"],
        measured_actions=len(durations), warmup_actions=len(messages),
        mean_reply_seconds=statistics.mean(durations), max_reply_seconds=max(durations),
        under_500ms=max(durations) < 0.5,
        websocket_replies=len(messages),
        max_websocket_reply_seconds=asyncio.run(verify_wire(bundle, messages, codec)),
        scope="CPU synthetic public boards and local WebSocket player; hosted container startup and strength unproven",
    )
    (args.output / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["under_500ms"]:
        raise SystemExit("Warm hosted action path exceeds the 500ms deadline")


if __name__ == "__main__":
    main()
