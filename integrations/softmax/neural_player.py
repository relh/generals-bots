"""Serve a frozen Puffer/Fabric Generals policy over the Coworld player wire."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from metta_training.environment import NumericObservation
from websockets.asyncio.client import connect

from integrations.native_policy_bundle import NativePlayerPolicy
from integrations.spatial_policy_bundle import SpatialPlayerPolicy

from .neural_codec import decode_policy_action, encode_wire_observation
from .protocol import VERSION

if TYPE_CHECKING:
    from metta_training.inference import FrozenPolicy


def select_action(policy: FrozenPolicy | NativePlayerPolicy | SpatialPlayerPolicy, message: dict, codec_kwargs: dict) -> list[int]:
    values, mask = encode_wire_observation(message, **codec_kwargs)
    prediction = policy.predict(
        0, NumericObservation(values=[values.tolist()], action_masks=[mask.tolist()])
    )
    return decode_policy_action(
        prediction.probabilities, factorized_actions=codec_kwargs.get("factorized_actions", True),
    )


async def play(url: str, bundle: Path) -> None:
    native = (bundle / "native-policy.json").exists()
    spatial = (bundle / "spatial-policy.json").exists()
    if not native and not spatial:
        from metta_training.inference import FrozenPolicy
        from metta_training.policy_bundle import load_frozen_policy_bundle

    config = None if native or spatial else load_frozen_policy_bundle(bundle)
    build = json.loads((bundle / "build.json" if native or spatial else config.build).read_text())
    options = build["config"]["python_environment"]["options"]
    codec_kwargs = (
        {"expander_general_distance_prior_hinted": True}
        if options.get("general_distance_hint_features")
        else {"expander_neighbor_threat_prior_hinted": True}
        if options.get("neighbor_threat_hint_features")
        else {"expander_packed_context_prior_hinted": True}
        if options.get("expander_hint_features") and options.get("packed_context_hint_features")
        else {"expander_context_prior_hinted": True}
        if options.get("expander_hint_features") and options.get("context_hint_features")
        else {"expander_prior_hinted": True} if options.get("expander_hint_features")
        else {"sprint_prior_hinted": True} if options.get("sprint_hint_features")
        else {"prior_hinted": True} if options.get("prior_hint_features")
        else {"hinted": True} if options.get("hint_features")
        else {"packed_directional": True} if options.get("packed_directional_features")
        else {"directional": True} if options.get("directional_features")
        else {"lean": True}
    )
    codec_kwargs.update(
        move_hint_scale=options.get("move_hint_scale", 1.0),
        split_hint_scale=options.get("split_hint_scale", 1.0),
        factorized_actions=options.get("factorized_actions", True),
        directional_time_features=options.get("directional_time_features", False),
        public_scalar_features=options.get("public_scalar_features", False),
        public_scalar_ablation=options.get("public_scalar_ablation", False),
    )
    policy = SpatialPlayerPolicy(bundle) if spatial else NativePlayerPolicy(bundle) if native else FrozenPolicy(config)
    policy.reset("coworld-classic")
    # Compile both the wire codec and graph before the first 500 ms deadline.
    kinds = [[1] * 21 for _ in range(21)]
    owners = [[0] * 21 for _ in range(21)]
    armies = [[0] * 21 for _ in range(21)]
    kinds[10][10], owners[10][10], armies[10][10] = 4, 1, 1
    warmup = {
        "height": 21, "width": 21, "type_grid": kinds, "owner_grid": owners, "army_grid": armies,
        "my_land": 1, "my_army": 1, "opp_land": 1, "opp_army": 1, "turn": 0,
    }
    values, mask = encode_wire_observation(warmup, **codec_kwargs)
    policy.predict(0, NumericObservation(values=[values.tolist()], action_masks=[mask.tolist()]))
    policy.reset("coworld-classic")
    replies, slowest = 0, 0.0
    async with connect(url, ping_timeout=None, max_size=128 * 1024, open_timeout=30) as ws:
        async for raw in ws:
            message = json.loads(raw)
            if message["type"] == "hello":
                if message["protocol_version"] != VERSION or message["ruleset"] != "classic":
                    raise ValueError("Unsupported Generals Coworld protocol")
            elif message["type"] == "observation":
                if message.get("eliminated"):
                    continue
                started = time.monotonic()
                action = select_action(policy, message, codec_kwargs)
                await ws.send(json.dumps({"type": "action", "turn": message["turn"], "action": action}))
                slowest = max(slowest, time.monotonic() - started)
                replies += 1
            elif message["type"] == "final":
                print(f"[neural] replies={replies} max_reply_seconds={slowest:.4f}", file=sys.stderr, flush=True)
                return
            elif message["type"] in ("failure", "error"):
                raise RuntimeError("Coworld reported a player failure")


def main() -> None:
    asyncio.run(play(os.environ["COWORLD_PLAYER_WS_URL"], Path(os.environ["GENERALS_POLICY_BUNDLE"])))


if __name__ == "__main__":
    main()
