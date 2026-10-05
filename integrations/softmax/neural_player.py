"""Serve a frozen Puffer/Fabric Generals policy over the Coworld player wire."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from websockets.asyncio.client import connect

from integrations.spatial_policy_bundle import SpatialPlayerPolicy

from .neural_codec import decode_policy_action, encode_wire_observation
from .protocol import VERSION


def select_action(policy: SpatialPlayerPolicy, message: dict) -> list[int]:
    values, mask = encode_wire_observation(message)
    prediction = policy.predict(0, SimpleNamespace(values=[values], action_masks=[mask]))
    return decode_policy_action(prediction.probabilities, rng=policy.action_rng)


async def play(url: str, bundle: Path) -> None:
    policy = SpatialPlayerPolicy(bundle)
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
    values, mask = encode_wire_observation(warmup)
    policy.predict(0, SimpleNamespace(values=[values], action_masks=[mask]))
    policy.reset(os.urandom(16).hex())
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
                action = select_action(policy, message)
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
