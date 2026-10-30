"""Exercise 32 player replies against independent Classic fixture legality.

Run in the serving container with /app/policy; timings are local, not hosted.
"""

import asyncio
import json
import platform
import time
from pathlib import Path

started = time.monotonic()
print("WIRE_IMPORT_START", flush=True)
from websockets.asyncio.server import serve  # noqa: E402

from integrations.softmax.neural_player import play  # noqa: E402
from integrations.softmax.protocol import parse_action  # noqa: E402

print("WIRE_IMPORT_DONE", time.monotonic() - started, flush=True)


async def main():
    replies = []
    connected = None

    async def exchange(ws):
        nonlocal connected
        connected = time.monotonic() - started
        print("WIRE_CONNECTED", connected, flush=True)
        await ws.send(json.dumps(dict(type="hello", protocol_version=1, ruleset="classic")))
        for turn in range(32):
            height, width = [(18, 21), (21, 18), (19, 20), (21, 21)][turn % 4]
            kinds = [[1] * width for _ in range(height)]
            owners = [[0] * width for _ in range(height)]
            armies = [[0] * width for _ in range(height)]
            kinds[2][2], owners[2][2], armies[2][2] = 4, 1, 20
            owners[2][3], armies[2][3] = 1, 10
            kinds[-3][-3], owners[-3][-3], armies[-3][-3] = 4, 2, 20
            message = dict(
                type="observation",
                height=height,
                width=width,
                type_grid=kinds,
                owner_grid=owners,
                army_grid=armies,
                my_land=2,
                my_army=30,
                opp_land=1,
                opp_army=20,
                turn=turn,
            )
            sent = time.monotonic()
            await ws.send(json.dumps(message))
            response = json.loads(await asyncio.wait_for(ws.recv(), 30))
            elapsed = time.monotonic() - sent
            if turn % 8 == 0:
                print("WIRE_RECEIVED", turn, elapsed, flush=True)
            action = parse_action(response, turn, height, width)
            # Validate this plain-terrain fixture independently. Calling the
            # 11-channel reference codec here would compile a second JAX graph
            # after the 16-channel player already warmed up.
            if not action[0]:
                _, row, col, direction, half = action
                assert owners[row][col] == 1 and armies[row][col] > 1, action
                dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[direction]
                target_row, target_col = row + dr, col + dc
                assert 0 <= target_row < height and 0 <= target_col < width, action
                assert kinds[target_row][target_col] != 2, action
                assert half in (0, 1), action
            replies.append(elapsed)
        await ws.send(json.dumps(dict(type="final")))

    async with serve(exchange, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        await asyncio.wait_for(play(f"ws://127.0.0.1:{port}", Path("/app/policy")), 120)
    assert len(replies) == 32
    print(
        json.dumps(
            dict(
                platform=platform.platform(),
                cold_ready_seconds=connected,
                legal_replies=len(replies),
                first_reply_seconds=replies[0],
                mean_reply_seconds=sum(replies) / len(replies),
                max_reply_seconds=max(replies),
                scope="Local Linux container; not production hardware latency",
            )
        )
    )


asyncio.run(main())
