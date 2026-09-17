"""Run the dependency-free Expander directly, importing before connection."""
import asyncio
import json
import os
import sys
import time
from types import SimpleNamespace

from websockets.asyncio.client import connect
from competition.agents.expander_python.agent import Agent
from .protocol import VERSION


async def play(url):
    agent = None
    count, maximum = 0, 0.0
    async with connect(url, ping_timeout=None, max_size=128 * 1024, open_timeout=30) as ws:
        async for raw in ws:
            message = json.loads(raw)
            if message['type'] == 'hello':
                if message['protocol_version'] != VERSION:
                    raise ValueError('unsupported protocol')
                agent = Agent(message['slot'], message['height'], message['width'])
            elif message['type'] == 'observation':
                if message.get('eliminated'):
                    continue
                if agent is None:
                    raise ValueError('observation before hello')
                started = time.monotonic()
                obs = SimpleNamespace(**message, H=agent.H, W=agent.W)
                action = agent.act(obs)
                await ws.send(json.dumps(dict(type='action', turn=message['turn'], action=action)))
                maximum = max(maximum, time.monotonic()-started)
                count += 1
            elif message['type'] == 'final':
                print(f'[expander] finished; replies={count}; max_reply_seconds={maximum:.4f}', file=sys.stderr, flush=True)
                return
            elif message['type'] in ('failure', 'error'):
                raise RuntimeError('server reported failure')


def main():
    try:
        asyncio.run(play(os.environ['COWORLD_PLAYER_WS_URL']))
    except Exception as exc:
        # Connection errors can include credential-bearing URLs.
        print(f'[expander] session failed ({type(exc).__name__})', file=sys.stderr, flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
