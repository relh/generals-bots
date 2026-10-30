# Coworld Classic 1v1

The local server and policy pipeline use the same pinned Classic engine and
shared map configuration: independently sampled 18–21 tile dimensions, fog of
war, neutral castles, capture-only victory, and a **2,000-turn cap**. See
[engine provenance](../../generals/core/COWORLD_ENGINE.md) and the
[policy runbook](../../docs/policy/runbook.md).

## Local play

```bash
pip install -e '.[softmax]'
python -m integrations.softmax.local --human
python -m integrations.softmax.local --seed 7 --keep-open
```

Open the printed player link for human play. Arrow keys/WASD or adjacent clicks
queue moves; H selects half moves, E undoes the last queued move, Q clears queues,
and Space clears the selected tile. Queues execute one action per turn. Human
play advances at two turns per second and allows one second for action delivery.

Use `--max-turns 40` for a bounded smoke, `--port` to choose another port, and
`--player-image IMAGE` to exercise a built player image. Local runs write
`local-output/episode-*`. Share completed `results.json` and `replay.json`;
`config.json` contains player authentication tokens.

## Rules and protocol

Move all but one army, or half the source army rounded down, to an orthogonally
adjacent cell. Mountains block movement. Friendly armies combine; attacks
subtract defending armies, and capture requires strictly more attackers.
Generals and owned castles grow on even ticks; owned land grows every 50 ticks.
Neutral castles begin with 40–50 defenders. Castle building and Deathtouch are
disabled. General capture scores +1/−1; a turn-limit draw scores 0/0.

Bot action delivery has a 500 ms deadline. Missing actions become passes;
20 consecutive misses cause forfeiture. A valid pass resets the counter.
Competitive matches generate a fresh private seed, recorded only after the
match in its replay. Explicit seeds are for reproducible local checks.

Players receive public observations over authenticated WebSockets. See
[PLAYER_PROTOCOL.md](PLAYER_PROTOCOL.md) for JSON and stdio formats. To connect
an external stdio bot:

```bash
python -m integrations.softmax.player -- /path/to/bot
```

The runner supplies `COWORLD_PLAYER_WS_URL`; compile or install the bot into its
player image before use. The bundled Expander is a scripted execution control.

## Frozen neural policy

Neural serving loads an exported spatial bundle through `SpatialPlayerPolicy`.
It requires the bundle's weights, model metadata, and exact sampling settings;
serving does not require the private Metta training runtime or factory source.

```bash
docker build --platform linux/amd64   --build-context policy=/absolute/path/to/exported-bundle   -f integrations/softmax/Dockerfile.neural -t generals-policy:local .
python -m integrations.softmax.local --player-image generals-policy:local --seed 42
```

Verify bundle parity, legal wire actions, zero timeouts, and completed captures.
A functioning serving image does not qualify competitive strength. The latest
recorded baseline remains 9/32 against Daveey and 18/32 against the incumbent;
see [current state](../../docs/policy/current-state.md).

## Spectators and replays

`/client/global` exposes live public scores and status. Live boards remain
restricted to authenticated players. Completed `/replay.json` contains board
frames, submitted/applied actions, seed, timeout flags, and results. See
[GLOBAL_PROTOCOL.md](GLOBAL_PROTOCOL.md).

To serve a standalone completed replay:

```bash
integrations/softmax/tools/build_replay_viewer.sh integrations/softmax/dist/replay-viewer
# Copy the completed replay to dist/replay-viewer/replay.json.
python -m http.server 8090 --directory integrations/softmax/dist/replay-viewer
```

Open `http://localhost:8090/#replay=replay.json`. The replay bundle supports
pause, seek, playback speed, and gzip replay bytes.

## Build and verify

Regenerate the embedded schema and documentation after changes:

```bash
python -m integrations.softmax.tools.manifest
python -m integrations.softmax.tools.manifest --check
```

With the Coworld SDK and Docker installed, build the project with an explicit
release version and certify `integrations/softmax/dist/coworld_manifest.json`.
Inspect real episodes, scores, player shutdown, logs, and replay output before
uploading a release. Keep release source URLs and container identities fixed.

Targeted integration checks:

```bash
JAX_PLATFORMS=cpu pytest -q tests/test_softmax.py
GENERALS_BROWSER_TESTS=1 pytest -q tests/test_softmax_browser.py
```

The browser harness uses Chromium/Playwright. Deployment dependencies are
pinned in `requirements.lock`; deliberate updates must regenerate that lock.
