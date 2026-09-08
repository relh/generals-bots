# Sentinel development runbook

Our fork: https://github.com/relh/generals-bots/tree/relh/sentinel-agent.

Sentinel is the current strategic candidate. The spatial learned policy is a
separate research candidate; its measured strength must be established before
promotion. Both receive only the engine's fogged observation and public scores.

## Install and run

```sh
python -m venv .venv
.venv/bin/pip install -e '.[dev,train]'
```

Run Sentinel through the real competition stdio interface against the C++
Expander. The competition preset supplies rectangular 18–21 maps, fog, castle
building, deathtouch at turn 800, and the 1,200-turn draw cap.

```sh
PATH="$PWD/.venv/bin:$PATH" JAX_PLATFORMS=cpu .venv/bin/python competition/matchup.py \
  competition/agents/sentinel_python/run.sh competition/agents/expander_cpp/run.sh \
  --mode competition --seed 31000
```

The adapter defaults to competition rules. Set `SENTINEL_MODE=classic` when
running the ordinary game. Startup compilation is separate from warmed action
latency; see the profiling tools below.

## Train and resume

```sh
JAX_PLATFORMS=cpu .venv/bin/python -m generals.training.train \
  --output .cache/runs/my-spatial --board-size 8 --iterations 256
JAX_PLATFORMS=cpu .venv/bin/python -m generals.training.train \
  --output .cache/runs/my-spatial --resume .cache/runs/my-spatial/checkpoint.pkl \
  --iterations 512
```

Use `--shaping-scale 0` for the terminal-only control. The default bounded
potential shaping scale is 0.2. Read the [training contract](../../generals/training/AUDIT.md)
for terminal, timeout, curriculum, and checkpoint semantics. Checkpoints are
trusted local pickle artifacts. The [learning experiment](learning.md) records
matched initialization and budgets, frozen evaluation, and observed outcomes.

On this workstation, an old `LD_LIBRARY_PATH` overrides the installed CUDA
libraries. GPU commands use this prefix:

```sh
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_COMPILATION_CACHE_DIR="$PWD/.cache/jax-compilation" .venv/bin/python ...
```

Coordinate GPU jobs when measuring throughput. A stopped process is not a
running trainer: inspect `metrics.jsonl`, `episodes.jsonl`, checkpoint timestamps,
and the process before reporting status. Compilation can delay the first metric.

## Evaluate a frozen candidate

```sh
JAX_PLATFORMS=cpu .venv/bin/python -m generals.evaluation.cli \
  --candidate sentinel --suites classic8 classic12 competition \
  --opponents random expander hunter harvester --boards 64 --seed 73000 \
  --output .cache/runs/sentinel-heldout-v2
```

Seed 73000 is reserved for the first frozen v2 holdout. Development used seed
31000; the reward pilot uses a separate development family. Once inspected,
holdout maps become development data for subsequent versions. Do not rerun a
tuned candidate on them and call the result held out.

Every base map is played from both starts and player IDs. The four games share
a map, so intervals resample maps rather than treating all games as independent.
CSV rows include outcomes, draws, turn counts, command diagnostics and builds.
Each run saves exact boards, rule settings, source hashes, and a private frozen
checkpoint copy when applicable. Use `--candidate learned --checkpoint PATH`
for the spatial network. Never change policy source during a run.

Hunter and Harvester share behavior where no neutral castles exist; competition
results against both therefore are not independent evidence. None of these
scripted baselines establishes standing against external tournament players.

## Diagnose and profile

```sh
JAX_PLATFORMS=cpu .venv/bin/python -m generals.evaluation.replay \
  .cache/runs/sentinel-heldout-v2 --max-samples 4
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests examples/_experimental/ppo -q
```

Replays preserve complete trajectories, keys, fogged observations, actions,
diagnostics and Sentinel decisions. Source or outcome mismatches are reported;
an explicit diagnostic override is not exact replay evidence.

See [strategy](strategy.md), [audit findings](audit.md),
[performance measurements](performance.md), and the [promotion gates](PLAN.md).
`scripts/profile_agent.py` compares old Python-dispatched PPO collection with a
compiled scan using identical trajectories. That experiment's speedup is not a
throughput claim for the more demanding spatial mixed-opponent trainer.
