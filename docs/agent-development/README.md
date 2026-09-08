# Sentinel development runbook

Our fork: https://github.com/relh/generals-bots/tree/main.

Start with the [measured results](results.md). Sentinel passed the first local
holdout gate; the learned policy and external opponent comparisons are reported
separately.

Sentinel is the current strategic candidate. The spatial learned policy is a
separate research candidate; its measured strength must be established before
promotion. Both receive only the engine's fogged observation and public scores.

The [controlled v3 cycle](v3-cycle.md) adds remembered invasion threats and
sustained defense in an experimental policy. V2 remains the default. To evaluate
the prototype, use `--candidate sentinel-v3`; the `sentinel-v3-memory` and
`sentinel-v3-defense` aliases isolate its two changes. Replays include each
player's memory and decision telemetry. Standalone builds select the same variants
with `scripts/build_sentinel_bundle.py --variant v3 --output PATH`.
Use `scripts/compare_agent_runs.py CONTROL_DIR CANDIDATE_DIR --output PATH` for
paired score differences; it verifies actual boards, rules, and complete cases.

The external runner, `scripts/stdio_arena.py`, retains every reply and completed
outcome before cleaning up both processes. To continue an interrupted run, repeat
its original command with `--resume --check-resume` first, then remove
`--check-resume`. It verifies policy/runtime identities, retained boards, complete
case traces, and process ownership before running only missing cases. Each
execution retains an immutable metadata segment and runner source snapshot.
Changed-source resumes require an explicit reviewed equivalence record and are
excluded from strict strength comparisons. Do not edit the original metadata.

After the preregistered v3 evaluation finishes, check its fixed promotion gates:

```sh
.venv/bin/python scripts/check_v3_promotion.py \
  --local-v2 LOCAL_V2 --local-v3 LOCAL_V3 \
  --external-v2 EXTERNAL_V2 --external-v3 EXTERNAL_V3 \
  --qualification V3_QUALIFICATION --output promotion.json
```

The checker validates full case budgets, paired maps, frozen sources/archives,
raw replies and cleanup records, and all sixteen runtime probes. Exit status 0
means the evidence passes every promotion gate; status 1 means rejection or
incomplete evidence. The JSON distinguishes invalid evidence from a failed
strength gate. These are local measurements, not an official ladder placement.

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
  --output .cache/runs/my-spatial --board-size 8 --iterations 256 --shaping-scale 0
JAX_PLATFORMS=cpu .venv/bin/python -m generals.training.train \
  --output .cache/runs/my-spatial --resume .cache/runs/my-spatial/checkpoint.pkl \
  --iterations 512
```

Use `--shaping-scale 0` for the terminal-only control. The default bounded
potential shaping scale is 0.2. Read the [training contract](../../generals/training/AUDIT.md)
for terminal, timeout, curriculum, and checkpoint semantics. Checkpoints are
trusted local pickle artifacts. The [learning experiment](learning.md) records
matched initialization and budgets, frozen evaluation, and observed outcomes.

The first controlled pilot favored terminal-only training on all eight measured
development matchup scores. Shaping remains available for controlled experiments;
the longer campaign continues the terminal-only checkpoint. GPU training uses the
measured compiled optimizer; CPU retains the reference implementation. Checkpoints
record any change of optimizer implementation and training source on resume.

```sh
env -u LD_LIBRARY_PATH .venv/bin/python -u scripts/train_campaign.py \
  --platform cuda --resume .cache/runs/spatial-terminal/final_checkpoint.pkl \
  --output-root .cache/runs/spatial-terminal-extended
```

This bounded campaign stops at 16,777,216 total training transitions and evaluates
frozen checkpoints at iterations 1024, 2048, 4096, and 8192. Its `status.json`
records the current phase, process IDs, latest metrics, and any failure.

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

Seed 73000 was consumed by the first frozen v2 holdout. Development used seed
31000; the reward pilot uses a separate development family. Once inspected,
holdout maps become development data for subsequent versions. Do not rerun a
tuned candidate on them and call the result held out.

Every base map is played from both starts and player IDs. The four games share
a map, so intervals resample maps rather than treating all games as independent.
CSV rows include outcomes, draws, turn counts, command diagnostics and builds.
Each run saves exact boards, rule settings, source hashes, and a private frozen
checkpoint copy when applicable. Use `--candidate learned --checkpoint PATH`
for the spatial network. Never change policy source during a run.

Generate the fixed promotion-gate report after the complete arena run:

```sh
JAX_PLATFORMS=cpu .venv/bin/python -m generals.evaluation.report \
  .cache/runs/sentinel-heldout-v2 --require-gate
```

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
[performance measurements](performance.md), [deployment checks](deployment.md),
and the [promotion gates](PLAN.md).
`scripts/profile_agent.py` compares old Python-dispatched PPO collection with a
compiled scan using identical trajectories. That experiment's speedup is not a
throughput claim for the more demanding spatial mixed-opponent trainer.
`scripts/profile_optimizer.py` measures the latter's optimizer change on matched
inputs. `scripts/profile_sentinel.py` measures synchronized scalar and batched
agent inference. The standalone bundle and CPU deadline probe are described in
the deployment checks; no tool automatically submits a bot to the tournament.
