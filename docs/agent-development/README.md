# Sentinel development runbook

Our fork: https://github.com/relh/generals-bots/tree/main.

Start with our [playing doctrine](playing-doctrine.md): the owner's direction for
castle capture or construction according to the rules, tick-aware local growth,
contact-dependent FFA strategy, and a purposeful but concealed 1v1 approach.
It is the strategy contract for new work. Construction/deathtouch experiments
below use the competition variant and do not establish classic or FFA strength.

Our [competitive objective](competitive-goal.md) is to beat every available
distinct opponent. **That goal is not achieved.** See the [opponent coverage
census](opponent-coverage.md), [v4 development evidence](v4-development.md), and
[earlier measured results](results.md). Completing an experiment does not close
the competitive goal.

The [v5 interception experiment](v5-development.md) improves local Hunter and the
small Juraj V3.5 sample but regresses on Amin; it is not promoted. Its complete
episode diagnostics identify repeated defensive relocation and lost construction
windows as targets for the next experiment. The default remains frozen v2.

The [v6 commitment experiment](v6-development.md) reduces defensive backtracking
and raises the small Amin development score from 43.75% to 46.875%, with no
measured gain against Juraj V3.5. It is not promoted. Use `--candidate sentinel-v6`
for local evaluation or `--variant v6` for a standalone archive;
`sentinel-v6-disabled` / `v6-disabled` reproduce V5. Its stateful memory is carried
through local, replay, reference and stdio execution. The actual prebuilt cache
passed all 16 shapes; uncached startup exceeded the limit.

The [v7 concentration experiment](v7-development.md) completes 256 paired
development games and is not promoted. It gathers and deploys real armies, but
regresses against Hunter, Amin and both Juraj versions. Many successful captures
spend several collection actions against small garrisons. Use
`--candidate sentinel-v7` or `--variant v7`; `sentinel-v7-disabled` / `v7-disabled`
preserve exact V6 actions, telemetry and defender memory. The built cache passes
all 16 shapes and a complete active-memory wire history; uncached startup exceeds
the limit. The next experiments must test when gathering is worth its action
cost and compare the least costly sufficient routes with existing field forces.

The [v8 action-cost experiment](v8-development.md) tests cheaper sufficient
collection and direct deployment separately and together. The combined variant
improves the small Amin development score but fails Hunter preservation; it is
not promoted. Use `--candidate sentinel-v8` or `--variant v8`; `v8-cheap` and
`v8-direct` isolate the options. `v8-disabled` preserves V7 and
`v8-no-concentration` preserves V6 actions and telemetry inside the 15-leaf
wrapper. Full plan audits distinguish actually issued
routes and observed captures from available alternatives.

The [v9 remembered-general experiment](v9-development.md) completes 432 games
against six opponents. It improves Amin, Hunter and original my_bot9 over V8 but
still fails stronger-control preservation; it is not promoted. Use
`--candidate sentinel-v9` or `--variant v9`; the disabled aliases preserve V8.
Its 19-leaf memory retains only a publicly revealed stationary general location.
Telemetry distinguishes proposed V8 decisions from actually issued actions.
The built cache passes all 16 shapes and an entire active-memory wire history.
The [full-episode diagnosis](v9-diagnosis.md) records remaining defense failures
for the next bounded experiment.

The [v10 home-mobilization experiment](v10-development.md) lets a real home army
join a newly selected interception after deathtouch activates. It repairs one
known my_bot9 failure in both mirrored complete games. Its [completed 576-game
comparison](v10-development-gate.md) confirms 16/16 my_bot9 wins and converts
two losses each into draws against Amin and Juraj V3.5, but fails preservation
against stronger controls. Deployment and runtime checks pass; it is not
promoted. Use `--candidate sentinel-v10` or
`--variant v10`; `v10-v6` selects the stronger V6 parent for a separate ablation.
The corresponding `-disabled` aliases reproduce their selected parent exactly.
The default remains V2 and the competitive goal remains active.

The [four-arm development plan](v10-plan.md) compared both V10 parent choices
with V6 and V9 across 576 fixed games. The [runtime qualification](v10-runtime-qualification.md)
records cached startup checks and complete-history coverage separately.
The [expanded public-opponent census](v10-opponent-coverage.md) identifies and
checks additional implementations beyond those six opponents; synthetic protocol
checks do not establish their playing strength.

The [V11 immediate-capture screen](v11-development.md) completes twelve fixed
games and rejects the candidate: Amin wins regress to losses, Juraj remains a
loss, and my_bot9 wins are preserved. Warm throughput and focused correctness
checks pass, but the strategy gate fails. The [preregistered plan](v11-plan.md)
keeps this prototype separate from deployment and broader strength claims.

The [V12 public-route guard](v12-development.md) also fails its twelve-game
screen: independent checks confirm every replacement preserves the static route
deficits, yet Amin wins become losses and Juraj remains a loss. Thirteen focused
tests and matched warm-runtime checks pass; no promotion follows. This separates
a valid local certificate from a useful complete-game strategy.

The [V13 collection-only comparison](v13-development.md) preserves ready direct
attacks but still regresses Amin and loses to Juraj. Its eight tests and runtime
checks pass; the twelve full games reject the strategy hypothesis. Earlier
packet allocation remains unresolved, and V13 is not promoted.

The [V14 service-ledger comparison](v14-development.md) adds a persistent action
budget and live V6-based controls. All eighteen games and ledger checks complete,
but Amin and Juraj remain losses. Serviced transfers do not establish frontier
progress or economic advantage; V14 remains experimental.

Sentinel is the current strategic candidate. The spatial learned policy is a
separate research candidate; its measured strength must be established before
promotion. Both receive only the engine's fogged observation and public scores.

The [completed v3 cycle](v3-final.md) tests remembered invasion threats and
sustained defense in an experimental policy. Local preservation gates passed,
but external improvement was unproven and runtime faults prevented promotion.
V2 remains the default. To evaluate
the prototype, use `--candidate sentinel-v3`; the `sentinel-v3-memory` and
`sentinel-v3-defense` aliases isolate its two changes. Replays include each
player's memory and decision telemetry. Standalone builds select the same variants
with `scripts/build_sentinel_bundle.py --variant v3 --output PATH`.
The experimental `sentinel-v4` candidate adds a visible two-step threat check
before building a castle; `sentinel-v4-adjacent` is its frozen-v2-equivalent
one-step ablation. Build its standalone archive with `--variant v4`.
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
