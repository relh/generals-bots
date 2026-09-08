# Sentinel agent development

We are developing our own fork of generals-bots, retaining the simulator's rules
and measuring agent quality through actual game outcomes.

## Baseline

Commit 3373878 preserves the initial PPO trainer fixes and paired evaluation.
The frozen 4x4 model lives at `.cache/runs/cpu-baseline/model.eqx`.
The 14,720-game evaluation lives at `.cache/runs/evaluation/REPORT.md`.
On empty boards it wins 85.4% against its random training opponent, 35.7%
against Expander, and 16.7% against Hunter. With terrain those rates fall to
56.1%, 28.3%, and 10.9%. It cannot directly run competition maps or build castles.

## Hypotheses and work streams

1. Correct training objectives and state/action representation: explicit outcome
   rewards, bounded potential shaping, canonical pass, legal castle builds,
   variable board sizes, PPO minibatches, complete resumable checkpoints.
2. Strategic reference agent: defend the general, consolidate armies, exploit
   economy, scout under fog, plan attacks, and respect competition endgame rules.
3. Instrumentation: preserve each game's outcome and seed; track invalid attempts,
   passes, splits, building, economic state, timing, and loss replays.
4. Performance: measure old rollout dispatch against compiled collection on the
   same device/input, separate compilation, and prove state/trajectory agreement.
5. Learning: map/opponent mixtures; compare terminal-only versus potential-shaped
   rewards and the strategic teacher. Preserve budgets and initialization when
   attributing gains to one change.

## Evaluation gates

Development seeds may be used for diagnostics and tuning. Final seed families
must be held out until a candidate is frozen. Record model/source hashes and
report every matchup, including draws. Pair both starting positions and player
IDs. Cluster uncertainty by the base map, not by the duplicated seat swap.

An initial dominance target is at least 85% wins against each local scripted
baseline with the lower 95% map-cluster interval above 70%, across at least
64 held-out maps per meaningful suite. Competition needs its actual 18–21
rectangular maps, fog, building, deathtouch, and 1,200-turn cap. Passing toy
boards alone is insufficient. Self-play and stronger external opponents remain
separate evidence; local dominance is not a global leaderboard claim.

Promotion additionally requires: focused correctness tests, observation-only
agent inputs, successful save/load, finite training metrics, reproducible
commands, and bounded inference costs. No threshold will be weakened merely
because a candidate fails.

## Run discipline

Use the existing `.venv`. For GPU commands, clear the shell's older CUDA override:
`env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false`.
Use `JAX_PLATFORMS=cpu` for independent tests. Coordinate GPU profiling/training
so simultaneous workloads do not contaminate measurements. Place transient
runs under `.cache/runs/`; commit reusable tools and result summaries.

Active branch: `relh/sentinel-agent`. Keep upstream separate from our fork.
