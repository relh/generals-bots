# Training audit and replacement foundation

The original 4×4 checkpoint learned a reward that does not reliably pay for
winning. `examples/_experimental/ppo/train.py` uses `composite_reward_fn`, whose
base reward counts newly owned **live generals**. The engine converts captured
generals to castles. A winner normally still owns one live general, so its base
terminal reward is zero. A loser does lose its general, producing an asymmetric
signal. The reported original performance remains a valid empirical result for
that checkpoint; it is not evidence that the original objective was correct.

Other original constraints: a board-size-specific flattened value head, one
separately sampled pass action per cell, no build actions, only random opponents,
empty 4×4 training maps, one full-batch optimization update per rollout, timeout
treated as terminal, model-only saves, and no optimizer/PRNG/environment snapshot.
The old random-action helper also truncates the candidate list to 100 moves.

## Replacement contract

`generals.training` is an independent training implementation. It leaves the old
checkpoint, experiments, engine, and their replayability intact.

* All games use `GeneralsEnv.step`, including modifier ordering, terminal outcome,
  time limits, and pool-based resets. `--mode competition` uses the authoritative
  competition preset: 18–21 rectangular maps padded to21, build castles,
  deathtouch at800 and truncation1200. The ordinary8×8 training task is explicitly
  a smaller, different task.
* Learner side and opponent are randomized at episode boundaries. Default
  opponents are Random, Expander, and Hunter; optional Sentinel uses its normal
  observation/action API. Training win rates are separated by opponent in
  `episodes.jsonl`, including all timeout draws.
* Features use only the engine's public fogged Observation and public scoreboard.
  Army/count features are logarithmically scaled. The critic uses spatial mean
  and max pooling, so checkpoint parameters work on arbitrary rectangles.
* Policy has eight move channels (four directions × full/half), one build channel,
  and exactly **one** pass. Build costs derive only from owned structures using
  the exact distance surcharge kernel. Build masking is enabled only by the
  active environment rules. Hidden structure identity is never consulted.
* Base terminal reward is +1/−1 from `info.winner`; timeout draws are0.
  Observable potential is `0.5 army advantage + 0.3 land advantage + 0.2 economy`,
  with normalized signed advantages and diminishing castle value. Its magnitude
  is at most1. Shaping is `scale*(gamma*Phi(next)-Phi(current))`; the default scale
  is0.2. True terminal potential is0. There are no unconditional move/split/build
  bonuses. Set `--shaping-scale 0` for the outcome-only control.
* Configured game caps are terminal draws by default: zero bootstrap and zero
  final potential. This matches the competition1200-turn rule. GAE cuts traces
  at every environment reset; ordinary rollout cutoffs bootstrap normally.
  For experiments where an environment time limit is an artificial sampling
  cutoff in an underlying longer game, `--no-time-limit-terminal` bootstraps the
  **pre-reset final state** and preserves its potential. That option is rejected
  for competition mode. True wins/losses never bootstrap.
* PPO uses shuffled minibatches, configurable epochs, normalized advantages,
  clipped surrogate loss, gradient norm clipping, entropy regularization and KL
  early stopping. Logged metrics include KL, clip fraction, losses, gradient norm,
  entropy, potential, terminal/shaping components and rollout/update wall time.
* Atomic checkpoints contain network, optimizer, RNG, complete pool and current
  game states, assigned sides/opponents, in-progress episode returns, config,
  curriculum stage, iteration/episode/step counts, JAX version and device.
  Resume restores those values; only the requested target iteration changes.
  Deterministic continuation is expected on the same software/device/backend;
  cross-backend floating-point equality is not guaranteed. Pickle artifacts are
  for trusted local use only.

## Commands

```sh
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_training_foundation.py -q
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu .venv/bin/python -m generals.training.train \
  --output .cache/runs/spatial --iterations 100 --board-size 8
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu .venv/bin/python -m generals.training.train \
  --output .cache/runs/spatial --resume .cache/runs/spatial/checkpoint.pkl --iterations 200
```

The default terrain pool samples mountain density uniformly from0 to0.25 and
0–2 neutral castles, with spawn distance at least3. Pools are regenerated every20
iterations by default. `--curriculum 6,8,12 --stage-iterations 100` changes board
size on that fixed schedule while retaining optimizer and parameters; unfinished
episodes discarded at a stage change are explicitly logged.

## Limits and next experiments

This is a correct runnable foundation, not a demonstrated dominant trained agent.
The network is memoryless, with a local convolutional receptive field and global
pooling; it has no fog memory, learned recurrent strategy state, explicit path
planner, teacher imitation, self-play league or online opponent adaptation.
Training is currently 1v1 only. Default scripted bots themselves do not build.
The critic has no padded-board mask for pooling; competition padding can affect
its global averages (without leaking hidden state). This is an inductive-bias
limitation, not a checkpoint size restriction.

Each opponent-mixture transition traces all opponent branches in JAX. Expensive
scripted pathfinding and build-cost masking may dominate larger-map rollouts.
Profile the new trainer separately from the old4×4 baseline before projecting
large-run throughput. Initial timing includes compilation; subsequent iterations
measure warmed throughput. Network-only skill claims require held-out balanced
evaluation against every opponent, meaningful game caps, draws in the denominator
and uncertainty intervals. Compare shaped and outcome-only seeds at equal game
step budgets before claiming shaping improves strength.
