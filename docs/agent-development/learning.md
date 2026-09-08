# Controlled spatial PPO pilot

Prepared experiment; results pending. The GPU launch is coordinated with the
strategy and arena work to avoid overlapping measurements.

Two arms start from exactly identical network weights, optimizer state, RNG,
map pool, environment states, assigned player sides, and assigned opponents.
The only treatment difference is potential-shaping scale: terminal-only `0`
versus shaped `0.2`. Both use terminal ±1 outcomes and zero-outcome terminal draws.

Shared configuration: seed73, 32 environments, 64 steps per rollout, 256
iterations (524,288 environment transitions), 8×8 mixed terrain, pool128 refreshed
every20 iterations, discount0.995, four PPO epochs, minibatches256, width32,
learning rate0.0003, clipping0.2, gradient norm cap0.5, entropy coefficient0.01,
KL threshold0.03, and an episode-wise mixture of Random/Expander/Hunter opponents.
The game cap is500 turns. These are smaller ordinary-rule games, not competition
maps or a claim of competition readiness.

The driver at `.cache/runs/spatial-shaped/pilot.py` runs terminal-only first,
then shaped, sequentially. It records commands, configuration differences, and
frozen final-checkpoint SHA256 values in `pilot_manifest.json`. Initial and final
complete snapshots, training logs, per-iteration metrics and per-episode outcomes
are retained separately for both arms.

Frozen initialization and both final policies are evaluated on the same
development seed41000, 32 boards each of `classic8` and `terrain4`, against Random,
Expander, Hunter and Harvester. Every board is evaluated under both spawn-label
assignments and both candidate seats. Timeout draws remain in all denominators;
score means win=1, draw=0.5, loss=0. Confidence intervals cluster on board identity.
The 4×4 suite measures size transfer rather than the trained map distribution.

This single-seed pilot can expose reward failures and direction of progress. It
cannot establish robust multi-seed superiority, dominance, or the best shaping
coefficient. Training throughput and training-opponent wins are reported
separately from the frozen-policy arena measurements. No teacher imitation or
self-play changes are included in this comparison.
