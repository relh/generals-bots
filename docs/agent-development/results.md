# Sentinel v2 measured results


Frozen strategy revision: `36985f5`; policy SHA-256
`be909e6fa3d5b46a3dd2454eaff8a030a8e7d7158f90d484044c36fa06e4650e`.
This is the first evaluation of v2 on seed family 73000, reserved before the run.
No policy tuning occurred during or after this evaluation. Formatting before
freezing preserved the policy's non-import AST exactly against `5c07216`.

The run comprises 3,072 complete games: 64 base maps per suite, four paired
start/player-ID assignments, and four opponents. Every matchup had zero Sentinel
invalid moves and zero rejected command encodings. Competition includes actual
18–21 rectangles, fog, castle building, deathtouch at 800, and the 1,200-turn cap.
Hunter and Harvester are behaviorally identical when no neutral castles exist;
their matching competition results are not independent opponent evidence.

Seed: 73000. Complete planned run: True. Local gate passed: True.

| Matchup | Maps | W / L / D | Win rate | 95% map interval | Gate |
|---|---:|---:|---:|---:|---|
| classic8/random | 64 | 256 / 0 / 0 | 100.0% | 100.0%–100.0% | pass |
| classic8/expander | 64 | 241 / 15 / 0 | 94.1% | 90.2%–97.7% | pass |
| classic8/hunter | 64 | 230 / 18 / 8 | 89.8% | 84.4%–94.5% | pass |
| classic8/harvester | 64 | 228 / 20 / 8 | 89.1% | 83.6%–93.8% | pass |
| classic12/random | 64 | 256 / 0 / 0 | 100.0% | 100.0%–100.0% | pass |
| classic12/expander | 64 | 242 / 10 / 4 | 94.5% | 90.6%–97.7% | pass |
| classic12/hunter | 64 | 224 / 30 / 2 | 87.5% | 82.0%–93.0% | pass |
| classic12/harvester | 64 | 224 / 30 / 2 | 87.5% | 82.0%–93.0% | pass |
| competition/random | 64 | 256 / 0 / 0 | 100.0% | 100.0%–100.0% | pass |
| competition/expander | 64 | 250 / 6 / 0 | 97.7% | 94.5%–100.0% | pass |
| competition/hunter | 64 | 246 / 8 / 2 | 96.1% | 92.2%–99.2% | pass |
| competition/harvester | 64 | 246 / 8 / 2 | 96.1% | 92.2%–99.2% | pass |

Gate: at least 64 maps, at least 85% wins with draws in the denominator,
and a lower 95% map-bootstrap win-rate bound above 70%, for every planned matchup.
Both starts and player IDs are paired; uncertainty resamples whole base maps.
Bootstrap intervals are descriptive: all-win samples produce a degenerate interval,
which does not establish a true 100% win probability. This gate does not verify
seed secrecy or external tournament strength; those require separate evidence.

Source hashes, frozen checkpoint identity, exact rules, and commands are in metadata.json.
Per-game diagnostics and boards are retained beside this report.

## Reproduce and inspect

```sh
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -m generals.evaluation.cli --candidate sentinel \
  --suites classic8 classic12 competition --opponents random expander hunter harvester \
  --boards 64 --seed 73000 --output .cache/runs/sentinel-heldout-v2
JAX_PLATFORMS=cpu .venv/bin/python -m generals.evaluation.report \
  .cache/runs/sentinel-heldout-v2 --require-gate
```

Use the frozen revision for exact historical source replay. The original run's
metadata, exact boards and CSV are retained under `.cache/runs/sentinel-heldout-v2`.
Eight sampled losses replayed with exact outcomes, turn counts, and all counters;
their trajectories and diagnostic reports are in its `replays/` directory.
Sampling targeted failures, so category frequencies are not population estimates.
See [failure analysis](failures.md) for specific decision timelines.

## Learned policy and external opponents

Sentinel is the strategic policy above. The separate spatial PPO learner has
not passed this gate. Its controlled [reward pilot](learning.md) improved over
initialization but favored terminal-only training over shaping in all eight
measured development matchup scores. The extended terminal-only campaign targets
16,777,216 total training transitions with four frozen evaluation milestones.

The deadline-aware external comparison finished with **12 wins, 14 losses, and
6 draws** against the pinned Amin NumPy PPO checkpoint, across eight maps and
32 paired games. Win rate was 37.5% (95% map-bootstrap interval 18.75–62.5%);
score was 46.875% (28.125–68.75%). Both agents had zero runner faults. This does
not establish dominance against that checkpoint or an external leaderboard rank.
See the [external comparison](external-comparison.md) and
[deployment and opponent provenance](deployment.md).

The extended learner's first milestone, 2,097,152 transitions, won 67.2% against
Hunter on 32 development 8×8 maps and 50.0% on 32 development 12×12 maps. Training
continues toward the bounded 16,777,216-transition target. Its live status and
milestone reports are under `.cache/runs/spatial-terminal-extended/`.

Verification: 197 full-suite tests plus 20 focused tests passed, including two
additional diagnostic regressions (199 unique tests). Changed production code
and tools pass Ruff. The standalone package's pinned-runtime timing checks are
reported separately from playing strength in the deployment document.
