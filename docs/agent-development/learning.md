# Controlled spatial PPO pilot

Terminal-only PPO improved over initialization and outscored the shaped arm in
all eight development matchups. Continue the terminal-only checkpoint for the
next larger-budget run. This is a single-seed result; it does not establish that
potential shaping is generally harmful or that the learned bot is dominant.

The controlled GPU pilot completed 1,048,576 training transitions and 3,072
frozen-policy evaluation games. Training and evaluation ran sequentially in the
allocated GPU window.

Two arms start from exactly identical network weights, optimizer state, RNG,
map pool, environment states, assigned player sides, and assigned opponents.
The only treatment difference is potential-shaping scale: terminal-only `0`
versus shaped `0.2`. Both use terminal ±1 outcomes and zero-outcome terminal draws.

Shared configuration: seed 73, 32 environments, 64 steps per rollout, 256
iterations (524,288 environment transitions), 8×8 mixed terrain, pool 128 refreshed
every 20 iterations, discount 0.995, four PPO epochs, minibatches 256, width 32,
learning rate 0.0003, clipping 0.2, gradient norm cap 0.5, entropy coefficient 0.01,
KL threshold 0.03, and an episode-wise mixture of Random/Expander/Hunter opponents.
The game cap is 500 turns. These are smaller ordinary-rule games, not competition
maps or a claim of competition readiness.

The original driver at `.cache/runs/spatial-shaped/pilot.py` ran terminal-only first,
then shaped, sequentially. It records commands, configuration differences, and
frozen final-checkpoint SHA256 values in `pilot_manifest.json`. Initial and final
complete snapshots, training logs, per-iteration metrics and per-episode outcomes
are retained separately for both arms.

Frozen initialization and both final policies are evaluated on the same
development seed 41000, 32 boards each of `classic8` and `terrain4`, against Random,
Expander, Hunter and Harvester. Every board is evaluated under both spawn-label
assignments and both candidate seats. Timeout draws remain in all denominators;
score means win=1, draw=0.5, loss=0. Confidence intervals cluster on board identity.
The 4×4 suite measures size transfer rather than the trained map distribution.
`classic8` also differs from training: mountain density 18–26%, 2–4 neutral castles
with 20–40 armies, general separation at least 6, and an 800-turn cap. Training uses
density 0–25%, 0–2 castles with 10–24 armies, separation at least 3, and a 500-turn cap.
Thus both suites test transfer; this pilot does not include a strictly matched
training-distribution arena.

This single-seed pilot can expose reward failures and direction of progress. It
cannot establish robust multi-seed superiority, dominance, or the best shaping
coefficient. Training throughput and training-opponent wins are reported
separately from the frozen-policy arena measurements. No teacher imitation or
self-play changes are included in this comparison.

## Pilot results

Arena score uses win=1, draw=0.5, loss=0; percentages below are scores, not win rates.

| Suite / opponent | Initial score | Terminal-only | Shaped | Shaped − terminal, 95% paired CI |
|---|---:|---:|---:|---:|
| classic8 / expander | 2.3% | 16.4% | 6.2% | -10.2 pp [-21.9, +0.8] |
| classic8 / harvester | 0.0% | 20.3% | 17.2% | -3.1 pp [-13.3, +7.0] |
| classic8 / hunter | 1.6% | 23.4% | 18.8% | -4.7 pp [-14.8, +5.5] |
| classic8 / random | 49.2% | 81.2% | 69.5% | -11.7 pp [-18.8, -4.7] |
| terrain4 / expander | 21.9% | 47.7% | 40.6% | -7.0 pp [-21.9, +7.8] |
| terrain4 / harvester | 4.7% | 28.9% | 21.9% | -7.0 pp [-18.8, +3.9] |
| terrain4 / hunter | 8.6% | 36.7% | 21.1% | -15.6 pp [-29.7, -0.8] |
| terrain4 / random | 58.6% | 69.5% | 63.3% | -6.2 pp [-23.4, +10.2] |

Intervals resample the 32 board clusters, preserving paired seats/spawn assignments.
They describe map-sampling uncertainty for this training seed; eight simultaneous comparisons
are exploratory and the intervals are not multiple-comparison adjusted.

Terminal-only improvement over initialization has a positive paired interval in
seven of eight matchups. Shaped versus terminal-only intervals exclude zero in
two matchups before multiple-comparison adjustment. The consistent direction
supports selecting terminal-only for the next experiment, not a broad claim that
shaping cannot help. The terminal-only bot still loses 104/128 games to Expander,
94/128 to Hunter, and 100/128 to Harvester on the harder classic8 suite.

No candidate invalid moves or malformed commands occurred in any of the 3,072
evaluation games.

### Training throughput

- spatial-terminal: 524,288 environment transitions, 2,028 completed training episodes; median warmed throughput 2,336 steps/s and effective warmed throughput 2,059 steps/s (first five iterations excluded); 0 KL early stops.
- spatial-shaped: 524,288 environment transitions, 1,918 completed training episodes; median warmed throughput 3,060 steps/s and effective warmed throughput 2,770 steps/s (first five iterations excluded); 0 KL early stops.

Full per-matchup win/loss/draw counts and paired initial-to-final deltas are in
`.cache/runs/spatial-shaped/comparison.json`; each arm retains arena `games.csv`,
`summary.json`, exact boards, source/checkpoint hashes and training JSONL files.

## Reproduce the experiment design

The tracked workflow creates one shared initialization, preserves separate full
snapshots for the two reward arms, runs training sequentially, freezes final
checkpoints, evaluates initialization and both finals, and writes a paired
bootstrap comparison report:

```sh
env -u LD_LIBRARY_PATH .venv/bin/python scripts/compare_rewards.py \
  --platform cuda --output-root .cache/runs/reward-reproduction \
  --seed 73 --iterations 256 --num-envs 32 --steps 64 \
  --board-size 8 --pool-size 128 --gamma 0.995 --shaping-scale 0.2 \
  --evaluation-seed 41000 --boards 32 --suites classic8 terrain4
```

Use `--dry-run` to inspect the resolved experiment plan, or `--report-only` to
regenerate comparisons from completed evaluations. The budget is per arm:
`iterations × num_envs × steps`. The manifest records the resolved configuration,
commands, source/device metadata and checkpoint hashes. Historical pilot
artifacts retain their original source fingerprints; later optimized drivers
reproduce the design without promising bit-identical cross-version trajectories.

## Bounded continuation campaign

The prepared next experiment resumes the frozen terminal-only checkpoint at
iteration 256 and trains to 1024, 2048, 4096 and 8192. This adds 16,252,928
transitions, for 16,777,216 total. After each segment it freezes the checkpoint,
evaluates 32 development boards per suite (`classic8`, `classic12`) against all
four scripted opponents using seed 51000, then resumes the complete training
state. The separate final-test seed is not used by this campaign.

```sh
env -u LD_LIBRARY_PATH .venv/bin/python -u scripts/train_campaign.py \
  --platform cuda --resume .cache/runs/spatial-terminal/final_checkpoint.pkl \
  --output-root .cache/runs/spatial-terminal-extended
```

The supervisor writes `status.json` with stage, progress and process IDs,
`manifest.json` with configurations and checkpoint hashes, complete training
JSONL, separate child logs, and a `REPORT.md` containing win/loss/draw counts and
board-bootstrap win-rate intervals. A failed child stops the campaign. Resume an
interrupted campaign in a fresh output directory using its latest complete
`training/checkpoint.pkl`. No in-progress episode is reset solely because a
training segment ends.

This continuation uses the newly compiled GPU PPO update driver, which preserves
the pilot's algorithm and budget. Independent parity checks found identical RNG
and control counters, with maximum GPU parameter difference 2.98e-8 from floating
arithmetic. Checkpoints preserve initialization provenance separately from the
current training source/driver, and resume events record that transition.
Historical and optimized trajectories are not promised to remain bit-identical.
