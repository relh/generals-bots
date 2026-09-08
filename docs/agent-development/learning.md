# Spatial PPO pilot and continuation

Terminal-only PPO improved over initialization and outscored the shaped arm in
all eight development matchups. That result selected terminal-only for the
16.8M-transition continuation, which is now complete. This is a single-seed result; it does not establish that
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

The continuation campaign resumed the frozen terminal-only checkpoint at
iteration 256 and completed 1024, 2048, 4096 and 8192. This added 16,252,928
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

### Completed continuation measurements

All four frozen milestones completed development evaluation. Each row contains
1,024 games: 32 boards per suite, two seats and two spawn-label assignments,
against four opponents. Score averages all eight suite/opponent matchups equally.
These development maps are reused for checkpoint selection; they are not a final
test. Every milestone snapshot remains retained with its manifest SHA256.

| Iteration / total transitions | Wins | Losses | Draws | Win rate | Score | Nonrandom score |
|---|---:|---:|---:|---:|---:|---:|
| 1,024 / 2,097,152 | 686 | 235 | 103 | 67.0% | 72.0% | 64.9% |
| 2,048 / 4,194,304 | 674 | 282 | 68 | 65.8% | 69.1% | 59.9% |
| 4,096 / 8,388,608 | 720 | 222 | 82 | 70.3% | 74.3% | 67.4% |
| 8,192 / 16,777,216 | 765 | 205 | 54 | 74.7% | 77.3% | 71.7% |

The 4.2M checkpoint regressed against all six nonrandom matchups while improving
against Random on classic12. All six individual paired score-difference intervals
include zero. Training wins still increased over the final 100-iteration windows
against Expander (75.2% to 77.4%) and Hunter (79.7% to 80.4%). On development games,
pass frequency rose from 2.1% to 8.2%, and half-army moves from 7.2% to 14.0%.
These are observed behavior changes; they do not establish the cause of the
regression or prove overfitting.

The 8.4M checkpoint recovered on classic8: win rates are 81.2% against Expander
and 76.6% against Hunter/Harvester. Its classic12 results remain weaker: 48.4%,
50.0%, and 50.0%, respectively. Thus more training has not produced uniform
improvement across board sizes. All eight paired score intervals comparing 8.4M
with 2.1M include zero; checkpoint ranking remains exploratory at 32 maps per
suite. Paired differences use 100,000 whole-board resamples with seed 91083,
retaining seats/spawn assignments; intervals are not multiplicity adjusted.

The completed final checkpoint has the highest equal-weight development score,
77.3%, and was selected before generating the fresh test maps. Its classic8
win rates are 88.3% against Expander, 73.4% against Hunter, and 75.0% against
Harvester; classic12 rates are 59.4%, 60.9%, and 64.1%. Progress remains uneven:
classic8 Hunter/Harvester are below the 8.4M milestone, and classic12 Random is
below the 4.2M milestone. The selection rule rewards the aggregate and does not
claim the selected checkpoint dominates every earlier checkpoint.

The selected complete snapshot is
`.cache/runs/spatial-terminal-extended/checkpoints/iteration-00008192.pkl`, SHA256
`048a1f8fdbad895f8db81818c35ee32f19596ec335cf44d311a1d1359a5fda1b`.
Selection was recorded in `fresh_evaluation_plan.json` before running reserved
seed **61073**, with 64 boards per suite and four opponents (2,048 games).
The fresh evaluation completed without tuning or reselecting checkpoints.

Detailed counts, paired differences, behavior rates, and training windows are
retained in `.cache/runs/spatial-terminal-extended/milestone_analysis.json`;
`REPORT.md`, `manifest.json`, `checkpoints/`, and `evaluations/` retain the raw
milestone evidence. The trained policy has no recurrent memory and was trained
only on 8×8 maps with builds disabled; these measurements do not establish
competition readiness or external-opponent strength.

### Fresh-seed result for the selected checkpoint

The frozen 16.8M policy completed **1,514 wins, 385 losses, and 149 draws** on
seed 61073: **73.9% wins** (95% board-cluster interval **69.8–77.9%**) and
**77.6% score** (**74.0–81.0%**). Excluding Random, it won 1,060/1,536 games
(69.0%) with 385 losses and 91 draws, for 72.0% score. This corroborates useful
strength against the tested scripts on fresh maps; it does not establish
competition or external-opponent dominance.

| Fresh suite / opponent | Wins | Losses | Draws | Win rate, 95% CI | Score |
|---|---:|---:|---:|---:|---:|
| classic8/random | 242 | 0 | 14 | 94.5% [88.3, 99.2] | 97.3% |
| classic8/expander | 213 | 41 | 2 | 83.2% [75.4, 90.2] | 83.6% |
| classic8/hunter | 187 | 65 | 4 | 73.0% [65.6, 80.5] | 73.8% |
| classic8/harvester | 183 | 69 | 4 | 71.5% [63.7, 78.9] | 72.3% |
| classic12/random | 212 | 0 | 44 | 82.8% [75.8, 89.1] | 91.4% |
| classic12/expander | 155 | 46 | 55 | 60.5% [51.6, 69.5] | 71.3% |
| classic12/hunter | 160 | 84 | 12 | 62.5% [53.1, 71.1] | 64.8% |
| classic12/harvester | 162 | 80 | 14 | 63.3% [54.7, 71.9] | 66.0% |

Each matchup contains 64 distinct boards and four seat/spawn-label cases per
board. Intervals use 100,000 board-cluster bootstrap samples with seed 61074;
aggregate intervals resample boards separately within each suite, retaining
all opponents and paired cases for each board. These intervals describe map
sampling for one fixed training seed and opponent set, not training-seed
variation or performance against an arbitrary bot. All 2,048 games had zero
candidate invalid moves and malformed commands. Every source hash recorded at
evaluation launch still matched at completion.

The final policy was selected by development score, not training reward. The
continuation logged 7,936 optimizer iterations after the 256-iteration pilot,
54 KL-triggered early stops, and no nonfinite scalar metrics. All four milestone
checkpoint hashes verify. Training diagnostics and throughput remain separate
from the frozen evaluation results.

Raw fresh-test evidence is under `.cache/runs/spatial-terminal-selected-fresh/`:
`quality_report.json`, `summary.json`, `games.csv`, frozen checkpoint, exact board
arrays, source/runtime metadata, and `launch.json`. The game CSV SHA256 is
`c18d68437233d2a6db36df3d94f25a48f0a40a6ff2dc71aa14999c69d39669c4`;
metadata SHA256 is
`4ba2254d2d24cf14282b58cfe4ae199069aea645ae0107d04269c4af4270df63`.
The selected snapshot's SHA256 is recorded above. Reproduce the fresh evaluation
with a new output directory:

```sh
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda .venv/bin/python -m generals.evaluation.cli \
  --candidate learned \
  --checkpoint .cache/runs/spatial-terminal-extended/checkpoints/iteration-00008192.pkl \
  --seed 61073 --boards 64 --suites classic8 classic12 \
  --opponents random expander hunter harvester \
  --output .cache/runs/spatial-terminal-selected-fresh-reproduction
```

### Next learning hypothesis

Use iteration 8,192 as the strongest measured aggregate checkpoint from this
campaign, keeping earlier snapshots for regression comparisons. The next
controlled hypothesis should be that **exposure to larger maps improves transfer**:
compare an 8×8-only continuation with an alternating 8×8/12×12 curriculum,
starting from identical selected weights and optimizer state, with the same
transition budget, opponent mixture, reward, and optimizer settings. Use at
least three training seeds, a new development set, and a separately reserved
final test; seed 61073 is now consumed. Report classic8 regressions as well as
classic12 gains. This is a proposed experiment, not an implemented change or
an additional training launch.

The observed size gap and the 8×8-only training distribution motivate that test,
but do not prove its cause. The current actor also has a limited spatial
receptive field and no persistent memory; a curriculum may be insufficient.
Teacher imitation, global spatial context, or memory are separate hypotheses
that should not be bundled into the same reward/curriculum comparison. The
current policy has not been evaluated against Sentinel or the external neural
checkpoint, and no learned-policy competition-rule evaluation was added here.
