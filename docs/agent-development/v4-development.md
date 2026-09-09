# Sentinel v4 development

V4 is an experimental candidate for the active [competitive-superiority goal](competitive-goal.md).
**V4 is not promoted; v2 remains the default.** V4 preserved local results and
scored better in a small Juraj sample, but failed to establish a winning advantage
against Amin. Neither candidate beats every tested opponent. This completed
experiment leaves the competitive goal active, without an official leaderboard rank.
The policy and arena sources stayed frozen during these development comparisons.

The hypothesis is narrow: avoid spending a stack on a castle when a visible enemy
can reach the construction site within two moves. V4 preserves v2's campaign moves,
home-defense conditions, and economic checks, adding a two-step visible-threat
constraint to castle construction. It has no fog memory. The envelope respects
known obstacles but does not model enemy stack merging; horizon one reproduces the
v2 construction ablation. This tests whether a small construction safeguard helps
without the broader defensive restrictions that failed to improve v3 reliably.

The frozen policy is `generals/agents/sentinel_v4_agent.py`, SHA256
`2de5694638e6461df08d27a8b31c04d2ccdd9ececc6fe0f4a9b0b0fa89c22744`.
Archive `.cache/runs/sentinel-v4/candidate.zip` has SHA256
`4a1740278c7bd42c7c36c39efebae62ea625e96927c0ace0b2c5c0c54dea2d93`.
Its manifest also pins the unchanged v2 and v3 helper dependencies. V2 control
policy SHA256 is `be909e6fa3d5b46a3dd2454eaff8a030a8e7d7158f90d484044c36fa06e4650e`.

## Completed development evidence

Juraj comparison: four consumed development maps from seed 83000, both seats and
both spawn labels, giving 16 games per policy under actual competition rules.
Score is wins plus half of draws, divided by games.

| Policy | Wins | Losses | Draws | Win rate | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen v2 | 11 | 5 | 0 | 68.75% | 68.75% |
| Frozen v4 | 13 | 2 | 1 | 81.25% | 84.375% |

The paired score difference is **+15.625 percentage points**, with a 95% paired
map-bootstrap interval of **[+6.25, +25]** (100,000 resamples). Four maps are the
independent units, not 16 games. Juraj retains its original entropy/clock RNG;
matching maps, seats, and labels does not match its internal randomness. This
exploratory interval therefore describes the observed map pairs without measuring
uncertainty from repeated opponent RNG draws. Both runs recorded zero reply faults,
invalid actions, process forfeits, and cleanup failures. All games are retained.
Concurrent CPU lanes on a shared host also limit inference-performance conclusions.
Raw results, checked identities, and analysis are in
`.cache/runs/sentinel-v4/juraj-development/paired-analysis.json` and its `v2/`, `v4/` directories.

The standalone runtime sample covers all 16 competition rectangles, 30 frames in
each fresh process: **480 responses, zero faults**, all below the configured 10s
first-response and 150ms ordinary-response deadlines. First responses ranged
1.131–1.213s; maximum ordinary response was 4.30ms. Maximum sampled RSS was
234,811,392 bytes. The offline cache build took 70.925s. These are synthetic frames,
a reused disk cache, one CPU affinity, and the pinned official runtime on a shared
host; RSS is sampled rather than enforced by a hard cgroup. The evidence is
`.cache/runs/sentinel-v4/qualification-summary.json`, which hashes every raw shape report.
This measured runtime check does not itself establish tournament qualification.

## Amin strategy adapter and reproduction

The synchronous [strategy arena](../../scripts/strategy_arena.py) loads the reviewed
Amin checkpoint's original Python sources and original wire parser. It rejects a
changed archive or directory before import. The pinned archive is
`.cache/runs/external-opponents/amin-main8-iter160/submission.zip`, SHA256
`d69cc5d28e4805faf629c5c072a6d23011c727ca4a3b3da9f5e890ea82b78755`, from upstream
commit `34506fe2d684f163ada3a0cc3401d192436bfc37`. No opponent code or model repair is applied.

Adapter parity passed **18 complete input histories, 1,528 frames**, against the
unchanged external `run.sh`. Two histories contain 383 and 665 frames from complete
previous games; all **1,048** actions also exactly match their original recorded
raw actions. The remaining 480 frames cover all 16 shapes. Evidence is
`.cache/runs/sentinel-v4/amin-adapter-parity/parity.json`; input histories and runtime,
source, and archive identities are retained alongside it. This is action parity on
these inputs, not a universal proof for every possible game state.

The strategy arena uses the actual competition transitions and four cases per map.
Every game resets policy memory and player RNG; compiled stateless policy code is
reused. It records raw/applied actions, full wire observations, outcomes, board
arrays, inference wall/process-CPU times, and identities. Infrastructure errors abort
and preserve partial evidence. There are **no response deadlines or elapsed-time
fallbacks**: a slow inference still supplies its action. Consequently these matches
measure strategy and cannot qualify deployment or replace fault-inclusive stdio
results. Separate deadline tests remain necessary.

From the repository root, use new output directories; existing outputs are refused:

```sh
runtime=.cache/runs/competition-runtime-venv/bin/python
amin=.cache/runs/external-opponents/amin-main8-iter160
v4=.cache/runs/sentinel-v4

env -u LD_LIBRARY_PATH "$runtime" scripts/strategy_arena.py parity \
  --external-directory "$amin/agent" --external-archive "$amin/submission.zip" \
  --history "$v4/parity-histories/v2-game0.json" \
  --history "$v4/parity-histories/r2-game0.json" --synthetic-shapes \
  --output .cache/runs/reproduction/v4-amin-parity

for candidate in sentinel sentinel-v4; do
  env -u LD_LIBRARY_PATH "$runtime" scripts/strategy_arena.py run \
    --candidate "$candidate" --boards 8 --seed 83000 \
    --external-directory "$amin/agent" --external-archive "$amin/submission.zip" \
    --output ".cache/runs/reproduction/amin-strategy-$candidate"
done
```

The tool verifies Python 3.12.10, NumPy 2.4.6, JAX/JAXlib 0.11.0, and SciPy 1.18.0,
and requires CPU execution. Official-runtime imports succeeded without equinox.
Validation comprises 13 new arena/adapter tests, 11 policy tests, and 35 integration
tests: **59 distinct passing tests**. The arena tests include real tiny engine games,
per-game memory/RNG reset, preserved invalid raw commands, full-history stdio parity,
and partial-error evidence without invented replacement actions.

## Local and Amin development results

On eight consumed development maps from seed 93000 (32 cases per opponent),
both v2 and v4 finished **32W/0L/0D against Expander** and **30W/2L/0D against
Hunter**, with zero invalid moves or malformed commands. The paired score
difference is zero for each opponent on every map; each empirical bootstrap
interval is consequently [0, 0]. This describes this sample, not certainty that
the policies have equal strength on unseen maps. The strict comparator verified
identical actual boards, rules, opponent sources, and complete case budgets.
Evidence: `.cache/runs/sentinel-v4/local-development/paired-analysis.json`.

The synchronous Amin comparison completed eight consumed maps from seed 83000,
with 32 games per policy:

| Policy | Wins | Losses | Draws | Win rate | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen v2 | 12 | 14 | 6 | 37.5% | 46.875% |
| Frozen v4 | 12 | 16 | 4 | 37.5% | 43.75% |

The paired difference is **−3.125 percentage points**, with an eight-map bootstrap
95% interval **[−18.75, +9.375]**. Map 1 changes from a v2 score of 1 to a v4 score
of 0.5; map 2 changes from 0.25 to 0.5. The other six map scores are unchanged.
This is no evidence of v4 superiority, and neither observed score exceeds 50%.

The independent audit checked all 40,140 turn frames / 80,280 player calls,
identical case budgets, boards, rules, runtime and frozen source identities.
There were no infrastructure errors, malformed commands or Sentinel invalid
actions. Amin returned 378 invalid actions against v2 and 374 against v4; the
runner retained them and applied the official invalid-action pass semantics.
There were no elapsed-time replacement actions. Evidence and per-map scores are
in `.cache/runs/sentinel-v4/amin-strategy-development/paired-analysis.json` and
`per-map-scores.csv`; the two run directories retain complete wire/action traces.

The checked-in [evidence summary](v4-development-evidence.json) and
[map scores](v4-development-map-scores.csv) cover all **224 completed games**
in this cycle. Intervals use 100,000 map resamples with NumPy `default_rng(19983)`;
the JSON specifies ordering, RNG scope and the reproduction formula, and hashes
the source artifacts. Consumed seeds 83000 and 93000 remain development evidence.
Reserved seed 143000 has not been generated or inspected. A future candidate
needs a frozen plan and fresh results before promotion.

## Complete-episode failure diagnosis

In matched Amin game 5, v2 wins at turn 1186 and v4 loses at 1029. Exact replay
reproduced all 2,215 candidate observations and raw actions. The first divergence
is at turn 607: v4 rejects construction with six defenders remaining because its
two-step envelope plus buffer requires seven. The visible enemy could send only
two armies and an intervening owned cell already held three, so the construction
check is overcautious in this position. Running frozen v2 on every observation
from v4's episode finds this as the sole policy disagreement.

The eventual defeat exposes a different, shared weakness. V4 later leads 585–483
in total army, but a continuously visible invasion from turns 1016–1028 receives
too little response. Its nearby southern force moves away at 1023, reverses at
1024, and cannot intercept before the enemy sends 38 armies into a home holding
34. This is a concrete target for visible-path interception and reserve planning.
It does not justify restoring v3's stale fog memory or prove that a particular
rescue would win. Full timelines and input/source hashes are retained in
`.cache/runs/sentinel-v4/amin-strategy-development/failure-diagnosis.json`.
