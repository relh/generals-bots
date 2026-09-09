# Sentinel v5 development

The competitive goal remains active and unachieved. **V5 is not promoted;
v2 remains the default.** V5 improves the small Juraj V3.5 sample but still scores
below 50% against it and regresses on Amin. Tactical rescues do not establish a
winning strategy.
The [plan](v5-plan.md) tests whether visible-route interception improves complete
games beyond v2's short-range reserve rule. V5 starts from frozen v2, without
v4's construction change or v3's fog memory.

## New opponent exposes a larger gap

Against the unchanged deterministic Juraj V3.5 rewrite, frozen v2 finished
**2W/14L/0D**, score **12.5%**, on four consumed maps from seed 83000. Both seats
and spawn labels give 16 cases. The four map scores are [0, 0, 0.5, 0].
This is a different opponent from Juraj V3.4, against which the earlier v2
sample was 11W/5L. Version names do not imply measured relative strength.

The independent audit verified all 11,390 turns / 22,780 replies, retained boards,
immutable execution segments, exact standalone policy/opponent bytes, runtime
versions and simulator identities. Both bots recorded zero reply faults, invalid
actions, stale replies, skipped observations, or forfeits. All 32 processes were
reaped. Evidence: `.cache/runs/sentinel-v5/juraj-v35-development/baseline-analysis.json`.
The public opponent's source/archive identities and compiler difference are
recorded in the [coverage census](opponent-coverage.md).

## Evidence behind the intervention

In the earlier Amin game 24, a legal forward reinforcement at turn 232 prevents
the recorded defeat at turn 238, leaving home with 11 armies. The actual engine
replay keeps all later actions fixed, so it establishes a tactical opportunity,
not a complete-game win against an adapting opponent. Other inspected losses
lack a large enough nearby army; recalling an unusable force cannot fix them.
Cache report: `.cache/runs/sentinel-v5/design-audit/REPORT.md`.

The complete first Juraj V3.5 loss reproduces winner and turn across all 466
recorded engine transitions. V2 loses before deathtouch while leading 511–423
in total army. At turn 459 it moves an 18-army screen off a homebound corridor;
at 461 it misses a nearby counterattack. These are observed opportunities rather
than verified winning counterfactuals. They support testing the same intervention
against another opponent. Cache report:
`.cache/runs/sentinel-v5/juraj-v35-game0-audit/REPORT.md`.

## Candidate contract

V5 considers currently visible enemy stacks within ten route steps, and at most
twelve interception targets. It checks friendly arrival time and troop collection
along owned routes, excluding garrisons already counted as stationary defense.
Reinforcing an owned tile at the same time as an enemy attack is permitted by
the engine's move order. Ordinary combat can be stopped through combined forward
attrition and home defense; projected deathtouch requires preventing the touch.
Guaranteed home production before contact is included.

The planner checks the baseline action's effect on an existing defensive screen,
retains immediately winning captures, and returns the baseline when no feasible
intervention is found. Candidate routes are approximate: alternate approaches,
enemy growth/merges, several simultaneous invaders, and future fog movement are
not solved. No tactical or runtime check establishes competitive superiority.

Use `--candidate sentinel-v5` in the local or synchronous strategy arena;
`sentinel-v5-disabled` is the frozen-v2 ablation. Standalone bundles select
`scripts/build_sentinel_bundle.py --variant v5 --output PATH`.

## Frozen strategy and runtime evidence

V5 source SHA-256:
`d8e7f41421095607bc53a7f4372eba32ea7a9e5e74614fdee8e270d78698941b`.
Standalone archive SHA-256:
`e5770991c08685ff595568ae5796b65045f33b899471d8eb8b8d13217c36dd25`.
The original v2/v3 helpers remain byte-identical. Sixty-six focused tests pass:
12 policy tests, 42 arena/replay/comparison tests, and 12 adapter/bundle tests.
The disabled ablation matches v2 actions and original telemetry; the enabled
policy supports JIT/vmap. A three-step engine test verifies actual owned-tile
transfers and excludes the existing screen from newly added defense.

The Amin adapter again matched the original bot on 18 histories / 1,528 actions,
including 1,048 originally recorded actions from two full games. Both strategy
runs completed 32 cases on eight consumed maps from seed 83000:

| Opponent / evaluation | V2 W/L/D | V5 W/L/D | V2 score | V5 score | Paired difference, 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Expander, local arena | 32/0/0 | 32/0/0 | 100% | 100% | 0 pp [0, 0] |
| Hunter, local arena | 30/2/0 | 32/0/0 | 93.75% | 100% | +6.25 pp [0, +18.75] |
| Amin, synchronous strategy | 12/14/6 | 10/14/8 | 46.875% | 43.75% | −3.125 pp [−9.375, 0] |
| Juraj V3.4, deadline enforced | 11/4/1 | 15/1/0 | 71.875% | 93.75% | +21.875 pp [+6.25, +34.375] |
| Juraj V3.5, deadline enforced | 2/14/0 | 6/10/0 | 12.5% | 37.5% | +25 pp [0, +50] |

Intervals use whole-map paired bootstrap resampling (100,000 samples,
`default_rng(19983)`), with four seat/spawn cases per map. These small consumed
samples are exploratory. Amin's mean map score changes only on map 1, where two
v2 wins become draws. On map 5, two draws become wins and two wins become draws,
leaving that map's mean unchanged. Juraj V3.5 improves on maps 0 and 3, by 0.5
score each. Neither candidate
establishes a winning advantage against both opponents.

Juraj V3.4 uses four consumed maps from seed 83000; its map differences are
[0, 0.25, 0.375, 0.25]. All four policy-specific environment variables were
explicitly unset in both commands and recorded. This original opponent seeds
from entropy and a clock, so only maps, seats and spawn labels are paired;
internal random choices differ. The observed interval does not measure repeated
opponent-RNG uncertainty. The separate V3.5 rewrite has deterministic selection.

Local runs use eight consumed maps from seed 93000, 32 cases per opponent.
Hunter changes only on map 0; Expander's map scores are all unchanged. Both
policies produce zero invalid or malformed commands. The same frozen simulator,
actual boards and complete cases were checked by the strict comparator. On this
shared host, V5's local batch run was slower: 220.697 vs 124.667 seconds for
Expander and 192.698 vs 140.227 for Hunter, including their measured run overhead.
These different policies and episodes do not isolate a compiler-performance
effect; single-action runtime qualification is reported separately below.

The synchronous Amin audit validated all 41,410 turns / 82,820 player calls with
no infrastructure errors, malformed commands, or Sentinel invalid actions.
Amin returned 378 invalid actions against v2 and 422 against v5; raw actions and
official invalid-action pass semantics are retained. No timing-based replacement
actions occur in these strategy-only runs. Juraj V3.5's 32 complete games contain
49,280 replies, with zero faults, invalid actions, stale replies, or forfeits for
either player and all 64 child processes reaped. Actual frozen archive, extracted
directory, cache, binary, simulator and segment identities were checked, alongside
runtime paths and versions. Historical stdio metadata does not retain hashes of
the interpreter and installed package bytes, so those bytes cannot be proven
retrospectively identical from these records.

Juraj V3.4's 32 games retain 22,853 turns / 45,706 replies, also with zero faults,
invalid or malformed actions, stale replies, skipped observations or forfeits;
all 64 child processes were reaped. The strict comparator and independent raw
audit verified actual boards, rules, opponent and policy artifacts, immutable
segments and the explicitly unset launch environment without source exceptions.

The standalone qualification passed all 16 shapes × 30 frames, **480 responses
with zero faults**. Cached first-response times ranged 1.733–8.356 seconds;
warmed responses had median 6.142 ms, p95 8.407 ms and maximum 13.735 ms.
Peak sampled RSS was 260,120,576 bytes. The worst startup leaves about 1.64 seconds
under the configured 10-second limit, so this sample does not establish robust
startup margin. These are fresh processes using a built cache on a shared host,
synthetic frames, and sampled RSS rather than a hard cgroup limit. The uncached
18×21 probe passed separately; its startup was about 7.72 seconds. Its actions
matched the built/reused probes exactly on the same input frames.

Cache build time was 122.501 seconds internally and 124.182 seconds measured by
the outer probe, with 3,869,910 final bytes including the build report. The
qualification did not change any bytes in the directory captured by the concurrent
Juraj V3.5 run. Runtime success does not establish playing strength or an official
tournament qualification.

Raw evidence lives under `.cache/runs/sentinel-v5/`: `amin-strategy-development/`,
`juraj-v35-development/`, `amin-adapter-parity/`, `qualification-summary.json`, and
the individual qualification reports. The full policy audit reproduces **all
21,072 V5 actions across 32 complete Amin games** from their retained public wire
observations. It records 334 interception overrides, 42 guard overrides, and 330
frames with a feasible plan; feasibility can overlap an unchanged or guarded
choice and is not an additional override count. Source/input hashes and all
scalar telemetry are retained in `amin-policy-audit/`. This audit establishes
policy-action parity on these trajectories, not a separate outcome replay.
Overrides occur in 18 of 32 games. No Amin loss becomes a win: the recorded
238-turn defeats are delayed to 601, while other interventions trade draws and
wins. This is why survival-time improvements alone cannot satisfy the goal.
All **256 evaluation games** in this cycle are complete. The checked-in
[evidence summary](v5-development-evidence.json), [map scores](v5-development-map-scores.csv)
and [complete case table](v5-development-cases.csv) retain results and reproduction
information. Reserved seed 143000 has not been used; all current comparisons are
explicitly development evidence.

## Failure diagnosis and next experiment

In Amin game 5, a v2 win at 1186 becomes a v5 draw at 1200. The first divergence
at turn 330 responds to a credible nearby threat; the actual enemy instead passes
by home. Later, at 877–881, all five interception overrides move the same
37-army stack back and forth while the visible enemy moves away. Recomputing the
cheapest attack corridor changes which location counts as newly added defense.
V5 also overrides all seven v2 construction choices available at 757–763.
Both full episodes have 11 passes, so this is not a pass-count stall.

The selected-frame diagnosis checks 62 exact policy decisions and retains full
episode build/pass counts and source/case hashes in `game5-failure-diagnosis.json`.
It does not quantify the cause of the aggregate score difference. The next
candidate should test progress toward an owned defensive plan, releasing it when
current visible evidence invalidates it, and avoid counting the same relocated
army as fresh defense repeatedly. This is distinct from remembering hypothetical
enemy armies in fog. Any such change must preserve the real equal-arrival rescue,
multi-step collection, already-covered growth, and hopeless-recall controls and
then improve complete-game results before promotion.
