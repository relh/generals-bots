# V6 defensive commitment development

V6 is an experimental candidate. The competitive goal remains unachieved and
V2 remains the default. V6 commits to an owned defender and fixed target, checks
observed arrival consistency and current visible need, and briefly protects a
recently placed screen from being recruited straight back. Urgent guards and
winning captures take priority. Disabling commitment reproduces frozen V5.

All 256 planned games finished and passed their comparison audits. The small
Amin gain, unchanged Juraj V3.5 results and uncertain Juraj V3.4 result do not
establish competitive superiority. See the checked-in
[evidence summary](v6-development-evidence.json),
[paired map scores](v6-development-map-scores.csv) and
[128 paired cases](v6-development-cases.csv).

The policy and evaluation sources were frozen before comparison. Development
uses previously consumed seed 83000 for external opponents and 93000 for local
opponents, with both seats and general-label assignments. Seed 143000 remains
reserved and uninspected. No official submission or leaderboard rank is claimed.

## Complete Amin comparison

Both versions finished all 32 cases on eight maps using the unchanged pinned
Amin policy and synchronous reference adapter. V5 recorded 10 wins, 14 losses
and 8 draws; V6 recorded 12 wins, 14 losses and 6 draws. Scores were 43.75% and
46.875%, a paired difference of +3.125 percentage points with map-bootstrap
95% interval [0, +9.375]. Only cases 9 and 10 changed outcome, from draws to wins;
both belong to map 2. No loss became a win. This small consumed-map gain does
not establish a winning advantage over Amin. The historical V2 score on these
maps was also 46.875%; this cycle directly compares V5 with V6.

Original-opponent parity passed 18 histories and 1,528 frames, including 1,048
previously recorded frames and synthetic inputs for all 16 shapes. Every fresh
V5 public observation, raw/applied action and outcome matched its prior complete
32-game run (21,072 turn frames), excluding timing. The comparison applies every
returned action and makes no response-deadline claim.

The raw audit covered 42,212 turn frames and 84,424 policy calls. Neither
candidate emitted an invalid or malformed command; the original Amin opponent
emitted 422 invalid actions against V5 and 396 against V6. All 21,140 V6 actions
were reproduced with sequential memory. Its telemetry records 94 plan starts,
150 continued transport frames, 94 held frames and 104 changes to V5 proposals,
with zero observation gaps.

## Local preservation

Both versions won all 64 local games: 32 each against Expander and Hunter on
eight maps per opponent. The strict paired comparison found identical case
outcomes, turns and recorded action diagnostics, with zero invalid or malformed
commands. Each opponent's paired score difference and map-bootstrap interval
are zero. This is preservation on the consumed sample, not raw action equality
or independent evidence against additional external designs.

Recorded GPU run wall times were 175.933/176.867 seconds for V5 against
Expander/Hunter, versus 226.564/212.188 seconds for V6. These shared-host run
measurements include compilation and execution and do not isolate a policy
kernel speed ratio. All 37 frozen source hashes matched before and after both
sequential runs.

## Juraj V3.5 comparison

Both versions finished 6 wins and 10 losses on four paired development maps,
scoring 37.5%. Every map mean was unchanged; the paired difference and its
map-bootstrap interval are zero. V6 has no measured improvement or winning
advantage against this opponent. All 32 complete games, 26,672 turn frames and
53,344 replies passed the raw audit with zero faults, invalid/malformed actions,
stale replies, skipped observations or forfeits. All 64 child processes were
cleanly reaped. The original deterministic Juraj V3.5 binary stayed unchanged.

## Juraj V3.4 comparison

V5 finished 13 wins and 3 losses, scoring 81.25%; V6 finished 12 wins, 3 losses
and 1 draw, scoring 78.125%. The paired difference is -3.125 percentage points
with 95% map-bootstrap interval [-34.375, +18.75]. The four map differences
were [0, +0.125, -0.5, +0.25]. This noisy result does not establish improvement.
The opponent's original entropy/clock-driven random draws are not paired;
`JURAJ_RNG_SEED`, `JURAJ_V3_SPLIT`, `JURAJ_V3_CASTLES` and `JURAJ_V3_TRACE`
were all unset. The four-map confidence interval does not remove that limitation.

All 32 complete games, 22,211 turn frames and 44,422 replies passed the raw audit
with zero faults, invalid/malformed actions, stale replies, skipped observations
or forfeits. All 64 child processes were cleanly reaped. Both Juraj comparisons
used the original pinned source archives/binaries, actual cached candidate
bundles and immutable execution metadata, with no source-equivalence exceptions.

## What the complete backtracking case explains

Game 5 still draws after 1,200 turns. A complete sequential-memory replay
reproduced all 1,200 V6 actions. Actual consecutive full reversals fell from
12 to 3; both versions made 11 passes and built no castles. The first changed
action was at turn 643, before the previously diagnosed 877–881 oscillation.
A hold at 643 followed by release at 644 demonstrates the intended behavior;
it does not prove every later hold useful. The audit distinguishes inherited
V5 proposal telemetry from changes to V6's final executed action.

The enemy general at (14, 5) was publicly visible at turns 72–78, then disappeared
into fog and was never revealed again. After turn 800, V6 never got an owned
square closer than three Manhattan steps to that previously observed general.
The campaign forgets this stationary objective when it leaves current vision,
and its field forces remain fragmented. Remembering a publicly discovered,
stationary objective is a concrete next hypothesis; this diagnosis does not
claim a winning counterfactual or justify remembering unobserved moving armies.

A public-history census bounds this hypothesis: only two of six V6 draws had
previously revealed the enemy general, and only two of fourteen losses ever
revealed it. Twelve losses therefore also expose a discovery or offensive
progress gap. These paired case counts do not identify twelve independent
failures or prove that earlier discovery would have prevented defeat.

The next primary campaign hypothesis is persistent offensive concentration
toward a useful destination. Three losing placements (each with a mirrored
case) ended with more observed army and land than Amin, but largest field stacks
of only 12–13. A bounded collection plan must use real owned donors, account for
travel and leave-behind troops, preserve current defensive coverage and reconcile
its issued moves. Search persistence and revealed-general retention should be
separate ablations. More builds or a larger stranded stack alone do not establish
improvement; the next cycle still needs complete-game evidence.

## Runtime and validation

85 focused and integration tests passed: 13 V6 policy tests, 12 frozen-V5
regressions and 60 adapter, bundle, replay, comparison and reference-runner tests.
The tests cover actual owned transfers, equal-ETA rescue, fixed-target progress,
arrival loss/depletion, enemy growth, obsolete plans, gaps/expiry, guard and
winning priorities, disabled action/telemetry parity, and batched memory. The
public game-5 fixture is a bounded reconstructed tactical continuation, not a
complete hidden game state. A pre-existing test's mountain encoding was corrected
from -1 to the engine's -2; prior policy and simulator sources stayed unchanged.

Uncached complete-step compilation/execution took 10.455–11.068 seconds on two
real public observations, excluding process startup. V6 therefore does not meet
the ten-second cold startup limit. Its actual offline cache build completed in
204.031 seconds internally (206.081 seconds measured by the outer probe).

With that cache, all 16 fresh-process shape probes passed: 480 responses with
zero faults, first responses 1.986–3.607 seconds, warm maximum 15.892 ms, and
sampled peak RSS 294,281,216 bytes. All 30 built/reused 18×21 actions matched;
bundle and cache bytes remained unchanged throughout qualification.

The actual standalone bundle also reproduced all 1,200 raw actions from the
complete game-5 active-memory history under the response limits, with zero faults,
clean EOF exit and unchanged cache identity. An initial helper attempt compared
normalized PASS coordinates against raw output and rejected an otherwise exact
reply; its failed report is retained. The corrected helper separately checks
raw replies against raw actions and normalized replies against applied actions.
This was an analysis-helper correction, with no policy or runner changes.

## Artifacts and limits

V6 policy SHA-256:
`bc6764f3bf7ae64ea380e91e7821bfd2d3c66e1aaec8b45a5a8da3ddb8cf95fa`.
Standalone archive SHA-256:
`0fb150b8662480c3a5e799f410f2bef244abfe7cd4d977bcb10a31decfd49e42`
(25,997 bytes, 16 source files, 72,426 unpacked source bytes). The actual built
bundle contains 132 files and 5,712,177 bytes including its build report.

Raw runs, original wire histories, source freeze, commands, complete diagnostics,
qualification and hashes are retained under `.cache/runs/sentinel-v6/`.
The pinned runtime is Python 3.12.10, JAX/JAXlib 0.11.0, NumPy 2.4.6 and SciPy
1.18.0. CPU affinity is recorded; the host is shared and RSS is sampled. The
measurements do not establish performance on every official host.

V6 still approximates one visible threatening stack. Alternate routes, merges,
multiple attackers and long-horizon strategy remain limited. Arrival consistency
is a conservative observation check, not proof of the engine's exact intervening
sequence. A fallback action can affect another screen, and expiry can permit a
new plan; the bounded memory is not a lifetime cap on defensive assignments.
Other public leaders remain untested as described in the
[opponent census](opponent-coverage.md).

The full competitive goal remains active. This cycle delivers the validated
experimental policy, evaluation integration and evidence to our fork's main;
it does not promote V6 or close the remaining opponent-strength gaps.
