# V20 branching collector: implementation and preflight

V20 assembles owned side branches into an offensive rally under a shared
six-action budget. It has not played its complete development screen and is
not promoted. The [plan](v20-plan.md) defines the experiment; the
[post-V19 audit](v20-audit.md) remains a separate historical investigation.

The pure planner is `generals/agents/branch_collection.py`; the one-parent-call
wrapper is `generals/agents/sentinel_v20_agent.py`. No earlier policy changed.
Use `SentinelV20Agent(build_castles=True, deathtouch_turn=800, max_turns=1200)`
for this competition experiment. `branch_collection=False` returns exact
V10 outputs and native nineteen-field memory. Enabled memory has 28 scalar
int32 leaves. No default, standalone bundle or live submission was changed.

## What is newly exercised

A synthetic public fork needs ten armies at its rally. Either single donor
path supplies eight; two serial branch donations supply eleven in two actions.
The complete wrapper issues those moves, reconciles their recipients, then
allows the parent's sufficient direct attack to take over. A separate synthetic
six-edge tree delivers 35 where every simple path delivers at most 15 against
required force 24. These demonstrate representational capability, not wins.

An admission census calls V20 on every original default-V10 observation,
original native parent memory and key, with empty branch transport. It checks
2,531 consumed decisions, not changed candidate trajectories:

| Original history | Decisions | Branch admissions | Exact parent actions | Exact parent native / inherited telemetry |
| --- | ---: | ---: | ---: | ---: |
| Winning Amin | 1,033 | 0 | 1,033 | 1,033 / 1,033 |
| Losing Juraj V3.5 | 422 | 6 | 416 | 422 / 422 |
| Winning my_bot9 | 1,076 | 22 | 1,054 | 1,076 / 1,076 |

Juraj admissions occur at 183 and 343–347. My_bot9 admissions occur at 221,
232–249, 515–516 and 551. Every admission changes the physical action. No
branch starts anywhere along the original winning Amin history, so this census
establishes exact behavior along that control trajectory. It is not a new
paired game or evidence about other Amin positions. Once a branch starts,
subsequent original observations cannot stand in for the changed trajectory.
The census therefore provides a reason to test the failing Juraj case, not a
claim that the new collection survives or wins.

## Correctness and review

All 40 final tests pass in pinned JAX 0.11.0. They include an independent
exhaustive tiny-tree oracle, zero-army intermediate success/stall, six-step
budget limits, deterministic ties, typed scalar/JIT/batched memory, real
wrapper admission and parent takeover, residual collection, early deployment,
fixed-expiry rejection, missing-recipient rejection, reset, parent priorities,
and actual engine land growth. Deployment cannot borrow army still at the old
rally. Nine archived fixtures retain exact wire/key/native/output provenance;
legacy V10-v6 references are checked against their actual parent and are not
misrepresented as recorded default-V10 calls.

The static review caught stale incoming-defense priority on reset and missing
residual-work expiry checks; both were fixed before the first compiled wrapper
probe. The first eight wrapper tests had two incorrect expectations: the parent
cannot yet collect after the first donation because its 1.5 multiplier still
fails, and reset telemetry correctly differs from fresh-start telemetry. Their
assertions were corrected without a policy change. The next 38-test suite and
two engine checks passed. After fixing planner lint (line wrapping and replacing
a lambda assignment), all 40 tests passed together again. Earlier attempts,
source snapshots and logs remain bound in the preflight evidence. Ruff passes.

## Full-call runtime

Nine fixed contexts pass the 50% throughput gate in pinned JAX/JAXlib 0.11.0
on Zephyrus CPU 5. Each arm receives ten warmups, then five rotated repetitions
of 100 fully synchronized complete calls. Ratios use total measured time over
those 500 calls, not the ratio of median individual latencies.

| Context | Actual wrapper behavior | V20 / V10 throughput |
| --- | --- | ---: |
| Synthetic fork start | Branch collection starts | 56.9% |
| After first actual fork donation | Branch collection continues | 60.5% |
| After second fork donation | Parent takes over | 82.0% |
| Manually supplied deployment state | Parent takes over | 72.9% |
| Original Amin 348 | Defense priority | 86.4% |
| Original Juraj 420 | Defense priority | 92.4% |
| Original Juraj 348 | Inspect, no branch available | 56.5% |
| Original Juraj 183 | Branch collection starts | 53.2% |
| Original my_bot9 221 | Branch collection starts | 57.6% |

All disabled full tuples match their corresponding V10 calls. The two real
admissions also match the independently recorded census outputs exactly.
Four contexts actively collect; no measured full-wrapper context actively
executes branch deployment. Its state logic is unit-tested, while the measured
deployment candidates actually hand off to the parent. The narrowest runtime
margin is Juraj 183: its median individual V20 call is 2.27 ms versus V10's
1.07 ms, although the total-sample throughput ratio is 53.2%. Preserve that
variation; this is not a uniform speed or deadline guarantee.

First compilation calls for new measured V20 shapes/rules take about 11–13.3
seconds, excluding imports. These are not standalone startup measurements.
The initial profile import failed before measurement because pytest was absent
from the pinned environment; the corrected helper appends the development
site directory after verifying the pinned JAX provider. A later optional 9×9
fixture failed parsing after six completed measurements. Its original fixture
and failed attempt remain intact; the remaining required context ran in a
separate additive attempt. The two newly identified game-sized admissions then
ran separately, without repeating any of the seven completed contexts.

The [preflight evidence](v20-preflight-evidence.json) binds final sources,
qualification attempts, original fixtures, admission records, timing samples,
runtime providers and source reviews. Root verified 345 bindings and recomputed
all nine throughput ratios from the saved samples. Historical release files
are immutable inputs; their older source versions are not relabeled as current
code. The game checker can explicitly recompute
the actual V10 reference tuple from each recorded public input, native parent
memory and original key, then check the new branch logic independently. Such
fresh reference inference must not be described as an archived inner oracle;
exact inactive/full-parent and inherited-telemetry checks remain required.

## Remaining gate

No V20 complete games, fresh holdouts or standalone deadline qualification have
run. The next release needs the frozen full-game harness and independent branch
checker, followed by the 24-game control/candidate screen in the plan. The
current parent/opponent histories remain immutable controls. Complete outcomes,
not successful arithmetic, admission counts or preflight runtime, determine
whether this version advances. The broader competitive goal remains active.
