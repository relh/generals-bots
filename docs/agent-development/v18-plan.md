# V18: separate-garrison expansion

**Preregistered experiment; qualification pending.** V17 lost four of six
selected development cases. V18 tests whether diverting an uncommitted owned
transfer to capture empty neutral land with a separate two-army garrison earns
enough durable income to repay the delayed transfer. V2 remains the default.
This experiment does not establish the [competitive objective](competitive-goal.md).

## Evidence and hypothesis

The complete V17 audits found defensive concentration failures and costly
territory allocation. Subsequent public-state censuses found many possible
rear captures in both losing candidates and winning controls. Availability is
not evidence that replacing those controls' actions is harmless. A proposed
hidden-force scalar alarm also failed to distinguish the controls: public
scores leave large uncertainty, and assuming every owned tile contains at
least one army is invalid under the engine's capture and construction rules.
V18 therefore tests a bounded action-allocation hypothesis, without treating
the scalar alarm as a defense certificate.

## Frozen policy contract

`SentinelV18Agent(rear_expansion=True)` wraps one actual default-parent V10
call. Disabled operation returns its exact complete tuple. Enabled operation
retains the parent's exact native nineteen-field memory and inherited telemetry.
There is no new commitment or remembered contact state.

Arbitrate only while an opponent tile is currently visible and V10 selects an
owned transfer. Preserve captures, builds, passes, visible enemy generals and
enemy generals remembered in either incoming or returned memory. Both incoming
and returned offensive phases must be zero and both defender slots uncommitted.
Preserve defense priority, intercept guard/override or positive home deficit,
adjacent threat, a home reserve deficit, every active defender commitment event,
pursuit and mobilization. The source defines the exact telemetry predicates.

Enumerate full moves from a different ordinary owned tile with exactly two
armies to an adjacent visible ordinary neutral tile with zero armies. Neither
endpoint may border a currently visible enemy with positive transferable force.
Exclude generals, castles, mountains, fog and hidden structures at both endpoints.
Rank destinations by shortest public terrain distance to home, then number of
adjacent visible empty neutral tiles, then flattened source/direction order.
Ordinary fog is traversable in this terrain-distance estimate; it is not a
known safe route. Five times the integral distance dominates the zero-to-four
branch count. Unreachable targets are excluded.

Check only the selected move with the inherited V7 current-visible radius-ten
home-coverage approximation. If it fails, retain V10's move without searching
another candidate. This cannot certify hidden forces or future opponent replies.
The expansion moves one army and leaves one behind; the original packet keeps
its army but loses its action. Inherited issued-action events additionally
require `rear_parent_action_issued`. Since the source differs, an issued rear
move always changes the actual action. Memory remains the actual parent output.

## Qualification before games

Use six original V10 public calls from the completed V17 controls: Amin 150,
Juraj 78, Amin 129, Amin 169, Amin 58 and my_bot9 64. These cover proposed
expansion opportunities, collection, defense, precontact and an actual capture.
Their original wires, keys, incoming memory and full parent outputs are frozen
in `tests/fixtures/rear_expansion/`. Opportunity labels are not assertions that
the final safety gate issues a move. Verify that distinction before games.

Require exact disabled output, unchanged enabled memory/inherited telemetry,
hand-checked enumeration/ranking, priority preservation, and actual engine
capture/production transitions. An independent public-only checker must review
the new arbitration and reject deliberately corrupted records. Record actual
test counts and all failed attempts; no historical V17 test count substitutes.

Compare synchronized full V18, disabled V18 and V10 calls in those six contexts,
using the pinned Python 3.12.10/JAX 0.11.0 CPU runtime on Zephyrus CPU 5, with
compilation cache disabled. After ten warmups, use five rotated repeats of one
hundred calls per arm/context. Each enabled and disabled context must retain at
least 50% of V10's pooled median throughput. Record imports and first calls
separately, including reused shapes. This is neither all-shape qualification
nor a deployment deadline test. Freeze the plan and timing helper before timing;
freeze final source, tests, reports, fixtures and all runners before games.

## Fixed development screen

Run the unchanged V10, V10-v6 and V16 controls plus enabled V18, completing
controls before candidate games. Use original seed-83000 boards and per-seat
action keys, construction, deathtouch 800 and a 1,200-turn cap:

| Original adaptive opponent | Cases | Games across four arms |
|---|---|---:|
| Amin main8 iter160 | 1, 2 | 8 |
| Juraj V3.5 | 12, 15 | 8 |
| my_bot9 | 4, 7 | 8 |

These are three already consumed physical positions with label mirrors.
Reserved seed 143000 remains untouched. Preserve original observations,
raw/applied joint actions, keys, native incoming/returned memory and telemetry
from actual calls. Require complete control parity against original records,
including real V16 native-memory oracles. The previously justified fused-score
V16 checker must be frozen from the outset; V18 has no V16 feeder or V17 chain
phase and requires its own checker. Keep public-wire reconstruction and full
history review independent. Preserve invalid actions, failures, deadlines,
terminal exits and cleanup, distinguishing synchronous candidate/Amin timing
from enforced external-opponent stdio deadlines.

Complete all 24 games and integrity checks before interpreting results. Do not
retune or restart healthy games. Confirm each rear capture in the next public
observation and measure retention through production ticks and 50/100 turns,
with terminal censoring explicit. Audit delayed transfers, field concentration,
complete losses and capital exposure. Attempts and local capture totals cannot
replace durable income or wins.

V18 must win all six cases to advance and preserve the strongest selected
control on each position. Even six wins require broader distinct-opponent
comparisons, fresh paired maps with a win-rate confidence lower bound above
50% per opponent, and valid deployment runtime before promotion.
