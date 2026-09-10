# V16: gather along an existing route to the same objective

**Implemented; nineteen behavioral tests and matched warm runtime pass.
Complete-game qualification is pending.** V16 is experimental and V2 remains the default.
The [competitive goal](competitive-goal.md) is not achieved. The
[preregistered comparison](v16-plan.md) requires eighteen complete games against
V10 and the stronger V10-v6 controls before judging this development candidate.

The [audit](v16-audit.md) found that motion and successful small captures can
coexist with fragmented territory and poorly concentrated armies. At the first
Juraj divergence, V10 starts another collection while the selected V6 campaign
move already feeds a larger packet toward the same objective. A longer serial
route can gather real garrisons along the way. This is the bounded part of our
[playing doctrine](playing-doctrine.md) tested here; it does not implement full
FFA contact decisions, a global deathball or a new capital-search policy.

## Decision and memory

Use `SentinelV16Agent(build_castles=True, deathtouch_turn=800, max_turns=1200)`
from `generals.agents.sentinel_v16_agent` for this construction-competition
experiment. `feed_front=False` returns the exact frozen V10 action, telemetry
and nineteen-field memory. Existing policies and the default are unchanged.

Enabled V16 adds a route candidate inside the V8 concentration layer of the
V10/V9 stack. It considers the actual V6 full transfer only when that transfer
advances through owned cells toward the new collection's same visible objective.
The route must take at most nine actions, pass the current home-coverage screen,
and leave more force after capture **and** more remaining force per action than
the proposed collection. Equal scores retain the original choice. New feeder
admission excludes visible enemy generals and preserves higher priorities.

The route calculation charges every departure, credits observed owned garrisons
once along a decreasing-distance path, and subtracts the target's current army.
Its first branch is the actual reference move, followed by the best remaining
shortest route. The collection comparator includes an upper bound on deployment
donations; overlap with its collection route can overcredit the collection,
making admission conservative against the feeder. These are static estimates,
without future growth, enemy replies or hidden armies. A slower, stronger attack
is a declared value hypothesis, not proof of superiority.

Phase 3 reuses the existing offensive memory to carry the packet, objective,
remaining distance, observed expected army and fixed expiry. Each subsequent
observation must support continued progress. Missing observations, lost or
depleted packets, invalid objectives, blocked routes and inadequate force abort
the plan. Defense, selected construction and visible-general tactics keep their
priority. The unchanged outer pursuit and home-mobilization layers can clear
unissued offensive transport. The home screen remains an approximate check
against currently visible threats, not a guarantee against adaptive opponents.

## Validation and instrumentation

The nineteen focused tests cover exact disabled behavior, the Juraj admission,
both productive Amin preservation fixtures, actual engine transfers and capture,
donor accounting, invalidation, defense/build/general/pursuit priorities, and
batched independent memory. Ruff passes. An initial test attempt stopped on a
fixture's incorrect `GameState` field name; its record is retained. The corrected
full suite passed in 128.01 seconds.

On the original public Juraj100 fixture, the candidate actually issues the
reference move and stores phase 3 at (10,17), with eight actions remaining and
expiry 110. Its projected remaining force is 13 over nine total actions, and its
first-step home screen passes. Both the original static audit and the exact
runtime collection comparator give four remaining over five actions. Amin129
and Amin142 preserve the parent's action, full memory and inherited telemetry.
These are consumed-state behavior checks, not new game results.

Feeder telemetry records availability, selection, issuance, route estimates,
reference and collection proposals, and applicable abort conditions. Multiple
abort conditions can hold. `feeder_differs_from_reference` compares with V6;
it does not mean that V10's original collection was retained. Inner offense and
feeder events must be masked by both `actual_v8_action_issued` and
`mobilization_parent_action_issued` before counting actual actions. A proposed
attack is confirmed as a capture only by the next recorded public observation.

The [preflight evidence](v16-preflight-evidence.json) records five rotated repeats
of 100 synchronized full calls per arm and fixture, on CPU 5 with the official
Python 3.12.10/JAX 0.11.0 runtime and persistent compilation cache disabled.

| Fixture | V16 median call | Throughput retained versus V10 |
|---|---:|---:|
| Juraj100 admission | 1.645 ms | 78.29% |
| Juraj101 active route | 1.615 ms | 81.81% |
| Amin129 preservation | 1.960 ms | 81.10% |
| Amin142 preservation | 1.860 ms | 80.35% |

All four exceed the preregistered 50% floor. Juraj101 uses the actual candidate
memory returned at 100 and the original V6 public observation after the matching
joint action; this is a constructed continuation, not archived V16 memory.
Compatible parent controls clear offensive transport. The carried phase 3 and
second Amin fixture reuse their existing compiled shapes. All 82 bound inputs
remain unchanged, and disabled calls preserve the full parent tuple.

Cold enabled compilation takes 15.03/15.70 seconds for the two distinct shapes,
excluding imports; this does not qualify uncached standalone startup. These
checks establish neither broader strength, all-shape deadlines, GPU performance
nor training throughput. The next gate is the fixed complete-game comparison.
No V16 training or complete game has run at this checkpoint.
