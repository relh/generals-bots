# V16: serial reinforcement toward the same front

**Preregistered development screen; implementation and qualification pending.**
The completed [V16 audit](v16-audit.md) selected no policy. This separate
experimental phase tests whether serial reinforcement can turn dispersed army
into useful force at the same front while preserving successful attacks and
home defense. V2 remains the default; the [competitive goal](competitive-goal.md)
is unmet.

## Hypothesis and counterexamples

The original Juraj V3.5 V10 loss contains successful captures but poor retention,
fragmented ownership and small field packets. It never has an affordable BUILD
in its 422 decisions. Raising a construction score cannot assemble those armies.
The stronger V10-v6 control develops a connected front and wins. Conversely,
V10's concentrations win the selected Amin position where V10-v6 loses. A repair
must preserve both examples; replacing all concentration with the weaker parent
does not solve the problem.

Test a serial feeder that reinforces an existing attack toward the same front,
rather than treating each small capture as completion of the force-development
task. Admission, continuation and release must use public observations and
explicit own-plan memory. The experiment must distinguish force delivered to an
attack from transfers that merely move the packet or delay a productive route.
The concrete admission distinction is a stronger source whose existing parent
campaign transfer advances along an owned route toward the concentration's same
objective. At Juraj 100, the small collection captures at 104, while the stronger
control's feeder reaches that same target at 108 with fourteen remaining rather
than five. Amin 129 and 142 are negative controls: their larger parent proposals
attack different fronts, so source size alone must not displace those useful
collections. Static owned-garrison delivery can compare alternatives, but does
not certify future production, route safety or the opponent's response.
The [playing doctrine](playing-doctrine.md) keeps retained production, purposeful
capital search, home defense and competition construction in scope.

This is a force-efficiency hypothesis, not the rejected proposal to finish every
released defender route. That proposal cannot activate on either selected losing
control in the audit. Nor do static feasible routes or successful fixtures prove
that reinforcement will improve a complete adaptive game.

## Implementation contract before release

`SentinelV16Agent(build_castles=False, deathtouch_turn=None, max_turns=1200,
feed_front=True)` installs the feeder inside the frozen V10 stack. Disabled
`feed_front=False` returns the exact V10 tuple. Both modes retain native
nineteen-field `StrategicMemory`; enabled operation reuses inner
`OffensiveMemory.phase == 3` for serial deployment, with a packet, objective,
remaining distance and fixed expiry. V9 pursuit and V10 mobilization retain
priority as unchanged outer wrappers and can clear an unissued inner plan.

Admission compares the actual full owned V6 transfer toward the collector's
same visible objective with the new collection. The feeder must have strictly
more estimated postcapture force and strictly more force per action, tested by
cross-multiplying positive residuals and action counts. Its owned shortest
route is bounded to nine moves. Delivery credits current owned garrisons once
and charges each departure; the collector upper estimate includes deployment
contributions and may conservatively overcredit overlapping collection cells.
Neither estimate predicts growth, hidden force or adaptive replies. Preserve
ready direct deployment, existing plans and actual higher-priority actions.

Distinguish proposed inner feeder decisions from actual final execution.
`feeder_*` events require both `actual_v8_action_issued` and
`mobilization_parent_action_issued` to count as executed. Existing `offense_*`
events describe the actual inner action and require those same outer masks.
Retain collector, reference and feeder proposals separately. Reconcile unissued
plan memory explicitly; an inner start suppressed by an outer action is not an
executed feeder start.

The implementation owner must finalize and test admission, continuation, abort
rules and scalar telemetry before source freeze.
Record that contract with the test and runner prerequisites; no game may begin
with an unresolved interface or a silently changed predicate. In particular,
tests must cover useful feeder progress, captured or inconsistent packets,
obsolete objectives, continuity/reset, higher-priority actions, and exact
disabled behavior. A completed-game loss cannot be excused by a fixture passing.

Freeze source, this plan, helpers, original opponents and inputs, focused tests
and the matched runtime report before releasing games. Profile complete warmed,
synchronized V10 and V16 calls in the same runtime, including actually active
feeder memory and blocked/priority cases. The four fixed fixtures are Juraj 100,
Amin 129, Amin 142 and active phase-three Juraj 101. The last uses the candidate
returned memory from Juraj 100 with the original V10-v6 public observation at
101; bind and document compatibility rather than claiming it is archived V16
memory or a new complete game. Each preregistered fixture must retain
at least **50% of parent throughput**. Preserve cold compilation separately.
This gate does not establish standalone startup, all-shape deadlines or strength.
Record exact commands, versions, affinity and source hashes. No launches are
authorized by this document alone.

## Fixed complete-game budget

Use three arms: original V10, original V10-v6, and enabled V16. Reuse only the
concrete seed-83000 boards and original per-case keys already consumed below:

| Original adaptive opponent | Cases | Physical positions | Games across three arms |
|---|---|---:|---:|
| Amin main8 iter160 | 1, 2 | 1, mirrored | 6 |
| Juraj V3.5 | 12, 15 | 1, mirrored | 6 |
| my_bot9 | 4, 7 | 1, mirrored | 6 |

The budget is **18 complete games on three consumed positions**, six cases per
arm. Mirrors are paired cases, not independent maps. Reserved seed **143000
remains untouched**. Every game starts at turn zero against the unchanged
adaptive original opponent, using competition construction, deathtouch at 800
and a 1,200-turn cap. Replay original control oracles first; saved future opponent
actions must never replace adaptive responses to a changed candidate trajectory.

Preserve complete original public observation wires, raw and applied joint
actions, action keys, incoming and returned native memory, decisions and all
scalar telemetry. V10 and V16 have nineteen native memory fields and V10-v6 seven; validate
the exact nested schemas before execution. Record telemetry during the
original call, including the outer action-issuance mask. Verify initialization,
continuity, action normalization and exact control actions/outcomes. Distinguish
original raw-action oracles from any separately recorded memory oracle; do not
claim historical memory parity without an actual archived source.

Record malformed and invalid actions separately for each role, actual runtime
faults/deadlines where enforced, process exits and cleanup. Amin's synchronous
strategy adapter has no deployment deadlines; label that limitation explicitly.
Well-formed invalid actions retain the frozen runner's normalization and remain
in the evidence. Infrastructure failures are failures, not silent passes. Keep
all attempts, commands, logs, input/output hashes and source identities. Do not
restart healthy runs, discard faulted games, retune mid-budget or substitute a
short window for a complete outcome. Interpret aggregate strength only after
all eighteen games terminate and integrity checks complete.

## Mechanism and decision

For every complete candidate history, report actual feeder starts,
continuations, captures, releases and abort reasons; action cost and army
delivered/remaining; and what parent action was displaced. Separate proposed
events from issued ones and legal moves from confirmed next-observation effects.
Measure captured land retained through the next production tick, actual income,
castle ownership, route progress and capital exposure. Keep diagnostic access
to hidden state separate from policy inputs, and label evidence of exposure
according to its source. Account for successful controls and all losses rather
than attributing an outcome to the first differing action.

The candidate must win all three selected positions in both mirrors to pass
this development screen, preserving each position's stronger retained control.
Surviving longer, gathering more army or improving a counter is insufficient.
Report faults and original-opponent invalids as qualifications; retain their
games regardless of direction. Failure rejects this candidate for advancement
under this screen, without proving the broad reinforcement idea impossible.

Even six wins would not establish promotion or dominant strength: these are
three repeatedly inspected positions. A passing candidate still needs broader
comparisons against distinct opponents, paired uncertainty on fresh reserved
maps, and deployment qualification. Classic play, FFA and the full economic and
capital-search doctrine remain outside this narrow competition 1v1 screen.
