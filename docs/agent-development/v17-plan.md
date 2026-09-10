# V17: productive capture continuation

**Preregistered experiment; implementation and qualification in progress.** V16's
completed routes did not preserve the stronger controls. The preceding
[public-state audits](v17-audit.md) identify costly objective choices and a
separate failure to concentrate defensive force early enough. This experiment
tests the former. It does not claim to repair all force allocation, implement
the full FFA doctrine or achieve the [competitive objective](competitive-goal.md).
V2 remains the default.

## Hypothesis and contract

When V16 starts an immediate direct attack on an ordinary visible enemy tile,
compare capture chains from that same selected packet. Consider at most three
consecutive captures of currently visible ordinary tiles, starting with an
enemy tile and permitting enemy or neutral continuations, with no owned
transfers, revisits, structures or assumed empty fog. Charge every departure
and defender once. Rank feasible chains by captured tile count, then surviving
force, preserving the original first action on exact ties. This uses an explicit
short route's material value; it does not globally prefer weak opponents or
cancel useful collections because another packet can expand elsewhere.

Commit a selected chain of at least two captures so the policy can execute
the valued continuation. Revalidate ownership, current army, visible
counterforce, home coverage, observation continuity and a fixed expiry before
each action. Defense, construction, general opportunities and outer pursuit or
mobilization retain priority. Preserve collection and feeder admission when
the immediate direct-capture trigger does not apply. Disabled operation must
return the complete frozen V16 tuple, including memory and telemetry.

`SentinelV17Agent(capture_chains=True)` places this arbitration inside the frozen
V16 campaign. Disabled `capture_chains=False` preserves V16. Both retain native
nineteen-field memory. Offensive phase 4 stores the current packet, at most two
pending destinations in `rally`/`objective`, remaining count, expected postcapture
army and expiry fixed to the final planned action. Phase 4 is cleared before the
single frozen-parent call, retaining its real defender state. Completion returns
to ordinary selection; no extra postcapture marker extends commitment.

Each remaining suffix must still be visible, distinct, adjacent and affordable
with residual force strictly greater than the largest current neighboring enemy
surplus after removing earlier projected captures. This does not model enemy
mergers. Only the selected next action receives the inherited radius-ten home
coverage check; if it fails, conservatively use the parent rather than searching
for a different safe alternative. Home safety of future moves is rechecked when
those observations arrive, not certified for the entire route at admission.

Retain the original inner proposal and the capture-chain proposal. Inherited
V16 events require `chain_parent_action_issued` and both unchanged outer masks
`actual_v8_action_issued` / `mobilization_parent_action_issued`. Chain events
require both outer masks. The parent mask compares actual action equality:
starting a chain with the same first move can change commitment while still
issuing the parent action. Distinguish this from a changed move and discard
unissued route memory. `chain_available` reports enumeration availability;
original comparison scores require `chain_comparison_available` and are -1
outside a new admitted comparison. Static residual estimates do not predict growth or
adaptive opponent replies. Freeze the finalized tested implementation before
running the budget; no mid-run predicate changes are permitted.

The Juraj 109 and Amin 214 recorded states motivate alternative captures.
Amin 223 and 163 are scope/preservation checks. Earlier Juraj 100 and Amin
129, 142 and 250 must retain their original collection or feeder admission.
Tests also cover disruption, insufficient force, priority, observation reset,
actual continuation and exact disabled behavior. A useful short route cannot
excuse losing a complete game or justify a claim about unseen enemy force.

## Qualification before games

Freeze source, focused tests, this plan, fixture inputs, runtime report and
runner helpers before any game. Compare full synchronized warmed V17 and V16
calls under the same pinned CPU runtime, with rotated repeated measurements.
The eight named recorded states above and an active carried continuation are
the preregistered contexts. Each must retain at least 50% of parent throughput.
Validate exact disabled output separately. An active continuation must have a
documented compatible observation; do not present synthetic memory as an
archived candidate trajectory. Keep cold compilation and environment details
separate. This gate is not standalone deadline or all-shape qualification.

## Fixed development screen

Run four arms: original V10, original V10-v6, frozen V16 and enabled V17. Use
only the same concrete seed-83000 boards and original action keys already
consumed in V16:

| Original adaptive opponent | Cases | Games across four arms |
|---|---|---:|
| Amin main8 iter160 | 1, 2 | 8 |
| Juraj V3.5 | 12, 15 | 8 |
| my_bot9 | 4, 7 | 8 |

The budget is 24 complete games on three inspected physical positions, each
with its label mirror. Reserved seed 143000 remains untouched. Use competition
construction, deathtouch at 800 and a 1,200-turn cap. Replay complete control
oracles first, then run changed policies from turn zero against unchanged
adaptive opponents. Saved future opponent actions cannot stand in for responses
to a changed trajectory.

Record every public observation, raw/applied joint action, original action key,
incoming/returned native memory and scalar telemetry in the actual policy call.
Verify complete control parity against real archived sources; label any missing
historical memory instead of inventing it. Retain all failures, attempts,
invalid/malformed actions by role, enforced deadlines, exits and cleanup. Amin's
synchronous adapter has no deployment deadlines. Candidate synchronous runtime
and original-opponent stdio deadlines remain separate measurements.

Do not retune mid-budget, restart healthy jobs, discard failed games or interpret
aggregate strength before all 24 games and integrity checks finish. Audit actual
chain starts, continuations, captures, residual force, releases and displaced
actions. Confirm captures in the next observation, measure ownership retention
through production ticks, and examine complete losses and capital exposure.

V17 must win all six cases to advance, preserving each position's stronger
control. Better local counters, survival time or aggregate wins cannot replace
that gate. Even six wins would require broader distinct-opponent comparisons,
fresh reserved paired maps and deployment checks before any promotion. The
earlier defensive allocation gap remains open unless complete evidence shows
the changed strategy resolves it.
