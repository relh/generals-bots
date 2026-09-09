# V8 action-cost experiments

The competitive objective remains active and unachieved. V7 reached the fork's
main at `ff3e71efce0e477eddbf03733f7177d01abf3855`, with no promotion. Its complete
audit confirms 788 of 792 normal objective attacks, but 520 of those captures
spend at least four collection actions against at most three defenders. Median
collection length is six actions. The next experiment tests action cost, keeping
the same selected visible objectives, staging candidates and owned-route planner.

V8 has two independent changes:

1. `cheapest_collection=True` chooses the shortest sufficient collection route
   among V7's feasible sources, breaking ties by the old delivery score and then
   flat index. It does not optimize arbitrary objectives or alternate rallies.
2. `direct_deployment=True` compares an existing packet's actual owned-route ETA
   with the chosen collection/deployment ETA. A sufficient faster packet enters
   deployment immediately and follows the route through explicit memory. This
   executes the alternative rather than merely skipping to an unrelated campaign
   action. Future garrisons are not credited in its initial force bound.

Both preserve one V6 defensive decision, winning priority, current visible home
coverage, observed force/ownership reconciliation and a fixed expiry. Direct
selection checks the proposed action; if it fails home safety, V6 acts instead
of searching another source in that frame. Current visible checks do not predict
hidden armies or future merges. Direct routes remain restricted to the inherited
objective filter and at most nine initial owned-route steps.

The memory stays V7's 15 scalar int32 leaves with nested defender state. The
aliases are `sentinel-v8` (both), `sentinel-v8-cheap` (collection only),
`sentinel-v8-direct` (direct only), `sentinel-v8-disabled` (exact V7), and
`sentinel-v8-no-concentration` (exact V6 through V7's disabled path). Corresponding
standalone variants omit `sentinel-`. Existing V2–V7 sources remain frozen.

## Preregistered development comparison

Freeze sources and complete policy/integration tests before matches. Stage one
compares V6, V7, V8-cheap, V8-direct and V8-both on eight consumed seed-83000 Amin
maps and eight consumed seed-93000 Hunter maps. Preserve all four seat/general
label cases: 32 games per policy per opponent, 320 total. Revalidate original
Amin adapter parity and retain complete public histories. These are separate
synchronous strategy and local GPU experiments, not deadline qualification.

Select one V8 variant for the remaining development gates using complete results.
Prefer variants that preserve V6's Hunter score and have no invalid/malformed
actions; among them choose the highest Amin score. If none preserves Hunter,
choose the highest Amin score solely for further diagnosis, explicitly failing
the preservation gate. Break score ties by Hunter score, then prefer cheap-only,
direct-only, and both in that fixed order. This adaptive choice is development
selection, not a fresh strength estimate or promotion.

Compare selected V8 with V6 on eight Expander maps at seed 93000 (64 games), and
each original Juraj version on four maps at seed 83000 (64 games across both
versions). Reuse the completed Hunter comparison without repeating it. This
provides 448 planned games before any newly qualified opponent. Preserve original
Juraj defaults, including V3.4's unpaired entropy/clock randomness.

The source audit also examines the previously unmeasured public `my_bot` family
from upstream PR 138, pinned at `f624c741ad5084be63fc17bacd3961599b8dcc82`.
Source features do not establish its strongest variant. Qualify the original
protocol, dependencies, rules assumptions and runnable artifact before adding
matches; report unsupported or mismatched rules rather than silently rewriting
the opponent. Any additional budget is recorded before those matches.

## Evidence and delivery gates

Tests must prove actual shorter collection/capture, multi-step direct deployment,
force and leave-behind bounds, defense/home safety, observed aborts, projected
deathtouch, exact disabled memory/action/telemetry parity, and JIT/vmap. Runtime
needs a new deterministic archive, actual offline cache build, all 16 fresh-process
cache reuse probes, and complete active direct/collection history parity.

Audit complete selected-policy histories, separating available routes, chosen
routes and issued actions. Check observed arrivals/captures and collection cost;
do not equate a smaller ETA or a larger army with winning value. Use map-cluster
uncertainty and include every planned case and fault. V2 remains default unless
separate promotion evidence justifies a change. Seed 143000 stays reserved and
uninspected; the all-opponent goal cannot close on consumed development maps.

Deliver only owned validated changes to origin/main after fresh-main and collision
checks, preserve unrelated work, and leave the primary workspace on main.
