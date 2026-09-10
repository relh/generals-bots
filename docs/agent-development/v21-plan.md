# V21 experiment contract: persistent tree collection

Status: implementation and qualification in progress; no games authorized by
this document and no strength claim. The competitive objective remains beating
every distinct available opponent with fresh paired winning evidence and valid
runtime. This experiment does not replace that objective.

[V20's complete screen](v20-results.md) failed four wins/two losses. Its six
initial branch transfers all yielded to new parent offense on the next turn;
none completed branch assembly. Some takeovers nevertheless transported the
recipient and captured the objective. The experiment therefore changes collection
arbitration rather than forcing every tree to finish or tuning force thresholds.

## Selected implementation

A new collector sits inside V9/V10 at the V8 boundary. It performs one frozen V8
solve using the original OffensiveMemory subset, preserving the actual returned
defender. Enabled memory extends that subset with a nested nine-scalar TreeState,
for 28 scalar int32 leaves across StrategicMemory (24 in its base). The disabled V21 instance
is exact V10, including its native19 memory and telemetry. No frozen V8, V9, V10,
V20 or branching-DP source changes are part of this experiment.

The pure tree planner retains V20's target/rally ranking, shortest-owned rooted
tree and exact six-action budgets, cheapest sufficient plan, 1.5 concentration
test, single-path exclusion, expected recipient army, current home check and
fixed expiry. It does not credit unissued donors to a deployed packet. Phase2
never resumes gathering. Construction/deathtouch rules remain explicit inputs.

When no tree is active, existing ordinary transport and actual newly issued V8
offense keep priority. An active, feasible tree may continue instead of a newly
started ordinary collection. It yields to actual builds, winning-general moves,
incoming/returned defense, guard, positive home deficit, adjacent threat, reserve
shortage and actual ready/deploying ordinary offense. Merely available direct
proposals do not qualify as issued ready attacks.

A new serial collection may take over a feasible active tree only if it uses the
actual last recipient, targets the same objective, requires no more gathering
actions than the remaining tree budget and no more total actions than the tree's
current remaining work, and finishes within the original expiry. The returned
ordinary expiry is capped at that original expiry. Real returned serial memory
survives. A ready direct attack is a separate higher-priority ordinary plan,
not a continuation credited with unissued tree donors.

Every selected tree action clears unissued ordinary offensive state, even if
both proposals have the same physical action; it preserves returned defense.
The tree stores actual recipient, expected arrival, shrinking budget and original
expiry. Invalid, inconsistent, expired or unsafe tree work falls back immediately.
A completed tree capture clears both tree and ordinary transport.

The existing V9 pursuit and V10 mobilization remain outside the collector. Their
changed actions clear the extended state through the collector's initial_memory.
Same-action pursuit can retain a valid tree under V9's unchanged agreement rule;
this seam must be tested explicitly. Inactive V21 is not claimed to reproduce
V20's complete outer-wrapper arbitration.

## Recording and qualification

Ordinary proposal events require `tree_parent_action_issued`,
`actual_v8_action_issued` and `mobilization_parent_action_issued` before execution
credit. Tree proposal fields require `tree_action_issued`, which combines the
collector selection and both outer masks. The recorder preserves full named
nested memory, observations, keys, raw/applied actions, telemetry and all original
invalid actions. A proposal, an owned transfer and a confirmed capture remain
separate events.

Required falsifying coverage includes:

- Exact disabled V10 and native28 batching/reset/gap behavior.
- A full-stack six-edge assembly with a competing new, different-objective
  ordinary collection, followed by actual tree deployment and capture.
- Productive same-objective recipient serial handoff, remaining-budget and
  expiry limits, and real ready-direct takeover.
- Urgent defense/build/winning-general priority, outer pursuit/mobilization
  clearing and the same-action pursuit seam.
- Missing recipient, depleted force, changed ownership, infeasible remaining
  work and phase2 non-gathering behavior.
- Original archived public situations and meaningful real engine transitions;
  controlled private-proposal tests alone cannot establish full-stack execution.

After source/fixture checks, qualify an independent public checker and matched
full-call runtime. Each selected profile context must retain at least half of
V10 throughput, including active collection and deployment. Preserve cold and
warm timings and every failed attempt; do not relabel source inference timing as
standalone stdio deployment qualification.

## Complete development screen and advancement

Only after qualification and an explicit root source/helper freeze, run four
arms V10, V10-v6, V20 and V21 across the same six consumed seed83000 cases: Amin1/2,
Juraj V3.5 12/15 and original my_bot9 4/7. Use construction, deathtouch800 and cap1200.
All18 controls must match their complete original V20-cycle histories before any
V21 game. All24 games and integrity checks must finish before outcome selection.
Do not restart healthy runs, replay fixed future opponent actions against a
changed candidate, or discard failed attempts.

V21 must win all six selected cases to advance. Even six wins only pass this
consumed-position screen: broaden distinct-opponent coverage and obtain fresh
paired score-confidence lower bounds above50% against every tested opponent,
with valid runtime, before claiming the broader objective. Reserved seed143000
stays untouched during this screen. V2 remains the default and no deployment or
official submission is authorized by this experiment.
