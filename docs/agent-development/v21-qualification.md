# V21: persistent tree collection qualification

V21 passes the bounded source and full-call runtime checks below. No V21 full
games have run, and this is not evidence of improved playing strength. V2 remains
the default. The next planned gate is the complete 24-game development screen in
[the experiment contract](v21-plan.md), followed by fresh paired opponent coverage
if the candidate advances.

## Behavior

V20's six initial tree transfers all yielded to ordinary offense on the next
turn in its [failed development screen](v20-results.md). V21 moves tree collection
inside the existing V9/V10 stack, at the V8 boundary. It calls frozen V8 once and
lets a feasible active tree continue over a new ordinary collection. Actual
ready attacks, defense, builds and winning-general captures retain priority.
A productive serial handoff must use the last recipient, target the same
objective, fit the remaining work and budget, and keep the original deadline.

Selected tree actions retain the actual returned defender and clear unissued
ordinary offensive memory. Changed outer pursuit or mobilization clears the
nested tree; same-action pursuit preserves it under the unchanged V9 rule.
Execution counts require the final collector and outer-action masks. Existing
planners, older agents, CLI and default selection are unchanged.

## Checks and preserved failures

The 23-case suite passed across 22 unchanged cases in the second attempt and the
repaired eight-step engine test in the fourth attempt. AST comparison verifies
that only that fixture and test changed between those attempts. Source remained
SHA256 `353c5c6c06f1450b5155908e8d2392a219ce5755bad977d5e06cb4e09f376a38`
throughout qualification. Ruff checks passed.

The eight-step test performs six actual gathers, one owned deployment and one
capture. Competing ordinary collection proposals do not replace the active tree;
the budget shrinks, the expiry stays fixed, and no unissued ordinary plan survives.
The actual engine confirms ownership and arrival after every move, ending with
10 armies on the captured target. This is a declared synthetic position, not a
full game or counterfactual replay of an opponent.

Twelve seam tests additionally exported 17 complete V21 calls. Three separate
checks from saved trajectory inputs cover lost recipient ownership, a broken
owned deployment route, and an understrength phase-two packet with a nearby donor.
All three release the tree. The last case explicitly adjusts expected army to
isolate deployment sufficiency; it is not presented as an original game state.

Failed attempts remain preserved: the first suite had four incorrect fixture
assumptions about local-pool eligibility and pursuit agreement; the second exposed
a correctly prioritized home-defense guard. A connected mountain barrier moves
that fixture's enemy beyond the guard horizon without changing collection routes.
The third attempt stopped before any policy call because its independent terrain
assertion omitted structures hidden by fog. No policy threshold or priority was
changed to obtain passing results. The original guard decline remains a checked
case.

## Independent checker

All 45 saved calls match exact final actions, all 28 memory fields and full
telemetry. The checker obtains a fresh frozen-V8 reference from each original
effective ordinary input, reconstructs collection independently in NumPy, then
passes that result through frozen V9/V10 selectors. It never calls V21 to build
its expected output. All 218 deliberate corruption probes were rejected.

The 45 comprise nine profile contexts, seven profile trajectory calls, eight
separately recorded test trajectory calls, 17 seam calls, three contract-gap calls
and the original guard decline. Overlapping inputs retain their original source
identities; they are not 45 distinct game positions. Fresh reference calls are
labelled as such, not as originally recorded inner oracles.

Static comparison also confirms unchanged proposal scoring, DP arithmetic and
public helpers. Their existing V20 qualification—25,376 exact score comparisons
and 49 budget checks—is inherited with verified source and provider hashes. Those
grids were not rerun or counted as new V21 evaluations.

## Runtime

Every context exceeds the required 50% of V10 throughput. Disabled V21 matches
V10's complete tuple in all nine contexts; its minimum throughput ratio is 98.3%.

| Context | V21 / V10 throughput |
|---|---:|
| Six-edge start | 58.5% |
| Active collection with competing ordinary plan | 61.3% |
| Tree deployment | 60.6% |
| Tree attack | 60.2% |
| Original Juraj turn 183 projection | 54.9% |
| Original my_bot9 turn 221 projection | 57.8% |
| Original Amin turn 348 defense | 89.8% |
| Juraj turn 291 serial handoff | 55.3% |
| my_bot9 turn 222 ready handoff | 87.8% |

Measurements use pinned JAX/JAXlib 0.11.0 on Zephyrus CPU5, compilation cache
disabled, ten warmups and five rotated repetitions of 100 synchronized full calls
per variant and context. Test export ran separately on CPUs4/16 during part of
the profile. Cold V21 calls took 11.6–12.5 seconds and are reported separately.
These checks establish the selected full-call gate, not standalone stdio deadline
qualification or full-game runtime validity.

The [evidence index](v21-qualification-evidence.json) binds source, fixtures,
attempts, complete outputs, runtime records and independent checks. Full-game
recorders and control barriers are being prepared; no game or outcome release is
implied by this document.
