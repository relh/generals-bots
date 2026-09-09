# What the complete V8 failures establish

The competitive objective is still unachieved. This investigation starts from
`93c18ce34f3ec232d18bd06891ee6a235f80996b`; it uses already consumed development
games and makes no fresh strength claim. Existing V2–V8 policies remain controls.

## Hunter: two failures, not four independent mechanisms

All four V8 losses on seed-93000 map 0 and the four paired V6 wins were replayed
to completion. All eight outcomes, terminal turns and diagnostic counters match
the original runs under their original JAX 0.11.1 GPU runtime. There are 3,576
transitions and 3,584 retained states. Both seat/label mirrors reproduce the same
candidate-relative histories. Public observations, actions, keys, telemetry and
15-leaf V8 / seven-leaf V6 memory are retained; no policy source substitution or
source-mismatch override was used.

In the early placement, V8 loses at 173 while V6 wins at 522. Their first action
difference at 78 redirects an already active direct plan through owned territory.
At 167 the candidate sees enemy armies 8 and 19 in adjacent cells. Its inherited
defender recalls a ten-army screen; the enemy merges to 26, and the next defense
calculation is infeasible. All actions after 161 are inherited defense/campaign
behavior. This supports a visible-merge and defense-continuation investigation;
it does not prove an alternative action would win against an adaptive opponent.

In the late placement, V8 loses at 873 while V6 wins at 220. At their first
difference, turn 76, both capture an adjacent ordinary enemy cell with one
defender, but in different directions. Neither first difference overrides a
selected city or visible-general objective. V8 sees the enemy general only at
180–185, preserves all six visible-general decisions, then loses that destination
when it returns to fog. At 222 a 17-army packet has a legal immediate northward
capture against one defender, but starts a six-action plan southward toward a
different ordinary enemy cell. It turns south again after that capture. The
physically available northward move has not been established as V6's proposal on
that later V8 state. V8's land falls from 31 at 100 to 14 at 800, against Hunter's
23 and 60. It never owns a castle. Late defense alone cannot repair that economy.

The complete replay proof is bound by SHA-256
`1bb52ef504bfbbe1b8d478492acdd3690d950ce1e556dcc509508603b1449642`
at `.cache/runs/sentinel-v9/hunter-diagnosis/proof.json`.

## Amin: preserving every immediate capture is not justified

The original actor runtime recomputed V6's exact proposal on all 22,166 V8 public
frames, using each recorded key and nested defender memory. Returned defender
memory and inherited scalar telemetry agree exactly; every changed action agrees
with the recorded offensive override. These are same-state proposals, not a
counterfactual V6 trajectory. All 32 complete histories were audited.

V8 replaces 3,286 of 10,832 legal V6 visible-capture opportunities: 2,640 with
owned transfers and 646 with different immediate captures. It also replaces 56
of 206 legal builds, all during continuing plans. All 20 proposed winning general
captures survive. Conflicts occur in both new and continuing plans, including
enemy castles; a collection-start-only veto misses most of them.

The new Amin win in case 1 is a counterexample to assuming all delay is waste.
At 129 V6 could capture an empty neutral cell. V8 instead executes a four-action
plan, captures enemy territory at 132, and actually performs the deferred neutral
capture at 133. It wins at 1,033. That observation proves deferral rather than
abandonment; it does not prove the four-action plan caused the win.

The full audit and aggregate hashes are respectively
`f8d62645ea109a3d865919306b44e6285836a084ad34d799e268cf08a1f7bb58`
and `88f5bd1aa312c6288b011c081178460f1c40605cc4b327ef48676d532018febc`
under `.cache/runs/sentinel-v9/amin-opportunity-audit/`. Win/loss conflict rates
are descriptive because trajectories and lengths differ.

## my_bot9: an economic deficit and a separate route-accounting defect

All eight V6/V8 map-1 games reproduce their outcomes and terminal turns across
4,258 stored action transitions. All 4,258 candidate raw replies, applied actions
and carried memories agree with recomputation under Python 3.12.10 / JAX 0.11.0.
The engine reconstruction uses the original coordinator JAX 0.11.1. Public wires
are reconstructed from retained boards/actions; original stdio wires were not
archived. Every one of these games has zero invalid actions or runtime faults
for either player, so the opponent's separate castle-price bug does not explain
these losses.

Cases 5/6 change from V6 wins at 183 to V8 losses at 613. V8 finishes with 128
army / 42 land against 336 / 70, with no castle for either side. The final enemy
22-army packet merges into 47, producing 68 and then capturing the 39-army home
in ordinary combat. The adjacent enemy already exceeded home before this merge;
the opponent delayed a winning attack. No rescue is established.

Cases 4/7 change from V6 wins at 296 to V8 losses at 1,037. V8 finishes ahead
523 army / 99 land / six castles against 469 / 37 / two, yet an enemy 41 captures
its 57-army home under deathtouch. Its final offense override is 1,023. At 1,027
the inherited interception planner accepts an apparent 50-army donation against
a 50-army requirement, beginning a five-action route from `(10,9)` to `(13,7)`.
At 1,028 it releases the exact expected 29-army defender: arrival consistency,
continuity, target validity, distance progress and timing all pass, but additional
defense 51 is less than the recomputed requirement 61.

The discrepancy exists before the enemy moves. Applying only the actually
issued first transfer vacates `(10,9)` and changes the enemy's minimum home-route
cost from 45 to 34. The same visible enemy army 95 now needs 61 additional defense,
while the original route planner still credits 50. Its off-corridor donation
classification used the pre-transfer route. This is a concrete stale accounting
defect, separate from enemy merges or observation-memory corruption. A repair
must recalculate the route after donations; merely retaining an insufficient
commitment is not a demonstrated fix. No alternate match or successful rescue
is claimed. Exact original-helper inputs and issued-action calculations are in
`.cache/runs/sentinel-v9/mybot-diagnosis/`.

## Next discriminating work

Test retention of a previously observed stationary enemy general as a separate
public-history strategy change. Keep actual visibility, ownership and armies
unchanged; known location must not fabricate current combat information. Preserve
inherited defensive and winning priorities, and include exact disabled parity.
Attack progress is a mechanism metric, not evidence of greater competitive
strength. Compare full games with frozen V6 and V8 before promotion.

The route-accounting defect and visible enemy reinforcement require separate
defensive experiments. A universal capture veto, a collection-only veto, a larger
garrison and unconditional commitment retention do not follow from these records.
Seed 143000 remains reserved and uninspected.
