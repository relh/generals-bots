# V19 complete development screen: rejected

**V19 fails stronger-control preservation: two wins and four losses.** All 24
frozen games finish, the eighteen controls match their original trajectories
before candidate play, and complete integrity/mirror checks pass. V19 is not
promoted. V2 remains the default; the competitive objective remains unmet.

The [qualified implementation](v19-preflight.md) is commit
`e6f3160cfa0f21d7a2d9e88bb4dfcf66e55e8a25`, delivered to origin/main before
execution. Its source SHA is
`9322e207c9f02c46856692b9c544b3e53920b07e5d0ace927d588f94794a2a72`.
The [plan](v19-plan.md) required all six candidate cases to win before advancing.
These are consumed seed-83000 development placements, with construction,
deathtouch 800 and cap 1200. Reserved seed 143000 is untouched. No fresh paired
advantage, confidence-bound success or broad opponent dominance is claimed.

## Complete outcomes

Both label mirrors have identical outcomes and complete relative trajectories.
Numbers are terminal turns.

| Opponent / case IDs | V10 | V10-v6 | V18 parent | V19 |
|---|---|---|---|---|
| Amin 1/2 | Win 1033 | Loss 370 | Loss 546 | **Loss 431** |
| Juraj V3.5 12/15 | Loss 422 | Win 1075 | Loss 447 | **Loss 575** |
| Original my_bot9 4/7 | Win 1076 | Win 296 | Win 888 | **Win 888** |

## What the new mechanism actually did

Across six V19 games, ten retained transits execute, six directly replace an
inner parent action, and two finish at home. **No new retained field hold occurs.**
The separately qualified synthetic holding branch therefore has no adaptive
execution evidence in this screen. Ordinary inherited V6 holds are distinct.

The unchanged rear-expansion layer executes 538 captures, all confirmed by the
next public observation. At their next land tick, 498 remain continuously owned,
20 are lost and 20 are censored. At fifty turns the split is 490/28/20; at one
hundred turns it is 452/46/40. These are real captures and retention, not proof
that their action cost produces timely defense or successful attack.

### Amin: delivery succeeds, the strategy still loses

The first action divergence from V18 is turn 348, on identical public input and
native memory. Four retained moves carry the existing defender from (8,13)
through (8,14), (8,15), (8,16) and into home (8,17). Expected next armies are
16/16/18/43; actual next observations show 16/17/18/44. Deterministic growth
explains the differences. Home arrival clears memory immediately, as specified.

The resulting home army is 44 at 352. Rear moves delay the next home proposal
through 354; the parent then halves home 45 at 355 and home 27 at 363. Further
home donations occur at 376, 404 and 420. None is a retained action. The last
retained move is 351 and the last rear move is 403.

A visible 71-army warning appears at 415. The inherited planner reports no
feasible interceptor throughout 415–430. At 420 the parent halves home 21 while
reporting deficit 28. At 430 the enemy has 33 adjacent to home 16; a friendly
22-army packet is at (14,1), twenty-two Manhattan steps away. Movable field
surplus is 411, but is not a one-turn defense. The final attack sequence is
entirely parent-selected, and the game ends at 431.

V19 makes 69 rear captures per mirror, versus V18's 77. The full audit compares
all eight original histories, including V10's different successful defense,
replenishment, seven builds and eventual deathtouch finish. It does not transplant
those later control actions into V19 or claim that home delivery caused the loss.
The measured conclusion is narrower: this completed delivery does not preserve
the winning Amin result.

### Juraj: memory changes a capture, then a later attack wins

At 439 retention preserves the same original move into defender target 129,
with original expiry 448. Observation 440 confirms eighteen armies there. The
public trajectory still matches V18, but incoming native memory differs. V19's
actual V6 now selects LEFT into enemy cell 128 instead of V18's UP reversal.
The nonowned capture has priority, so new retention releases; seventeen sent
against one leave sixteen owned at 128 on observation 441. This is a
memory-mediated divergence, not a direct retained-action override.

At 441 the candidate attempts to retreat to 129 while the original opponent
moves its visible 57 from 127 into 128. The next observation has enemy 40 at
128 and only the old one-army garrison at 129: the retreat did not arrive under
simultaneous resolution. This legal submitted action is not an invalid command.
The resulting smaller threat has current deficit four, allowing a new original
V6 route at 442–444 to deliver home thirty. At 446 the home has 31 against
adjacent enemy 32; the actual opponent goes down instead of the capital attack
that ended V18. No new retained holding occurs in this sequence.

No further defender starts after 442. At 566 a visible 72 has current deficit
36 and no feasible intercept. It is absent from view at 567–569; no identity
through fog is inferred. At 570 a visible 76 has deficit 44. By 574 an adjacent
62 faces home 17; the game ends at 575. The final threat has no existing defender
for this bounded rule to retain, and no rear override occurs during 566–574.

V19 ends with 734 total army against 595, but its largest field garrison is 26.
Although home is connected to 699 army by then, connection alone does not supply
an on-time collection schedule. Earlier, much of its army was disconnected:
home's component had only eleven land and 71 army at 500. V19 builds once at
531; the winning V10-v6 control makes sixteen builds in its complete game,
starting at 447. All productive control releases/builds/captures remain exact.
The extra 128 turns over V18 and additional territory do not satisfy the win gate.

### my_bot9: retained defense never activates

V19 has no eligible retained action in either mirror and wins at 888, matching
V18's complete public/action trajectory. Each makes 89 rear captures. This is
preserved parent behavior, not an observed benefit from retained defense. The
stronger V10-v6 control still wins at 296.

Original my_bot9 invalid construction attempts remain part of the experiment.
The V18 and V19 games each contain seventeen such attempts per mirror, normalized
to PASS by the unchanged engine. They are not repaired or excluded, and these
wins do not establish performance against a repaired opponent.

## Integrity, runtime and attempt accounting

Root confirmed clean terminal exits for both original runner sessions: Amin 4559
and external 37698. Before V19 started, the shared barrier independently compared
all eighteen controls over 12,306 frames, including original public observations,
action keys, native memory, scalar telemetry and both players' raw/applied actions.

Completed checks cover all 16,094 policy decisions, including 3,788 V19 calls,
all twelve mirror comparisons and 22,668 byte-exact external public wires.
Candidate invalid and malformed actions are zero throughout. The original Amin
makes 228 off-board RIGHT attempts across the eight Amin games. Original my_bot9
makes 68 invalid builds across the V18/V19 control/candidate games; external
protocol/deadline faults are zero and children exit cleanly.

All frozen policy, checker and runner bytes remain unchanged. No healthy game
was restarted and no outcome-driven tuning occurred within the budget. One wire
checker invocation ran before its required strict-check file existed; it failed
without a proof. The same checker passes after that prerequisite completes.
Full review session 50301 exits zero on its first attempt. Its report SHA is
`8190327a24569590c5424aeb099c12d4dfe84561a119693708e969b4d747b80e`.
Audit preparation errors and their corrected attempts are retained separately;
none changes a frozen game or qualifier.

The full-call qualification remains the stated synchronous development workload.
Official startup/standalone deadlines are not certified, and episode timing across
different trajectories is not a same-input speed comparison. Complete failure
audits and all original attempts are bound by the [results manifest](v19-results-evidence.json).

This screen rejects the selected rule. It establishes correct bounded transit,
one memory-mediated tactical change and no adaptive new field holding. Durable
income, successful delivery and longer survival still fail to produce the
required wins. No next policy or counterfactual rescue is selected here.
