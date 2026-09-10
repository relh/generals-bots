# V17 prerequisites: capture value and force placement

The completed V16 experiment lost the selected Amin and Juraj positions despite
executing its feeder routes correctly. These read-only audits distinguish two
remaining problems. They use original public observations, actions and memory;
they run no policy inference, new games or hidden-state counterfactuals.
The [machine evidence](v17-audit-evidence.json) retains the reports and hashes.

## A gathered packet needs a productive next objective

A census checks all 18 completed V16-family games and 11,814 decisions. Across
the three representative V16 positions, 25 new direct plans have an archived
V6 proposal that can capture a different visible target from the same source.
The label mirrors repeat these states; they are not 50 independent examples.
Of those 25 choices, 21 start with owned transfers, 16 reverse the preceding
move, and only four immediately attack the selected objective. A reversal alone
does not prove a mistake or justify cancelling every deployment.

At Juraj turn 109, the gathered 14-army packet attacks enemy 2 north. That
destination has no adjacent visible empty neutral plain. The archived reference
instead attacks enemy 1 south, adjacent to two such plains. V16's capture is
confirmed at 110 and first lost at 143; subsequent plans reverse through owned
territory. This suggests comparing explicit capture continuations, including
departure and combat costs, instead of choosing the largest enemy garrison.
It does not establish that changing this action wins the game.

Four exact fixtures cover Juraj 109 and Amin 214, where the reference's cheaper
immediate capture exposes more neutral frontier; Amin 223, where the current
choice has better frontier; and Amin 163, an owned transfer outside an immediate
capture comparison. Earlier V10 traces lack the necessary same-call reference
proposals: all 232 direct starts in those three control histories are explicitly
marked missing, never reconstructed from another trajectory.

## Late reaction cannot assemble the missing defense

In the complete V16 Amin loss, a 247-army enemy first becomes visible at turn
520, eleven public-terrain steps from home. All ten owned screen cells on
shortest invasion paths fail an optimistic shortest-owned-route gathering
comparison. The best ten-move home route supplies 64 from current garrisons;
five guaranteed home births bring it to 69, against 223 required after charging
the enemy's departures and currently visible path attrition. At 521 the threat
is within the existing ten-step horizon and the bound still fails. Every
recomputed screen through 530 fails as well.

Even granting eleven actions and instant delivery of the richest distinct
field donors, ignoring all travel, provides only 124 at home. This excludes a
one-step horizon extension as a repair within this existing-force gathering
class. It does not exclude a capital race, earlier construction, different
earlier strategy or other unmodeled actions.

The force is dispersed, largely within connected territory:

| Public turn | Own land / total army | Movable field surplus | Largest field tile |
|---|---|---:|---:|
| 450 | 107 / 475 | 349 | 9 |
| 500 | 115 / 566 | 432 | 10 |
| 520 | 119 / 559 | 411 | 10 |

All owned tiles are home-connected at 450 and 520; 114 of 115 are connected at
500. Public enemy concentrations were visible earlier: 55 at 169, 59 at 247,
74 at 327, and 114 at 440. The last leaves sight at 445. These observations
support earlier preparation and accounting for deliverable force. They cannot
identify the later 247-army packet under fog. Enemy scoreboard totals outside
view remain uncertainty, not a known hidden formation.

This supports the owner's [doctrine](playing-doctrine.md): a useful deathball
must advance, while a global army lead cannot substitute for timely home
defense. The [V17 experiment](v17-plan.md) isolates productive capture
continuation. Earlier force allocation remains a separate unresolved problem;
passing the local fixtures cannot establish its solution or competitive strength.
