# V16 results: successful delivery, failed strength gate

**All eighteen games completed; V16 is rejected for advancement.** It wins two
of its six games and loses four. V2 remains the default and the
[competitive goal](competitive-goal.md) is unmet. The
[source/preflight checkpoint](v16-development.md) and [fixed plan](v16-plan.md)
remain unchanged as release records; this page records the subsequent outcome.

Each table entry occurred in both mirrored cases of one already-consumed
physical position. These are three positions, not eighteen independent maps.

| Opponent / cases | V10 | V10-v6 | V16 |
|---|---|---|---|
| Amin / 1, 2 | win at 1033 | loss at 370 | **loss at 531** |
| Juraj V3.5 / 12, 15 | loss at 422 | win at 1075 | **loss at 335** |
| my_bot9 / 4, 7 | win at 1076 | win at 296 | win at 769 |

The declared gate required preserving each position's stronger control. Longer
survival than the weaker Amin control, or a faster my_bot9 win than V10, cannot
compensate for the lost Amin wins or the unresolved Juraj loss. Reserved seed
143000 remains untouched. No promotion, training run or deployment followed.

## The routes work; their strategic value does not pass

Across the six candidate games, twelve feeder plans issue 82 actual actions.
Ten end in next-observation-confirmed target captures; two abort when their
owned route is cut. No inner feeder action is falsely counted after an outer
override. The static route arithmetic and actual transport pass the independent
checks. Counting those completions as success would reward behavior that fails
the complete-game objective.

Juraj supplies a particularly useful discriminator. V16 follows the winning
V10-v6 control's complete joint action history through 108, gathers as intended,
and captures (11,10) with fourteen armies. Both players' public observations
still match at 109. The first difference then comes from the same packet:
V16 attacks north into enemy 2; V6 attacks south into enemy 1. The southern
target has visible empty neutral territory south and west; the northern target
has fog north, a mountain east and enemy 2 west. Those are public features,
not knowledge of the hidden capital.

V16 reverses south at 110 and continues tactical redirections. V6 instead
captures neutral territory west at 110 before attacking north. By 150, V16
has 35 land/110 armies split into components of 26 and 9 tiles; V6 has a
connected 55 land/142 armies. V16's second feeder also completes, at 133.
Correct delivery has not solved objective choice or useful continuation after
capture. The preserved public109 fixture and full follow-through distinguish
that question from a gathering failure; they do not prove that changing one
direction would win.

Amin preserves the original useful 129/142 decisions, then diverges at 250.
Its three feeder plans per mirror finish at 255, 291 and 408 with 15, 17 and
23 armies remaining. The first target is lost before the next land tick; the
other two survive their next ticks and are lost later. All routes complete,
and subsequent packet movement is recorded. The last feeder action is at 408;
there is no active feeder when the game ends at 531.

At 520, a 247-army enemy stack becomes visible eleven
Manhattan steps from home. At 521, the policy reports a 188-army home deficit
and no feasible interception. At 530 the enemy has 224 adjacent to a 20-army
capital; a remote 26-army packet moves east, and the next transition loses.
Our aggregate army is still 549 versus 460. Concealment also improved—the
opponent first sees our capital at 530, versus 360 for V10—but neither late
exposure nor an aggregate army lead supplies timely local defense. This is
ordinary combat, before deathtouch activates at 800. No early-build rescue or
single-action causal explanation is established.

In my_bot9, the opponent cuts the feeder's owned corridor at 207. The route
and force checks correctly abort at 208, and the fallback continues moving
the 29-army packet. Releasing named memory does not mean its army stopped.
The complete game still wins at 769.

## Integrity and limits

The [machine evidence](v16-results-evidence.json) covers all 11,814 recorded
decisions: 8,544 control frames and 3,270 V16 frames. Independent review verifies
complete control joint actions, public histories, available native-memory
oracles, and all nine mirrored pairs. Candidate invalid and malformed actions
are zero. Original Amin invalid actions total 192 and remain in the results;
none occur in V16's final 520–530 attack window. All twelve external opponent
processes exit cleanly, with no external faults or invalid actions.

Both game runners and frozen strict postchecks finish on their first attempts.
One Amin descriptive-analysis attempt and one root packaging attempt require
schema corrections; their original versions are retained and no game or policy
inference is repeated. The nineteen behavior tests and 78–82% warm-throughput
retention remain valid, but do not qualify standalone startup or playing strength.
The source-level synchronous candidate evaluation does not enforce official
candidate deployment deadlines. These competition 1v1 results establish neither
classic nor FFA performance.

Next work must evaluate productive objectives and movement after capture,
alongside whether force can reach a threatened capital in time. The exact
Juraj109 public state is a discriminating fixture; Amin's complete losing
trajectory is a preservation warning. A new objective or defense rule must
earn another explicit comparison rather than inherit a success claim from
completed routes, total army, concealed home or shaped reward counters.
