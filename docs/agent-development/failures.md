# Frozen Sentinel v2: two complete loss audits

Audited September 8, 2026. These losses expose a gap between winning territory
and keeping enough troops near the general. One case repeatedly releases a
rescued garrison; the other ignores a large, visible army outside the short
defensive horizon and then forgets it in fog. Neither loss involved invalid
commands, a runtime fault, or active deathtouch.

The policy remains frozen at SHA-256
`be909e6fa3d5b46a3dd2454eaff8a030a8e7d7158f90d484044c36fa06e4650e`.
These inspected holdout cases are now diagnostic data, not fresh validation for
a future v3. The hypotheses below have not been implemented or established as
strength improvements.

## Evidence and method

Sources are the complete NPZ episodes and verified replay JSON files in
`.cache/runs/sentinel-heldout-v2/replays/`:

- `classic12-hunter-b5-r0-s1-swap0.{npz,json,md}`: loss at turn 112.
- `competition-expander-b28-r0-s1-swap0.{npz,json,md}`: loss at turn 309.

Both reproduce the recorded outcome, terminal turn, and every action counter
exactly. We reviewed all 112 and 309 pre-action states, actions, observations,
and policy telemetry, rather than only the JSON's final 20 turns. The recorder
contains hidden state; claims about warnings below explicitly use Sentinel's
`observation_1_opponent_cells`. Hidden routes are labeled hindsight. Coordinates
and pre-action turns are zero-based. Army and land pairs are Sentinel/opponent.
Full per-turn summaries and their generating script are beside the replay files.

## Classic 12×12: reinforce, release, repeat

Sentinel's general is at `(7,9)`, Hunter's at `(5,0)`. Neutral castles and
mountains divide the approaches. Sentinel expands mainly north and west;
Hunter's first attack comes around the south. At turn 50 Sentinel already leads
15/12 in land and 41/38 in total army. Early expansion alone does not establish
an error: the first approaching Hunter stack is hidden until turn 72.

| Turn | Observation and decision | Consequence |
| --- | --- | --- |
| 69 | General has 16; sends half north. No approaching enemy is visible. | Home falls to 9 after growth. This is a hindsight vulnerability, not a demonstrated ignored warning. |
| 72–74 | Enemy 13 becomes visible at `(8,10)`, two steps from home. Sentinel routes its nearby stack through `(7,7)` and `(7,8)` to reinforce. | Home reaches 23 at turn 75. Reactive defense works here. |
| 75 | Enemy 13 remains visible two steps away; reserve is only 6.5. General sends 11 south to `(8,9)`. | Hunter simultaneously moves 12 to `(7,10)`, adjacent to home. The newly assembled defense is split again. |
| 76–79 | The outgoing stack captures behind the attacker and continues south. Hunter attacks home on turn 78. | Home drops from 14 to 3. The raid wins land but does not remove the adjacent attacker before its strike. |
| 82–88 | Enemy general becomes visible at 82 and remains the objective for the rest of the episode. Home dispatches half its 5 troops at 82 and half its 6 at 88. Another visible Hunter wave is approaching. | Reinforcements travel toward the distant enemy general rather than building a durable home reserve. |
| 95–98 | Enemy stack moves from 12 at `(6,7)` to 10 at `(7,7)` and 8 at `(7,8)`. Home has 6–7; distant troops keep advancing west. | A second attack leaves the general at 1 by turn 98. |
| 100–111 | Sentinel still leads 72/59 in army at 100. At 106 it sends half its general's 6 left while another wave is visible farther away. At 110, enemy 14 is two steps away; at 111, enemy 12 is adjacent. | The final reinforcement adds only 1 to home's 5. Hunter sends 11 and wins at 112. |

Turn 75 is the earliest clear decision where a successfully restored defense
is released despite a known nearby threat. It is a plausible intervention
point, not proof that this move alone caused defeat. The later pattern matters:
the three-step threat estimate is discounted, and there is no persistent defense
phase that retains gathered troops through successive waves. Seeing the enemy
general supplies a distant campaign objective even when local defense is fragile.

We tested exactly one branch: replace Sentinel's turn-75 action with a pass,
then resume both unchanged policies on their new observations with the original
PRNG continuation. This **lost at turn 98**, earlier than the recorded turn 112.
Sentinel made the same southward half sortie at turn 76, then continued raiding
and feeding the western campaign. A one-turn pause is therefore not a verified
fix. The experiment does not test a sustained defensive plan or establish that
turn 75 was recoverable under the remaining frozen behavior.

Reproduce the branch from the repository root:

```sh
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu PYTHONPATH=. taskset -c 8 \
  .venv/bin/python .cache/runs/sentinel-heldout-v2/replays/audit_counterfactual.py
```

The adjacent JSON `classic12-hunter-b5-turn75-pass-counterfactual.json` records
the source hash, intervention, all subsequent actions, and result. The bounded
CPU run took 8.41 seconds including compilation; this is not an inference
benchmark. The source hash was unchanged before and after.

## Competition 20×19: a visible invasion becomes a forgotten invasion

Sentinel's general is `(4,2)`, Expander's `(0,16)`. Sentinel campaigns south and
east, builds a castle at `(13,5)` on turn 171, and gains an economic lead. Its
first sortie north from the general is at turn 243, after substantial southern
expansion. That sortie also heads east. The enemy general is never visible.

| Turn | Observation and decision | Consequence |
| --- | --- | --- |
| 100 / 250 | Land leads are 42/28 and 96/71; army leads are 109/98 and 370/311. | A substantial economic advantage is available to defend. |
| 276 | General sends 18 of its 36 troops north. | This stack follows the northern corridor east, away from home. |
| 280–284 | The future attacking stack is visibly 72 at `(0,11)`, then 72 at `(1,11)`, 65 at `(1,10)`, 63 at `(1,9)`, and 62 at `(2,9)`. Home has only 20–22. | Reserve stays 3 because the threat is beyond three route steps. Sentinel's nearby campaign stack continues east. |
| 285–290 | Enemy remains visible: 60 at `(2,8)`, then 58 at `(1,8)`, finally 56 at `(1,7)`. | Sentinel continues capturing toward the upper-right corner; it does not establish a homeward defense plan. |
| 291–294 | The stack enters fog, then briefly reappears with 52 at `(1,5)` on turn 294. Home has 27 and reserve remains 3. | There is another observable warning 14 decisions before the final attack. |
| 295–307 | In hindsight, the stack follows the top and left edges toward home, losing troops along the route. Sentinel cannot see this stack during these turns. | A stateless policy retains no threat estimate from its earlier sightings. |
| 306 | Sentinel leads 499/371 in army and 119/80 in land, with home at 34. It builds a second castle far away at `(11,13)`. | Economic strength is concentrated outside the defensive area. |
| 307–308 | At 307, home sends 17 right; the attacker is still hidden two steps west. At 308, enemy 42 is finally visible adjacent at `(4,1)`. Reserve jumps from 3 to 42. | Sentinel returns 19 from `(4,3)` to home's 18, totaling 37. Expander sends 41 and captures the general at 309. |

The earliest clear warning in this episode is turn 280: a visible 72-army stack
is much larger than the home garrison and is moving across the northern front.
By turn 284 its movement toward the left is evident, while Sentinel has a
20-army stack at `(1,7)` that could be considered for retreat or interception.
This is a candidate planning window; no counterfactual rescue was tested.
Turn 294 supplies a later warning using only the actual observation. Labeling
the defeat merely an unseen flanking attack would miss these visible sightings.

The final sortie is not sufficient to explain the loss. Keeping all 34 troops
home at turn 307, with the same next enemy action, would leave 35 after growth;
the pre-existing adjacent stack could add only 2 on turn 308. That is still 37
against 41. Likewise, the remote castle build at 306 is not demonstrated to be
the decisive error. The two recorded castle losses occur with elimination and
are consequences of losing the general, not evidence that castle capture caused
the defeat. Deathtouch was scheduled for turn 800 and never activated.

## Separate v3 experiments

1. **Sustained defense against successive waves.** Compare a temporary defensive
   objective with the current per-turn reserve discount. Test whether retaining
   or intercepting with a gathered stack prevents repeated depletion without
   reintroducing the v1 corridor stalemates. The failed one-pass branch makes a
   single-turn patch an unsupported remedy.
2. **Arrival-time estimates beyond three steps.** Estimate how soon a visible
   enemy can reach home and how much friendly army can arrive first. Test on
   multiple approaches and simultaneous moves, not only adjacent attackers.
3. **Decaying fog threat memory.** Track last-seen large stacks with reachable
   regions and an age-dependent confidence bound. Turn 294 is a useful crafted
   test input. This requires an explicit stateful deployment/training interface;
   v2's stateless API cannot retain those sightings.
4. **Defensive scouting and local reinforcement measures.** Compare nearby fog
   coverage and rally routes with distant exploration. Instrument projected
   home survival and troops able to arrive before a threat, alongside total army
   and land. Both losses show why those global totals alone are inadequate.

Evaluate such changes on development maps, preserve the v1 anti-stalemate tests,
and reserve new unseen maps and public-opponent games for a fresh gate. These
two selected losses identify mechanisms to test; they do not estimate how often
each mechanism occurs across all opponents or maps.
