# V15: test local opening income against a purposeful attack line

## Question and evidence

Can small, visible neutral captures before first contact improve retained land
income without losing the successful attacks of stronger controls?

The owner's [playing doctrine](playing-doctrine.md) calls for local growth around
land ticks, purposeful gathering and capital search, with different strategic
needs in 1v1 and FFA. Construction remains in scope for competition mode. The
completed V14 service ledger did not improve strength; a serviced transfer did
not necessarily develop land or deliver an offensive packet.

The V15 opening audit inspects the original first 200 decisions of V10, V10-v6
and V14 on three consumed positions, with mirrored histories. It introduces no
new gameplay or hidden-state policy inputs. The findings distinguish opportunities
from proven mistakes:

- Amin's turn 49 has a legal half-general capture north, leaving the parent's
  three-army reserve, immediately before land production. Its selected remote
  transfer instead leads to a capture after the tick. My_bot9 also has a local
  neutral alternative on turn 49.
- Earlier alternatives at 16/17 divert a packet whose existing route already
  produces two retained captures. Earlier local expansion is therefore a
  tradeoff, not a free gain.
- Amin's collections at 129 and 142 capture enemy tiles, including a capture
  just before tick 150. My_bot9's ready attacks and Juraj's productive attack
  route are also negative controls against indiscriminate expansion.
- More land immediately is insufficient: Amin V10-v6 has 72 land at 150 versus
  V10's 64, but retains only six of its next twenty capture events through 200,
  versus V10's twenty-two of twenty-eight, and eventually loses.

The full audits and input hashes remain in
`.cache/runs/sentinel-v15/frontier-audit-{amin,external}/`. Embedded public
fixtures bind their original wire and memory provenance separately where needed.
The earlier full-only small-garrison prototype passed twelve tests before these
audits sharpened the design below. That prototype was not evaluated in games;
its tests do not qualify the revised policy.

## Frozen comparison to prepare

V15 wraps one frozen V10 call. Its two modes share the same proposal and priority
rules. `tick` is the default and acts only on decisions congruent to 49 modulo
50: moves execute before the ensuing whole-land growth. `fanout` permits the
same proposal between ticks, testing whether broader compact development pays
for the displaced line progress.

The proposal captures a currently visible empty neutral plain. It considers
full and half moves spending one to three armies, excludes owned castles as
donors, and requires a general-source move to retain the parent's current
reserve. Rank by least army spent, then nearest destination to home in Manhattan
distance, then most adjacent visible empty neutral plains. Manhattan distance
measures locality, not route length or safety under fog. Large field packets
cannot be spent just to acquire one empty tile.

Only an actual parent PASS or transfer into our own territory may yield. Keep
BUILD, captures, all actually issued offensive plans (including new collection
and ready deployment), committed defense, pursuit and mobilization. Memory
records first observed hostile contact before arbitration; once contact is
recorded, neither mode overrides again that game. It survives fog and forward
observation gaps, resetting on a shape/nonmonotonic-time reset or explicit new
initialization. Previously observed general knowledge also counts as contact.
This is a public-information heuristic, not proof that we have remained unseen
or that an unobserved opponent cannot attack.

Enabled memory contains the original nineteen-field parent and one int32 contact
flag. Disabled operation returns the exact original V10 tuple and native
nineteen-field memory. If an action changes, reconcile unissued offensive memory,
retain returned defender and public general knowledge, and require
`frontier_parent_action_issued` before attributing inherited issued-action events.
Record proposal, parent action, availability, window, contact, priorities,
eligibility, issued action and next tick separately.

This does not implement distinct FFA contacts, per-opponent scores, capital
obfuscation, sustained deathball search or a complete economic planner. The
current observation contract cannot distinguish individual FFA enemies. Their
implementation and multiplayer evaluation remain part of the full goal.

## Tests and runtime gate before games

Check recorded opening/tick behavior, preserved productive attack plans, exact
disabled behavior, actual full/half expenditure and reserve, fog/structure
exclusion, contact persistence/reset and isolated typed batched memory. Check
the actual engine's capture at 49 versus 50 and general growth at step two.
These establish contracts, not strength.

Profile frozen V10, V15-tick and V15-fanout in the same official-version runtime
on Juraj 17, Amin 49 and Amin 129. Use five rotated repetitions of 100 warmed,
fully synchronized complete calls per arm and fixture, retaining cold compilation
separately. Active and blocked cases must be distinguished. Retaining less than
50% of parent throughput disqualifies strategy evidence until the runtime issue
is fixed. This is not standalone startup or all-shape deadline qualification.

## Complete-game screen

Freeze source, helpers, original inputs, tests, profile and this plan, then
deliver the validated candidate to `origin/main` before releasing games. The
screen has four arms: original V10, original V10-v6, V15-tick and V15-fanout.
Use the unchanged seed-83000 concrete positions and original keys:

| Original opponent | Cases | Physical positions | Games across four arms |
|---|---|---:|---:|
| Amin main8 iter160 | 1, 2 | 1, mirrored | 8 |
| Juraj V3.5 | 12, 15 | 1, mirrored | 8 |
| my_bot9 | 4, 7 | 1, mirrored | 8 |

All 24 games start at turn zero against unchanged adaptive original opponents,
with competition construction, deathtouch at 800 and a 1,200-turn cap. No
healthy reruns, source changes mid-run or early-window substitutes for complete
outcomes. Preserve raw/applied joint actions, public wires, native 19/7/20/20
memory, keys, decisions, diagnostics and process cleanup. Original controls
must reproduce complete archived joint actions and outcomes. Verify native
initialization and continuity; do not invent an archived seven-field memory
oracle where only original raw histories are available.

For each V15 mode, independently recompute contact/tick/arbitration and compare
paired complete results. Require wins on all three selected positions in both
mirrors to proceed to a broader gate; a mode that fails is rejected, even if it
owns more land or survives longer. Preserve each position's stronger control,
rather than accepting gains against only a weaker parent.

Diagnose actual captures retained through ticks, ordinary tile income, castle
ownership, displaced route progress, contact timing, home exposure and complete
losses. Diagnostic hidden state must never enter the policy. These are three
consumed development positions, not fresh strength evidence. Reserve 143000
remains unopened. Any passing mode still needs broader opponent coverage,
reserved-map paired uncertainty and actual deployment qualification before
promotion. The full competitive goal stays active.
