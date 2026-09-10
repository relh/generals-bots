# V14: balanced action service does not establish stronger play

V14 fails the [preregistered eighteen-game comparison](v14-plan.md). Its ledger
and memory behavior execute correctly, but the candidate loses both Amin and
Juraj placements. Winning my_bot9 does not compensate for those failures. V14
is not promoted; V2 remains the default and stronger controls are preserved.

| Original opponent | Cases | V10 | V10-v6 | V14 |
|---|---|---|---|---|
| Amin main8 iter160 | 1, 2 | 2 wins, 1,033 turns | 2 losses, 370 | 2 losses, 524 |
| Juraj V3.5 | 12, 15 | 2 losses, 422 | 2 wins, 1,075 | 2 losses, 913 |
| my_bot9 | 4, 7 | 2 wins, 1,076 | 2 wins, 296 | 2 wins, 992 |

All eighteen games start at turn zero and terminate by capture with unchanged
castle construction, deathtouch at 800 and a 1,200-turn cap. These are three
consumed physical placements with mirrored seats/general labels and three
policies, not eighteen independent strength samples. Seed 143000 remains
unopened. All twelve controls match their original complete joint raw/applied
histories and outcomes. All nine candidate-relative mirror pairs agree.

## What was tested

An activation preflight rejected the previously proposed V6 plus remembered-
general pursuit composition before implementation: its selected Amin controls
never see the enemy general before losing at 370, so pursuit cannot change that
prefix. V14 instead addresses a mechanism that can act in the early game.

The ledger charges one unit per issued collection transfer. Eligible
uncommitted V6 campaign moves repay one unit, saturated at zero. A new collection
with debt yields only to a different eligible MOVE. Existing plans and ready
direct deployments remain available; coincident collection/campaign moves keep
their plan and charge once. No PASS or BUILD is forced as repayment. The policy
uses the actual inner V6 proposal without another policy solve and clears only
unissued offensive memory after a deferral. Returned defender state and public
general knowledge remain. Inherited events require `service_parent_action_issued`.

Across the six V14 games, all 4,858 original decisions pass independent ledger
recomputation:

| Quantity | Count |
|---|---:|
| Deferred new collections | 272 |
| Collection charges | 660 |
| Eligible campaign service moves | 2,478 |
| Debt units actually repaid | 660 |
| Maximum outstanding debt | 6 |

The additional 1,818 service moves occur at zero debt and do not repay anything.
There are no runtime memory resets; every game ends with zero debt. Exact
accounting is not evidence that the two action classes have equal strategic
value, nor that serviced moves advance the economy.

## Complete-episode consequences

Amin has 53 deferrals per mirror: 27 owned transfers, 25 enemy captures and one
neutral capture. The 86 debt units repaid per mirror consist of 49 owned transfers,
32 enemy captures and only five neutral captures. All deferred attacks really
capture their destinations in the next public observation.

At the first changed decision, 193, V14 spends three turns on campaign moves
before running the original four-action collection at 196–199. Both players'
public views and the underlying nineteen-field parent memory rejoin at 200–203
and later at 206–224; the extra V14 ledger is separate state. The first opponent
action difference occurs at 266. The first deferral alone therefore does not
establish a persistent public-state divergence responsible for the final loss.

Over the same first 524 decisions, V10 makes 295 owned transfers, 94 neutral
capture attempts and 123 enemy capture attempts. V14 makes 306, 79 and 127.
The ledger neither reduces aggregate transport nor increases neutral expansion
in this trajectory. None of the three Amin arms builds a castle in the shared
first 524 decisions. V10 later builds seven, first on action 655 (visible at 656),
after V14 has already lost; that later production gain did not
exist during the shared early comparison window. The final V14
invasion reaches the adjacent (9,17) cell with 67 armies against 26 at home;
the recorded policy reports no feasible interception, and capture follows at 524. The final
candidate moves are remote transfers. No earlier individual deferral has been
isolated as a sufficient cause of the whole-game failure.

Juraj's first changed action at 106 does service the intended large packet.
Moves 106–108 advance it from (9,17) through (10,17), (10,16) and (11,16).
Debt then reaches zero and another collection takes turns 109–113; the packet
resumes only at 116. Counting those transfers as completed service therefore
does not ensure that the packet reaches and exploits a frontier.

| Juraj opening window | V10 transfers / neutral / enemy attempts | V14 transfers / neutral / enemy attempts |
|---|---|---|
| 100–149 | 31 / 4 / 15 | 34 / 4 / 12 |
| 150–199 | 33 / 2 / 15 | 28 / 2 / 20 |

V14 still has 42 land against 51 at 150. Its later position improves to 52 against 60
at 200, but neutral capture attempts remain unchanged. Across the 913-turn game,
121 debt repayments comprise 56 owned transfers, 58 enemy captures and seven
neutral captures. The 45 deferrals comprise 22, 22 and one, respectively. V14
builds castles at 700 and 870 and owns two at termination; the winning V10-v6
control first builds at 447 and owns sixteen at its termination. These are
complete-trajectory differences, not proof that forcing one earlier build wins.

The terminal Juraj loss is separate from ledger arithmetic. With debt already
zero, a defender plan at 909 responds to a visible 12-army enemy and moves a
nine-army donor into an owned two-army screen, expecting ten. The opponent
simultaneously merges into its own six-army cell, producing 17. The defender
releases, and the policy reports no feasible interception. The enemy proceeds to home and
captures it with eleven moving armies against sixteen under deathtouch. No
service deferral causes that final action directly, and no successful alternate
rescue has been demonstrated.

My_bot9 remains a win. V14 first reveals the enemy general at 891 and captures
it under deathtouch on action 991 with a 69-army source against 104 at the target.
The match ends at 992. Home mobilization does not explain this selected win.
The V10-v6 control wins much earlier; the success criterion is preservation of
wins, not treating shorter or longer games alone as strength.

## What changes next

The owner's [playing doctrine](playing-doctrine.md) now directs tick-aware local
growth and distinct FFA/1v1 behavior, with castle acquisition matched to the
rules; competition construction remains in scope. These historical build-enabled
1v1 games tested neither that doctrine nor FFA strength. New work must measure
actual land-tick income and, for FFA, expose distinct public contacts and scores.

Both pursuit-only composition and generic campaign-service scheduling have now
failed necessary parts of the objective. The latter serviced real actions but
left neutral expansion scarce and delayed economic investment. The next bounded
work should separate actual frontier progress and neutral-development choices
from the enemy-focused campaign's transfer ranking. It must preserve useful
concentration near the other front and respect the successful controls; neither
a neutral-capture quota nor an earlier castle is already proven to do that.
Explicit expansion proposals or service earned on completed frontier progress
are hypotheses requiring their own cost and complete-game checks. A balanced
counter or another late survival improvement cannot substitute for wins.

## Verification and reproducibility

The candidate, plan, tests and five embedded public fixtures were committed
before games at `97b70e6f88aa91661e833d5650a07a2bb735babd`. All 45 frozen source
bindings remain unchanged. Eleven focused tests pass: nine in the initial run
and two after correcting test assertions that misread an action list as a
dictionary. The policy did not change. Style and diff checks pass. A broken
shared `uvx` tool environment was bypassed with an isolated ignored Ruff 0.16.6
installation; the repository runtime was not changed.

The matched official-version CPU5 profile uses five rotated repetitions of 100
synchronized complete calls per arm and fixture:

| Diagnostic fixture | V10 median | V14 median | Parent throughput retained |
|---|---:|---:|---:|
| Juraj 106, debt 3 | 1.202 ms | 1.283 ms | 93.66% |
| Amin 193, debt 3 | 1.369 ms | 1.424 ms | 96.18% |
| Amin 129, debt 0 | 1.365 ms | 1.418 ms | 96.30% |

The added debt scalars are explicitly diagnostic state alongside original parent
memory, not archived V14 histories. Distinct-shape V14 compilation takes 13.14
and 13.55 seconds excluding imports; Amin 129 reuses the shape compiled at 193.
The profile does not qualify uncached startup, other shapes or submission
response deadlines. Candidate game inference is synchronous; original external
opponents retain their separate stdio deadlines.

The complete records contain 13,402 candidate decisions across all arms, with
native 19/7/20 scalar int32 memory, public observations, keys, original and applied
actions, proposals and telemetry. V10's original nineteen-field memory archive
also matches. No archived internal-memory oracle is available for the selected
external V10-v6 controls; their newly recorded seven-field state is validated
for initialization and continuity, without claiming historical memory equality.
Candidate invalid/malformed actions and external faults are zero. All 158 original
Amin invalid actions remain retained; external invalid actions are zero. All
twelve external opponent processes terminate cleanly and are reaped.

`v14-development-evidence.json` embeds complete results, independent reviews,
runtime/tests, diagnoses and artifact identities. Full histories and helpers
remain under `.cache/runs/sentinel-v14/`, outside the committed repository.
The full competitive goal remains active; this experiment establishes no fresh
winning advantage against the complete local or public opponent set.
