# V15: retained opening income does not preserve the winning routes

Both V15 modes fail the [preregistered 24-game screen](v15-plan.md), losing all
six of their selected games. The strategy is rejected and neither mode is
promoted. The default remains V2 and the [competitive goal](competitive-goal.md)
remains active.

| Opponent / mirrored cases | V10 | V10-v6 | V15 tick | V15 fanout |
|---|---|---|---|---|
| Amin / 1, 2 | 2 wins, 1,033 turns | 2 losses, 370 | 2 losses, 239 | 2 losses, 369 |
| Juraj V3.5 / 12, 15 | 2 losses, 422 | 2 wins, 1,075 | 2 losses, 422 | 2 losses, 840 |
| my_bot9 / 4, 7 | 2 wins, 1,076 | 2 wins, 296 | 2 losses, 803 | 2 losses, 604 |

All games start at turn zero against unchanged adaptive original opponents and
finish by capture. These are three consumed physical positions with mirrored
seats/general labels, not 24 independent strength samples. Construction,
deathtouch at 800 and the 1,200-turn cap are unchanged. Reserve 143000 remains
unopened. This screen does not evaluate classic FFA or the full
[playing doctrine](playing-doctrine.md).

## The intended behavior executed

V15 proposes a visible empty neutral capture spending one to three armies,
retaining the parent's general reserve when moving from home. The default tick
mode intervenes only immediately before whole-land production; fanout permits
the same choice between ticks. Both remember first hostile contact and stop
overriding thereafter. Parent captures, construction, defense and all actually
issued attack plans retain priority.

| Opponent | Tick overrides per mirror | Fanout overrides per mirror |
|---|---:|---:|
| Amin | 2 | 23 |
| Juraj V3.5 | 0 | 10 |
| my_bot9 | 1 | 10 |

All 92 issued overrides capture their intended tile in the next original public
observation; 90 retain it continuously through the next land tick. The two
exceptions are mirrors of Amin fanout's turn-106 capture, lost before tick 150
and later recaptured. Counting that recapture as continuous retention would
overstate the result. These are actual acquisitions and income opportunities,
yet they do not establish a better complete strategy.

Tick's Juraj games have no override and reproduce V10's entire joint history.
They provide no evidence about an alternative frontier trajectory there.
All other tick histories first differ at 49; fanout first differs at 16.

## Income, packet movement and capital exposure

On Amin, tick mode captures north of home at 49 and an empty western tile at 99.
Both survive their next production event. At tick 50 it has 15 land / 41 armies
versus V10's 14 / 40; at 100 it has 44 / 110 versus 43 / 108. It nevertheless
loses at 239. In its last preterminal observation it still leads the aggregate
scoreboard, 84 land / 226 armies versus 72 / 197. The issued reinforcements do
not bring sufficient force to the capital before capture.

Amin fanout makes 23 early acquisitions but has no land advantage at 100:
43 land / 109 armies versus V10's 43 / 108. Between 100 and 107 it postpones the
same proposed nine-army packet transfer on eight consecutive decisions. Every
chosen little capture is real; the larger packet is delayed. This is a concrete
transport cost, not proof that that delay alone causes the final loss.

The broader external trajectories also separate income from strength. At 100,
my_bot9 fanout has 37 land / 89 armies versus V10's 24 / 65; it still loses.
Juraj fanout has 40 / 99 versus V10's 36 / 86, but later trails economically.
It builds its first castle at 824, while the winning V10-v6 control first builds
at 447 and owns sixteen castles at termination. These are trajectory
differences, not proof that forcing an earlier build would win.

The original opponent wires expose a further cost to inspect:

| Opponent | First view of our capital: V10 | V10-v6 | Tick | Fanout |
|---|---:|---:|---:|---:|
| Amin | 360 | 288 | 238 | 166 |
| Juraj V3.5 | 243 | 271 | 243 | 485 |
| my_bot9 | 394 | never before our win | 464 | 329 |

This uses the opponent's own archived public view for retrospective diagnosis;
it is never supplied to our policy. Exposure moves earlier in some trajectories
and later in others. Neither additional land nor later disclosure alone predicts
a win. No single-intervention counterfactual proves that a particular opening
move or disclosure event caused a complete loss.

The contact latch works as specified. Amin tick first sees a hostile tile at
128 and fanout at 130; neither overrides afterward, including during later fog.
Their postcontact parent decisions differ because their preceding trajectories
differ. Preserving the original controller's priorities does not preserve the
original winning game.

## Complete defensive failures

Amin tick sends a half-general packet away at 225. An enemy stack advances on
the final approach from 229, reaching 48 armies adjacent to home at 238 against
25 defenders. The policy reports no feasible interception in the recorded late
interval; reinforcement is insufficient and the general falls at 239 despite
the aggregate army advantage.

Amin fanout starts an interception at 359 with projected arrival 33 and added
defense 20 against a required 18. On 360 the enemy temporarily disappears from
the public view. The parent releases the commitment without an expiry, gap or
army-consistency failure, then sends half the home army west. The enemy
reappears, reaches home with 41 against 11, and wins at 369. This is a specific
release during lost visibility; an alternate successful rescue has not been
demonstrated. Both losses precede deathtouch.

Against my_bot9, tick's single turn-49 acquisition remains owned through the
last observation, and its land lead over V10 grows to 58 versus 32 at 200.
That does not produce a sustained economic lead over its adaptive opponent:
by 350 it has no castle against two. At 802 an adjacent nine-army enemy gets
its deathtouch move before the candidate's half-general counterattack from 28,
ending the game at 803. Fanout builds one castle but faces five enemy castles
by 600. The final enemy merge produces 94 armies; an ordinary 93-army capture
overwhelms home and its reinforcement, ending at 604.

The full opponent reports and original terminal records are retained in the
machine evidence. Original my_bot9 emits sixteen physically invalid actions in
each fanout mirror. The runner preserves and normalizes those actions under the
existing rules; it does not repair the opponent or count them as runtime faults.
The resulting losses remain measured against that original implementation.

## What changes next

This rejects the two tested opening interventions on these selected 1v1 cases;
it does not establish that local growth is wrong in FFA. The user's distinction
between compact growth, purposeful attack lines, capital concealment and a
sustained deathball requires more than a neutral-capture override.

The parent campaign repeatedly chooses among visible enemy border tiles. Its
concentration layer selects one enemy target and clears offensive memory after
attacking it. Neither records sustained capital-search progress after capture.
The next audit must follow complete packet routes and retained captures,
disclosure and defensive releases across the successful and losing controls.
That can distinguish persistent attack progress, search and temporary-threat-fog
failures before choosing another policy change. A new transfer quota or another
opening threshold is not justified by these results alone.

## Verification and delivery

The frozen source, plan, four public fixtures and preflight evidence were
delivered before gameplay at `7ce3dfd9b2217c9ed59cd752a148aa1252dd3a42`.
Eighteen focused tests pass. The matched official-version warm profile retains
87.7–91.1% of V10 throughput across both modes and three fixtures. Cold V15
compilation takes 14.46–14.86 seconds excluding imports; cold startup,
unmeasured shapes and actual submission deadlines remain unqualified.

All 15,098 original candidate decisions are retained, including 6,554 enabled
V15 decisions. Independent checks reconstruct every frontier proposal, legal
candidate count, expenditure/locality ranking, contact latch, priority,
eligibility, selected action and offensive-memory cancellation. Native memory
contains 19 / 7 / 20 / 20 scalar int32 fields for V10 / V10-v6 / tick / fanout.
All twelve original control joint histories and outcomes match, all twelve
mirror pairs agree, and all sixteen external opponent processes exit cleanly
and are reaped. No historical external seven-field memory oracle is invented:
fresh continuity and original joint action parity are verified separately.

Candidate invalid/malformed actions are zero. All 190 original opponent invalid
actions are retained: 158 Amin and 32 my_bot9. Runtime faults are zero. Both
runners and all frozen owner postchecks complete on their first attempt; the
independent reviewer also exits zero. The root summary's first attempt failed
on an external reply schema assumption; correcting its parser required no new
inference, game or source change.

`v15-development-evidence.json` embeds results, diagnoses, independent checks
and artifact identities. Complete raw histories and helpers remain under
`.cache/runs/sentinel-v15/`. A failed experiment is not competitive superiority;
the full opponent-coverage, fresh-evaluation and deployment goals remain open.
