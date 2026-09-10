# V18 selected-screen results

**V18 is rejected: two wins and four losses fail the required six wins.** All
24 adaptive games, strict integrity checks and the independent complete-history
review finished. V18 captures durable territory but does not preserve the
stronger controls. V2 remains the default and the competitive goal is unmet.

## Complete fixed screen

The [preregistered experiment](v18-plan.md) uses three previously consumed
seed-83000 positions with label mirrors, construction, deathtouch at 800 and a
1,200-turn cap. Every pair has identical label-normalized histories; these are
not six independent maps. Reserved seed 143000 remains untouched.

| Opponent and cases | V10 | V10-v6 | V16 | V18 |
|---|---|---|---|---|
| Amin 1/2 | Win 1,033 | Loss 370 | Loss 531 | Loss 546 |
| Juraj V3.5 12/15 | Loss 422 | Win 1,075 | Loss 335 | Loss 447 |
| Original my_bot9 4/7 | Win 1,076 | Win 296 | Win 769 | Win 888 |

Each cell describes both complete mirrors. The 18 control games reproduce
their original public observations, action keys, native memory, telemetry and
joint actions. Controls complete before V18 starts in each runner. Original
opponents adapt to the changed trajectory; no recorded future replies replace
their decisions.

The implementation and [preflight](v18-development.md) were delivered at
`5856f4d58091fe985ec34413fd3c678f76b3c79b`: 35 focused cases, six matched warm
contexts and 19 independent checker cases. Enabled throughput is 80.4–93.7%
of V10 in the fixed contexts. These remain limited correctness/runtime results,
without official startup, all-shape or playing-strength qualification.

## The captures execute and often produce

All 522 issued rear captures are confirmed from the next original public
observation. None is terminal-censored or unconfirmed. Every inherited action
event is masked when the rear move replaces its parent proposal.

| Per mirror | Rear captures | Held to next land tick | Held 50 turns | Held 100 turns |
|---|---:|---:|---:|---:|
| Amin | 77 | 68 | 68 | 63 |
| Juraj | 95 | 90 | 87 | 69 |
| my_bot9 | 89 | 82 | 79 | 73 |

Across both mirrors, 480 captures remain continuously owned through their next
land tick, 20 are lost beforehand and 22 lack a long enough remaining episode
to observe that tick. At 50 turns the held/lost/censored counts are 468/28/26;
at 100 turns they are 410/46/66. These are repeated observations on three
trajectories, not independent treatment effects.

The 522 replacements displace 352 home-source and 170 field-source proposals.
Longest consecutive rear runs are 24, 18 and 15 turns against Amin, Juraj and
my_bot9 respectively. A 15-turn field burst also occurs in the my_bot9 win, so
burst length alone does not distinguish failure. V14 already tested conditional
collection/campaign service and failed Amin/Juraj; adding service again would
need an explicit account of that prior result. V18's new evidence is durable
capture progress, not a proven rule for scheduling the parent afterward.

## Complete failures and remaining allocation problem

On Amin, the first change at 150 begins 24 consecutive replacements of the
same home half-transfer proposal through 173. A second rear burst spans 207–227.
By 500 V18 has 157 land and 700 armies against 100 land and 449 armies, but its
largest ordinary field stack is only ten. Much of the force remains dispersed.
The last rear move is 508; the terminal sequence uses parent actions. Home is
halved at 524, enemy force 106 becomes visible at 532, and the final observation
has enemy 56 adjacent to home 21. A remote 55-army packet is 21 Manhattan steps
away. More territory and successful local captures have not produced timely
defensive concentration. This does not prove that reversing one earlier action
would recover V10's win.

Juraj remains a loss despite 95 confirmed rear captures per mirror, 90 held to
their next land tick. Of the 95 replacements, 88 displace proposed home moves.
The last override is 424. At 438 the parent starts a defender with predicted
arrival army 18 against required defense 17; the visible 57-army threat then
disappears. The commitment releases at 439 despite expiry 448, and the actual
packet reaches its former target before reversing twice. Enemy force becomes
visible again at 442 and reaches home with 49 against 27 at 446. V18 loses at
447 while still ahead in total army, 469 to 446. The actual V10 parent loses at
422 and the stronger V10-v6 control wins at 1,075. This observed release/reversal
does not prove that extending the commitment would win, and any change must
preserve the stronger control's useful defender releases and castle production.

Against my_bot9, V18 has 64 land/166 armies at 200 versus V10's 52/141, and
114/795 at 800 versus 60/244 on their respective adaptive histories. It builds
seven castles for 353 armies. Yet it first sees the enemy general at 887,
one observation before winning, whereas V10-v6 sees it at 265 and wins at 296.
Economic advantage has not ensured prompt discovery and exploitation.

The original my_bot9 makes 17 invalid BUILD attempts per V18 mirror: turns
400–413, 463–464 and 499. All 34 are unaffordable under the unchanged engine's
proximity penalty of 14, while satisfying the bot's penalty of 10 plus its
eight-army reserve. They normalize to PASS and remain recorded. These wins do
not establish performance against a repaired opponent. Amin's 240 invalid
actions across eight games, including 24 per V18 mirror, also remain recorded:
all attempt to move right from the board's rightmost column and normalize to PASS.

## Integrity and scope

Both original runners exit zero. Amin's eight-game postcheck, external strict
checks, all 21,232 external public-wire reconstructions and the independent
24-game review pass on their first attempts. The review covers 15,576 actual
policy frames: 11,814 controls and 3,762 V18 frames, with 12 mirror checks.
Candidate invalid and malformed actions are zero throughout. External opponent
protocol/deadline faults are zero and children exit cleanly; synchronous Amin
and candidate timing remain distinct from enforced external stdio deadlines.

The [result artifact manifest](v18-results-evidence.json) binds complete games,
native traces, releases, every checker, full-episode audits and all attempts.
V18 does not advance to fresh reserved comparisons or promotion. The next
experiment must account for the cost of repeatedly delaying useful transfers
and for turning dispersed economic gains into timely deployable force.
