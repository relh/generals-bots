# V17: complete capture-chain screen on Zephyrus

**V17 fails preservation: two wins and four losses. It is not promoted.**
All 24 preregistered games completed on Zephyrus, with complete control parity,
runtime records and independent history checks. V2 remains the default. This
does not achieve the [competitive objective](competitive-goal.md).

The [frozen plan](v17-plan.md) required V17 to win all six selected cases.
Each table entry describes two matching label mirrors of one consumed physical
position, not two independent maps. All games use seed 83000, competition
construction, deathtouch at 800 and a 1,200-turn cap. Reserved seed 143000 remains
untouched. Opponents respond adaptively from turn zero.

| Opponent | V10 | V10-v6 | V16 | V17 |
|---|---:|---:|---:|---:|
| Amin main8 iter160 | win 1033 | loss 370 | loss 531 | loss 235 |
| Juraj V3.5 | loss 422 | win 1075 | loss 335 | loss 790 |
| my_bot9 | win 1076 | win 296 | win 769 | win 961 |

V17 starts 86 chains across its six games and issues 214 chain attacks, all
confirmed as captures in the next public observation. Those include 162 enemy
and 52 neutral captures. Eight chains abort. These counters do not compensate
for losing both positions where a stronger control wins.

## Complete failures

Amin diverges from V10 and V16 at turn 161: the 21-army packet at zero-based
(11,9) moves down instead of up. All four chains and ten chain attacks per mirror complete
their captures, but the game ends at 235. A 55-army enemy is publicly visible
at 169; a sequence of visible large forces ends at 175. No visible packet has
at least 20 armies again until 224. The record cannot identify the later force
as the earlier one through fog. Defense overrides occur at 169–171, yet home
makes half transfers at 173 and 218, leaving 16 and 12 armies respectively.

At 224 a 64-army enemy is visible eleven Manhattan steps from home, which has
15 armies. The next ten decisions report a home deficit without a feasible
intercept, while remote movement continues. At 234 an enemy with 47 stands
adjacent to the 20-army capital; the next transition loses. Our total army still
leads 221 to 212. A read-only shortest-owned-route calculation can gather 56
at home in eleven actions at 161, ignoring replies. At 224 that route class
supplies only 42, but a looser instant-delivery donor bound reaches 60. This is
evidence for earlier allocation work, not an impossibility proof or a guaranteed
counterfactual win. V17 has no turn-440–530 Amin history to compare with V16's
later failure window.

Juraj's seven chains and fourteen captures per mirror do not create a winning
campaign. Ten captures persist to the next land tick and four persist for 50
and 100 turns; all chains end by 390. During turns 100–199 V17 spends 60 actions
on owned transfers, 33 on enemy attacks and seven on neutral attacks. The
winning V10-v6 control spends 51, 27 and 22 respectively. V17 holds 47 tiles at
200 versus that control's 67. At 700, V17 has 478 armies against Juraj's 839.

A 60-army threat is visible at 777, then 62 at 779 while a remote packet makes
small captures. Defense begins at 780. The threat disappears from view at
783–784; home splits its 27-army garrison at 783. At 789, home has 28 against an adjacent
enemy with 33, while a remote 22-army packet twelve steps away moves elsewhere.
The next transition loses before deathtouch. Longer survival than V16 does not
preserve V10-v6's victory or establish adequate economy, deployment or defense.

## Preserved wins and opponent invalid actions

my_bot9 still loses both V17 games, at 961. Each V17 mirror starts 32 chains,
issues 83 chain attacks and confirms all 83 captures. The unchanged opponent
also issues **295 invalid BUILD actions per game**, each normalized to pass.
Its original proximity surcharge is 10 rather than the engine's 14. Every
invalid build is affordable under its own formula plus reserve but unaffordable
under the actual rules. The five repeated-build intervals are 390–439,
489–492, 584–599, 728–807 and 815–959. All my_bot9 control games have zero such invalids.
These are retained failures of the original opponent, not repaired actions or
runtime faults. Winning later than the clean V16 and V10-v6 controls does not
establish superiority over an opponent that computes affordable builds.

## Host qualification and integrity

The [machine evidence](v17-results-evidence.json) binds the source release,
both game reports, complete audits, amendments and all attempts. Zephyrus
rebuilds the exact candidate Python 3.12.10/JAX 0.11.0 runtime and engine
JAX 0.11.1 environment. All nine archived runtime comparisons and the external
artifact identities match. The host's 29 correctness tests pass in 105.26s;
all ten matched warm contexts pass, retaining 81.3–92.1% of V16 throughput.
These are separate Zephyrus measurements, not relabeled Titan results.

Both game runners exit zero on their first attempts. Complete review covers
15,786 decisions, including 11,814 control decisions and all twelve mirror
comparisons. Candidate invalid and malformed actions are zero. Amin's original
202 invalid actions and my_bot9's 590 invalid builds remain recorded. External
opponents have zero protocol/deadline faults and clean process exits. Independent
NumPy reconstruction matches all 22,896 external public wires byte for byte.

The first external strict postcheck fails at my_bot9 turn 728 because its
NumPy score expression rounds multiplication and subtraction separately.
Single-round float32 arithmetic ties cells 136 and 302 and selects 136, exactly
as the recorded policy does. Separate `libm.fmaf` arithmetic and an isolated
score-expression compilation in the pinned runtime confirm this; no policy or
game is rerun. An additive checker corrects only this arithmetic, retaining
exact objective assertions and every original lifecycle and parity check.
It passes 7,918 recorded diagnostic frames and rejects a deliberately wrong
objective. The amended strict check and public-wire check then pass. The
original failed attempt, helpers, release and histories remain unchanged.

Before games, an independent review-helper correction also distinguished actual
incoming phase 4 from V17's cleared parent input, so a legitimate released-chain
feeder start is not rejected. Six synthetic postprocessor regressions pass.
Neither checker correction changes policy behavior or turns a loss into a win.

Candidate inference and Amin's adapter are synchronous. Warm-call qualification
and complete strategy games do not establish official startup or deployment
deadlines. The external opponents retain their enforced 10-second first and
150-ms later deadlines. Classic and FFA strength, broad distinct-opponent
coverage and fresh paired winning evidence remain unqualified.

## Next investigation

Reject V17 as a preservation candidate. Investigate allocation after earlier
public concentrations, the release of defensive commitments during threat fog,
and home donations alongside purposeful field deployment and neutral expansion.
Compare those decisions with the complete winning controls before selecting a
new policy. A late horizon extension, a single donation veto or more successful
local captures has not been shown to repair these games. Preserve the stronger
controls and require a separately frozen experiment followed by fresh paired
evidence before promotion.
