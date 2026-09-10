# V13: preserving ready attacks does not preserve strength

V13 fails its [preregistered twelve-game screen](v13-plan.md). Restricting V11's
capture arbitration to new collection preserves ready direct deployments, but
both Amin wins still become losses. Juraj survives longer without winning;
my_bot9 remains a win, reached sooner. V13 is experimental and is not promoted.
The competitive goal remains unachieved, and stronger earlier controls remain.

| Original opponent | Cases | V10 | V13 | Terminal turns, V10 → V13 |
|---|---|---|---|---|
| Amin main8 iter160 | 1, 2 | 2 wins | 2 losses | 1,033 → 519 |
| Juraj V3.5 | 12, 15 | 2 losses | 2 losses | 422 → 554 |
| my_bot9 | 4, 7 | 2 wins | 2 wins | 1,076 → 483 |

These are three consumed physical placements, each mirrored across seats and
general labels, with both policies starting at turn zero. All twelve games
terminate by capture under unchanged castle construction, deathtouch at 800,
and a 1,200-turn cap. They are not independent fresh strength samples. Reserve
143000 remains unopened. All six V10 controls reproduce their original complete
histories and outcomes, and all six candidate-relative mirror pairs agree.

## What the ablation establishes

V13 wraps frozen V10 and takes the actual inner V6 proposal without a second
policy solve. It applies V11's immediate enemy-plain capture preference only
when the final parent action would start a new collection toward an ordinary
enemy tile with no larger defending army. Ready direct deployments, continuing
plans and outer priorities keep the parent's arbitration. Rejected collections
clear unissued offensive transport while retaining defender memory and public
general knowledge. Both the original parent action and the V6 proposal are
recorded; inherited events require `collection_capture_parent_action_issued`.

All 32 changed actions across the six V13 games independently satisfy those
public capture conditions: eleven per Amin mirror, three per Juraj mirror,
and two per my_bot9 mirror. Every ready direct start surviving outer arbitration keeps the actual
parent action. This confirms the intended distinction was executed; it does
not establish that equal target defenders imply equal strategic value.

The original useful Amin collection at 193 was an explicit preregistered
counterexample. V13 captures elsewhere at 193, starts that collection at 194,
and captures its original objective at 197 instead of 196. Both players' public
observations and the candidate's incoming memory rejoin at 198; actions agree
through 218. The next replacement at 219 again occurs on a common input. The
first changed decision alone therefore does not demonstrate a persistent
state divergence responsible for the eventual loss.

The later Amin history diverges and loses before deathtouch. No defender
commitment is active during the final ten decisions. The ordinary campaign
moves armies out of home and back again while a larger invasion approaches.
On turn 517 it tries to reinforce home from a screen that the opponent attacks
in the same turn. The engine's chasing-before-reinforcing order resolves the
attack first, captures the source and cancels the outgoing reinforcement.
The next observation has 25 at home against the adjacent 97-army invader;
home is captured at 519. These are valid commands and game mechanics, not a
runtime fault. No individual earlier replacement has been isolated as a
sufficient cause of this terminal force disadvantage.

Juraj's three replacements occur at 129, 164 and 340. Each actually captures
its target; their observed ownership lasts 9, 176 and 96 observations. The
candidate still has no feasible interception against the final visible invasion
and never reveals the enemy general or builds a castle. The final attacker
sends 40 against 17 at home. Longer survival is not a winning advantage.
My_bot9's two replacements at 100 and 150 also capture their targets; the
candidate reveals the enemy general at 481 and captures it with an ordinary attack
at 482, winning at 483. Neither pursuit nor home-mobilization overrides explain
that particular win.

A secondary archived V11 comparison confirms the structural difference on the
external games. With identical public and memory prefixes, V13 preserves a
ready two-action deployment that V11 interrupts at Juraj 178 and my_bot9 80.
Those single decisions have not been isolated as causes of the final results.

## The remaining strategic gap

The opening audit exposes an earlier issue that this rule leaves intact. In the
selected Juraj game, V10 first differs from the stronger V6 at 100. Two locally
successful small collections delay the same larger packet's eight-transfer route
by fifteen turns. By 123, V6 has made nine owned transfers and fifteen captures;
V10 has made seventeen transfers and seven captures. By 149, their captures are
27 versus 17. These are complete-trajectory observations, not evidence that
removing one collection would reproduce the V6 win against an adaptive opponent.

V11 and V13 together reject a simple universal ordering between collecting and
immediately capturing another enemy tile. Direct readiness alone does not make
that ordering preserve strength. The next comparison should separate the
stronger V6 campaign from the later offensive concentration layer while retaining
previously observed enemy-general pursuit and the useful home-mobilization work.
Existing V10-v6 removes concentration and pursuit together; existing V9-disabled
still includes V8 concentration. The missing factorial control is V6 plus
remembered-general pursuit without that concentration layer. Its Amin strength
is unproven, and it must face the stronger controls in complete games before
any promotion. Further local defense vetoes do not substitute for this test.

## Correctness, runtime and evidence

Eight focused tests pass: six initial cases and two added real Amin fixtures.
They cover exact disabled V10 output including telemetry, original Juraj
collection arbitration, continuing plans, structure objectives, batched memory,
ready deployment at 163 and the productive-collection counterexample at 193.
Ruff and diff checks pass. The source, tests, fixtures and plan were merged to
`origin/main` before games at `7a81ab3a34809e7949d483157678c4cba5408701`.
All 44 frozen source bindings remain unchanged throughout evaluation.

The official-version CPU5 benchmark uses five rotated repetitions of 100
synchronized whole-step calls per policy and fixture:

| Fixture | V10 median | V13 median | Parent throughput retained |
|---|---:|---:|---:|
| Juraj collection, 129 | 1.187 ms | 1.281 ms | 92.65% |
| Amin ready deployment, 163 | 1.351 ms | 1.413 ms | 95.65% |
| Amin productive collection, 193 | 1.390 ms | 1.440 ms | 96.51% |

All pass the 50% floor. First V13 calls on the two distinct shapes take about
13.21 and 13.55 seconds, excluding imports. The 193 fixture reuses the 163
shape's compiled policy; it is not an independent cold compilation measurement.
This does not qualify uncached startup, other shapes or actual submission
response limits. Candidate game inference is synchronous; untouched external
opponents retain their separate stdio limits.

The complete records contain 8,174 original candidate decisions. Independent
checks verify per-seat keys, nineteen scalar int32 memory fields, original
control histories, mirrored histories and each public capture claim. Candidate
invalid/malformed actions and opponent runtime faults are zero. All 140 original
Amin invalid actions remain in the records; external opponent invalid actions
are zero. All eight external children terminate cleanly and are reaped.

Diagnostic helper schema assumptions were corrected with failed attempts
retained: the independent review initially expected a per-reply opponent invalid
flag, the external trajectory reducer expected an opponent `raw_action` field,
and the secondary V11 comparison initially compared reply timing metadata.
Corrected checks derive public action validity and use original reply text.
No games or policy inference were repeated for those fixes.

`v13-development-evidence.json` embeds results, runtime/tests, independent checks,
diagnoses and artifact identities. Full local histories and helpers remain in
`.cache/runs/sentinel-v13/`; they are not embedded in the repository. The public
policy fixtures and candidate are committed. A passed correctness check, faster
selected win or completed experiment does not satisfy the full competitive goal.
