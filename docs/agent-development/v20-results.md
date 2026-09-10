# V20 complete development screen: rejected

V20 records four wins and two losses and fails the six-win advancement gate.
All 24 games complete, the 18 controls reproduce their original histories, and
complete integrity and history reviews pass. V20 is not promoted. V2 remains
the default; the competitive objective remains unmet.

The [qualified implementation](v20-development.md) was delivered on main at
`3ab1f71b9b62806f618c2b3293b8eaeda1202b5c` before this screen. The wrapper SHA is
`f68a2cb8650e1d3984ef77f5c50a12dd56c805056382f10f538a0ed7ccca0f46`; the planner
SHA is `789cf890b36cead97d1ef0a2ab61f110e945696b5a90d7ece304891550d428e7`.
The [plan](v20-plan.md) uses consumed seed-83000 placements with construction,
deathtouch 800 and cap 1200. Reserved seed 143000 remains untouched. These
mirrored development cases are not fresh confidence-bounded winning evidence.

## Complete outcomes

Both seat mirrors have identical relative histories. Numbers are terminal turns.

| Opponent / case IDs | V10 | V10-v6 | V19 | V20 |
|---|---|---|---|---|
| Amin 1/2 | Win 1033 | Loss 370 | Loss 431 | **Win 1033** |
| Juraj V3.5 12/15 | Loss 422 | Win 1075 | Loss 575 | **Loss 402** |
| Original my_bot9 4/7 | Win 1076 | Win 296 | Win 888 | **Win 558** |

V20 preserves V10's selected Amin wins and wins against my_bot9, but does not
preserve V10-v6's Juraj wins. A shorter my_bot9 game does not establish a higher
winning probability. V10 and V10-v6 remain distinct stronger controls.

## What the branch mechanism actually did

Across all six V20 games, only six branch actions execute: four against Juraj
and two against my_bot9. Every action is a single owned-cell gathering move.
Every branch releases on the immediately following turn under parent priority.
There are zero continued branch moves, deployments, branch attacks or captures.
The completed screen therefore supplies no adaptive example of completed
multi-branch assembly. The preflight's internal deployment tests remain a
separate, explicitly limited qualification.

Amin never starts a branch. Its complete V20 public observations and actions
match V10, so its wins provide no evidence of benefit from branch collection.

Juraj first diverges at turn 183 on identical public input and identical V10
parent memory. V20 chooses a gathering action with planned delivery 7, required
force 7 and five gathering actions budgeted. The parent resumes at 184. A second
branch starts at 290 and releases at 291. Each mirror loses at turn 402, twenty
turns earlier than V10. The changed history does not isolate a counterfactual
cause of the earlier loss.

The two takeovers differ materially. At 184, another parent packet takes over
while the original recipient waits until 213; that other packet captures the
original objective at 188. At 291, the parent actually uses the second branch
recipient in a serial collection and captures its objective at 296. Both
objectives survive their next land tick and fifty turns, then are lost at
observations 239 and 347, before one hundred turns. These are productive parent
attacks, not completed branch assemblies.

At turn 401, V20 has 268 field armies but its largest plain-cell packet is only
9; home holds 18 against an adjacent threat of 28. Public home-deficit warnings
appear from 397, with no feasible interceptor reported through the final turn.
The parent halves home 15 at 398. These final actions are parent decisions,
long after the last branch action at 290. Fragmented available force and late
infeasible defense remain unresolved.

my_bot9 first diverges at 221 on identical public input and V10 parent memory.
The plan budgets six gathering actions for delivery 4 against required force 4.
Only its first gather executes, followed by parent takeover at 222. The win at
558 follows that changed trajectory; it is not evidence that a branch assault
completed. The parent takeover uses a separate ready 21-army packet; the
branch recipient does not move again until 423. Another parent attack captures
the original objective at 234 and holds it through one hundred turns.

The next untested hypothesis is to integrate an active tree into the collection
phase so that an ordinary new collection does not automatically discard it,
while retaining urgent defense and ready attacks and allowing a sufficient
serial continuation to take over. These productive takeovers are counterexamples
to blindly forcing every tree to finish. No successor policy is selected here.

## Evidence and runtime

The global control barrier independently verifies 12,332 original frames before
any V20 game starts: both public perspectives, keys, native input/output memory,
telemetry, and joint raw/applied actions. All 18 controls match their complete
V19-cycle archives. The full screen contains 16,318 frames, including 3,986 V20
frames; all twelve relative mirror comparisons pass.

A separate pinned-JAX producer recomputes V10 once from every V20 frame's actual
incoming parent memory, original key, observation and rules. It never evolves a
separate parent trajectory and never runs inside gameplay. The independent NumPy
checker verifies each new action, complete native28 transition, branch telemetry
and inherited parent telemetry/state against these **fresh references**, not an
invented archived inner oracle. All 21,168 external public wires independently
reconstruct exactly from the saved states. Original opponent invalid actions
remain recorded and normalized; candidate invalid/malformed actions are zero.

All six V20 warm-call medians lie between 1.08 and 1.99 ms; the largest recorded
warm call is 9.32 ms. No warm call exceeds 150 ms. First calls on newly compiled
V20 game shapes take 6.62–9.00 s; mirrored cases reuse compilation. Amin timings
include recorder serialization and file IO; external timings measure synchronous
source inference. These are not standalone stdio startup/deployment measurements.
The separately qualified nine-context pure-policy profile still supplies the
preflight throughput gate, with minimum enabled ratio 0.5324 relative to V10.

## Preserved checker failure and correction

Both game runners and both fresh-reference producers succeed on their first
attempts, without game or reference reruns. Amin's original postcheck succeeds.
The first external strict check, session 98907, exits 1 after frame checks because
a reference-record variable named `identity` shadows the opponent-identity
function. Its frozen source, failed terminal and log remain unchanged.

An additive replacement renames that variable to `reference_identity`, retains
every original validation, and binds a root-reviewed amendment plus the failed
attempt. Replacement session 98395 exits 0. The original public-wire checker
then exits 0. The complete-history review, session 71283, exits 0 across all
24 games. No validation tolerance, policy or completed trajectory changes.

The [screen evidence manifest](v20-screen-evidence.json) binds the original
releases, recordings, references, checks, amendment, full chronological capture
and retention reviews, and delivery collision census. Raw capture reviews require
next-public ownership confirmation and continuous retention through the next
land tick and fifty/one-hundred turns; terminal or fog censoring is explicit.
