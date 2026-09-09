# V10: mobilizing the home army under deathtouch

V10 converts both selected my_bot9 losses into wins when run from turn zero: frozen V9 loses at turn 1037; V10 wins at 1076. Each V10 game issues exactly one new mobilization action. These are two mirrored cases of **one already examined development map**, not independent strength samples. V2 remains the default, the competitive goal remains active, and no promotion or training claim follows from this result.

The frozen policy SHA-256 is `7095167e547af5edbf55b528939e4fc7537c954b97ad7370c258ce8090209b2c`. [Machine evidence](v10-evidence.json) records plans, reports, helpers, raw outputs and hashes. Integration validation and bounded warm profiling passed; deployment qualification remains pending. The completed games establish strategy behavior without official response deadlines.

## Change and boundary

`SentinelV10Agent(parent_version=6|9, mobilize_home=True, ...)` wraps either frozen parent. Disabled operation returns the parent's exact action, memory and original telemetry. It preserves the parent's seven or nineteen scalar memory fields.

After deathtouch activates, an inactive prior defender and a newly issued, feasible field interception permit one additional donor: the owned general's army. V10 compares a shortest owned route from home to the **same interception target**, requiring sufficient delivered force, no later arrival than the existing plan or enemy deadline, a strict speed or force improvement, and a nonworsening immediate visible home-coverage estimate. Adjacent threats, guards and existing defensive commitments retain priority. This changes donor eligibility; it does **not** repair the parent's approximate donation accounting or guarantee survival against detours and merging enemies.

When the home move replaces the parent proposal, V10 clears the unissued defender and offensive transport plans, retains stationary general knowledge and strategic timing, and resumes the parent on the next actual observation. `mobilization_parent_action_issued` distinguishes inherited proposal telemetry from executed behavior.

## Evidence sequence

The public-state diagnosis reproduced an existing accounting defect: moving the selected field army at turn 1027 reduces weighted home-route cost from 45 to 34 before any enemy action. The same visible enemy army of 95 consequently requires 61 extra defense instead of the previously estimated 50. Static route-family failures did not prove that adaptive defense was impossible.

An exact control harness first restored both original opponents through all 1,027 prefix replies and reproduced the remaining actions, public wires, candidate keys/memory and terminal states. The actor uses original Python 3.12.10/JAX 0.11.0; the separate engine coordinator uses Python 3.12.11/JAX 0.11.1. Changed continuations supply newly generated observations to the original adaptive opponent, never saved future replies.

The preregistered intervention then tested all six legal full/half moves from home `(14,5)`, holding 52 armies, plus control. Only turn 1027 was overridden; both mirrors completed before interpretation. All 14 continuations, 600 transitions and 614 saved states were retained. Their outcomes are identical under label permutation:

| First action | Outcome in each mirror | Terminal turn |
| --- | --- | --- |
| Control | Loss | 1037 |
| North full | Win | 1076 |
| North half | Loss | 1051 |
| West full | Win | 1078 |
| West half | Win | 1101 |
| East full | Loss | 1062 |
| East half | Win | 1084 |

The complete mechanism audit explains why immediate improvement is insufficient. North full forms a blocking packet, then a separate castle army attacks the next wave's source; a campaign army eventually captures the enemy general. East full stops the first wave but loses its depleted screen to chip attacks and a second wave. North half leaves one visible enemy beside home; it later grows to two and wins through deathtouch. Neither a fixed compass choice nor maximizing immediate route-cost improvement describes the successful behavior generally.

The actual V10 policy subsequently selected the northward home move without a forced action. Its four planned suffixes reproduced the control losses and candidate wins. A separate four-game experiment started both policies from initial memory at turn zero:

| Policy | Cases | W / L / D | Terminal turn | Mobilizations per game |
| --- | --- | --- | --- | --- |
| Frozen V9 | 4 and 7 | 0 / 2 / 0 | 1037 | 0 |
| Actual V10 over V9 | 4 and 7 | 2 / 0 / 0 | 1076 | 1 |

All four full games completed, totaling 4,226 transitions. Candidate and opponent invalid actions and malformed commands were zero throughout the full games, policy suffixes and intervention suffixes. Cleanup reaped the actors with exit code zero. The initial policy-suffix attempt failed on a cache-helper import (`ModuleNotFoundError: memory`); its report and stderr remain bound alongside the successful second attempt. This was an infrastructure failure, not a lost or discarded game.

## Remaining work

Eleven focused policy tests passed, recorded as nine initial tests and two additional tests. Another 105 integration tests passed, totaling 116 distinct tests. The scoped Ruff check passed. All four aliases are available through local evaluation, replay and standalone packaging, including exact disabled parents and complete imported source dependencies.

On the official Python 3.12.10/JAX 0.11.0 runtime, one CPU core and the actual 18×21 public fixture, V10 over V9 retains 91.67% of V9's warmed throughput on mobilization and 94.86% on the initial inactive state. Corresponding V10 median full-step times are 3.537 ms and 1.423 ms; each measurement uses 500 synchronized calls in five rotated-order repeats. The V6-parent variant retains 91.14% and 94.18% of V6 throughput. These repeated snapshots measure inference, not additional games.

Cold V10 compilation alone takes 14.714 seconds with the disk cache disabled, excluding imports and process startup. This exceeds the 10-second first-reply budget. A separately qualified prebuilt cache, all sixteen shapes and a complete active-memory standalone history remain required before deployment. The generated V10 source archive is not deployment-qualified.

A separate 500-call measurement starts with the actual active defender memory at turn 1028. V10 retains 95.71% of V9 throughput (3.232 ms median), and its V6-parent variant retains 96.82% of V6 throughput. The parent releases its commitment on this turn: this tests revalidation and release with existing memory, not successful ongoing transport. All measured comparisons clear the 50% research throughput floor.

The broader gap remains substantial. Complete reconstruction of Juraj V3.5 case 12 shows V6 winning at 1075 while V9 loses at 422. V9 has a development and force-distribution deficit hundreds of turns before the lethal wave becomes visible. Its final loss is ordinary combat before deathtouch, and V8/V9 actions are identical throughout that case. No tested adaptive alternative establishes a rescue there; this narrowly triggered V10 change does not address that earlier failure by assumption.

The original my_bot9 source is unchanged, including its build-price assumption of 10 versus the engine's 14. That broader comparison caveat does not explain these selected losses: neither side emitted an invalid action in these experiments. Pre-cut public wires and intermediate states were reconstructed from retained boards and actions; original stdio inputs were not archived.

A meaningful strength decision still requires preservation and improvement across all six measured opponents, including separate V6- and V9-parent hypotheses. No new six-opponent budget has completed for V10, and reserved seed 143000 remains unused. The selected-map result supports this implementation hypothesis; it does not establish dominance or justify changing the default.
