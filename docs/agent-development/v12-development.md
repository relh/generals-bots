# V12: route certificate passes, strategy screen fails

V12 fails its [preregistered selected comparison](v12-plan.md). The action guard
preserves its independently checked static route-cost condition, but turns both
Amin wins into losses. Juraj remains a loss at the same turn, and my_bot9 remains
an identical win. V12 is not promoted and does not merit the broader gate. The
competitive goal remains unachieved; V2 and stronger earlier controls are retained.

| Original opponent | Cases | V10 | V12 | Terminal turns, V10 → V12 |
|---|---|---|---|---|
| Amin main8 iter160 | 1, 2 | 2 wins | 2 losses | 1,033 → 635 |
| Juraj V3.5 | 12, 15 | 2 losses | 2 losses | 422 → 422 |
| my_bot9 | 4, 7 | 2 wins | 2 wins | 1,076 → 1,076 |

Twelve games cover three consumed physical placements, with seat/general-label
mirrors and two policies. They are not independent fresh strength samples. All
six V10 controls reproduce their archived complete histories and returns; every
candidate-relative mirror pair agrees. All games start at turn zero on the fixed
seed-83000 boards with castle construction, deathtouch at 800 and a 1,200-turn cap.
All terminate by capture. Reserve 143000 remains ungenerated and uninspected.

## The measured correction and its limit

The [plan](v12-plan.md) records an actual additive-donation error found in V11's
later Juraj trajectory. On its turn-863 public observation, the proposed transfer
opens a cheaper route through the vacated donor. Independent Dijkstra and the
instrumented frozen policy agree on the projected route costs. The later defender
release is consistent with its still-insufficient defense; forcing a permanent
hold is not justified by that diagnosis. Those twenty public-frame policy and
memory checks are separate from the new complete games.

V12 starts from V10, checks each final proposed action against the full current
public cost field, and substitutes a bounded, independently checked campaign
alternative when any modeled visible enemy deficit increases. It clears unissued
transport memory after replacement and preserves publicly remembered general
knowledge. The reference fixture now selects a different owned transfer instead
of worsening its modeled deficit, while real useful-build and release controls
still pass. In the complete V10-based Juraj game, however, the agent dies at 422
and never reaches the motivating V11 turn-863 state.

Only six actual replacements occur across all six V12 games:

| Opponent | Turn | Candidate index / actual probes | Maximum deficit, before / replacement |
|---|---:|---|---|
| Amin, each mirror | 249 | 2 / 1 | 13 / 13 |
| Amin, each mirror | 495 | 0 / 1 | 10 / 10 |
| Juraj, each mirror | 418 | 2 / 1 | 19 / 19 |

There are no replacements against my_bot9. Independent stdlib Dijkstra checks
confirm that **every per-enemy deficit is unchanged**, not merely the maximum.
No replacement is PASS; no search is exhausted. Skipped duplicate, ineligible or
identical-parent candidates explain why candidate index 2 can use only one probe.
No measured game exercises two through nine probes.

The complete records contain 9,328 original candidate decisions, with keys,
actions, public observations, memory and telemetry. Inherited parent events are
proposals whenever `route_guard_parent_action_issued` is false. Generic rejected
parent actions were not archived separately; independent verification covers the
emitted replacement certificates, not a replay of every rejected parent proposal.
The first changed Amin and Juraj decisions can additionally be compared with the
controls because their preceding histories are identical.

## Complete-episode consequences

Amin's first candidate action difference occurs at 249, public difference at 250,
and opponent action difference at 283. At the common-input turn 249, V10 moves a
12-army packet from (8,15) to (8,14); V12 instead captures neutral (11,2) from
(12,2). The original decision is an ordinary campaign move, with no interception,
commitment, offensive-plan or pursuit override. The new guard therefore suppresses
useful campaign behavior more broadly than the motivating false interception
certificate. At 495 it also rejects an unissued proposed defender start and makes
another remote neutral capture. These are whole-policy trajectory changes; neither
individual veto has been isolated as a sufficient cause of the eventual defeat.

In the final Amin wave the guard stays active but accepts actions that leave an
already positive deficit unchanged. An invasion visible with 177 armies at 616
later approaches home with 139 at 634 against 54 at home, and captures at 635.
Keeping the static deficit unchanged does not require an action that averts defeat.

Juraj's only difference is at 418. The guard rejects a half-home sortie that would
increase modeled deficit from 19 to 28, then moves an unrelated remote packet.
Every opponent action remains identical to V10 through capture. In the last
ordinary attack, the opponent sends 38 from its 39-army approach: V10 recalls nine
into home to reach 19; V12 already retains 19 at home. Both lose at 422. My_bot9's
full raw/applied histories, public observations and memory remain identical to V10.

The next bounded diagnostic should distinguish invalid *defensive-plan claims*
from ordinary offensive/campaign spending. Validate the same public-route flaw
only where a proposed interception claims adequate reinforcement, and test whether
that scope preserves the successful Amin campaign decisions. This is a hypothesis,
not evidence that a narrower guard wins or solves the earlier Juraj economic gap.
The stronger V6 controls and useful concentration/pursuit behavior must remain in
subsequent complete comparisons. No extra safety veto alone establishes superiority.

## Correctness, runtime and evidence

Thirteen focused tests pass: twelve initial cases plus a separate mixed batched
case. They include engine-backed winning, losing and tied ordinary attacks;
noncanonical PASS/build handling; exact disabled V10 behavior; actual public
positive/negative controls; fixed memory schema and general-knowledge preservation.
Ruff and diff checks pass. The prototype and plan were committed before games at
`fc7b63c424356a4f5b09c1ddcf69d3fa01cce4b8`; all 43 frozen source bindings remain
unchanged throughout evaluation.

The matched official-version CPU5 benchmark uses five rotated repetitions of
100 synchronized whole-step calls per policy and fixture:

| Fixture | V10 median | V12 median | Parent throughput retained |
|---|---:|---:|---:|
| Weakening transfer | 3.023 ms | 3.691 ms | 81.89% |
| Unchanged-bound build | 2.554 ms | 2.854 ms | 89.49% |
| Productive build | 1.358 ms | 1.598 ms | 84.95% |

All pass the 50% floor. V12 cold compilation takes 16.24–16.79 seconds, excluding
imports; uncached startup is not qualified. These are three fixed CPU fixtures,
not all-shape, GPU, memory-limit or worst-case nine-probe qualification.

Candidate invalid and malformed actions are zero. Original Amin invalid actions
remain retained: 58 in each V10 control and 12 in each V12 game, 140 total, with
no malformed commands. External opponents have zero invalid actions and runtime
faults, and all owned child processes are reaped. Original stdio opponent response
limits remain separate from synchronous candidate inference. The unchanged Amin
adapter retains its 18-history / 1,528-frame parity proof. This comparison does
not qualify V12 for a deadline-enforced submission.

`v12-development-evidence.json` records the results, guard events, runtime/tests,
independent reviews and artifact hashes. Full local histories and helpers remain
under `.cache/runs/sentinel-v12/`; the earlier predicate audit is under
`.cache/runs/sentinel-v11/defender-predicates/`. Those ignored cache archives are
not embedded in the repository; reproducing their hash checks requires the cache.
The policy, plan and embedded public test fixtures are committed.
