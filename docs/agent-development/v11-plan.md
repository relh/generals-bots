# V11: immediate enemy-capture arbitration

The competitive goal remains a winning advantage against every distinct opponent.
V10's completed comparison improves late survival but leaves early economic and
field-force failures. This selected development experiment tests one proposed
repair before any broader evaluation or promotion.

## Hypothesis and counterexamples

On the retained Juraj V3.5 game 12, V9/V10 spend eight actions from turn 129
collecting toward an ordinary enemy cell defended by one army. The target is
captured at 136 and held for six observations. During turns 129–135, V6's
same-observation proposal can immediately capture a different ordinary enemy
cell defended by one army. Enemy reinforcement closes that window and consumes
the parked source. This is an opportunity-cost hypothesis; no alternative win
has yet been established. The first offensive plans at 100 and 106 are productive,
and no V6 build proposal exists before the loss at 422. Neither a blanket plan
veto nor a missed-build-window explanation follows from these records.

V11 preserves V6's actual immediate capture of an ordinary enemy tile when V10
would start a multi-action offensive plan for an ordinary enemy tile with no
larger visible defending army. It leaves neutral-capture deferral, structures,
generals and continuing plans outside this rule. Existing defense, pursuit and
mobilization arbitration is respected. Rejected starts clear their unissued
offensive transport while preserving actual defender state and public general
knowledge. V11 observes the original public observation and its own memory only.

The proposal is not a dominance theorem. On the successful Amin placement,
the rule preserves the first productive neutral-deferring plan at 129 but flags
24 of 142 starts, including 22 actual captures and 19 held
for at least 25 observations. Some retention measurements are censored by the
episode end. A public-land-deficit veto also rejects the productive first plan.
These falsifiers require complete adaptive preservation games, not claims based
on selected local actions or fewer transfers.

## Fixed selected comparison

Freeze the source, helpers, input archives, tests and matched warm profile before
starting games. Compare frozen default V10 with V11 enabled; neither is promoted.
V11 disabled must reproduce V10's exact action, memory and telemetry.

The prototype is available through `generals.agents.sentinel_v11_agent.SentinelV11Agent`.
Construct it with the environment's `build_castles`, `deathtouch_turn` and
`max_turns`, initialize memory for each game, and pass the current public
observation, action key and returned memory to `step`. The disabled control uses
`preserve_enemy_captures=False`.

| Opponent | Retained seed / cases | Policies | Games |
|---|---|---|---:|
| Original Amin main8 iter160 | 83000 / 1, 2 | V10, V11 | 4 |
| Original Juraj V3.5 | 83000 / 12, 15 | V10, V11 | 4 |
| Original my_bot9 | 83000 / 4, 7 | V10, V11 | 4 |
| **Total** | Three placements with seat/general-label mirrors | | **12** |

Every game starts at turn zero and ends at capture or the unchanged 1,200-turn
cap. Keep castle construction and deathtouch at 800. Use retained concrete boards
and original action-key initialization. Preserve untouched adaptive opponents;
never substitute recorded future opponent actions into a new candidate game.
All maps are consumed development data. Seed 143000 remains untouched.

Run the two Amin V10 controls before its two V11 games. On the external lane,
run all four V10 controls before its four V11 games. The controls must reproduce
their prior complete raw/applied action histories and terminal outcomes. Retain
all errors and partial outputs; a loss is never a reason to restart. Complete
all 12 games before aggregate interpretation or choosing the next policy change.

## Runtime and records

Amin uses the existing synchronous official-version CPU adapter, with its bound
18-history / 1,528-frame action parity proof. External games use the coordinator's
original engine and a separate synchronous official-version candidate actor.
Original opponents use their unchanged stdio interface and response limits.
Candidate inference is synchronous strategy evidence without an official
deployment deadline. Record runtime versions, CPU allocations, source identities,
original and applied actions, public observations, candidate memory, proposal
versus issued-action telemetry, cleanup and final results.

Before games, measure V10 and V11 on the same official-version CPU runtime and
the actual Juraj/Amin fixtures. Separate cold compilation from warmed whole-step
timings and require at least half the parent's warm throughput. The candidate's
internal action instrumentation must reuse the actual V6 call, avoiding a second
policy solve. No cold-start or standalone-cache qualification is implied by this
selected experiment. Those deployment gates remain required before promotion.

## Decision

Report every complete game, including preserved wins, regressions, draws and
faults. Analyze each changed trajectory to identify actual arbitration and its
later adaptive consequences. The selected hypothesis merits a broader gate only
if it improves the Juraj placement and preserves both Amin and my_bot9 wins,
with valid actions and runtime. Both mirrored cases count as one physical
placement, not independent evidence. Passing this screen does not establish
superiority against these opponents or the unmeasured opponent set.

Retain stronger controls and the useful V10 mobilization work. Deliver validated
owned work to the fork's fresh `origin/main` and leave the checkout on `main`.
The full competitive goal stays active if this candidate loses or is inconclusive.
