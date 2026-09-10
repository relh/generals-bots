# V13: collection-only capture arbitration

The competitive goal remains beating every distinct opponent with fresh paired
strength evidence and valid runtime. It is not achieved. V13 tests a structural
ablation of rejected V11, over frozen V10, without V12's rejected route guard.

## Hypothesis and counterexamples

V11 displaced ready direct attacks with chains of small enemy captures. Its first
Amin divergence at 163 diverted an already sufficient packet, and later chains
consumed ready armies while repeatedly deferring their proposed objectives. The
motivating Juraj turn 129 decision instead began eight actions of collection toward an
ordinary enemy tile with one army while V6 could capture another enemy tile with one army immediately. Collection
and direct deployment have different opportunity costs.

V13 applies V11's existing immediate-enemy-capture criterion only to a **new
collection**. It retains ready direct deployments, continuing plans, structure
objectives, defense, remembered-general pursuit and home mobilization. There is
no new ETA threshold, opponent identity switch, or reserve-map tuning. As in
V11, only actual proposed starts surviving outer arbitration are eligible; an
unissued collection clears offensive transport while retaining actual defender
state and observed general knowledge. Actual V6 proposals come from its existing
inner call, without another policy solve. Generic parent and reference actions
are both recorded in the candidate telemetry for decision attribution.

This does not guarantee preservation of productive concentration. In the
unchanged Amin control it would reject twelve collection starts; eleven later
capture their targets and hold them for at least 25 observations. The first at 193
captures at 196 and retains its target for 63 observations. In Juraj, the first two
collections at 100/106 also delay a useful larger packet route relative to V6,
though they capture useful land. V13 leaves those initial starts intact. Their
aggregate tempo cost and the later force deficit may remain unresolved.

The selected experiment can disprove this arbitration rule; a successful local
fixture or narrower trigger is insufficient. Keep the stronger V6 control and
prior complete comparisons. A separate missing factorial experiment is V6 plus
remembered-general pursuit without offensive concentration; do not silently
combine it into this candidate.

## Fixed selected comparison

Freeze the source, helpers, input archives, tests and matched warm profile before
starting games. Compare frozen default V10 with V13 enabled; neither is promoted.
V13 disabled must reproduce V10's exact action, memory and telemetry.

The prototype is available through `generals.agents.sentinel_v13_agent.SentinelV13Agent`.
Construct it with the environment's `build_castles`, `deathtouch_turn` and
`max_turns`, initialize memory for each game, and pass the current public
observation, action key and returned memory to `step`. The disabled control uses
`preserve_collection_captures=False`.

| Opponent | Retained seed / cases | Policies | Games |
|---|---|---|---:|
| Original Amin main8 iter160 | 83000 / 1, 2 | V10, V13 | 4 |
| Original Juraj V3.5 | 83000 / 12, 15 | V10, V13 | 4 |
| Original my_bot9 | 83000 / 4, 7 | V10, V13 | 4 |
| **Total** | Three placements with seat/general-label mirrors | | **12** |

Every game starts at turn zero and ends at capture or the unchanged 1,200-turn
cap. Keep castle construction and deathtouch at 800. Use retained concrete boards
and original action-key initialization. Preserve untouched adaptive opponents;
never substitute recorded future opponent actions into a new candidate game.
All maps are consumed development data. Seed 143000 remains untouched.

Run the two Amin V10 controls before its two V13 games. On the external lane,
run all four V10 controls before its four V13 games. The controls must reproduce
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

Before games, measure V10 and V13 on the same official-version CPU runtime and
the actual Juraj turn 129 and Amin turns 163/193 fixtures. Separate cold
compilation from warmed whole-step timings and require at least half the parent's warm throughput. The candidate's
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
