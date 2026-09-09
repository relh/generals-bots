# V10: complete four-arm development comparison

**The goal of beating every distinct opponent remains unachieved.** All 576 planned games and the strict strategy, deployment, history and runtime checks completed. Default V10 improves several consumed-map results over V9, but still falls below stronger V6 controls against Hunter and both Juraj versions. V10-over-V6 matches V6's aggregate score against all six opponents and remains below V9 against Amin. Neither candidate is promoted; V2 remains the default.

This is the completed [fixed development plan](v10-plan.md), separate from the earlier [selected-position experiment](v10-development.md). The source and all prerequisites stayed frozen throughout the campaign. No losing game was discarded or restarted. Seed 143000 remains untouched for later final evaluation.

## Results

Each cell is **wins / losses / draws**, followed by score, where a draw counts as half a win. Each opponent has four seat/general-label cases per map: eight maps for Amin/Hunter/Expander and four for each external opponent. Four policies therefore produce 576 games and 144 policy/opponent/map rows. These are 36 opponent-map cases per policy; shared seed streams mean they are not 36 globally independent physical maps.

| Opponent | Games per policy | V6 | V9 | V10 over V6 | Default V10 over V9 |
|---|---:|---|---|---|---|
| Original Amin main8 iter160 | 32 | 12 / 14 / 6 — 46.875% | 22 / 10 / 0 — 68.75% | 12 / 14 / 6 — 46.875% | 22 / 8 / 2 — 71.875% |
| Local Hunter | 32 | 32 / 0 / 0 — 100% | 30 / 2 / 0 — 93.75% | 32 / 0 / 0 — 100% | 30 / 2 / 0 — 93.75% |
| Local JAX Expander | 32 | 32 / 0 / 0 — 100% | 32 / 0 / 0 — 100% | 32 / 0 / 0 — 100% | 32 / 0 / 0 — 100% |
| Original Juraj V3.5 | 16 | 6 / 10 / 0 — 37.5% | 2 / 12 / 2 — 18.75% | 6 / 10 / 0 — 37.5% | 2 / 10 / 4 — 25% |
| Original Juraj V3.4 | 16 | 14 / 2 / 0 — 87.5% | 11 / 3 / 2 — 75% | 14 / 2 / 0 — 87.5% | 13 / 3 / 0 — 81.25% |
| Original my_bot9 | 16 | 16 / 0 / 0 — 100% | 14 / 2 / 0 — 87.5% | 16 / 0 / 0 — 100% | 16 / 0 / 0 — 100% |

Default V10 turns two V9 losses into draws against Amin and two into draws against Juraj V3.5. It turns the two my_bot9 losses into wins. Those gains each involve one map and its mirrored cases. Against Juraj V3.5, its win rate remains 12.5%, with score 25%; draws must not be reported as wins. The unchanged pre-deathtouch policy cannot repair the previously diagnosed early economic collapse simply by mobilizing the home army later.

Primary parent-relative score differences and descriptive 95% intervals are below, in percentage points. They use 100,000 whole-map resamples with seed 19983 reset for every contrast. Mirrors remain within their map cluster. There are only four or eight map clusters per interval; all 24 fixed contrasts, including cross-parent comparisons, are retained in the machine evidence. Intervals are unadjusted for multiple comparisons and do not establish fresh-map superiority.

| Opponent | V10 over V6 minus V6 | Default V10 minus V9 |
|---|---|---|
| Amin | 0 [0, 0] | +3.125 [0, 9.375] |
| Hunter | 0 [0, 0] | 0 [0, 0] |
| JAX Expander | 0 [0, 0] | 0 [0, 0] |
| Juraj V3.5 | 0 [0, 0] | +6.25 [0, 18.75] |
| Juraj V3.4 | 0 [0, 0] | +6.25 [-9.375, 28.125] |
| my_bot9 | 0 [0, 0] | +12.5 [0, 37.5] |

Juraj V3.4 retains its original entropy/clock randomness. Equal maps do not pair its internal randomness, so changed outcomes cannot alone establish a policy effect. V6 versus V10-over-V6 has one loss-to-win and one win-to-loss on board 2, swapped starts. V9 versus default V10 includes a win-to-loss on board 1, swapped starts, seat 1, and a draw-to-loss on board 2, normal starts, seat 1. These cases remain in the result even though aggregate scores rise or stay equal. The prior V6 perfect V3.4 result did not reproduce in this refresh; no deterministic historical parity is claimed for that opponent.

## Complete changed episodes

All 11 changed external primary pairs are retained, including both parent-win regressions. Their 22 complete episodes reconstruct 20,107 recorded engine transitions with matching final winners and times. V10 source replay separately reproduces all 9,455 candidate raw/applied actions with sequential memory. All 22 histories have zero faults or invalid actions. The original stdio traces retained joint actions and final outcomes, not intermediate observations or memory: public observations and internal states in this diagnosis are reconstructed, not independent archived oracles. No opponent was rerun and no new match was played.

The Amin audit replays both policies' complete board-5 mirrored histories: 4,586 exact candidate raw actions and 9,172 checks of both roles' recorded raw/applied commands. All four histories have zero invalid or malformed commands. The 19 memory fields and telemetry are reconstructed sequentially; the synchronous originals do not contain an independent memory oracle. The two mirrors match in candidate-relative public observations and reconstructed state, so they remain one development position.

Candidate actions first diverge at turn 963, before the opponent's first action difference at 991. V10 actually mobilizes at 963, 1017 and 1144. The first visible enemy stack moves away from home; the first donation becomes a campaign force rather than an immediate interception. Later donations form defensive screens. A late 78-army screen absorbs almost all of an 80-army incoming move, allowing the neighboring garrison to stop the remaining force. This supports a real contribution to survival without isolating each mobilization's causal effect. The adaptive opponent follows a different later trajectory.

Both V10 Amin cases reach the 1,200-turn cap while still under pressure. Neither policy ever sees the enemy general. At turn 1199, V10 has 289 armies and 87 land against 1,201 armies and 189 land, with another enemy force approaching home. The observed gain is survival through the configured horizon; unplayed turns provide no evidence of future safety. The initial replay checker's raw-PASS normalization error is preserved separately from game or policy faults, along with the corrected complete audit.

In the two Juraj V3.5 recovered draws, V10 mobilizes at turns 844 and 1042. The policies survive the first attack on different force distributions; V9's later capture at turn 983 must not be attributed to a single immediate collision at 844. V10 never reveals the enemy general and ends with 965 armies / 101 land against 1,779 / 190. Its late defense buys survival through the cap while leaving the economic and offensive gap open.

The refreshed my_bot9 histories reproduce the intended defensive conversion. Both policies share joint actions through turn 1026; V10 mobilizes its 52-army home at 1027 and the adaptive opponent first changes at 1035. The home packet collects into a 61-army screen, absorbs the incoming 46-army move and retains 15. A separate field army later discovers the enemy general at 1045 and captures it at 1075, ending the game at 1076. The mobilized packet protects home; it is not the army making the terminal capture. Both mirrors reproduce this sequence, including zero opponent invalid actions throughout these particular complete histories.

Every changed Juraj V3.4 candidate history has zero mobilizations. The opponent's applied actions diverge first, on turns 4–25, before candidate observations and decisions diverge and long before the turn-800 activation threshold. This includes both parent-win regressions. Those complete histories exclude mobilization as the observed mechanism behind their changed outcomes; all remain in the comparison with the original randomness intact.

The earlier ordinary-combat specificity case also remains unchanged. On Juraj V3.5 game 12, V9 and V10 have identical raw/applied actions for both roles through the complete 422-turn loss; V6 wins its paired case at turn 1075. The next strategy experiment must address the opening economy and field-force decisions behind this gap while preserving the successful Amin plans. The late-mobilization rule alone cannot supply that repair.

## Validation and limits

The completion census verifies all 24 runs, 576 planned cases and frozen inputs. Strict per-surface analyzers verify rules, board identities, candidate options, sources, original opponent identity, action validity and complete records. The combined report independently checks every paired board and estimate. All candidate invalid and malformed action counts are zero. All 192 external games have zero candidate or opponent deadline/fatal-reply faults; original my_bot9 invalid castle attempts remain counted. Amin likewise retains the original checkpoint's invalid actions rather than repairing its policy.

The refreshed V6/V9 controls reproduce all 64 prior Amin histories, including public observations and raw/applied actions, and all 128 prior local recorded case rows. Across the two local primary comparisons, every recorded field except candidate name also matches its parent: outcomes, turns, seeds and action counters. The local runner did not archive original per-turn actions, so this is not a claim of raw-action or internal-memory parity.

Both new variants pass all 16 offline-cache shape probes and complete actual-history deployment checks. The [runtime evidence](v10-runtime-qualification.md) distinguishes the seven-field reconstructed parent6 memory from the 19-field default history proof. Both pass matched CPU and GPU warm throughput floors. GPU active-path throughput is nevertheless lower: scalar V10-over-V6/default retain 66.93%/73.99% of their respective parents, and batch32 retains 76.59%/80.24%. The batch repeats one observation/key/memory; it measures compiled layout throughput, not diverse rollout or training speed. Uncached compilation is not qualified for the first-reply deadline; explicitly built caches are required.

Amin and local games are strategy measurements without response deadlines. The external runner enforces deadlines, but shared-host timing and sampled process memory are not official isolated-container measurements. No official submission or leaderboard rank is claimed. The [expanded coverage census](v10-opponent-coverage.md) still contains unmeasured policies, checkpoints, stress modes and local baselines. Results against stochastic JAX Expander do not transfer to the deterministic stdio ports.

The [machine evidence](v10-development-evidence.json), [576 cases](v10-development-cases.csv) and [144 map scores](v10-development-map-scores.csv) retain the complete planned comparisons and input hashes. Detailed traces remain bound under `.cache/runs/sentinel-v10/`. Completing this experiment does not close the competitive goal.
