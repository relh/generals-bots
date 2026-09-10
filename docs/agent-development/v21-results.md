# V21 development screen: rejected

V21 won four of six selected games and lost both Juraj V3.5 games. It fails the
six-win advancement gate in [the experiment contract](v21-plan.md). These consumed
positions provide no fresh paired confidence claim. V2 remains the default, and
the broader competitive goal is unachieved.

All 24 adaptive games completed. Before any V21 game, all 18 controls matched the
complete original V20-cycle histories across 12,530 frames. The subsequent audit
covered all 16,516 frames, all 3,986 V21 full outputs, twelve mirror pairs and
19,156 external public wires. Both runners, all eight reference/checker stages,
aggregate validators and lane integrity checks exited successfully. Candidate
source stayed at the [qualified implementation](v21-qualification.md).

## Results

Each entry is the result and game length in both mirrored cases. Rules were
construction enabled, deathtouch800, cap1200, consumed seed83000. Opponents were
Amin main8 iter160 (cases1/2), Juraj V3.5 (12/15), and original my_bot9 (4/7).

| Agent | Amin | Juraj V3.5 | my_bot9 | Total |
|---|---|---|---|---|
| V10 | Win, 1033 | Loss, 422 | Win, 1076 | 4–2 |
| V10-v6 | Loss, 370 | Win, 1075 | Win, 296 | 4–2 |
| V20 | Win, 1033 | Loss, 402 | Win, 558 | 4–2 |
| V21 | Win, 1033 | Loss, 402 | Win, 558 | 4–2 |

V21 matches V20's complete physical histories: all public wires, both players'
raw and applied actions, invalid flags and keys are identical across all 3,986
candidate frames. Successful synthetic persistence and runtime qualification did
not change play on this screen.

## Executed collection

Across both mirrors, V21 issued six initial tree gathering moves: two per Juraj
game, one per my_bot9 game, and none against Amin. It issued no continued tree
action, deployment or attack. The Juraj records include one productive serial
handoff per game; the my_bot9 records include one ready-offense handoff per game.
A continued *proposal* at the serial handoff was not an executed tree move.

Juraj's first tree releases at turn184 because the opponent's actual move captures
the owned deployment bridge: the rally-to-objective owned route becomes
unreachable. The last recipient is still owned and sufficiently stocked; ordinary
priority did not suppress a feasible continuation. The second tree hands off
productively at turn291. The my_bot9 tree yields to ready offense at turn222.

The my_bot9 histories have one harmless representation difference after turn503:
changed pursuit resets V21's inactive nested tree timestamp to−1, whereas V20's
outer empty tree retains the current turn. Both remain inactive; this changes no
physical action. Other projected native fields match.

These counts use the collector, V9 and V10 final issuance masks. Ordinary offense
requires its own collector-parent mask as well. Full returned nested memory and
telemetry were checked independently, so inner proposals and actual actions remain
separate. Public ownership after a later capture does not identify the original
force or prove a causal benefit from collection.

The complete Juraj failure remains the same as V20's. At the final observation,
home holds18 against an adjacent29-army enemy with28 movable, while268 friendly field armies are
dispersed into81 movable packets; the largest ordinary packet is9. The opponent's
scoreboard army is439. No enemy general was observed in the episode. These are
public-state facts, not proof that a late single-action defense could save home.

## Evidence and limits

The [screen evidence index](v21-screen-evidence.json) binds original full games,
policy calls, public observations, keys, raw and applied actions, invalid attempts,
process receipts and complete episode reviews. Fresh frozen-V8 and injected
frozen-V9/V10 references used each original call input; no second V21 call or
counterfactual opponent replay was used to construct expected outputs.

A release-helper filename mismatch was handled additively: the final audit gate
reads the actual external `public-wire-postcheck.json` and
`strict-postcheck-terminal.json`. The frozen original helper and actual reports
were preserved. Amin's exporter completed directly in tool chunk `0a6317`; its
receipt explicitly records that identity and the absence of a persistent tool
session, rather than claiming an unobserved process handle.

The per-context warm runtime gate remains passed, but this synchronous development
screen does not qualify standalone startup/deadline behavior, classic rules or FFA.
Reserved seed143000 remains unused. Broader opponent coverage and fresh paired
winning evidence are still required before any dominance claim.
