# Independent correctness audit

The September 8, 2026 audit covered the new training and evaluation pipelines,
their interfaces to the existing simulator, and Sentinel's stdin/stdout adapter.
The review used source inspection and focused CPU reproductions. A separate
agent implemented the fixes, followed by an independent review of those fixes.

## Findings and verified fixes

| Finding | Reproduction | Current behavior |
| --- | --- | --- |
| Evaluations could mix changing checkpoints | The CLI hashed a checkpoint once but loaded it again for each opponent and suite; concurrent atomic training saves could change the model between loads. | The CLI copies the checkpoint into the evaluation directory once, hashes that copy, and uses it for every matchup. |
| Malformed commands could execute as moves | With ten armies at `(0,0)`, direction `99`, command kind `9`, or a build command with building disabled could move right through permissive engine indexing. | Arena and stdio match boundaries validate command shape, integer values, kind, source bounds, direction and split flag as applicable. Malformed commands become passes; the arena records a separate malformed-command counter. |
| Terminal draws were missing from training draw totals | A complete mutual-deathtouch rollout ended with `terminated=True`, `truncated=False`, outcome zero, but logged one completion and zero wins, losses or draws. | Outcome counting partitions all finished episodes into wins, losses and draws, with separate timeout and terminal-draw totals. |
| Wire decoding changed visible mountain observations | Encoding and decoding `[[1,-2,0],[0,0,2]]` for player 0 incorrectly changed the visible mountain's neutral-cell flag from false to true. | The decoder excludes mountains from neutral cells. The round-trip test now includes visible mountains for both seats. |

The fixes live in [arena.py](../../generals/evaluation/arena.py),
[evaluation CLI](../../generals/evaluation/cli.py),
[matchup.py](../../competition/matchup.py),
[Sentinel protocol adapter](../../competition/agents/sentinel_python/main.py),
and [training entry point](../../generals/training/train.py).
Boundary validation does not change the raw simulator API's historical behavior.
Physically impossible moves, such as attempts into hidden mountains, remain
distinct from malformed command encodings.

Focused regressions cover malformed commands, protocol round trips, actual
mutual-deathtouch collection, terminal freezing, and castle-build semantics.
The coordinating agent reported 20 targeted arena/strategy/protocol tests and
10 training tests passing after the fixes. The independent replay tests also
passed: stochastic key/seat/state/counter parity, fog observations, stored-board
label swaps, representative sampling, and source-provenance guards.

## Semantics reviewed

The replacement reward derives terminal outcomes from the winner, avoiding the
original trainer's reward based on surviving general tiles. Observable potential
shaping uses the same discount as learning returns and zero potential at task
termination. GAE bootstraps artificial timeouts from the final state before reset,
stops traces at every reset, and suppresses bootstrap for true task endings.
Competition's turn cap is treated as a terminal draw.

Policy and opponent inputs use fog observations and public scoreboard totals.
Learner sampling, opponent sampling, episode-side selection, opponent selection
and optimizer shuffles use separate PRNG splits. Evaluation balances player IDs
and general-label assignments; its confidence intervals resample whole boards,
keeping their paired games together. Checkpoints contain optimizer, random keys,
active games, pool, assignments and configuration; tests compare resumed training
with uninterrupted continuation on the same software and backend.

No additional reproducible defect was found in those reviewed semantics. This is
bounded evidence, not an assertion that the system is bug-free. The full training
contract and architectural limitations are recorded in
[the training audit](../../generals/training/AUDIT.md).

## Replay and provenance

[Replay tooling](../../generals/evaluation/replay.py) loads stored boards rather
than regenerating maps. It reproduces the arena's random-key split order and
records complete states, both players' fog observations, actions, keys, action
counters and Sentinel decision telemetry. JSON and Markdown reports show final
turns, termination, economic changes and observed failure categories.

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu .venv/bin/python \
  -m generals.evaluation.replay RUN_DIRECTORY --max-samples 4
```

Historical Sentinel sources require an explicit `--candidate-source` snapshot.
Critical source changes or outcome/turn/counter mismatches are rejected unless
`--allow-mismatch` explicitly requests labeled diagnostic output. Older runs
without source hashes remain marked as having incomplete historical provenance;
matching outcomes alone cannot establish identical historical software.

Before later changes to boundary validation, four v1 losses and one v1 draw were
reproduced exactly, including every recorded action counter. The observed failures
included two games with no expansion and a pass on every turn. Those traces
supported subsequent strategy experiments. Representative failure samples are
not unbiased estimates of failure frequency, and categories describe observations
rather than proving causes.

Preserve run metadata, source hashes or snapshots, frozen checkpoints, board
archives and per-game CSVs with every strength claim. Resume equality is bounded
to unchanged software and backend. Aggregate training reward, training win rates,
and a successful checkpoint load do not establish held-out competence or a
competition ranking. Shaping benefits require an equal-budget outcome-only
control; strategy changes require evaluation on new boards and seeds.

See [the measured performance audit](performance.md) for reproducible profiling,
trace evidence, compile-versus-steady-state distinctions, and the separate CPU
and GPU acceptance decisions. Throughput evidence is specific to its measured
workload and does not itself demonstrate stronger play.
