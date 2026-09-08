# Sentinel v3 final evaluation

**Decision: retain Sentinel v2 as the default.** The full v3 policy preserves
strong local baseline performance, but it does not demonstrate an external
improvement. The final external comparison also fails its preregistered
zero-reply-fault requirement. V3 remains an explicitly experimental agent.
Training and all evaluation jobs for this cycle are complete.

The candidate and thresholds were frozen before seed **113000** was released
in commit `76a5e76`. No policy tuning followed these results. Both policies played
64 competition maps against each local opponent and 32 maps against the pinned
Amin checkpoint, with four seat/general-label assignments per map: **1,280 games**
in total. All planned cases completed. Rules include rectangular 18–21 maps,
fog, castle construction, deathtouch at 800, and the 1,200-turn cap.

## Scores and uncertainty

| Opponent | Games per policy | V2 W/L/D | V3 W/L/D | V3 − v2 score, percentage points (95% CI) |
| --- | ---: | ---: | ---: | ---: |
| Expander | 256 | 250/6/0 | 252/4/0 | +0.78 [−1.56, +3.91] |
| Hunter | 256 | 248/8/0 | 250/6/0 | +0.78 [0, +2.34] |
| Pinned Amin checkpoint | 128 | 75/43/10 | 65/54/9 | −8.20 [−20.31, +3.91] |

Score assigns a win 1, draw ½, and loss 0. Intervals use 100,000 paired
map-cluster bootstrap samples, retaining all four cases together (RNG seed 19983).
They are unadjusted intervals for map sampling, not training seeds or leaderboard
rank. Neither local paired interval establishes superiority. The external
interval likewise fails the required strictly positive lower bound.

V3 wins **98.44%** against Expander (95% map interval 96.09–100%) and **97.66%**
against Hunter (94.53–100%). Both local preservation gates pass: at least 85%
wins, win-interval lower bound above 70%, and score regression no worse than
three percentage points. Every local candidate invalid/malformed counter is
zero. Local source hashes stayed unchanged and the retained board arrays match.

Externally, v2 scores **62.50%** and v3 **54.30%**. These are observed results
with runtime faults retained, not an isolated estimate of strategic strength.
The opponent is one publicly available checkpoint with no verified leaderboard
identity. This cycle establishes neither external dominance nor a ladder rank.

## Runtime and complete-episode audit

| Measurement | V2 run | V3 run |
| --- | ---: | ---: |
| Candidate reply frames | 69,079 | 79,124 |
| Candidate deadline faults | 3 | 3 |
| Opponent reply faults | 57 | 127 |
| Candidate invalid actions | 0 | 0 |
| Opponent invalid actions | 1,395 | 1,863 |
| Candidate ordinary-response median | 5.68ms | 6.17ms |
| Candidate ordinary-response p95 | 9.86ms | 10.43ms |
| Candidate ordinary-response maximum, including faults | 952.38ms | 14,783.21ms |
| Candidate maximum first response | 3.53s | 5.28s |
| Candidate peak sampled process-tree RSS | 233,078,784bytes | 260,022,272bytes |
| Mean completed episode length | 539.68 turns | 618.16 turns |

The opponent's totals comprise 13 deadlines plus 44 pending-late-reply skips
against v2, and 16 deadlines plus 111 skips against v3. Invalid game actions
became silent passes under the competition rules. No games were discarded,
and no process or fault-budget forfeits occurred. All 512 child processes were
reaped with exit code zero. One v2-run cleanup advanced past its EOF wait to
the TERM stage; all cleanup reports completed successfully.

These are coordinator-observed timings on a shared host, including protocol and
scheduling effects. They do not isolate policy compute time or establish the
cause of a deadline miss. CPU lanes were 22/23 with engine 0 for v2 and 20/21
with engine 1 for v3; our GPU evaluation coordinator used CPUs 2–7 and 14–19.
Unrelated host workloads remained present. Memory is sampled RSS, with no hard
cgroup or network sandbox. Accepted-only response maxima exclude failures and
must not be used as all-turn runtime bounds.

The separate frozen-v3 qualification passes all **480 synthetic replies across
16 shapes**: maximum first response 1.617s, ordinary response 8.09ms, and sampled
RSS 255,217,664bytes. The new checker independently verified every retained
probe and archive identity. That qualification does not erase game-run faults.
All external processes used Python 3.12.10, JAX/JAXlib 0.11.0, NumPy 2.4.6,
and SciPy 1.18.0.

## Evidence and follow-through

[Compact evidence](v3-final-evidence.json) records exact counts, intervals,
thresholds, runtime distributions, source identities, and hashes of local audit
artifacts. [Per-map results](v3-final-map-scores.csv) retain the 160 complete
map/matchup clusters needed to reproduce the score comparisons. The compact
evidence also records the original bootstrap map order and shared local RNG
sequence: exact Monte Carlo intervals depend on that ordering, even though
the population statistic is invariant to map order. The exported CSV plus
those settings reproduces every reported paired and win interval exactly. Raw boards,
every external reply, manifests, runner snapshots, and process reports remain
under `.cache/runs/sentinel-v3/fresh-local-113000/` and
`fresh-external-113000/`. Each fresh external run completed one original
immutable segment, with no resume or source-equivalence exception.

`scripts/check_v3_promotion.py` returned status 1 and `promote=false` because
of an external reply-fault/deadline confound. Its conservative early exit leaves
later gates unevaluated; the independent complete external audit and local
quality report supply the full diagnostics and preservation-gate results above.
All frozen bundle directory hashes still match their launch records. Independent
review reproduced the external bootstrap and reconciled every raw counter and
cleanup report without finding an evidence inconsistency.

Validation includes the 240-test full suite before the recovery additions,
followed by 41 runner and 43 comparison/promotion tests, with scoped Ruff and
diff checks passing. Recovery regressions cover saved outcomes before cleanup,
bounded child cleanup, process ownership, complete traces, safe resume, and
exact source-equivalence boundaries.

The [development diagnosis](v3-development.md) identifies stale threat memory
and risky castle construction as concrete weaknesses. They remain research
directions; this final sample was not used for another policy revision. Seed
113000 is now consumed. A later iteration needs a new frozen candidate, reserved
maps, and a runtime-controlled comparison before claiming improvement.

The independent [learning campaign](learning.md) also completed: 16,777,216
training transitions, four retained checkpoints, and 2,048 fresh classic games
for the selected final checkpoint. Its 73.9% win rate does not qualify it for
competition deployment. Terminal-only rewards led the controlled shaping pilot;
bounded potential shaping remains available but was not promoted on these data.
