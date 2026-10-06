# Current Classic policy state

Updated 2026-10-06. **Winning-policy acceptance has not passed; no champion was
changed.** The selected policy is still the radius-2 Classic source from B300
job `35892`. This page records the decision; the
[machine manifest](../../integrations/policy_baseline.json) holds exact artifact
paths and hashes, the [runbook](runbook.md) holds operational steps, and the
[roadmap](roadmap.md) holds the next decisions.

## Selected policy and game

| Item | Selected value |
| --- | --- |
| Checkpoint SHA-256 | `f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14` |
| Optimizer SHA-256 | `1c4832d6f5516ed85a97cf0b476b303ba8829467defeac4b5a7e6d1f6b11599b` |
| Lifetime agent steps | 2,499,805,184 |
| Hosted policy ID | `64649097-765f-4706-8310-910e57067a34` |
| Frozen sampler | Structured; move/split temperatures 0.05/0.15; neutral bonus 6; weak-owned and doomed-attack penalties 4 |
| Baseline hosted screen | Daveey 9/32; incumbent 18/32 |

The selected build-compatible native asset and portable bundle are in
`/tmp/generals-appledouble-clean-cold-migration-v1/`. A verified metadata
rebind preserved the policy and learner bytes, ABI and acting logits; it added
zero RL steps. The manifest binds both assets and the rebind proof. The official
Softmax Coworld Classic engine SHA-256 is
`f39e448a6b2822869d75cb07cce4cb43d589c4112fef04007ade951809d4a318`;
qualification uses independently sampled 18–21 tile maps, fog and a 2,000-turn
limit.

## Qualification and hosted decision

The selected source qualified on one H100 80GB with 4,096 environments, horizon
128, minibatch 8,192 and replay 0.5: **36,182.7 steady end-to-end SPS** after
warmup over 4,194,304 completed steps. All 13 opponent types were sampled in
both seats; illegal actions, nonfinite rewards and clipped rewards were zero.
The result archive hash is in the manifest. Any changed long-training setup
needs its own ≥30,000 SPS qualification.

A same-weight penalty-8 sampler passed local paired development but failed its
completed balanced 512-game hosted panel:

| Opponent | Seat 0 | Seat 1 | Total | Wilson 95% lower |
| --- | --- | --- | --- | --- |
| Daveey | 50/128 | 48/128 | **98/256 (38.28%)** | 32.54% |
| Incumbent | 68/128 | 81/128 | **149/256 (58.20%)** | 52.08% |

All 512 replay identities and outcomes were audited; the panel had no failed
requests, illegal actions, candidate timeouts or forfeits. Hosted acceptance
requires **at least 65% wins against each** named opponent, each Wilson 95%
lower bound above 50%, broad-pool preservation and clean execution. The
penalty-8 candidate was rejected and the selected source remains unchanged.
The panel path and summary, promotion-report and replay-audit hashes are in the
manifest.

## Development status

The safe-owned half-split, public capital-threat gathering and general-garrison
split sampler pilots had no reliable paired development gain. The source-mirror
PPO treatment and exact matched control each qualified on H100 (38,100.9 and
36,903.5 steady SPS), but the fresh paired 4,096-game Classic panel rejected
the treatment: mirror minus control signed-score delta −0.00293, clustered 95%
CI [−0.03278,+0.02575]. None changed the selected policy.

The hard-opponent weighting probe `job-izcgj` qualified the exact treatment on
H100: 4,096 environments/H128/minibatch 8,192/replay 0.5, 4,194,304 steps,
**34,494.9 steady SPS** over the final 1,048,576 steps after six warmup epochs.
All 13 opponents appeared on both seats; illegal actions and nonfinite or
clipped rewards were zero. Its qualification marker SHA-256 is
`6fd0a28c23fdfe252ae7b86fc60b08df0b5e595bc504d0d63f39d341fda0df44`.
This is a throughput result, not a strength result.

Two bounded H100 development jobs are active; the manifest records their
sealed context hashes and last observed states:

| Job | Isolated branch | Decision pending |
| --- | --- | --- |
| `job-6cwtq` | `codex/hard-opponent-trial` at `542de7f` | Matched 16,777,216-step source/control PPO arms, changing only the weights of frozen `d2c30` and `classic_siege` from 17/16 to 34/32. Assess fresh paired Classic strength before any hosted panel. |
| `job-wz2x8` | `codex/classic-siege-early-border` at `5fed4e2` | Old siege bot versus the same bot with its turn-800 guard removed from full-army border reinforcement, on 1,024 paired fresh Classic first episodes per arm (904 siege rows, 452 per learner seat). No split-rule change. |

Both jobs have zero restarts. The matched run has a 60-minute/$2.97 cap; the
siege pilot has a 30-minute/$1.485 cap. Treat statuses as historical
observations until provider readback and artifact
verification. The separate `job-m9ina` paired siege-opponent panel **rejected**
the proposed pressure bot. In 3,640 paired siege games, the frozen learner won
2,162 (59.40%) against the old bot and 3,615 (99.31%) against the new bot;
the new bot was much weaker on both learner seats. Candidate-minus-control
learner win-rate delta was +39.92 points, clustered 95% CI [+38.09,+41.72].
All 4,096 episodes per arm were completed, paired by initial state, seat and
opponent label, with zero illegal moves. The verified archive SHA-256 is
`813be34c1185a42545b21e6d1187bd80c20737c5a28623f898fd016dedd5df36`.
The new bot made far more half moves, dispersed stacks into neutral cells and
rarely reached its remembered-general target; the aggregate trace cannot
isolate one cause of the loss.
No training-population or serving change follows. Hosted replays suggest that
general defense and army gathering are promising mechanisms; use fresh paired
development and independent confirmation rather than those hosted losses as a
tuning set.

For Slurm jobs, repository `AGENTS.md` requires B200/B300, maximum Nice
`2147483645`, controller readback of Priority 1 and a finite limit, and a new
≥30,000 steady end-to-end SPS gate for each long-training setup. Preserve the
current champion until the full hosted acceptance gate passes.
