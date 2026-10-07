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

The full matched hard-opponent weighting run `job-6cwtq` **rejected** the
treatment. Both arms completed 16,777,216 steps on H100 with fresh optimizers
and exact matched settings except frozen `d2c30` and `classic_siege` weights
17/16→34/32. Control and treatment reached 37,027.3 and 34,576.8 steady
end-to-end SPS, all 13 opponents on both seats, and zero illegal, nonfinite or
clipped rewards. On 4,096 paired fresh Classic games, control won 2,759 and
treatment 2,700; treatment-minus-control signed-score delta was −0.027832,
clustered 95% CI [−0.059182,+0.003395]. Its reweighted-opponent delta was
also negative (−0.038055), and one broad seat stratum regressed. Verified
artifact SHA-256:
`87710e728221e5d9fd9b322a777e67d1a41bd15b195147f2204e3573a23a5b73`.
No independent confirmation or hosted panel follows.

The separate `job-m9ina` paired siege-opponent panel **rejected**
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
The narrower `job-wz2x8` early full-army border reinforcement pilot also
**failed**. In 904 paired siege rows (452 per seat), the learner won 537
(59.40%) against the old bot and 684 (75.66%) against the candidate;
candidate-minus-control learner win-rate delta was +16.26 points, paired 95%
CI [+12.54,+20.18]. Both seats worsened. The 1,024 episodes per arm matched
on maps, seats and labels, with zero illegal actions. Verified archive SHA-256:
`d5d340133d640691a66689f52c533c2b5f15538dc56b9fb33ab5221b2d8513b4`.
Stop this siege-opponent line; neither variant qualifies for training.
No training-population or serving change follows. Hosted replays suggest that
general defense and army gathering are promising mechanisms; use fresh paired
development and independent confirmation rather than those hosted losses as a
tuning set.

The isolated rank-8 source-conditioned global residual on
`codex/source-global-product` at `817a258` preserves the public observation
and action ABI. Its zero-head transplant reproduces selected source logits
exactly before training; all ten frozen neural opponents were migrated once
with old/new direct logit parity. Its bounded H100 `job-ffctd` **qualified**:
4,194,304 steps at **37,787.9 steady end-to-end SPS** over the final
1,048,576 steps after six warmup epochs, with 4,096 environments/H128/
minibatch 8,192/replay 0.5, all 13 opponents balanced by seat, and zero
illegal, nonfinite or clipped rewards. The 512-game source/serving gate and
native CPU preflight passed before PPO. The job had zero restarts and billed
$0.7293; verified result archive SHA-256:
`46caeb037c564cdf55f178c4e327d5bc46fd7ea66a57462f6513fd756de4ff23`.
The bounded matched Product ablation `job-6y2gf` **did not clear the paired
Classic development gate**. Both Product-ABI arms completed 16,777,216 PPO
steps from the same source, optimizer seed and opponent pool; the control
masked only the new Q-head gradients. On one H100, control and Product reached
36,544.66 and 36,788.27 steady end-to-end SPS with 4,096 environments/H128/
minibatch 8,192/replay 0.5. On 4,096 paired fresh first episodes (2,588 unique
initial states), control went 2,729W/1,336L/31D and Product went
2,737W/1,320L/39D. Product-minus-control paired signed-score delta was
+0.005859, initial-state-clustered 95% CI [−0.022529,+0.034493]. The
verified terminal artifact SHA-256 is
`678fe6903e6782be772e033f37fc1612a20b80f2615dee51c6fb3f6c82257b49`.
The trained Product Q-head norm was only 1.6655e-5; Product U/V stayed
bitwise identical to the control. Sentinel seat 1 also regressed by 0.12931
signed-score over 116 games, breaching the preregistered broad-stratum guard.
The independent lineage and execution audit passed; the strength decision is
negative. No independent confirmation, hosted evaluation or promotion follows.
The selected source and serving version remain unchanged; exact checkpoint,
comparison and audit hashes are in the manifest.

A repaired Product implementation is now under a **bounded throughput probe**:
H100 `job-24kw9`, isolated `codex/product-logical-migration` revision `0fa8765`,
sealed context `ctx-9bf8ff4e`. The correction treats Product U/V/Q as logical
matrices and applies a 128× gain. Zero Q still reproduces the selected source
exactly; all 11 migrated policy byte sets are identical under both topologies.
The probe must demonstrate ≥30,000 steady end-to-end SPS, all 13 opponents on
both seats, clean reward/action/gradient audits, and a >1e-3 Q effect with
native/serving parity. It has a 60-minute/$2.97 cap and zero restarts. There
is no game-strength result or promotion decision from this repair. The
manifest binds its context archive and input seal.

For Slurm jobs, repository `AGENTS.md` requires B200/B300, maximum Nice
`2147483645`, controller readback of Priority 1 and a finite limit, and a new
≥30,000 steady end-to-end SPS gate for each long-training setup. Preserve the
current champion until the full hosted acceptance gate passes.
