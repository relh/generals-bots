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

The repaired logical-matrix Product probe `job-24kw9` **failed its live SPS
guard** at epoch 4: 21,436.7 SPS over the interval ending after 2,097,152
completed steps. It produced no trained checkpoint or activation measurement,
and no game-strength result. Its source 512-game gate and native parity passed;
all 13 opponents appeared on both seats, with zero nonfinite or clipped rewards
through the partial run. Verified result artifact SHA-256:
`9e6aa536be345d6433248ddef1291be613ba94c45d2fd906f2bf8be544a91afb`.
The repair uses logical Product U/V/Q matrices and a 128× gain; zero Q matches
the source exactly and 11 migrated policies have byte-identical weights under
both topologies. The live guard fired before the preregistered final-two-epoch
steady interval after six warmup epochs. Its verified audit receipt SHA-256 is
`e8a9126276b5e6f941d6fd18b7b2f5d67ba935d35f8e02a5a4e540cb1da7e2ac`;
it billed $1.0186.

The corrected bounded H100 probe `job-g45tq` at `3a57c87` completed
4,194,304 training steps and achieved **36,117.94 steady end-to-end SPS**:
1,048,576 steps / 29.032 seconds after six warmup epochs. Settings were
4,096 environments/H128/minibatch 8,192/replay 0.5; peak GPU memory was
67.79 GiB. All 13 opponents had balanced assignments on both seats, with
zero illegal actions, nonfinite rewards or clipped rewards. Source and trained
native-to-serving parity passed, including 46/46 matching top actions.

The provider job nevertheless **failed** (exit 1) because its final activation
audit compared raw sampler dictionaries: the source omitted
`full_action_temperature=1.0` and `route_half_weight=0.0`, while the trained
export included those same defaults. No original qualification marker exists.
Audit correction `bd1dacf` compares their effective values. A separate CPU
recovery on the unchanged checkpoint and bundle passed: Product legal-logit
change max 0.01004052, RMS 0.00086386, with all U/V/Q matrices moving.
This recovers the activation evidence without changing the provider outcome.
The terminal artifact SHA-256 is
`ee3b3caec3943ad630421b5e471f4cbc3e86fbe8eb044b986e4513c232b218af`;
terminal and recovery audit hashes are in the manifest. The probe billed
$1.2771. There is no policy-strength result; the selected source and champion remain
unchanged.

The corrected matched Product experiment `job-9sump` **failed the development
strength gate**. Both arms completed 16,777,216 steps on one H100 at
37,170.37/37,873.87 steady SPS (control/Product), with 4,096 environments,
H128, minibatch 8,192 and replay 0.5. Each measurement covered the final
1,048,576 steps after epoch 30. Both included all 13 opponents on both seats
with zero illegal actions or nonfinite rewards. Trained serving parity,
control invariance and Product activation passed: Product U/V/Q all moved,
with legal-logit change max 0.01582146 and RMS 0.00154340.

Over 4,096 paired fresh games (2,616 distinct initial states), control scored
2,801W/1,262L/33D versus Product 2,781W/1,283L/32D. Product-minus-control
signed-score delta was −0.010010, clustered 95% CI [−0.040109,+0.019303].
No broad stratum breached the −0.10 guard, but the overall improvement gate
failed. No independent confirmation or promotion follows; the selected source
and champion remain unchanged. The provider job succeeded on attempt 2 after
one infrastructure start failure, with zero restarts and one billed attempt
($2.728). Exact source, result and independent audit hashes are in the manifest.

Architecture correction: the actual global readout is a dense 32×3,530
matrix, with distinct columns for different source sites. Its default-semantic
edges are excluded from the pinned Fabric edge-sharing rule; DirectSpatial
and serving preserve their native parameter rows. The earlier rationale that
only the Product residual can distinguish source sites was incorrect.
The failed Product comparison remains valid, but does not establish that
more global source discrimination is the next missing capability.

For Slurm jobs, repository `AGENTS.md` requires B200/B300, maximum Nice
`2147483645`, controller readback of Priority 1 and a finite limit, and a new
≥30,000 steady end-to-end SPS gate for each long-training setup. Preserve the
current champion until the full hosted acceptance gate passes.
