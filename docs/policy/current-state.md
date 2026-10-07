# Current Classic policy state

Updated 2026-10-07. **Winning-policy acceptance has not passed; no champion
changed.** The selected policy remains the radius-2 Classic source from B300
job `35892`. No GPU experiment is active; a launch-path repair is being prepared.

The [machine manifest](../../integrations/policy_baseline.json) records exact
artifact identities, paths, qualification receipts and rejected experiments.
Use the [runbook](runbook.md) for operations and [roadmap](roadmap.md) for
planned decisions. Older prose remains in Git history.

## Selected policy and game

| Item | Selected value |
| --- | --- |
| Checkpoint identity | `f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14` |
| Lifetime agent steps | 2,499,805,184 |
| Hosted policy ID | `64649097-765f-4706-8310-910e57067a34` |
| Frozen sampler | Structured; move/split temperatures 0.05/0.15; opening move temperature 0.10 for 100 turns; neutral bonus 6; weak-owned and doomed-attack penalties 4 |
| Baseline hosted screen | Daveey 9/32; incumbent 18/32 |

The restored, build-compatible native asset and portable bundle live under
`integrations/softmax/local-output/route-shortcut-diagnosis-20261007/` in
`source-asset/` and `source-bundle/`. Recovery and metadata rebind preserved
checkpoint and optimizer bytes, ABI and acting logits; they added no RL steps.
The manifest pins the original identities and separate recovery evidence.

Qualification uses the pinned official Softmax Coworld **Classic** engine,
independently sampled 18–21 tile map dimensions, fog and a 2,000-turn limit.
The generic chasing/smaller-army engine is not equivalent. Engine hashes and
rule-parity evidence are in the manifest.

## Throughput and acceptance gates

The selected source qualified on **one H100 80GB**: 4,096 environments,
horizon 128, minibatch 8,192, replay ratio 0.5; **36,182.7 steady end-to-end
SPS** after warmup in a 4,194,304-step probe. All 13 opponent types appeared
on both seats; illegal actions, nonfinite rewards and clipped rewards were
zero. This qualifies that setup, not every subsequent change.

Every changed long-training setup must independently pass **≥30,000 steady
end-to-end SPS**, measured after compilation and warmup and including rollout,
transfers and optimization. High GPU utilization alone does not qualify it.
Potential shaping must use the learner's exact discount. Mixed-opponent
training must sample every opponent on both seats.

Hosted acceptance requires **at least 65% wins against each named opponent**,
each Wilson 95% lower bound above 50%, broad-pool preservation and clean
execution. The completed penalty-8 hosted panel failed:

| Opponent | Seat 0 | Seat 1 | Total wins | Wilson 95% lower |
| --- | --- | --- | --- | --- |
| Daveey | 50/128 | 48/128 | 98/256 (38.28%) | 32.54% |
| Incumbent | 68/128 | 81/128 | 149/256 (58.20%) | 52.08% |

All 512 identities/outcomes were audited, without failed requests, illegal
moves, candidate timeouts or forfeits. After local files disappeared, separate
Observatory recovery reproduced both counts and audited 277,894 turns across
512 unique seeds. Recovery hashes do not replace original evidence identities.

## Matched experiment: technically passed, strength rejected

`job-wgtyc` succeeded at **2026-10-07 11:34:43 UTC** on attempt 3 after two
provider preemptions, with zero runtime restarts. Independent terminal audit
verified all **237 retained files**, native learner resume, both 512-game
initializer sampling gates, legality/reward audits, and final serving parity
(46/46 states for both arms). No task job remains active. The provider charged
**$5.0534 for 6,126 seconds**, one billed attempt; the earlier two were unbilled.
Its predecessor `job-kzmub` failed before training due to the repaired stage
gate path; its failure evidence remains in the manifest.

Both arms completed **33,554,432 steps** on one H100 with 4,096 environments,
horizon 128, minibatch 8,192 and replay ratio 0.5. After two warmup epochs:

| Arm | 4 Mi qualification SPS | Continuation SPS | Continuation interval |
| --- | ---: | ---: | --- |
| Control | 35,528.49 | 38,261.28 | 28,311,552 steps / 739.953 s |
| Candidate | 37,570.35 | 37,659.31 | 28,311,552 steps / 751.781 s |

These longer intervals retain their hash-bound console receipts. The independent
audit separately checks the final two epochs: qualification 35,600.46/37,552.41
SPS and continuation 38,339.16/37,702.29 SPS (control/candidate). Peak GPU memory
was 66.9 GiB. GPU utilization averaged 21.18%/21.29% across the entire continuation
subprocesses, including startup; these are not steady-interval means. All 13
opponents had balanced training seats. Both continuations had zero illegal
actions, nonfinite or clipped rewards; control had one zero-reward terminal event.

The fresh 4,096-game panels rejected the fixed monotone force potential:

| Actor | Wins | Losses | Draws |
| --- | ---: | ---: | ---: |
| Selected source | 2,786 | 1,275 | 35 |
| Control | 2,727 | 1,335 | 34 |
| Candidate | 2,721 | 1,353 | 22 |

Candidate-minus-source signed-score delta was **−0.03491**, clustered 95% CI
**[−0.06520, −0.00489]**; candidate-minus-control was **−0.00586**, CI
**[−0.03666, +0.02466]**. Neither required improvement gate passed. **No independent
confirmation or promotion follows.** The selected policy and champion remain
unchanged. Exact checkpoints, artifact hashes, audit and billing receipts are
bound in the manifest.

## Rejected experiments: retain these lessons

Signed-score deltas below are candidate minus matched control unless noted.
Exact settings, artifacts and audits are indexed by experiment in the manifest;
these outcomes do not authorize promotion or repeat tuning on hosted losses.

| Experiment | Result and lesson |
| --- | --- |
| Penalty-8 sampler | Local gain did not survive the hosted acceptance panel above. |
| Safe-owned split, capital-threat gathering, garrison split | No reliable paired development gain. |
| Force-assembly reset curriculum (`job-xbqmn`) | Candidate 2,778 wins versus source 2,791/4,096; no demonstrated gain. Frontier-join and early-border ideas are prior work, not unexplored fixes. |
| Source-mirror PPO | Delta −0.00293, 95% CI [−0.03278,+0.02575]. |
| Full-action temperature2 | 33.55M-step candidate 2,715 wins versus original 2,742/4,096; no established gain. |
| Log-gap4 exploration | 8.39M-step candidate 2,633 wins versus original 2,780/4,096; delta −0.07251, CI [−0.10446,−0.04044]. |
| Hard-opponent weights (`job-6cwtq`) | Both arms trained 16,777,216 steps at 37,027.3/34,576.8 SPS; delta −0.027832, CI [−0.059182,+0.003395]; one broad seat stratum regressed. |
| Siege pressure bot (`job-m9ina`) | Learner wins rose 59.40%→99.31%: proposed opponent was much weaker on both seats. |
| Early-border siege bot (`job-wz2x8`) | Learner wins rose 59.40%→75.66%; both seats worsened. Stop this opponent line. |
| Original Product residual (`job-6y2gf`) | Delta +0.005859, CI [−0.022529,+0.034493]; tiny Q activation and sentinel-seat regression. |
| Repaired Product early guard (`job-24kw9`) | Stopped at epoch 4, 21,436.7 SPS before planned warmup ended; no trained checkpoint or strength result. |
| Repaired Product qualification (`job-g45tq`) | 36,117.94 SPS, 67.79 GiB peak; provider failed on omitted-versus-explicit sampler defaults. Separate CPU audit recovered activation on unchanged bytes; provider remains failed. |
| Repaired Product pair (`job-9sump`) | Both arms trained 16,777,216 steps at 37,170.37/37,873.87 SPS; parity/activation passed, but delta −0.010010, CI [−0.040109,+0.019303]. |
| Global route-shortcut removal (`job-wwvk3`) | Source 2,758 wins versus candidate 74/4,096; delta −1.300293, CI [−1.330346,−1.270591]. All 14 broad strata breached the guard. No confirmation. |

The Product comparisons do **not** establish missing global source
awareness: the existing global readout is already a dense **32×3,530 matrix**
with distinct source-site columns. Its default-semantic edges are excluded
from the pinned Fabric sharing rule, and native/direct/serving preserve those
parameter rows. The earlier contrary architecture rationale was incorrect.

## Public diagnosis and cleanup

In the final 50 states of 23 Daveey losses, 57 castle-goal states and 133
visible nearby-threat states had **zero overlap**. Castle priority is not
supported as the immediate cause. An initial reinforcement screen double-counted
evacuated defenders: accounting correction reduced six candidates to four;
strict earlier-arrival/intermediate-safety screening retained one of 133 states.
These conservative screens omit helpful merges, growth and alternate paths.
Rejected candidates are not proof of impossible defense; survivors are not
proof of rescue against an opponent response.

The learned route shortcut contributes roughly 12.73–14.44 logits at the
selected temperature. Removing it improved a few static defense probabilities
while preserving conditional splits, but the full paired ablation above
collapsed. Static attribution is not evidence of improved playing strength.

Earlier-window analysis verified 4,535 public states from the same losses.
Before the first static threat, 3,117 states had visible enemies; 2,596 offered
an immediate friendly merge increasing the largest stack, and selected actions
increased it in 598. Median largest-stack share was 10.48%. Both merge
directions are counted, but position and opportunity cost are not; correlated
known-loss states without winning controls do not prove consolidation helps.

Rejected log-gap runtime code was removed (210 net lines). Selected source
and ten frozen opponents retained bitwise logits/probabilities on 133 public
views and four temperature-boundary variants. Frozen reports require schema
v2; retired metadata fails explicitly. This cleanup claims no strength gain.

For Slurm, `AGENTS.md` requires B200/B300, Nice `2147483645`, controller
readback of Priority 1 and a finite limit. Preserve the champion until the full
hosted acceptance gate passes.
