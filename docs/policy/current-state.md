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

## Matched experiment: retry running

`job-kzmub` **failed before training** at 2026-10-07 09:12:58 UTC, on
attempt 1 with zero runtime restarts. Both native builds completed. Source
serving parity passed on 46 public states, and the 512-game identical-source
sampling gate completed 267W/240L/5D. The qualification launcher then looked
for `control/qualification/sampling-gate.json`, while the authentic gate was
saved at the shared experiment root. The artifact and all 34 retained file
hashes were independently verified; exact hashes are in the manifest.

`job-wgtyc` completed the control arm's **4,194,304 qualification steps**
as of 2026-10-07 10:10:59 UTC on attempt 3 and started publishing. One H100,
4,096 environments, horizon 128, minibatch 8,192 and replay ratio 0.5 achieved
**35,528.49 steady end-to-end SPS** after two warmup epochs: 3,145,728 steps
between native uptimes 114.749 and 203.290 seconds (88.541 seconds). All 13
opponents had balanced seats; console reports zero illegal actions. The final
dashboard showed 66.9/79 GiB VRAM, 6.1 GiB RAM and 54% GPU utilization (a
snapshot, not an interval mean). Trained control checkpoint `59f07adc` passed
GPU serving parity on 46/46 states (maximum probability difference 0.00000447),
with the source sampler unchanged. Its 4 Mi reward audit reports zero nonfinite
or clipped rewards.

The candidate completed its **4,194,304 qualification steps** by 10:25:22 UTC,
with the same hardware/batch settings and two warmup epochs: **37,570.35 SPS**
from 3,145,728 steps over 83.729 seconds (native uptime 56.218→139.947).
Its dashboard showed 66.6/79 GiB VRAM and 4.4 GiB RAM. Console audits report
zero illegal actions, nonfinite or clipped rewards. Candidate checkpoint
`37e23e20` then passed GPU serving parity on 46/46 states (maximum probability
difference 0.00000244), with the exact intended sampler. **Both runtime
qualification gates passed** by 10:31:47 UTC. Control's trained-initializer
self-match passed (276W/233L/3D in 512 games). By 10:51:40 UTC the control completed
**33,554,432 total steps**, including 29,360,128 continuation steps. Excluding
resumed warmup epochs 9–10, epochs 10–64 measured **38,261.28 SPS**: 28,311,552
steps in 739.953 seconds (uptime 49.007→788.960), with the same H100 and batch
settings; dashboard VRAM was 66.6 GiB. Continuation reports zero illegal
actions, nonfinite or clipped rewards, and one zero-reward terminal event.
Control's final checkpoint `f4b185ef` was exported and passed GPU/native versus
NumPy serving parity on **46/46 states**, with maximum probability difference
0.00000268. By 11:05:45 UTC the candidate's initializer
self-match passed: **273W/233L/6D in 512 games**, across 328 unique initial
states, with the exact `37e23e20` checkpoint and intended sampler for both
actors. Candidate continuation completed **33,554,432 total steps** by 11:18:59 UTC.
Excluding resumed warmup epochs 9–10, epochs 10–64 measured **37,659.31 SPS**:
28,311,552 steps in 751.781 seconds (uptime 50.132→801.913), with the same H100,
4,096 environments, horizon 128, minibatch 8,192 and replay ratio 0.5. All 13
opponents had balanced seats. Final audits cover all 29,360,128 resumed steps:
zero illegal actions, nonfinite or clipped rewards, and zero zero-reward terminal
events. Candidate checkpoint `7fc1d194` was exported and passed final GPU/native versus
NumPy parity on **46/46 states**, with maximum probability difference 0.00000203
and the unchanged sampler. By 11:26:00 UTC, **both arms had completed training
and runtime parity**, and the frozen 4,096-game source/control/candidate
evaluation had started. No strength result or selection is available yet. These are runtime console results. Checkpoint artifacts, learner-byte checks,
training audits and GPU CSV await terminal collection and independent verification. The log monitor now
reads every 64 KiB page with explicit attempt IDs; prior attempt logs were
recovered completely. The terminal collector now preserves sanitized billing
receipts, attempt history and preemption counts for the final audit.
The provider preempted the first two attempts; no replacement job was submitted,
and runtime restarts remain zero.

Repaired source `6b297db`, context `ctx-66b75696`, runs on one H100. The
provider's **150-minute limit and $7.425 quote apply per attempt**; preemptions
do not consume its restart budget. The internal execution deadline is 148
minutes per attempt. Aggregate billing is unverified. Our monitored external
stop deadline is **2026-10-07 11:56:55 UTC** (150 minutes from submission),
not an enforced provider-wide cap; no automatic watchdog is installed yet.
Qualification binds the authentic source gate at each stage; each
continuation requires fresh 512-game self-play of its exact trained initializer.
Runtime qualification permits continuation after each initializer sampling gate;
final independent verification and strength results remain pending. The selected
policy and champion remain unchanged.
Control and candidate start the exact selected policy with fresh optimizers;
each first trains 4,194,304 qualification steps. **Both** must pass ≥30,000
steady SPS, native serving parity and legality/reward/population audits before
either continues its own authentic learner to **33,554,432 total steps**.

Candidate adds a fixed 0.05 potential from turn 100 using own minus opponent
`F = sum(army²) / (10000 + sum(army²))`. Control retains the original reward.
Unlike the rejected normalized-concentration prototype, this potential cannot
increase merely because own troops disappear. It still rewards location-blind
merging and may discourage expansion; no strength benefit is established.

CPU admission used explicitly reconstructed receipts for a retained binary
fixture. It does not qualify a fresh CUDA build or GPU training. The final
4,096-game paired panel must improve over **both source and control**, with
positive clustered lower confidence bounds and signed-score delta at least
−0.10 in each opponent/seat group with at least 100 games. Smaller groups are
reported without automatic rejection. A positive result needs independent confirmation
before hosted qualification. The manifest binds input seals and timing budget.

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
