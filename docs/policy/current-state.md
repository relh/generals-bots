# Current Classic policy state

Updated 2026-10-07. **Winning-policy acceptance has not passed; no champion
changed.** The selected policy remains the radius-2 Classic source from B300
job `35892`. As of **2026-10-07 16:39 UTC**, row-rotation trial `job-zsz35` is **failed**:
its throughput gate stopped qualification at 29,811 SPS. No GPU job remains
active. Hardware telemetry and CPU-worker profiling are being prepared.
The completed first-contact diagnostic did not pass its improvement gate.

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

## Learning-path repair: coverage verified, throughput failed

At replay ratio 0.5, the pinned native learner selects 32 contiguous minibatches
of 64 environment rows from a 4,096-row rollout. The minibatch index restarts
at zero each epoch and the transpose preserves row order: only rows 0–2,047
receive gradient updates. Both seats and all 13 opponents remain represented,
but the rollout population audit is not an optimizer-sample coverage audit.
The measured end-to-end SPS remains valid as environment throughput.

Correction `d945bae` is integrated after ten focused CPU checks and exact
native patch-chain admission. It rotates the starting minibatch block across epochs,
keeping the gradient budget unchanged. It addresses permanent row starvation;
it still trains on half of each rollout at replay ratio 0.5. Native runtime
coverage, ≥30K SPS and a fresh paired strength comparison are required before
claiming the corrected training setup qualifies. No strength gain is established.

Trial `job-zsz35` was submitted **2026-10-07 16:04:23 UTC** with source
`b068a18` and context `ctx-739342ae`. The candidate repeats the retained
`job-wgtyc` control's 4,194,304-step qualification and 29,360,128-step continuation
with unchanged model, sampler, reward, optimizer settings and training seed.
Only the optimizer row schedule changes learning behavior. The shared audit
also now authenticates the starting learner and checks incremental steps, so
continuation audits correctly expect 29,360,128 newly collected actions.

Fresh 4,096-game panels compare candidate, source and historical control on
seeds 14001101/14001103 (bootstrap 14001111). Both paired improvements must
have positive clustered lower bounds and satisfy the stratum guard before a
fresh independent confirmation. A historical control on another allocation
is not an identical floating-point training trajectory.

The job has one H100, a 75-minute provider limit, 73-minute internal limit and
zero application restarts. A detached guard conservatively caps aggregate
runtime across preemptions at 75 minutes from a pre-allocation observation
(or submission when no such observation exists). Its PID and observed state
are recorded in the launch receipts. The validated maximum runtime quote was
**$3.7125**, excluding any separate build fees. GPU coverage, throughput and
strength are pending; there is no promotion or new qualification claim.

The image build completed and worker download began at 16:20:19 UTC. The external
guard now freezes its bound on `starting` as well as `running`, so startup cannot
escape the aggregate limit. Its conservative deadline is **2026-10-07 17:35:08 UTC**,
based on the last pre-allocation observation at 16:20:08. The guard was replaced
as PID 80704; the provider job was not restarted. A focused starting/preemption
fixture passed, and the new guard process and receipt were verified live.

`job-zsz35` terminated at **16:38:29 UTC** on attempt 1 after the qualification
gate measured **29,811.11 SPS**: 1,048,576 steps / 35.174 seconds between epochs
2 and 4, after two warmup epochs. The matching prior-control interval was
35,487.21 SPS. Native optimizer blocks were **0, 32, 0, 32** for epochs 0–3,
so the coverage correction executed as intended. No long continuation or
strength evaluation ran. Source GPU parity passed 46/46 states.

Environment, rollout-model and optimization times all increased about 19%
relative to the prior control. This does not isolate rotation as the cause.
Clocks, power and CPU capacity were not recorded. Independent terminal collection
verified 35 retained files; no trained checkpoint was saved. The last reward
audit covered 2,097,152 agent steps with no nonfinite/clipped rewards, but the
forced stop left no complete action-mask audit. Billing was **761 seconds,
$0.627**. The guard observed termination and exited.

The next bounded qualification will retain the 30K gate and learning settings,
record hardware capacity and clocks, and measure native CPU-worker choices on
fixed public inputs before selecting a worker count. Exact action/memory
agreement is required; no learning or strength claim comes from that benchmark.

## Matched experiment: technically passed, strength rejected

`job-wgtyc` succeeded at **2026-10-07 11:34:43 UTC** on attempt 3 after two
provider preemptions, with zero runtime restarts. Independent terminal audit
verified all **237 retained files**, native learner resume, both 512-game
initializer sampling gates, legality/reward audits, and final serving parity
(46/46 states for both arms). That job has no remaining allocation. The provider charged
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
**[−0.03666, +0.02466]**. Neither required improvement gate passed. A separate control-minus-source
diagnostic was −0.02905, CI [−0.05969, +0.00293]; the interval includes zero,
so a control decline is not statistically established. **No independent
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

The memory diagnosis covered all **32 retained source games** (23 losses,
9 wins; 13,582 public pre-action states). A previously seen enemy capital was
subsequently hidden in 3 losses/436 states and 6 wins/830 states. This opportunity
is not loss-specific; it justifies neither promoting memory nor dismissing its
potential value. Games and repeated states are not randomized controls.

The force-response diagnostic sampled 242 public states from 22 known-loss
games. Of 206 states offering an immediate friendly transfer that grows the
largest stack, candidate median probability on such actions was **4.28e−6**;
95 were below 1e−6. In 22 states the potential-maximizing transfer did not grow
the largest stack despite an available alternative. Median best scaled own-force
increment over passing was **2.56e−5**. This static own-transfer calculation
omits growth, combat and opponent response; largest-stack growth is not proven
strategic value. It supports neither a causal strength claim nor a coefficient
sweep.

The completed opening comparison covered all 32 games. At turn 50, wins and
losses had the same median army ratio (1.023) and largest-stack fraction (0.136).
At first contact, largest-stack fractions were also similar (0.220/0.226), but
its median distance from the capital was 1/9 and from visible enemies 13/2
(wins/losses). Contact occurred earlier in wins: median turn **71 versus 85**.
This timing difference, small unequal groups and sparse map dimensions prevent
a causal claim that holding a stack back helps.

The preregistered first-contact diagnostic `job-k8yfq` succeeded on its first
attempt at **2026-10-07 12:24:17 UTC**. Source `bb2b02d` is integrated into
this branch. It evaluated 256 unique fresh maps against two frozen opponents
(`d2c30`, `83dc`), balanced by seat, with four action branches and four replicas.
Independent audit verified all 17 retained files, snapshot/map identities,
public observations, paired random seeds and 125 duplicate-action pairs.

| Forced action versus sampled source | Mean signed delta | Map-cluster 95% CI |
| --- | ---: | --- |
| Largest-stack-growing route merge (primary) | −0.03125 | [−0.08105, +0.01758] |
| Best-probability different-source route move | −0.00879 | [−0.05469, +0.03906] |
| Pass | −0.04004 | [−0.09573, +0.01465] |

The primary gate failed despite 201 available merge maps (minimum 128).
None of these intervals establishes a benefit or a harm. This rules out
promoting this fixed first-contact merge heuristic from this experiment; it
does not establish that all merges or longer plans are ineffective. No teacher,
training or promotion follows. Inspect the learning path before another change.
The diagnostic took 200.29 seconds internally; the provider billed 203 seconds
on one H100, **$0.1672**. It performed no training and makes no training SPS
claim. Terminal full-state hash preimages were not retained; duplicate terminal
hashes, outcomes and turns were checked, while snapshot preimages were verified.

An exploratory check of the retained contact results selected an action using
three continuation replicas and scored it on the fourth. Mean signed delta
was −0.04297, map-cluster 95% CI [−0.08398, +0.00195]. This post-hoc calculation
uses simulated future outcomes unavailable to a public-only serving actor;
it establishes no conditional teacher or policy benefit and does not change
the preregistered negative decision. No new games or training were used.


Rejected log-gap runtime code was removed (210 net lines). Selected source
and ten frozen opponents retained bitwise logits/probabilities on 133 public
views and four temperature-boundary variants. Frozen reports require schema
v2; retired metadata fails explicitly. This cleanup claims no strength gain.

For Slurm, `AGENTS.md` requires B200/B300, Nice `2147483645`, controller
readback of Priority 1 and a finite limit. Preserve the champion until the full
hosted acceptance gate passes.

## Local opponent coverage

The local pool contains ten historical Fabric checkpoints and three scripts;
none is bound to the hosted Daveey actor `daveey-grl:v7`. No authorized published
bundle was found in the inspected metadata/SDK; this does not prove none exists.
The incumbent source declaration names `cee053c`, but an immutable deployed
image/source receipt is missing. Its reference Python and local native bot
matched 128/128 sampled incumbent actions from four retained games, with native
memory supplied from reference history. This is bounded action agreement, not
complete recurrent trajectory or image parity. A separate cold-memory check on
128 source-side views found two differences caused by padding.

Selected-source local wins were 548/906 against siege, 428/638 against `83dc`,
and 554/986 against `d2`. Hosted source results remain 18/32 incumbent and 9/32
Daveey. Different maps, counts and actors prevent causal comparisons. This
supports keeping fresh named-opponent hosted acceptance mandatory; it does not
support blaming padding or changing the active trial. Evidence is hash-bound
in the manifest; no new matches or training were performed for this audit.
