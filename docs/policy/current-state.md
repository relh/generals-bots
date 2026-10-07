# Current Classic policy state

Updated 2026-10-07. **Winning-policy acceptance has not passed; no champion
changed.** The selected policy remains the radius-2 Classic source from B300
job `35892`. As of **2026-10-07 17:59 UTC**, bounded bridge diagnostic
**`job-zwt6d` is building**, retrying the diagnostic after `job-6khn8` failed before training. The preceding row-rotation qualifications failed the
strict throughput gate; no long continuation or strength evaluation ran.

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

The candidate repeats the retained `job-wgtyc` control's 4,194,304-step
qualification and 29,360,128-step continuation with unchanged model, sampler,
reward, optimizer settings and training seed. Only the optimizer row schedule
changes learning behavior. The shared audit authenticates the starting learner
and checks incremental continuation steps.

Fresh 4,096-game panels compare candidate, source and historical control on
seeds 14001101/14001103 (bootstrap 14001111). Both paired improvements must
have positive clustered lower bounds and satisfy the stratum guard before a
fresh independent confirmation. A historical control on another allocation
is not an identical floating-point training trajectory.

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

### Profiled retry: throughput failed

`job-5cirw` was submitted **2026-10-07 16:59:07 UTC**, source `ba8461e`,
context `ctx-31be43a7`. Preparation and the bounded worker benchmark completed
in 26.3 seconds. Source GPU parity passed 46/46 top actions with maximum
action-probability difference 0.00000445; native trainer compilation succeeded. It retained the strict 30K gate and learning settings,
records CPU quota/affinity and GPU clocks/power, and selects native opponent
workers from 1/2/4/8 on fixed public replay inputs. Exact action/memory agreement
is required. The selected count applies to candidate training and all evaluation
actors; the CPU benchmark makes no training-throughput or strength claim.

One H100 runs 4,096 environments, horizon 128, minibatch 8,192, replay ratio 0.5.
The provider/internal/aggregate limits are 75/73/75 minutes, with zero application
restarts. The validated runtime quote is **$3.7125**, excluding separate build
fees. Detached guard **PID 84329** observed termination and exited. Its aggregate bound froze
on `starting` or `running` from the last pre-allocation observation and never
advances after preemption. The bound is frozen at **2026-10-07 18:22:54 UTC**
from the 17:07:54 pre-allocation observation. Source, admission, archive and request hashes are in
the manifest and launch receipts. GPU throughput qualification failed; strength remains untested.

The retry terminated at **17:23:19 UTC**, attempt 1, zero restarts/preemptions.
It selected **8 workers** from 1/2/4/8 on a **28-CPU quota**. The fixed fixture
median improved from 28.42 ms at 4 workers to 18.90 ms at 8, but training reached
only **28,788.05 SPS** (1,048,576 steps / 36.424 seconds, epochs 2–4 after two
warmup epochs). The last individual epoch reached 31.6K; it does not override
the failed prescribed interval. No continuation or strength panel ran.
Independent terminal collection verified **40 files**, including CPU/GPU telemetry
and worker-profile inputs. Hardware was **H100 80GB HBM3**, peak console VRAM
**66.9 GiB**, host RAM **7.1 GiB**. Billing was **776 seconds, $0.6402**.
That qualification job has no remaining allocation. Relative to the prior run, the measured
environment portion improved 2.946 seconds, while rollout inference and training
slowed 3.086 and 1.111 seconds. Same physical GPU, sampled clocks fixed at
1,980/2,619 MHz, no sampled clock events and no new cgroup throttling over the
retained 61-second fully allocated interval. These observations do not isolate
a cause; they provide no support for changing power limits or CPU quota.

### Bridge execution diagnostic retry: building

**`job-zwt6d`** was submitted at **2026-10-07 18:32:06 UTC**, context
`ctx-f76d9c6f`. At **18:33:06 UTC** its image was building, with no runtime
attempt yet. Same one-H100 A–B–B–A plan, source revisions and 1,063 unchanged
input files; only the diagnostic helper, report-writing template and input
manifest changed. Strict report-file parsing and retained compiler/runtime
stdout and stderr replace parsing embedded-Python stdout as JSON.

Corrected local admission activated the production bootstrap and Muon hooks;
baseline generated sources match the prior GPU build exactly. Both CPU
ownership probes passed. Actual CUDA lifetime checks remain mandatory before
training. Root verified all 1,066 staged hashes and the archive. The quote is
**$2.2275**, with **45/43/45-minute provider/internal/aggregate limits**, zero
restarts and live detached guard **PID 52951**. No state-free runtime changes
are included in this diagnostic, and its results cannot qualify sustained
training or policy strength.

### Bridge execution diagnostic: report-protocol failure

`job-6khn8` was submitted **2026-10-07 17:58:15 UTC**, context `ctx-12ac3324`,
for one H100. The image and baseline native executable built successfully. Attempt 1
failed at **18:22:38 UTC**, before training: the compiled CUDA lifetime
executable exited zero, but its wrapper could not parse stdout as JSON. Raw
stdout was not retained, so lifetime success and the precise output remain
unproven. All **118 retained files** were verified; billing was **124 seconds,
$0.1023**, with no restarts or preemptions. A corrected diagnostic will write
a dedicated report file and retain compile/run stdout and stderr. It compares the original bridge with four output-copy waits
consolidated into one, in **A–B–B–A order**. Each fresh run has 2,097,152 steps,
two warmup epochs and two measured epochs; workers stay fixed at eight.
All DLPack owners remain alive through the final output wait; input readiness
remains synchronized. Host timings and CUDA stream spans overlap and must not
be summed; end-to-end speed uses completed-step wall time.

The first sealed archive passed CPU source transformation and ownership checks,
but later review found those checks omitted active runtime Muon hooks. The retry
will activate the production bootstrap and compare generated sources against
the retained GPU build. Review rejected an earlier
unsubmitted archive because its instrumentation expected the wrong build-stage
loop. Actual CUDA compilation and lifetime probes must pass before training.
The production correction is `d1e9705`, runner `0bd5e1e`, baseline `0df0d24`;
exact inputs and receipts are bound in the manifest.

Limits are **45 minutes provider/aggregate**, 43 minutes internal, 480 seconds
per run, zero application restarts. The validated runtime quote is **$2.2275**.
Detached guard **PID 77874** observed terminal failure at **18:22:50 UTC**
and exited; its unused aggregate deadline was **19:03:05 UTC**. These instrumented runs are
bounded diagnostics; no continuation or strength evaluation follows, and full
uninstrumented **≥30,000 SPS** qualification remains required.

A separate active-path audit found that the direct actor ignores recurrent carry
but copies its unchanged 77,684-word state into JAX and back during rollout:
**1.185 GiB per buffer, 303.453 GiB of explicit copies per 128-step epoch** at
4,096 environments. These are code-derived byte counts, not measured speedups.
The zero-external-state implementation is integrated as **`ccf1895`**, with
explicit ABI v2 and no compatibility path. It removes carry allocations, copies,
resets and snapshots while preserving compiler scratch. All 12 policies match
bitwise on fixed CPU rollout outputs, sampled actions and cotangent gradients;
both layouts also match on multi-step training inputs with mixed terminals.
All 14 asset manifests and 12 serving bundles were migrated, preserving policy,
portable weight and both optimizer snapshots byte-for-byte. Root independently
verified the comparisons and migrated bytes. The selected asset references now
point to the migrated assets; historical experiment inputs remain hash-bound.
**CUDA execution and throughput qualification are still pending.** This does
not change the sealed bridge comparison or establish any policy strength gain.
The next qualification is preregistered in
`integrations/stateless_qualification/plan.json`: 4,194,304 uninstrumented steps,
workers eight, two warmup epochs, then all six remaining epochs measured
together, with the existing rolling 30K SPS guard retained. Both source and
checkpoint parity, complete action/reward audits and all 13 opponents in both
seats are required. Coordinator **`1181b57`** reuses the shared execution path;
local preparation against actual migrated assets passed, and its audit rejected
slow/incomplete runs, wrong checkpoint clocks or initial weights, and an
unexpected restored optimizer. It remains unsealed pending the bridge comparison.

Main and the sealed source branch are pushed to the fork; transient GitHub
server errors cleared on retry.

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
