# Current Classic policy state

Updated 2026-10-07, as of the 23:30 UTC corrected gather-probe submission.
**Winning acceptance has not passed; the selected policy and champion are unchanged.**
Training execution is qualified for the previously measured setup. Playing
strength remains the unmet goal.

The [machine manifest](../../integrations/policy_baseline.json) owns exact
artifact identities, retained evidence paths and experiment outcomes. Use the
[roadmap](roadmap.md) for decisions and [runbook](runbook.md) for operations.
Completed experiment narratives and retired source remain in Git history.

## Selected policy and game contract

| Item | Authoritative value |
| --- | --- |
| Selected checkpoint | `f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14` |
| Lifetime agent steps | 2,499,805,184 |
| Hosted policy | `64649097-765f-4706-8310-910e57067a34` |
| Architecture | Radius-2 spatial policy; 578,860 parameters |
| Hosted source screen | Daveey 9/32; incumbent 18/32 |
| Sampler | Structured; move 0.05, split 0.15, opening move 0.10 for 100 turns; neutral bonus 6; weak-owned and doomed-attack penalties 4 |

Use the pinned official Softmax Coworld **Classic** rules: independently sampled
18–21 tile map dimensions, fog and a 2,000-turn limit. The generic engine's
chasing/smaller-army priority is a different ruleset. Engine hashes and rule
parity evidence are pinned in the manifest. Potential shaping uses the learner's
exact discount, currently 0.999; all 13 training opponents must appear on both seats.

Native and serving assets use stateless ABI v2. Migration preserved selected
policy and optimizer bytes. Current assets are under
`integrations/softmax/local-output/stateless-abi-admission-20261007/migrated/`,
with `00-selected-native/` and `01-selected-serving/` as the selected pair.
Historical asset identities are evidence, not supported runtime formats.

## Verified execution and latest strength result

Common measured geometry: **one H100 80GB**, 4,096 environments, horizon 128,
minibatch 8,192, replay ratio 0.5 and eight environment workers. Measurements
exclude two warmup epochs and include rollout, transfers and optimization.

| Run | Evidence and decision |
| --- | --- |
| Stateless qualification `job-tvqh9` | 4,194,304 steps; 41,601 SPS over all six post-warmup epochs; all rolling windows ≥30K; 155 retained files independently verified |
| Continuation `job-mwvdb` | Exact qualified policy/optimizer restored; 33,554,432 cumulative run steps; 28,311,552 measured steps / 667.212 s = 42,432.618 SPS; minimum rolling window 37,330.485 SPS; 244 files verified |

Both passed source/final checkpoint GPU parity (46 fixtures), legal-action and
finite/unclipped-reward audits, and opponent/seat coverage. Sampled GPU memory
peaked at **63.35 GiB** for qualification and **63.89 GiB** for continuation.
Telemetry covers the whole subprocess at five-second cadence; GPU utilization
samples are not aligned to the measured steady interval.

The continuation candidate `28a090d4…` **failed strength selection**. Three
fresh 4,096-game panels produced source/control/candidate wins of
2,733 / 2,759 / 2,758. Candidate-minus-source signed-score CI95 was
[−0.016485, +0.044801]; candidate-minus-control was [−0.028961, +0.032458].
Sentinel seat 1 also regressed beyond the −0.10 guard. Do not extend, confirm
or promote this candidate. The comparison tested the repaired training path;
execution changes and separate allocations prevent attributing it solely to row rotation.

## Fresh-start experiment: strength still untested

`job-r4xkt` changed midgame-reset probability from **0.25 to zero**, holding
other settings fixed. The motivation was the trained policies' frozen-pool gains
and scripted-opponent losses; the retained 384-position pool has median turn
385.5. This supports a distribution hypothesis, not a causal conclusion.

The job completed 4,194,304 qualification steps but **failed the strict rolling
throughput gate** at 22:36:45 UTC: full post-warmup interval 3,145,728 steps /
96.953 seconds = **32,446 SPS**, final two-epoch interval **28,091 SPS**.
All 138 retained files were verified. Checkpoint/optimizer were retained;
no continuation or strength panels ran. Illegal actions were zero.

Its H100 reported constant **1,590 MHz** SM clocks versus **1,980 MHz** on the
successful continuation allocation. Inference, environment and optimization
all slowed; clock differences alone do not establish the cause. Peak sampled
memory was 65,420 MiB (63.89 GiB). No unchanged retry is planned.

`INITIAL_POSITION_MIX` observes startup distribution. The shared reset function
also applies zero midgame probability to automatic episode recycling.

## Live work: corrected gather memory comparison

**`job-tjqy8` failed at 23:22 UTC before training.** Both native builds and
46-fixture source GPU parity passed. The new gradient audit incorrectly wrapped
a Python-checked forward interface inside JAX tracing, causing
`TracerBoolConversionError`. All **224 retained files** were independently
verified; the allocation guard observed termination. No ABBA arm ran.

The repaired audit exercises production forward tapes and backward separately.
Its full CPU check passed on eight authentic replay states: bitwise probabilities,
loss and all 578,860 parameter gradients; an altered observation was rejected.
Production finite checks are unchanged. The fix and downstream runner passed
independent review.

**`job-8tdud`**, submitted **23:30:12 UTC**, is building. Its corrected package
contains 1,155 verified files; the live allocation guard is running. The prior
failed attempt cost $0.4752. The retry maximum quote is **$2.97**.

The proposed storage change replaces duplicate full transposed observations/masks
with float32 minibatch scratch: calculated saving **20.3508 GiB**. Actor storage,
row rotation and optimizer math are preserved. **Completed GPU correctness,
measured memory reduction and speed remain unverified.**

The planned comparison remains **baseline / gather / gather / baseline** on one
H100, each from the exact source with a fresh optimizer: four 2,097,152-step
diagnostics, two warmup epochs, the geometry above and fresh-game resets.
Limits are **60 provider/aggregate minutes, 58 execution minutes, zero restarts**.
Both gather runs must reach 30K SPS before a separate full qualification.
This comparison does not establish the cause of the prior reset-run slowdown.

## Current implementation and next decision

- Stateless execution removes unused external recurrent state. Native startup
  admission binds the actual actor, initializer and compiled build.
- PPO rotates its 32 updated minibatch blocks across all 64 blocks over two epochs,
  fixing permanently skipped environment rows without increasing update budget.
- One output-copy fence preserves DLPack ownership; its local saving is verified,
  but the earlier ABBA did not establish repeatable end-to-end acceleration.
- One explicit-plan trial runner and shared audits replaced completed experiment
  launchers, removing 1,037 net lines. Fixed configs and assets were preserved.
- Gather changes are awaiting GPU qualification. The calculated 20.35 GiB saving
  is not yet an observed memory reduction or evidence of higher SPS.

**Next:** inspect the GPU gate and same-allocation comparison from `job-8tdud`. Adopt
only a demonstrated useful change; otherwise diagnose the measured bottleneck.
The fresh-start runner now requires bound, independently audited probe success,
both gather diagnostics ≥30K SPS, measured memory reduction and matching
execution sources. Its plan remains pending that evidence. A successful
diagnostic must be followed by full 4Mi qualification before resuming the
fresh-start 32Mi strength experiment. Positive development results
then require independent confirmation and fresh balanced hosted matches.

## Winning acceptance

Require **≥65% hosted wins against each Daveey and incumbent**, each Wilson
95% lower bound above 50%, broad-pool preservation and no unexplained seat
regression. Freeze checkpoint and sampler; verify serving parity and clean
execution without unexplained illegal actions, timeouts or forfeits.
Every changed long-training setup independently requires **≥30,000 end-to-end
GPU SPS** after warmup. High utilization alone does not qualify it.

Prior route-prior, exploration, shaping, teacher and opponent-mixture trials
have not established a winning improvement. Consult the manifest before
revisiting them; require new evidence and a preregistered comparison.
