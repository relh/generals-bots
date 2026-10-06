# Current policy state

Updated **2026-10-06**. **The winning-policy objective remains unmet.** The
selected source is the radius-2 Classic policy from B300 job `35892`. The
penalty-8 sampler improved local play but failed the full hosted strength gate;
no champion change occurred. [Runbook](runbook.md), [roadmap](roadmap.md), and
[machine manifest](../../integrations/policy_baseline.json) cover operations,
next decisions, and artifact identities. Git history retains earlier attempts.

## Selected policy

| Item | Current value |
| --- | --- |
| Checkpoint SHA-256 | `f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14` |
| Optimizer SHA-256 | `1c4832d6f5516ed85a97cf0b476b303ba8829467defeac4b5a7e6d1f6b11599b` |
| Lifetime steps | 2,499,805,184 |
| Hosted policy ID | `64649097-765f-4706-8310-910e57067a34` |
| Frozen sampler | structured; move/split temperatures 0.05/0.15; neutral bonus 6; weak owned and doomed attack penalties 4 |
| Hosted screen | Daveey 9/32; incumbent 18/32; neither qualifies winning strength |

The current build-compatible native asset and bundle are under
`/tmp/generals-appledouble-clean-cold-migration-v1/`; their SHA-256 values
are in the manifest. Its verified metadata rebind changed the model fingerprint
from `d30f5fae…` to `cead5dce…` with identical policy and learner bytes, the
same ABI, zero added RL steps and 18-state logit parity (proof SHA
`94972ea9e22e84cac5b11b2c97c6e8d5a5676f93c13c73c3ff88809fcaed75cd`).
The qualified H100 build used this exact native asset manifest. The baseline
hosted replay audit is
`/tmp/relh-generals-serving-35892-final/hosted-replay-panel-audit.json`
(SHA `ba1ed86cff62b777f77a8ddd8df5a288165538080675f7456fa21760425cebdb`).

Official Coworld Classic engine SHA is
`f39e448a6b2822869d75cb07cce4cb43d589c4112fef04007ade951809d4a318`;
qualification uses 18–21 tile maps, fog and a 2,000-turn limit. One H100 80GB
qualified 4,096 environments/H128/minibatch 8,192/replay 0.5 at **36,182.7
steady end-to-end SPS** after two warmup epochs over 4,194,304 steps. Peak
memory was 68,517 MiB; illegal actions, nonfinite rewards and clipped rewards
were zero, with all 13 opponents sampled on both seats. Result archive SHA:
`fe574ca31f60d61a825a84c1a4922362b6d378f3e403e453cd88a758cdb8abe5`.
For future Slurm work, use maximum Nice `2147483645`, verify Priority 1 and a
finite limit, and follow the [runbook](runbook.md). Current `AGENTS.md` requires
B200 or B300 for new Slurm jobs and a fresh ≥30K SPS gate before long training.

## Hosted decision

The same-weight penalty-8 sampler passed a 4,096-game independent paired local
confirmation: 2,848 wins versus 2,754 for penalty 4, paired signed-score gain
+0.0439453125, initial-state-cluster 95% CI [+0.018576786,+0.069750967].
Comparison SHA `61429545978242061231894e3b9244e0eff63952a5a9d2c08ed41e4df6d2cb41`;
archive SHA `b26125169a5e88771eb15c40b071a1b80e6d447e3be244d61aad58c5178afb1d`.
This local gain did not establish hosted strength.

Its frozen image digest was
`sha256:a5a4c29e39367b6892dd65dc50775097c14463537a963b0b70e2a1c9146a518e`;
Coworld policy version `e1644605-50ea-495a-9631-a2ff4d25b85f`. The
completed balanced 512-game hosted panel had zero failed requests and cost
$5.7034:

| Opponent | Seat 0 | Seat 1 | Total | Wilson 95% lower |
| --- | --- | --- | --- | --- |
| Daveey | 50/128 | 48/128 | **98/256 (38.28%)** | 32.54% |
| Incumbent | 68/128 | 81/128 | **149/256 (58.20%)** | 52.08% |

The panel at `/tmp/generals-penalty8-hosted-acceptance-chunked-20261006/` passed
its identity and completeness evidence gate but failed the strength gate:
**65% wins against each opponent** and Wilson lower 95% above 50% are required.
All 512 SHA-bound replays had unique seeds, general-capture endings, and zero
candidate timeouts, forfeits, unapplied turns or illegal actions. Panel summary,
promotion report and replay audit SHAs are in the manifest. The selected source
remains unchanged.

## Current experiment and next decision

The safe owned half-split sampler pilot `job-mvdyr` completed four paired
4,096-game Classic arms on development seeds 9674001/9674003. Baseline bias 0
won 2,869 games. Bias 2 won 2,874, paired signed-score gain +0.00317 with
initial-state-cluster 95% CI [−0.00245,+0.00887]. Bias 4 won 2,849
(delta −0.01050, CI [−0.02201,+0.00049]); bias 6 won 2,819
(delta −0.02734, CI [−0.04640,−0.00853]). Retain bias 0: there is no broad
positive result to confirm or promote. Verified result archive SHA:
`1b32f552ac52fbb1ccdf34f0cb5dc1079f4439e318c4ee642024a608b2121015`.
The completed H100 run billed 722 seconds/$0.5951.

The public capital-threat gathering diagnostic `job-t7ydi` completed three
paired 4,096-game Classic arms on seeds 9774001/9774003. Baseline won 2,845;
bonus 2 also won 2,845 (paired signed-score delta +0.00122, clustered 95% CI
[−0.00220,+0.00467]); bonus 4 won 2,837 (delta −0.00317, CI
[−0.00894,+0.00264]). Neither clears the development gate. The decision
retains bonus 0 and makes no serving change. Verified result archive SHA:
`ce68dfc4143e8f6f56046bbcb159c5ad87b17f69e583895a6adf1a62f30e76ca`.
The H100 run billed 535 seconds/$0.4411; the provider retried its initial
image build without charge after losing the first build worker.

The general-garrison split diagnostic `job-nqjyf` completed three paired
4,096-game Classic arms on seeds 9874001/9874003. Baseline won 2,806;
bias 2 won 2,813 (paired signed-score delta +0.00366, clustered 95% CI
[−0.00121,+0.00855]); bias 4 won 2,806 (delta −0.00024, CI
[−0.01066,+0.00985]) with opposite seat effects. Neither clears the
development gate. The frozen sampler and serving version stay unchanged.
Verified result archive SHA:
`efecdd6b4cedfb01b24d81e6443cb858213f2318cd88c03249a713b68dda134c`.
The H100 run billed 523 seconds/$0.4312.

A bounded source-mirror PPO throughput probe `job-w43vv` on
`codex/source-mirror-ppo` at `7497168` **qualified**. It replaced one weak
historical frozen opponent with the exact source actor; other opponents,
reward and sampler were fixed. One H100 completed 4,194,304 steps at 4,096
environments/H128/minibatch 8,192/replay 0.5. After six warmup epochs, the
final 1,048,576 steps took 27.521 seconds: **38,100.9 steady end-to-end SPS**.
Peak sampled device memory was 69,080 MiB. All 13 opponents appeared on both
seats; illegal actions, nonfinite rewards and clipped rewards were zero.
Verified result archive SHA:
`74d3d40f4ec2b3f2f170edf504b017cf96dace29d8a27fd712a37b5212e08722`;
qualification marker SHA:
`089291cef732a3b610a5f19840023e79284fd430eed17a4aaabd010c47154532`.
The new training checkpoint SHA is
`40e7662aac04d676649e3ba51e5e70e7e3de5ac0bdba663cf86456fbd3f0cf94`.
The matched original-population PPO control `job-gjxu3` also **qualified**:
same source checkpoint, clean asset, seed 9107331 and PPO geometry, restoring
frozen slot 0 and its original weight. It completed 4,194,304 steps on H100;
the final 1,048,576 after six warmup epochs took 28.414 seconds, or **36,903.5
steady end-to-end SPS**. All 13 opponents appeared on both seats; illegal
actions and nonfinite/clipped rewards were zero. Its final checkpoint SHA is
`88de88396b2ef2213d39c3f16cafb97bf25bba7cedbc743c795a62a093e5085d`;
verified result archive SHA:
`71963b2f69b1a2b52d555582c6c3142160250d525db237123df63ef3a99b611f`.
Fresh paired Classic development evaluation `job-7bm8b` **rejected the mirror
candidate**. Source, matched control and mirror each played 4,096 first
episodes on exactly paired initial states, seats and opponent labels (2,628
unique states, seeds 10432717/10432719). Source won 2,811; control won 2,792;
mirror won 2,783. Mirror minus control paired signed-score delta was −0.00293
with initial-state-cluster 95% CI [−0.03278,+0.02575]; mirror minus source
was −0.01123, CI [−0.03985,+0.01804]. The full verified result archive SHA is
`35951c69e384b7249c56ab13ff15c594e6aa0d3e5110a4426514737bbba4ea03`;
COMPLETE marker SHA `6afff207db431d6be153344e7c155153d7a7ce66fff830a7946ebf02f8d754d5`.
Published bundle bytes match the authenticated PPO checkpoints. There is no
development evidence for advancement or hosted acceptance. All 26
opponent-by-seat cells were represented, the map-cluster intervals were
independently reproduced from retained arrays, and the H100 run billed 1,070
seconds/$0.8822. Prior
`job-dzu73` failed before PPO because the probe invoked a sampling gate that
expected an absent distilled bundle ($0.1089). Its first retry `job-63iic`
passed the 512-game source sampling gate but failed native preflight: the
staged original cold asset's model fingerprint `d30f5fae…` differed from the
qualified build's `cead5dce…` ($0.3333; verified output SHA
`a22c9dd025220027824c376ae60cb774224c5d0771ae40797fd182ce20a988e2`).
The successful retry used the already qualified, proof-bound metadata-rebound
asset with exact source, ABI and policy bytes. Its sealed context was
`ctx-f0e50843`, archive SHA
`7f2c0aae1efe09dff3e1530f4c4fe16a6162787ed5377cc32b7ae8cd590744dc`.

A separate hard-opponent weighting throughput probe `job-j4fpe` on
`relh/hard-opponent-weighting` at `22e32c4` failed before training. It starts from the
same qualified source asset and original 13-opponent pool, changing only the
weights of frozen `d2c30` and `classic_siege` from 17/16 to 34/32. The
planned 4,194,304-step probe used 4,096 environments/H128/minibatch 8,192/
replay 0.5. GPU idle verification passed with no compute PID, but the
source-sampling subprocess exited on SIGSEGV after native bootstrap and before
any game. The verified terminal archive SHA is
`f9f7a0f538f01422c1ce4715212d9601a07fa08e860a8e31f8723546c61045cd`;
the attempt billed 60 seconds/$0.0495. The successful mirror gate used the
same evaluation code, policy and CUDA image; the hard-weight gate differed in
seeds, two explicitly exported default sampler values and running before the
native build. The evidence does not distinguish a transient native crash from
a seed-specific reset failure.
The first `job-wrj6w` failed during image build because restrictive context
permissions hid its Dockerfile from the rootless builder; it billed $0. The
corrected archive passed tar-header and unprivileged extraction checks.
The next `job-ju3rh` reached H100 but stopped at the GPU idle check before
games/training (60 seconds/$0.0495); its log lacked process identities, so
the exact cause is unproved. The repaired, bounded `job-izcgj` on
`codex/hard-opponent-trial` at `cf98d7d` **qualified** the exact treatment:
one H100 80GB, 4,096 environments/H128/minibatch 8,192/replay 0.5, and
4,194,304 completed steps. Its last 1,048,576 steps after six warmup epochs
took 30.398 seconds: **34,494.9 steady end-to-end SPS**. All 13 opponents
appeared on both seats, with zero illegal actions, nonfinite or clipped rewards,
and nonfinite gradients. Both 512-game source gates completed 256 games per
seat, including the formerly crashing seeds 10441691/10441693. The verified
artifact SHA is `523093a6ca6c9d1335f1f4b1c532d99c47d748f85bae53e6e9f87ed5ac5bd7bf`;
qualification marker SHA `6fd0a28c23fdfe252ae7b86fc60b08df0b5e595bc504d0d63f39d341fda0df44`.
Its context `ctx-5a1213c5` had archive SHA
`8ae62472eede2e1d543cc08b613c9c30382cff094ee54074bb62576910a64289`;
the completed job billed 1,039 seconds/$0.8569. The proposed 16,777,216-step
matched treatment/control run is being staged against this exact marker.

An isolated public-view `classic_siege_padded` opponent upgrade is submitted
as H100 `job-m9ina` on `codex/classic-siege-pressure` at `375d3c2`. It adds
early frontier gathering and legal half captures while keeping the batched
callback and explicit memory ABI. The fixed development gate compares old
versus new on 4,096 paired Classic first episodes per arm (3,640 siege rows,
1,820 per learner seat), using fresh map/sample seeds 10941321/10941323.
Its context is `ctx-4fb34c52`, archive SHA
`599f9f9e677c99f1058d960d05889a44290e06a4b25dda1de97361df15285be8`.
The 60-minute/$2.97 H100 job has zero restarts; no result or training-population
change exists yet.

Hosted replay analysis points to general defense and army gathering: every
recorded loss ended in general capture, often despite an economic lead at turn
200. Against Daveey, the candidate used **241 half moves in 123,787 moves
(0.195%)**, versus Daveey's **8,968 in 119,399 (7.51%)**; candidate losses
ended at mean turn **413** versus **610** for wins. These post-hoc associations
are diagnostic, not a tuning set. Derivation:
`/tmp/generals-penalty8-hosted-acceptance-chunked-20261006/behavior-diagnostic.json`,
SHA `3efec93f7d1c26755b0d9a3191b8084c94b51c8d4830562599e31db58a8215c0`.
The prior warmstart and force-assembly curriculum did not establish broad
improvement. Select another mechanism only from paired development evidence,
confirm it on independent maps, verify train/serve parity, then run fresh
balanced hosted acceptance. Preserve the existing champion until both named
strong opponents meet the winning gate.
