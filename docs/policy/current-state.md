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

A bounded source-mirror PPO throughput probe is resubmitted as `job-w43vv` on
`codex/source-mirror-ppo` at `7497168`. It replaces one weak historical frozen
opponent with the exact source actor; other opponents, reward and sampler are
fixed. One H100 will run 4,194,304 steps at 4,096 environments/H128/minibatch
8,192/replay 0.5, requiring at least 30K steady end-to-end SPS and clean
audits. Its 30-minute/$1.485 cap and success marker bound the job. Prior
`job-dzu73` failed before PPO because the probe invoked a sampling gate that
expected an absent distilled bundle ($0.1089). Its first retry `job-63iic`
passed the 512-game source sampling gate but failed native preflight: the
staged original cold asset's model fingerprint `d30f5fae…` differed from the
qualified build's `cead5dce…` ($0.3333; verified output SHA
`a22c9dd025220027824c376ae60cb774224c5d0771ae40797fd182ce20a988e2`).
The new retry uses the already qualified, proof-bound metadata-rebound asset
with exact source, ABI and policy bytes. Its sealed context is `ctx-f0e50843`,
archive SHA `7f2c0aae1efe09dff3e1530f4c4fe16a6162787ed5377cc32b7ae8cd590744dc`.
It has no training result yet and cannot establish strength without a matched
control and fresh evaluation.

A separate hard-opponent weighting throughput probe is resubmitted as H100
`job-ju3rh` on `relh/hard-opponent-weighting` at `b3f12c0`. It starts from the
same qualified source asset and original 13-opponent pool, changing only the
weights of frozen `d2c30` and `classic_siege` from 17/16 to 34/32. The
4,194,304-step probe uses 4,096 environments/H128/minibatch 8,192/replay 0.5,
has a 30-minute/$1.485 cap and zero restarts, and must pass ≥30K steady SPS,
all-opponent both-seat coverage and clean reward/action audits. Its sealed
context is `ctx-9fef3793`, archive SHA
`0c15c8503cc5e9139ffea7e39e37fecc3992af351c2512c788e90f24e7cc359f`.
The first `job-wrj6w` failed during image build because restrictive context
permissions hid its Dockerfile from the rootless builder; it billed $0. The
corrected archive passed tar-header and unprivileged extraction checks.
The proposed 16,777,216-step matched treatment/control run remains held.

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
