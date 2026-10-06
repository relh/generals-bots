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

The selected bundle and native asset are under
`/tmp/generals-current-policy-input-v2/`; their SHA-256 values are in the
manifest. The baseline hosted replay audit is
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

The safe owned half-split sampler pilot is resubmitted as `job-mvdyr` on branch
`relh/safe-owned-half-split-pilot` at `eeda0f8`, with four paired 4,096-game
arms on development seeds 9674001/9674003. The preceding `job-53h6p` failed
before evaluation because its runner expected `input/bundle` but the sealed
context used `input/bundles/penalty8`; its 60-second H100 attempt cost $0.0495.
The new package checks the correct bundle path and manifest hash before games.
No game result exists yet.

A separate public capital-threat gathering diagnostic is submitted as
`job-t7ydi` on `relh/capital-threat-gather-pilot` at `3cf46a0`. It compares
bonuses 0/2/4 on 4,096 paired Classic development games per arm, seeds
9774001/9774003. The trigger uses only visible enemy strength near our general
and legal owned-to-owned moves toward it. Its context archive SHA is
`29209f8603bccf124e0d5f1e8a6222c09d628b399224dfae4fbbf1440a7728f7`.
No result or serving change exists yet.

A third sampler diagnostic, general-garrison split, is prepared but not
submitted on `relh/general-garrison-split-pilot` at `182956f`. It compares
half-move biases 0/2/4 when the public own general holds 5–19 armies, using
4,096 paired development games per arm on seeds 9874001/9874003. It was
motivated by observed full departures from the general in hosted replays; that
panel is diagnostic only and is not used as training data. The isolated branch
contains the frozen and serving sampler path and passed 17 focused tests.

A bounded source-mirror PPO throughput probe is submitted as `job-dzu73` on
`codex/source-mirror-ppo` at `0d6bd40`. It replaces one weak historical frozen
opponent with the exact source actor; other opponents, reward and sampler are
fixed. One H100 will run 4,194,304 steps at 4,096 environments/H128/minibatch
8,192/replay 0.5, requiring at least 30K steady end-to-end SPS and clean
audits. Its $1.485 cap and success marker bound the job. The first submission
`job-jqgpa` was cancelled during image build with **zero charge** after a
JSON tuple/list comparison bug was found in preflight. The corrected package
has archive SHA `ed24d908952e0001c53ef5d376ce914c1b1edf91e3fef2dbcf9f3fd77e1571d3`.
It has no result yet and cannot establish strength without
a matched control and fresh evaluation.

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
