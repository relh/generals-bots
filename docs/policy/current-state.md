# Current policy state

Updated **2026-10-06**. **The winning-policy objective remains unmet.**
Training and serving are qualified; the matched warm and distilled candidates
regressed on held-out games and are rejected. Use the [runbook](runbook.md) for operations, [roadmap](roadmap.md)
for decisions and [machine manifest](../../integrations/policy_baseline.json)
for current artifact identities. Git history preserves superseded attempt records.

## Selected baseline and qualification

Retain the radius-2 baseline from B300 job **35892**, with log-gap scale 0.
Its latest hosted screen won **9/32 against Daveey v7** and **18/32 against the
incumbent**, unchanged from the preceding candidate. Local improvement was
inconclusive; neither these screens nor runtime smoke establish winning strength.

| Identity or evidence | Recorded value |
| --- | --- |
| Policy SHA-256 | `f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14` |
| Optimizer SHA-256 | `1c4832d6f5516ed85a97cf0b476b303ba8829467defeac4b5a7e6d1f6b11599b` |
| Lifetime steps | 2,499,805,184; last baseline continuation added 33,554,432 |
| Hosted policy | `64649097-765f-4706-8310-910e57067a34` |
| Daveey seats | Seat 0: 3W/13L; seat 1: 6W/10L |
| Incumbent seats | Each seat: 9W/7L |
| Hosted execution | 64 unique fresh seeds; no failed requests, timeouts or illegal replay moves |
| Paired local panel | 4,096 games: parent 2699W/1364L/33D; baseline 2713W/1348L/35D |
| Signed-score delta | +0.007324; clustered 95% CI [−0.022176, +0.036830] |

Baseline evidence: `/tmp/relh-generals-portable-result-35892/` and
`/tmp/relh-generals-serving-35892-final/`; durable S3 result key
`relh/generals-classic-results-20261004T004731Z-ee3fda7c`, archive SHA-256
`744e8d92b59a019cf0738acf56a62535a927dbc483fb2d0cec7d167fc9abb8b1`.
It sustained 83,326 audited SPS with 8,192 games, horizon/minibatch 256/8,192,
replay 0.5 and four native opponent workers; all 13 opponents sampled both seats.
A broad post-warmup interval was 25,165,824 steps / 301.407s = 83,494 SPS;
first epoch took 111.555s, sampled GPU utilization 30–100%.

Acceptance requires at least **65% wins against each named strong opponent**,
balanced fresh hosted seats, a 95% confidence lower bound above 50%, preservation
against the broad pool, verified parity and clean execution. No champion change
is justified by the current evidence.

## Held-out decision: retain source, reject defense warmstart

**`job-9fkii` SUCCEEDED**, source `512af1f91ce83d43fab8e0864f204eb064340d01`.
One H100 billed **729s/$0.6006**. All four **4,096-game first-episode Classic
panels completed**, using preregistered seeds 51213/17431 and the same pool/seats.
There are **2,577 distinct initial-state hashes** across 4,096 game entries;
paired 95% intervals resample initial-state clusters, not independent game rows.

| Frozen arm | W/L/D | Paired signed-score delta vs source | Clustered 95% CI |
| --- | --- | --- | --- |
| Source | 2773/1297/26 | Reference | — |
| Control | 2763/1305/28 | −0.00439453 | [−0.03349,+0.02460] |
| Warm | 1757/2156/183 | −0.45776367 | [−0.49303,−0.42337] |
| Distilled | 1682/2246/168 | −0.498046875 | [−0.53374,−0.46251] |

**Reject warm and distilled; control has no demonstrated advantage. Retain the
source baseline.** Tactical retention did not translate into broad improvement.
This is development population evidence, not fresh hosted qualification;
**no hosted promotion occurred**. Collected `EVALUATED.json`, paired reports,
frozen bundles and episode arrays:
`/tmp/generals-heldout-4x4096-512af1f-results-job-9fkii/`.
Archive SHA: `435242250cd9fe416da169e51ad8b2c8ae7f46b21a9d331a91a1717cfd483428`.

## Qualified H100 execution

**Confirmation job `job-s79eq` SUCCEEDED and qualifies 4,096/H128 on H100.**
Source `e649d73d28a5ccbe1995837010db50d8d35e7199`; minibatch 8,192, replay 0.5,
**4,194,304 completed steps**, last two epochs **36,182.7467 end-to-end SPS**,
sampled peak **68,517 MiB** on H100 80GB. Final reward audit covers all
4,194,304 agent steps: **0 nonfinite rewards, 0 clipped rewards, 0 illegal
actions**, with all 13 opponents sampled on both seats. One H100 billed
**608s/$0.5016**. Verified collector:
`/tmp/generals-h100-qualification-4096h128-results-job-s79eq/`.
Archive SHA: `fe574ca31f60d61a825a84c1a4922362b6d378f3e403e453cd88a758cdb8abe5`;
profile result SHA: `c21491a7b94f0190ed023908f7f5ae3cce59b386e41599f63ab4f6030fd81a98`.
The 2.21 MiB policy fits H100 at 4,096/H128. The previous 8,192/H256 setup
retained 172.503 GiB of known buffers; its 205.584 GiB device-used reading includes
allocator caches, not peak live tensors. Do not infer intrinsic model memory
from that geometry. Accounting: `/tmp/generals-native-memory-accounting-20261005.json`.
Keep both matched arms at the same qualified geometry.

## Completed matched training

**Continuation `job-vezu3` SUCCEEDED**, 1,760 billed H100 seconds/**$1.452**.
Control and warm each completed **8,388,608 PPO steps**, 4,096 env/H128,
minibatch 8,192/replay 0.5, at **36,171.513 / 36,265.339 steady SPS** respectively.
Both final audits have **zero illegal actions, nonfinite and clipped rewards**;
policy sampler, seats and 13-opponent pool match. Sampled H100 peaks were
**69,078 MiB control / 68,517 MiB warm**, for this smaller geometry.
Control is preserved from `job-uyssa`; the continuation published it and trained
warm, without repeating control PPO. Native/serving parity passed on 46 public
states, max probability errors **4.38e-6 / 5.60e-6**.

| Frozen result | SHA-256 |
| --- | --- |
| Control checkpoint | `8d5c65e0de7da8074aa353386cb22a2bf40f1a773ff88db47d692b807769c7bf` |
| Warm checkpoint | `c44316f304f833332d0cb2804d3fdf3551aeaa49313217f4263c129f6f409c46` |
| Control bundle manifest | `9dd95fb69258eb1bf666626985f5094a2d5e6cf5f2af1ad4570831220eac88d8` |
| Warm bundle manifest | `4de62a328e7afe5cc5ed622c8d8a7bd8b293afab892bec543b0e38973addf503` |

Distillation used 256 batches×128 labels; on 512 independent held-out maps,
safe-defense action probability mass improved **0.19454→0.63000**, top1 accuracy **0.1914→0.6367**.
These tactical metrics did not translate into broad strength: the completed
held-out panels reject warm and distilled.
Artifacts: `/tmp/generals-matched-continuation-results-job-vezu3/`, archive SHA
`d57aafc8e5dbb34b1b55dac869a0b1d2e904a8cbbb900a5f6248999f72545456`.

## Diagnostics and interpretation

Training-pool diagnostic is adverse: control **7,192/10,896 wins (66.0%)**
versus warm **4,430/10,198 (43.4%)**, worse across all **13 opponents/26 seats**;
the final-quarter gap persists. This is correlated, reused training-pool evidence
and **cannot select a winner**. Post-PPO tactical retention is measured on reused teacher states below;
the completed held-out panel confirms broad regression. SHA-bound diagnostic:
`/tmp/generals-matched-training-pool-comparison-vezu3.json`, SHA
`6cb6ace308a3820cb5b60e9cb911b86792d0a37c0bf4bd83f13b44cec26f9b06`.

Tactical retention diagnostic on ARM64 CPU uses the same **512 preexisting
teacher held-out states**: public-serving safe probability source **19.45%**,
control **19.93%**, distilled **63.00%**, warm **61.17%**, with zero illegal top1
actions. Warm retained most CE gain after 8M PPO. This reuses the CE metric
holdout: **neither fresh confirmation nor full-game strength**. Completed broad
`job-9fkii` panels reject warm despite this tactical retention. Analysis:
`/tmp/generals-vezu3-heldout-defense-serving-cpu-v2/analysis.json`, SHA
`0f2f74222314573464bbb265e4af9715d081f4ebcd4470c8acd58b9f20014cba`.

## Current artifact contract and proofs

Supported native graph: 16 planes, F32/G32, five priors, flat actions, radius
1.01 or 2.01. Factory SHA:
`48767fb4ee333ae0b1a02ae644fbdf6f52f7f6df6c90c97ab3fc3888ba0c0d8a`.
Native assets bind weights/learner, architecture, sampler, seeds and objective;
portable manifests bind `asset.json`, `policy.bin`, `weights.npz` and have no learner.
The current clean radius-2 model SHA is
`cead5dce2bc2f507363854d62bfa9ef9f2a76c7a231afabca6ba658fef6a1c61`, ABI
`0c7a1fb0dfb646b394dd94fbabbad397182bfc3fe62fd739889f2cfd6a8de851`.

| Proof / artifact | Pointer and SHA-256 |
| --- | --- |
| Exact source-cleanup graph/layout/VJP/sampler equivalence, both radii | `/tmp/generals-native-proof/equivalence-proof.json`; `c1125da1b820a46fecb0163e7dd6d5b264709eb3df3569b0bb587a9a722d7a87` |
| AppleDouble metadata rebind: 243 executable sources unchanged, ABI preserved, all 23 tensors bitwise, 18 states max logit 1.19e-6 | `/tmp/generals-appledouble-clean-cold-migration-v1/proof.json`; `94972ea9e22e84cac5b11b2c97c6e8d5a5676f93c13c73c3ff88809fcaed75cd` |
| Clean cold native asset, unchanged baseline policy | `/tmp/generals-appledouble-clean-cold-migration-v1/asset/asset.json`; `58925af1dbeaa46e17230d0ea856739232d4e057b28a14417e3d9ea65903a5b9` |
| Current 13-policy source inputs | `/tmp/generals-current-policy-input-v2/`, with the proven metadata-clean cold asset substituted in the authenticated capsule |
| Fresh public curriculum: 2,048 training /512 independent confirmation views, legal labels | `/tmp/generals-fresh-public-curriculum-770-780-v1/`; `3c2d31a86863bbaa555adbfa1d78742f0fa8120cb56587b1b187b2c8f2923efb` |

Source/metadata cleanup added zero RL steps and preserved original assets.
Proofs include all 299 portable tensors and a 151-array public-environment trace.
Slurm work must retain maximum Nice 2147483645/Priority 1, controller readback and
finite bounds. Prior scheduling receipts and failure outcomes remain in
[Git history](https://github.com/relh/generals-bots/blob/eb54d60/docs/policy/current-state.md)
and the retained result directories, without changing scheduler state.

## Next decision

Retain the source baseline and preserve rejected checkpoints/results. Diagnose
losing held-out replays by opponent and seat before another intervention; the
CE defense objective improved its tactical metric while harming full-game play.
Keep qualified 4,096/H128 H100 execution and serving parity for controlled next
experiments. A future broad winner still requires independent confirmation and
balanced fresh hosted acceptance before promotion.
