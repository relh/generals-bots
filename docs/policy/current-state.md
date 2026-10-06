# Current policy state

Updated **2026-10-06**. The winning-policy objective remains unmet.
Use the [runbook](runbook.md) for supported operations and [roadmap](roadmap.md)
for remaining work. Source history and full experiment artifacts preserve details.

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

## H100 sizing outcome and revised probe

**First sizing job `job-i8tp6` FAILED**, submitted 2026-10-06 01:25:53 UTC via
Richard-authenticated Givemeanode: one H100, source
`8e41477ee1224127ca2ce5cb684c42acf39c25d5`, context `ctx-ed1b264a`.
Its same-sampler source gate passed **254W/253L/5D in 512 games**, balanced
256 games per seat. Both 2,048/H256 and 4,096/H128 native builds passed, but
**both 120s CPU preflights timed out before any training epoch**. No training
SPS or training memory measurement was obtained; neither geometry is qualified.
Bounds were GPU duration 30 min, build timeout 60 min, queue TTL 60 min,
zero restarts and resume `none`; native CUDA/current assets, no QEMU.
Context archive SHA-256:
`820dc8d81a0213aa780fa97aff6e7838085be05bf97d6bf928f7dbc8c3ef2892`.
Terminal evidence: `/tmp/generals-h100-probe-results-job-i8tp6/`, including
`extracted/generals/probe-results.json` and each profile's preflight log;
submission/package receipts remain `/tmp/generals-h100-probe-owned-job.json`
and `/tmp/generals-h100-probe-package.json`.

**Revised probe `job-kcten` FAILED** on one H100: **651 billed seconds,
$0.5368 charged**, source `ed34d683bbc7082ddfa7bd679d8629bd88d43a34`, context
`ctx-b8121c41`. Its source sampler again passed 254W/253L/5D, balanced seats;
both native builds passed. Both training attempts were rejected by `prepare_run`:
`Native asset target identity differs: model_sha256`. The cold asset binds
`d30f5fae…`, while both new builds report `cead5dce…`.
**No epochs, training SPS or training memory measurement** resulted. GPU samples
were 0% utilization with a 555 MiB peak, reflecting preparation only.
The revised driver used 420s startup and bounded 600s training per profile,
without duplicate standalone CPU preflight; its 30-minute cap was $1.485.
Verified terminal archive: `/tmp/generals-h100-probe-results-job-kcten/`, SHA-256
`dbc67a8b0393b1844783f75517f1edcc0ce1bb83c447572ff23ecaa6bcd9825e`.
The identity mismatch was traced to **179 AppleDouble metadata sidecars**
in the old Fabric fingerprint; all **243 executable Python sources are unchanged**.
CPU rebind proof preserves ABI `0c7a1fb0…`, all **23 portable tensors bitwise**,
and logits on **18 hosted states** within **1.19e-6**; weights and learner bytes
are retained, adding zero RL steps. Proof:
`/tmp/generals-appledouble-clean-cold-migration-v1/proof.json`, SHA-256
`94972ea9e22e84cac5b11b2c97c6e8d5a5676f93c13c73c3ff88809fcaed75cd`.
The clean cold asset manifest SHA is
`58925af1dbeaa46e17230d0ea856739232d4e057b28a14417e3d9ea65903a5b9`.

**Third H100 sizing job `job-zfz7e` is terminal and unqualified.**
2,048/H256 was stopped at **24,731.7 SPS**, below the 30K gate.
4,096/H128 completed 3,145,728 steps, sustaining **36,643 SPS over its final
1,048,576 steps**, with **68,241 MiB sampled peak device memory**. This geometry
fits H100, but qualification failed: `Complete finite reward audit is missing`.
Neither profile qualifies; these are throughput measurements, not strength results.
Evidence: `/tmp/generals-h100-probe-results-job-zfz7e/`.

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
**Matched control/warm staging is underway; no new strength result exists.**

B300 **36081 was CANCELLED while PENDING at 01:24:40 UTC** at the user's request
for smaller measurements: runtime 0, no node, no epochs, Nice 2147483645/Priority 1
unchanged. Proof: `/tmp/generals-job-36081-superseded-sizing.json`.

## Latest completed execution: 35956

**FAILED, workload exit 1**, source `4c7dfc26e850799dc07bc1a0ea7b78d166e18ea7`;
controller retention expired. Direct Enroot passed allocated host/container GPU
ownership/UUID and idle smoke, then control CUDA parity: **46/46 top actions**,
batch 8, CPU layout/GPU inference, max logit/probability errors **1.43e-6/1.85e-6**.
Current native build and fresh-optimizer CE-weight CPU preflight passed.
Sampling failed before games: `Spatial frozen opponents require balanced seats`.
The evaluator now sets `balance_opponent_sides=True`; 33 focused checks and an
actual retained-CE CPU self-match passed. **No warm PPO or broad panels ran**;
completed control/CE work was not repeated.

Verified result/evidence: `/tmp/generals-policy-overhaul-results-35956/`,
`analysis.json`, GPU-step receipts, control parity and warm preflight/sampling logs.
Archive SHA: `a93a68b51af78c3070358aa2910e735d4d820a1d560626c28895708e4fd59695`.
Enroot uses private `/var/tmp` storage; capacity gates are in the runbook.
Installed audit `/tmp/generals-installed-enroot-audit.json` SHA:
`a7d831ac21b15eb8769ca1b95df5616781b825897c0bd80e3eb670b2ffc2b0cd`.

## Memory and smaller-batch decision

The policy weights are **2.21 MiB**; B300 is not an intrinsic policy requirement.
At 8,192 environments / horizon 256, native Puffer retains two float32 observation
copies (110.25 GiB), two float32 action-mask copies (55.14 GiB), and three native
state copies (7.11 GiB): **172.503 GiB known buffers**. The measured **205.584 GiB**
is device total-minus-free from `cudaMemGetInfo`, including allocator caches,
not peak live tensors; **33.081 GiB remains unattributed**. Accounting:
`/tmp/generals-native-memory-accounting-20261005.json`.

The sizing profiles compare 2,048/H256 (43.126 GiB known buffers) with
4,096/H128 (44.904 GiB), each fresh from cold weights, minibatch 8,192/replay 0.5,
**3,145,728 steps / six epochs**, two warmup epochs, then raw epoch and device
memory measurements. Require **≥30K actual end-to-end SPS** before qualification;
4,096/H128 is now qualified by the completed eight-epoch confirmation. Selecting new
geometry requires **both control and warm arms at that geometry**; the retained
8,192-control is baseline evidence, not a matched control for 2,048/4,096.

## Attempt ledger

All listed B300 submissions record Nice **2147483645**, Priority **1**, one GPU,
eight CPUs and 96 GiB; finite caps are retained below. Controller records are in
result directories; 35936 and 35956 terminal controller retention expired, so
their outcomes come from workload receipts/logs and caps from submission readback.
Prior H100 migration attempts failed before updates; that old launcher is retired.

| Attempt | Outcome / error | Cap | Retained evidence |
| --- | --- | --- | --- |
| `job-xyrtm` | Failed: context compression unsupported; zstd tar required | Not recorded here | `/tmp/relh-generals-h100-terminal-job-xyrtm.json` |
| `job-xwjya` | Failed: empty Dockerfile | Not recorded here | `/tmp/relh-generals-h100-terminal-job-xwjya.json` |
| `job-vmn9x` | Failed: permission denied `/ctx/pilot` | Not recorded here | `/tmp/relh-generals-h100-terminal-job-vmn9x.json` |
| `job-hhccv` | Failed: CPU preparation QEMU executable format | Not recorded here | `/tmp/relh-generals-h100-terminal-job-hhccv.json` |
| `job-gsp6k` | Smoke/sampling passed; `Pinned Puffer trainer changed` before updates | Not recorded here | `/tmp/relh-generals-autoresearch-result-gsp6k/` |
| 35892 | Completed selected baseline; hosted/local gains inconclusive | 70 min | `/tmp/relh-generals-portable-result-35892/` |
| 35932 | Completed 8M steps; exploration regressed and was rejected | 80 min | `/tmp/generals-policy-overhaul-results-35932/` |
| 35933 | Failed 1:0, 1m37s: nested `build.log` collision, `FileExistsError`; no updates | 120 min | `/tmp/generals-policy-overhaul-results-35933/` |
| 35934 | Failed before PPO: training reward audits applied to signed self-match rewards | 120 min | `/tmp/generals-policy-overhaul-results-35934/` |
| 35935 | Failed 1:0, 15m39s: control 8M + CE completed; parity adapter installed twice | 120 min | `/tmp/generals-policy-overhaul-results-35935/` |
| 35936 | Workload exit 1; control parity SIGSEGV −11, no child traceback; no warm/evaluation | 90 min | `/tmp/generals-policy-overhaul-results-35936/` |
| 35949 | Failed 1:0, 1s: `/tmp` inode guard; no downloads/GPU query/parity/build/updates | 100 min | `/tmp/generals-policy-overhaul-results-35949/` |
| 35956 | Workload exit 1: GPU scope/parity/build/preflight passed; balanced-seat guard failed before sampling games | 100 min | `/tmp/generals-policy-overhaul-results-35956/` |
| 36081 | Cancelled while pending: no node/runtime/epochs; smaller sizing requested | 100 min | `/tmp/generals-job-36081-superseded-sizing.json` |
| `job-i8tp6` | Failed: sampler and both builds passed; both CPU preflights timed out at 120s before epochs | 30 min GPU | `/tmp/generals-h100-probe-results-job-i8tp6/` |
| `job-kcten` | Failed: sampler/builds passed; both train initializers rejected model identity; no epochs; 651s/$0.5368 billed | 30 min GPU / $1.485 maximum | `/tmp/generals-h100-probe-results-job-kcten/` |
| `job-zfz7e` | Unqualified: 2048/H256 below30K; 4096/H128 36,643 SPS/68,241 MiB but final reward audit missing | 30 min GPU / $1.485 maximum | `/tmp/generals-h100-probe-results-job-zfz7e/` |
| `job-s79eq` | Succeeded: 4096/H128 qualified at 36,182.7467 SPS, full finite/legal audit; 608s/$0.5016 billed | 30 min GPU / $1.485 maximum | `/tmp/generals-h100-qualification-4096h128-results-job-s79eq/` |

35949 required 60K available inodes; subsequent read-only inspection found
13,170 despite ~1.45 TB free bytes. Its unreached recovery-mount mismatch is
removed in the current backend. Verified three-file terminal archive SHA-256:
`7bca77aed10597e45c0bf817df11145ea86538a200a0c5b741906d72bd3c0c78`.

## Preserved control, intervention and decision

The matched defense driver is integrated and pushed through `de5f915`: it uses
the provider-visible GPU, avoids duplicate standalone CPU preflight, bounds
training startup to 420s and refuses unqualified geometry. **This experiment
has not run**; H100 geometry qualification passed and staging is underway.

35935's completed control checkpoint:
`c2d6737be7b09bbc956643f72247c2ef4335b839fb3327f8b899bef4144fe8a6`.
One NVIDIA B300 SXM6 AC, 8,192 games, horizon 256, minibatch 8,192, replay 0.5:
two warmup epochs took 131.398s, then **4,194,304 steps / 49.645s = 84,486 SPS**.
Mean last-60s GPU utilization was 44.17%; sampled device-used memory 210,518 MiB
(including caches). No illegal actions,
nonfinite rewards or clipped rewards; one terminal agent had zero reward. All 13
opponents sampled both seats. This qualifies this control's throughput only.
Its genuine same-sampler gate passed: 512 games, 254W/253L/5D, 314 unique maps,
256 games per seat. Warm-arm throughput and strength remain unqualified.

Preserved CE checkpoint:
`97bc62c79f33c9124a7394f29ef5caa94e4bf02b87d5b21e41bd6c11d7645816`.
256 supervised B300 updates took 7.085s, adding **zero RL steps**. On independent
512 training / 128 held-out maps, held-out defense survival improved
0.88% → 56.09% and teacher accuracy 0.78% → 57.81%; training accuracy was 100%.
These tactical metrics do not establish broad gameplay improvement.

Reject 35932 exploration: candidate score 64.66%, initialization 66.05%, cold
baseline 68.29%; paired signed delta −0.07251, 95% CI [−0.10446, −0.04044].
Its 76,357 SPS / legality / 46-state serving parity remain valid, but its source
self-match used actor log-gap 4 versus opponent 0 and was not a same-sampler gate.
Two minimal AMD64 hosted smokes lost both games but had no timeouts/illegal moves
and maximum replies 5.1/13.4ms; proof is `/tmp/relh-generals-minimal-serving-57be9f9/`.

Fresh public curriculum `/tmp/generals-fresh-public-curriculum-770-780-v1/` contains
2,048 unique training and 512 independent confirmation views, balanced by seat,
direction and all 16 shapes. All 2,560 labels were legal and survived their attack;
24 already-lost controls passed. Healthy counterfactuals are retained. No policy
optimization or confirmation evaluation occurred; proof SHA-256
`3c2d31a86863bbaa555adbfa1d78742f0fa8120cb56587b1b187b2c8f2923efb`.

## Current artifact contract

Only the 16-plane F32/G32 flat spatial graph, radius 1.01 or 2.01, is supported.
Factory SHA-256:
`48767fb4ee333ae0b1a02ae644fbdf6f52f7f6df6c90c97ab3fc3888ba0c0d8a`.
Native assets explicitly bind policy/learner hashes, architecture, sampler, seeds,
learner configuration/objective and opaque ancestor hashes. Portable manifests
bind only `asset.json`, `policy.bin`, `weights.npz`; serving has no learner.
Current 13-policy inputs: `/tmp/generals-current-policy-input-v2/`.

| Radius | Canonical model SHA-256 | ABI SHA-256 |
| --- | --- | --- |
| 2.01 | `d30f5fae0f1d4e8805476791817a1998535d21e30f979c4746b366b31e5ab3ab` | `0c7a1fb0dfb646b394dd94fbabbad397182bfc3fe62fd739889f2cfd6a8de851` |
| 1.01 | `81d9f696de1ef768c9af5f21d4def5cf27e19674081d4af6a3168e4996f639d1` | `fd02887c61dd313632e1d81e4b911e668d1668d505a185c5c116e7dc96216285` |

One-time `/tmp/generals-native-proof/equivalence-proof.json` proved exact offsets,
padding, graph/JAXPR/constants, priors/state/callbacks and forward/VJP/sampler
for both radii; SHA-256
`c1125da1b820a46fecb0163e7dd6d5b264709eb3df3569b0bb587a9a722d7a87`.
All 299 portable tensors and a 151-array public-environment trace matched;
proofs are packaged in the active capsule. Reused two-graph CPU execution is
identified honestly alongside the reviewed non-model source delta and current
sampling guard. Source cleanup added **zero RL steps**, preserving originals.

Next: collect the owned H100 probe's raw memory and steady epoch throughput,
use qualified 4,096/H128 geometry for matched control and warm arms.
Retain the baseline until broad and fresh hosted evidence qualifies a candidate.
