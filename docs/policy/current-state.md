# Current policy state

Updated **2026-10-05, 23:21 UTC**. The winning-policy objective remains unmet.
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

## Active continuation: 35956

**PENDING**, source `4c7dfc26e850799dc07bc1a0ea7b78d166e18ea7`, direct Enroot,
one B300 / eight CPUs / 96 GiB memory, Nice **2147483645**, Priority **1**,
finite **01:40:00** cap. At 23:20 UTC the B300 node was `MIXED+DRAIN`, reason
`temporary drain for cleanup maintenance`. Keep the required lowest priority;
node state and competing work are unchanged. No new GPU measurement, warm PPO,
evaluation or strength result exists for this job.

Receipts and signed transport share prefix
`/tmp/relh-generals-direct-enroot-b300-20261005t223500z-v2`.
The 313-file sealed input SHA-256 is
`f34d7e56ab893615ec14acc88092efdbe25309e39b4484db2e346bca52849c6d`.
Phase maxima total 4,560s, with 840s startup and 600s finalization; core dumps
are disabled in the host shell and driver. The signed transport permits a full
run only if allocation starts **strictly before 2026-10-06 03:16:50 UTC**;
effective expiry is 05:06:50 UTC. Proof:
`/tmp/generals-policy-job-35956-credential-window.json`. It remained pending at
23:21:36 UTC; monitor this start horizon without treating queue time as a result.

Required order: allocated host GPU cgroup visibility/idle ownership → matching
container UUID → preserved-control CUDA parity (46 hosted public states, batch 8,
300s child cap, CPU layout/GPU inference) → current build → fresh-optimizer warm
8,388,608-step arm and its sampler/≥30K SPS gates → source/control/CE/warm paired
4,096-game panels. Completed control and supervised work are **not repeated**.
Actual allocated host/container GPU scope and CUDA parity remain unverified;
installed cgroup configuration and prior batch-shell receipts are supporting evidence.

The backend uses fresh mode-0700 `/var/tmp` job storage with owned Enroot
DATA/TEMP/CACHE/RUNTIME/CONFIG paths. Startup/unpack/workload free-capacity gates
are respectively 32 GiB/60K, 15 GiB/40K and 8 GiB/20K inodes, with actual gauges
recorded. It rejects site-storage, recovery-mount and retained-container flags.
Installed source/hook audit and CPU lifecycle review passed, **not GPU execution**.
Audit `/tmp/generals-installed-enroot-audit.json` SHA-256:
`a7d831ac21b15eb8769ca1b95df5616781b825897c0bd80e3eb670b2ffc2b0cd`.

## Attempt ledger

All listed B300 submissions record Nice **2147483645**, Priority **1**, one GPU,
eight CPUs and 96 GiB; finite caps are retained below. Controller records are in
result directories; 35936's terminal controller retention expired, so its failure
comes from the workload receipt/logs and its cap from submission readback.
H100 attempts all failed before optimizer updates; that path is retired.

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
| 35956 | Pending; first allocated GPU scope/parity proof outstanding | 100 min | Active receipt prefix above |

35949 required 60K available inodes; subsequent read-only inspection found
13,170 despite ~1.45 TB free bytes. Its unreached recovery-mount mismatch is
removed in the current backend. Verified three-file terminal archive SHA-256:
`7bca77aed10597e45c0bf817df11145ea86538a200a0c5b741906d72bd3c0c78`.

## Preserved control, intervention and decision

35935's completed control checkpoint:
`c2d6737be7b09bbc956643f72247c2ef4335b839fb3327f8b899bef4144fe8a6`.
One NVIDIA B300 SXM6 AC, 8,192 games, horizon 256, minibatch 8,192, replay 0.5:
two warmup epochs took 131.398s, then **4,194,304 steps / 49.645s = 84,486 SPS**.
Mean last-60s GPU utilization was 44.17%; peak memory 210,518 MiB. No illegal actions,
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

Next: collect 35956's GPU-scope/parity gates, then its warm arm and paired panels.
Retain the baseline until broad and fresh hosted evidence qualifies a candidate.
