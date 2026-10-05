# Current policy state

Baseline evidence reconciled on **2026-10-05**. The overhaul is in progress.
This page describes recorded results; verify artifact availability and hashes
before resuming or deploying any policy.

## Latest qualification and selected decision

Current job **35956** is **PENDING** in partition `b300`, from sealed source
`4c7dfc26e850799dc07bc1a0ea7b78d166e18ea7`. Controller readback confirms Nice
**2147483645**, Priority **1**, and a finite **01:40:00** cap. It requests one
B300, eight CPUs and 96 GiB memory using the direct Enroot backend with owned
`/var/tmp` storage. Source, submission and signed transport receipts share prefix
`/tmp/relh-generals-direct-enroot-b300-20261005t223500z-v2`.
Its 313-file input archive SHA-256 is
`f34d7e56ab893615ec14acc88092efdbe25309e39b4484db2e346bca52849c6d`.

The first allocated-step gate must verify host GPU cgroup visibility, idle GPU
ownership and matching container UUID. That actual host-step scope is still
unverified; installed cgroup configuration and three prior batch-shell
single-GPU receipts support the implementation but do not replace this gate.
The preserved control's native CUDA parity audit then runs first: 46 hosted
public states, batch 8, 300-second child cap, CPU layout and GPU inference.
Only after those gates pass may the current build, fresh-optimizer warm arm
and matched broad panels run. Completed control/CE work from 35935 is preserved
and not repeated. There are **no new GPU measurements, warm PPO steps,
evaluation results or strength qualification** from 35956 yet.

Continuation **35949** terminated **FAILED, exit 1:0**, after **one second**
on `metta-fabric-b300-1`, from source
`4ca2e9926858a8f53a0f7a7d2141916fbe00ff4c`. Host `PREPARE` rejected insufficient
free bytes or inodes on its owned `/tmp` scratch filesystem. It required
32 GiB free and 60,000 available inodes; the error did not record actual gauges.
No input/image downloads, GPU visibility query, parity, build, warm training
or evaluation were reached. **Zero new training steps** were produced.
Controller readback confirmed Nice **2147483645**, Priority **1**, one B300,
eight CPUs, 96 GiB memory and the finite **01:40:00** cap.

A subsequent read-only node inspection found about **1.45 TB free bytes but
13,170 available inodes**, below the declared requirement. Storage selection
and the site's Enroot filesystems must be audited before another submission.
The next preparer also omits the obsolete `mount_recovery` key: the signed
35949 configuration set it true despite the sealed current input containing
no recovery tree. That later mismatch was not reached in 35949.

The original sealed current NativeSpatialAsset inputs and completed control/CE
policies from 35935 remain preserved; no completed computation was repeated.
Submission/source receipts share prefix
`/tmp/relh-generals-current-continuation-b300-20261005t221200z-v2-core`.
The input archive SHA-256 remains
`f569da1390893d384c8be3e7f2174bbaf1abaa66fec3c6795589008825bb6fcc`.
The 9,355-byte terminal archive has three files and SHA-256
`7bca77aed10597e45c0bf817df11145ea86538a200a0c5b741906d72bd3c0c78`;
parts and extraction were verified under
`/tmp/generals-policy-overhaul-results-35949/`.
35949 is terminal. The corrected replacement 35956 is submitted and pending;
no new warm PPO, evaluation or strength result is available.

The direct Enroot backend is now implemented at source
`4c7dfc26e850799dc07bc1a0ea7b78d166e18ea7`. It creates fresh mode-0700 job storage
under `/var/tmp`, derives its own DATA/TEMP/CACHE/RUNTIME/CONFIG paths, and runs
one sealed source runner inside the assigned Slurm GPU cgroup. It rejects old
site-storage, recovery-mount and retained-container configuration. Startup
requires 32 GiB/60,000 inodes, image unpack 15 GiB/40,000, and workload phases
8 GiB/20,000; actual filesystem gauges are recorded.

Read-only installed Enroot source/hook review and CPU lifecycle tests passed.
Packaged installed audit: `/tmp/generals-installed-enroot-audit.json`, SHA-256
`a7d831ac21b15eb8769ca1b95df5616781b825897c0bd80e3eb670b2ffc2b0cd`.
Those proofs establish implementation and ownership checks only. **The new
backend has not passed an allocated GPU launch or CUDA parity.** A newly sealed
capsule retains all 13 native/portable policies and the completed 35935
control/CE work. It reuses the actual native two-graph proof only after checking
all model-related source bytes remain identical, records the reviewed
non-model source delta, and reruns the current source sampling guard.

All five H100 attempts failed before optimizer updates on 2026-10-05:

| Attempt | Terminal failure |
| --- | --- |
| `job-xyrtm` | Context used an unsupported compression format; zstd tar required |
| `job-xwjya` | Empty Dockerfile |
| `job-vmn9x` | Build context permission denied at `/ctx/pilot` |
| `job-hhccv` | CPU preparation invoked a QEMU binary with the wrong executable format |
| `job-gsp6k` | Runtime smoke and sampling passed; train stopped at `ValueError: Pinned Puffer trainer changed` |

The H100 path is retired. Raw terminal/observation receipts are retained under
`/tmp/relh-generals-h100-*-job-*.json`, with the final attempt's extracted
result at `/tmp/relh-generals-autoresearch-result-gsp6k/`.

Trial **35935** terminated **FAILED, exit 1:0**, after 15m39s from source
`9ea67c73a489d40c3ab5013942de2bde82aa54f4`. Its fresh-optimizer control completed
**8,388,608 RL steps**, then the native/serving parity audit installed the direct
adapter a second time and failed. The warm PPO arm and broad evaluation did not
run. The completed control checkpoint is
`c2d6737be7b09bbc956643f72247c2ef4335b839fb3327f8b899bef4144fe8a6`.

The control used one NVIDIA B300 SXM6 AC, 8,192 games, horizon 256, minibatch
8,192 and replay 0.5. Two warmup epochs took 131.398s; then **4,194,304 steps in
49.645s = 84,486 SPS**. Mean GPU utilization over the last 60s was 44.17%; peak
memory was 210,518 MiB. No illegal actions, nonfinite rewards or clipped rewards
were recorded; one terminal agent had zero reward. All 13 opponents had samples
on both seats. This qualifies the measured control setup's throughput, not
warm-arm throughput or stronger play.

The genuine same-sampler source gate passed: 512 games, 254W/253L/5D, 314 unique
initial maps and 256 games per seat. Distillation completed **256 supervised
updates in 7.085s**, adding **zero RL steps**. Held-out defense survival
probability improved 0.88% → 56.09% and teacher action accuracy 0.78% → 57.81%;
training accuracy reached 100%. The separate sets contained 512 training and
128 held-out maps. Its checkpoint is
`97bc62c79f33c9124a7394f29ef5caa94e4bf02b87d5b21e41bd6c11d7645816`.
Tactical results do not establish gameplay improvement. Complete retained
analysis and hash-checked archives: `/tmp/generals-policy-overhaul-results-35935/`.

Retry **35936**, source `0bbec52b4aa2da1559c3a1827901279ceb601ed6`, restored the
completed control, distillation and build without repeating them. Smoke passed;
**control parity terminated by SIGSEGV (child exit −11)** after bootstrap,
direct-adapter activation and the graph-build notice, with no child Python
traceback. The workload ended with exit 1; controller terminal readback was
unavailable during collection. Warm training and held-out evaluation were never
reached. Raw logs, source receipt and verified archive are retained at
`/tmp/generals-policy-overhaul-results-35936/`.
Both submissions recorded Nice **2147483645**, Priority **1**, one B300, eight
CPUs and 96 GiB host memory; finite caps were two hours / 90 minutes respectively.
These submissions are terminal; the later 35949 continuation also failed.

Trial **35934** previously completed distillation but failed before PPO because
training-only reward audits were applied to signed evaluation rewards. That
scope error and 35935's duplicate activation have been repaired in source;
35936's native GPU parity crash still needs a successful actual-runtime proof.
Evidence: `/tmp/generals-policy-overhaul-results-35934/`.

Matched defense trial **35933** failed before any supervised or PPO updates
from source `bd9bb9dae2ea93083b63bfb497a6d347dabf6b32`. It started
2026-10-05 20:06:14 UTC and stopped after 1m37s, exit 1:0. Controller readback
confirmed one B300 GPU, eight CPUs, 96 GiB memory, Nice **2147483645**,
Priority **1**, and a finite two-hour cap. Smoke passed; the build phase hit
`FileExistsError` because nested subprocess logging collided with the outer
runner's `build.log`. Commit `e5c9c92` gives inner processes distinct owned
log names. The retry preserved the same sealed experiment; 35933 produced no learning
result. Verified failure evidence is retained at
`/tmp/generals-policy-overhaul-results-35933/`.

The consolidated code is on `relh/policy-overhaul`, with integration review in
[PR #5](https://github.com/relh/generals-bots/pull/5). Old experiment versions
are retained in Git history.

B300 job **35932** completed with exit 0 on 2026-10-05 from source
`57be9f95e4531c907ab312ceee0041f02cd311ab`. It used one B300, eight CPUs,
96 GiB host memory, 8,192 games, horizon 256, minibatch 8,192 and replay 0.5.
Controller readback confirmed Nice **2147483645**, Priority **1**, and a finite
80-minute cap; actual allocation time was 20m14s.

After two warmup epochs (141.083 seconds), the trainer advanced **4,194,304
steps in 54.93 seconds: 76,357 end-to-end SPS**. It completed 8,388,608 new
steps with no illegal actions, nonfinite rewards, clipped rewards or missing
terminal rewards. All 13 opponents had samples on both seats. Serving parity
passed 46 public states, with maximum logit difference 1.67e-6. The result
archive and parts were SHA-verified and collected under
`/tmp/generals-policy-overhaul-results-35932/`; analysis is in `analysis.json`.

On 4,096 matched development games, the trained exploration candidate scored
**64.66%** (wins plus half draws), its exploratory initialization 66.05%, and
the unchanged cold baseline **68.29%**. Candidate minus cold signed-score
change was −0.07251, clustered 95% CI [−0.10446, −0.04044]. **Reject the
exploration candidate; retain baseline `f4ef5616…` with log-gap scale 0.**
These development results do not establish hosted winning strength.

A subsequent stricter startup audit found that job 35932's self-match used
log-gap scale 4 for the actor and scale 0 for its opponent. The old report
incorrectly called it a same-sampler check. Throughput, legality, serving parity
and broad paired results remain valid; genuine same-sampler startup readiness
was not established. The current gate verifies both sampler dictionaries and
raw maps, seats and outcomes directly. The next trial uses scale 0 for both.

The minimal serving image also completed two native AMD64 hosted runtime
smokes with no timeouts or illegal actions. Maximum replies were 5.1 / 13.4 ms;
the unchanged baseline lost both. Frozen runtime policy ID is
`5ef78e23-c02e-4d13-bc8e-2f9ab47fb0a1`, image digest
`sha256:daa8f48fa7ea6ac01a0f07e5a452e6321f5dcd8abab7813cbe4bba3464753d25`.
This qualifies execution; it is not a new strength result.

The matched public-defense experiment has completed its control and supervised
intervention, but native parity blocked the warm arm and strength panels.
A stronger gameplay policy has not been established.

## Qualification

**The hosted winning-policy objective remains unmet.** The latest completed
hosted screen scored 9/32 wins against Daveey v7 and 18/32 against the incumbent
relh policy. These 32-game screens are insufficient for champion qualification.
The previous candidate had the same totals.

The initial promotion target is at least 65% wins against each strong named
opponent, with balanced seats and enough fresh games for the 95% confidence
interval's lower bound to exceed 50%. Broad strength also requires preservation
against the fixed evaluation pool and clean hosted execution.

## Selected baseline and evidence

| Item | Recorded identity or result |
| --- | --- |
| Latest completed Classic training | B300 job `35892`, radius-2 spatial policy |
| Policy SHA-256 | `f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14` |
| Optimizer SHA-256 | `1c4832d6f5516ed85a97cf0b476b303ba8829467defeac4b5a7e6d1f6b11599b` |
| Lifetime training steps | 2,499,805,184; latest continuation added 33,554,432 |
| Hosted policy ID | `64649097-765f-4706-8310-910e57067a34` (`relh-classic-radius2-35892-final`) |
| Hosted Daveey results | Seat 0: 3W/13L; seat 1: 6W/10L |
| Hosted incumbent results | Each seat: 9W/7L |
| Hosted execution | 64 fresh unique seeds; no failed requests, timeouts, or illegal replay moves recorded |
| Local paired panel | 4,096 games: parent 2699W/1364L/33D; final 2713W/1348L/35D |
| Local improvement estimate | Signed-score delta +0.007324; clustered 95% CI [−0.022176, +0.036830], inconclusive |

The recorded B300 setup used **8,192 parallel games, horizon 256, minibatch
8,192, replay 0.5, and four native opponent workers**. All 13 opponents had
samples on both seats. Audited steady training throughput was **83,326 SPS**.
A broad interval after warmup advanced 25,165,824 steps over 301.407 seconds
(83,494 SPS); first-epoch startup was 111.555 seconds. Sampled GPU utilization
ranged from 30–100%. These measurements qualify that recorded setup; a changed
configuration must demonstrate its own end-to-end throughput.

Retained evidence locations recorded in the handoff:

- Local results: `/tmp/relh-generals-portable-result-35892/`.
- Hosted results and audits: `/tmp/relh-generals-serving-35892-final/`.
- S3 result key: `relh/generals-classic-results-20261004T004731Z-ee3fda7c`;
  archive SHA-256 `744e8d92b59a019cf0738acf56a62535a927dbc483fb2d0cec7d167fc9abb8b1`.
- Full settings, receipts, and experiment lineage: [historical Classic status](https://github.com/relh/generals-bots/blob/106ac6af647d8a2148f9ebd1d43409b73edd4ddb/integrations/COWORLD_CLASSIC_STATUS.md).

## Learning diagnosis and next decisions

Wider spatial context learned nonzero weights and passed serving parity, but
its bounded continuation produced inconclusive local gains and unchanged
hosted win totals. Nine retrospective fatal-state probes still missed the top
defense, with extremely low rescue-action probability. The matched exploration experiment subsequently regressed and was rejected.
The defense warmstart improves tactical support but still needs broad strength
evaluation and fresh hosted confirmation.

Fresh public-view data is sealed at
`/tmp/generals-fresh-public-curriculum-770-780-v1/`: 2,048 unique training views
and 512 independent confirmation views, balanced across both seats, all four
attack directions and all sixteen Classic board shapes. All 2,560 teacher
labels are legal and survive the prescribed attack; 24 already-lost controls
also passed. Healthy public counterfactuals are retained for a possible
behavior-preservation objective if broad results require it. No learned
candidate has been evaluated on the new confirmation set. The training set
has four times the previous 512 unique views. Proof SHA-256:
`3c2d31a86863bbaa555adbfa1d78742f0fa8120cb56587b1b187b2c8f2923efb`.
This is data preparation, with zero new policy optimization or RL steps.

## Current artifact and source contract

The cleaned current factory contains only the sixteen-plane F32/G32 flat
spatial graph. Its source SHA-256 is
`48767fb4ee333ae0b1a02ae644fbdf6f52f7f6df6c90c97ab3fc3888ba0c0d8a`.
One-time isolated old/new proofs passed exact parameter offsets/padding,
complete graph/JAXPR/constants, priors, state, native callbacks, forward/VJP
and sampler comparisons for both radii. Proof:
`/tmp/generals-native-proof/equivalence-proof.json`, SHA-256
`c1125da1b820a46fecb0163e7dd6d5b264709eb3df3569b0bb587a9a722d7a87`.
This is source cleanup with **zero RL steps**; original weights, optimizer
state and research records remain untouched.

Canonical sorted current asset configuration identities:

| Radius | Model SHA-256 | ABI SHA-256 |
| --- | --- | --- |
| 2.01 | `d30f5fae0f1d4e8805476791817a1998535d21e30f979c4746b366b31e5ab3ab` | `0c7a1fb0dfb646b394dd94fbabbad397182bfc3fe62fd739889f2cfd6a8de851` |
| 1.01 | `81d9f696de1ef768c9af5f21d4def5cf27e19674081d4af6a3168e4996f639d1` | `fd02887c61dd313632e1d81e4b911e668d1668d505a185c5c116e7dc96216285` |

The native asset explicitly binds policy/learner hashes, architecture, sampler,
training seeds, current learner settings/objective and opaque ancestor hashes.
The portable bundle binds `asset.json`, `policy.bin` and `weights.npz` only;
serving excludes optimizer state. Current derived inputs are at
`/tmp/generals-current-policy-input-v2/`; successful current GPU execution and
fresh hosted strength remain required before qualification.

Active next decision: collect 35956's allocated host/container GPU-scope and
preserved-control CUDA parity gates before warm training and broad evaluation.
Reuse control/CE artifacts under the proven equivalent effective experiment.
Retain the baseline until broad and fresh hosted evidence qualifies a candidate.
