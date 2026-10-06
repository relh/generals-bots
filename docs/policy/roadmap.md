# Winning Classic policy roadmap

The user authorized ownership of the fork, refactoring, documentation cleanup,
parallel agent work, and pushes. Completion requires a qualified hosted policy;
infrastructure cleanup alone does not complete the goal.

| Stage | Intended result | Completion evidence |
| --- | --- | --- |
| 1. Consolidate | One development line, current docs, baseline inventory; Git retains superseded history | Current state with exact artifact identities and accessible evidence |
| 2. Game contract | Training, evaluation, replay and serving use the explicit Classic target | Required engine checks and retained hosted replay reproduction |
| 3. Puffer execution | Real-launch preflight, useful commands, durable run manifests, resume | Bounded optimizer run and resume; sustained ≥30K end-to-end SPS on its GPU setup |
| 4. Baseline | Fixed diverse opponents, balanced seats, held-out maps, ranked loss diagnosis | Reproducible paired results, uncertainty, and replay-supported learning hypotheses |
| 5. Learning | Controlled changes that address observed losses | Broad improvement with preservation against strong controls; qualified throughput |
| 6. Hosted qualification | Frozen serving bundle and confirmed winning advantage | Fresh hosted acceptance panels and recorded champion promotion |

Consolidation and the shared Classic contract are implemented. Control training
in 35935 sustained 84,486 SPS and passed its genuine same-sampler source gate.
The source cleanup has exact structural/numeric proof. Job 35956 then passed
actual direct Enroot GPU ownership/UUID, 46-state CUDA control parity, current
build and CE fresh-optimizer CPU preflight. It failed before sampling games on
an omitted explicit balanced-seat option; warm PPO and broad panels did not run.
B300 continuation 36081 was cancelled while pending, with no node/runtime/epochs
and unchanged maximum Nice/Priority 1, to measure smaller geometry as requested.
First H100 sizing job `job-i8tp6` failed: its source sampler and both native
builds passed, but both 120s CPU preflights timed out before training epochs.
Revised H100 job `job-kcten` also failed: sampler and both builds passed, but
both train initializers rejected the cold asset's model identity (`d30f5fae…`
versus built `cead5dce…`). It billed 651s/$0.5368 with no epochs or sizing
measurements; GPU samples were 0% utilization, peak 555 MiB. The mismatch was AppleDouble metadata in the old fingerprint: 243 executable
Python sources unchanged; CPU proof preserves ABI, all 23 tensors bitwise and
18 hosted-state logits within 1.19e-6. Third H100 sizing job `job-zfz7e` measured 24,731.7 SPS at 2,048/H256
(stopped below30K) and 36,643 steady SPS at 4,096/H128, with 68,241 MiB sampled
peak device memory. The latter completed six epochs but lacks its final finite
reward audit, so neither geometry is qualified. Eight-epoch confirmation
`job-s79eq` SUCCEEDED from `e649d73`: 4,096/H128, minibatch8,192/replay0.5,
4,194,304 steps, last-two-epoch 36,182.7467 SPS, peak68,517 MiB. Its complete
reward audit has zero nonfinite/clipped rewards and illegal actions, and all13
opponents sampled both seats. H100 geometry is qualified; matched control/warm
job `job-uyssa` failed finalization after clean control training from `7c60504` on one H100, bounded90min/$4.455,
zero restarts. Prior `job-int7f` BUILD_FAILED before allocation/zero billed at
the pip/version Docker layer; its log truncated before error details. The retry
exposed `/ctx/framework-source` permission denied and also BUILD_FAILED before
GPU execution: BSD tar included mode700 staging directories. The third attempt
uses files-only GNU tar/mode644, preserving source/input bytes and quiet installs. The files-only attempt `job-mvtti` then failed only on resolved curriculum
symlink spelling; 14 native sources and model/environment/revision matched.
The verified-path spelling fix passed targeted tests. The control completed8,388,608 steps at36,171.5 SPS with clean final reward/legal
audit, then publisher exceeded120s after graph reconstruction. Preserve its
control/learner outputs; warm PPO and broad panels remain incomplete. No matched
strength result exists. Known 8,192/H256 buffers account
for 172.503 GiB; the 205.584 GiB device-used reading includes caches and is not
an intrinsic requirement of the 2.21 MiB policy.
Learning improvement and hosted winning qualification remain open; see
[current-state.md](current-state.md).

Continuation `job-re2cn` BUILD_FAILED before GPU allocation/zero billed seconds:
pip `files.pythonhosted.org` ReadTimeout. Independent review also caught a latent
v1 source-revision seal mismatch. Continuation `job-vezu3` SUCCEEDED: preserved control published, distillation
and warm8M completed with clean reward/legal audits and46-state serving parity.
Both arms use4,096/H128/minibatch8,192/replay0.5; control/warm steady SPS are
36,171.513/36,265.339, H100 peaks69,078/68,517 MiB. Billed1760s/$1.452.
Held-out tactical survival/teacher accuracy improved, but broad warm strength
remains unknown: four4,096-game panels and matched paired CIs are still required.

## Experiment decisions

Retain baseline `f4ef5616…` with log-gap scale 0. The latest 8M-step exploration
candidate regressed against both its initialization and the unchanged baseline
on matched development games and was rejected. The stale launcher binding is
resolved; repeating that intervention or scaling its training is unjustified.

The matched public-defense warmstart experiment has completed control training in `job-uyssa`, with publisher finalization blocked
at qualified 4,096/H128 H100 geometry. The driver is integrated/pushed through `de5f915`,
uses the provider-visible GPU, avoids duplicate CPU preflight and allows 420s
bounded training startup. It refuses unqualified geometry; matched experiment results remain pending. Separate
synthetic training and held-out maps supply legal Sentinel tactical labels.
Distillation changes only policy weights; both control and warm PPO arms use
fresh optimizers, equal 8,388,608-step budgets, and the unchanged sampler,
position curriculum, opponent pool and rewards. Compare source, distilled,
control and warm bundles on the same broad development games before selecting
a candidate for fresh confirmation and hosted qualification. The frozen-match
balanced-seat call is repaired with 33 focused tests passing; verify its actual
sampling path before retrying the warm arm and broad panels. Direct Enroot and CUDA parity already have actual 35956
proof; retain completed control and supervised artifacts from 35935. Those
computations are not repeated. The failed retries provide no warm-arm result.

The intended sizing profiles compare 2,048/H256 (43.126 GiB known buffers) with
4,096/H128 (44.904 GiB), using minibatch 8,192, replay 0.5 and 3,145,728 steps
per probe (six epochs, two warmup). The eight-epoch confirmation passes measured fit and ≥30K end-to-end SPS;
33.081 GiB of the current device allocation is unattributed and cannot be assumed
to scale proportionally. Selecting new geometry requires new control and warm
arms at that geometry; the old 8,192 control remains baseline evidence. See
current-state for the source allocation accounting.

Preserve throughput, legality, train/serve parity, opponent-by-seat coverage and
frozen artifact identities. Reject failed hypotheses using their paired evidence;
choose subsequent changes from losing-replay diagnosis rather than unchanged
long runs. Additional architecture or memory work should address measured
failures in expansion, force concentration, half moves, defense or attack timing.

Use targeted tests for changed behavior and required execution gates. Broaden
verification when a concrete failure needs resolution. Integrate working
components promptly and debug the full operational path as it becomes usable.

Held-out evaluation `job-9fkii` is submitted from `512af1f`, validated one H100,
120min/$5.94 maximum, zero restarts, context `ctx-95843f1b`. It consumes the
verified `job-vezu3` artifact for four4,096-game Classic first-episode panels,
ordered source/warm/control/distilled, seeds51213/17431 and cap1,440s each.
The 145.39s512-game gate suggests~1,163s/panel, unmeasured. Paired analysis and
`EVALUATED.json` remain pending; no strength claim yet.

Training-pool diagnostic is adverse: control **7,192/10,896 wins (66.0%)**
versus warm **4,430/10,198 (43.4%)**, worse across all **13 opponents/26 seats**;
the final-quarter gap persists. This is correlated, reused training-pool evidence
and **cannot select a winner**. Post-PPO defense retention remains unknown;
frozen held-out `job-9fkii` results remain pending. SHA-bound diagnostic:
`/tmp/generals-matched-training-pool-comparison-vezu3.json`, SHA
`6cb6ace308a3820cb5b60e9cb911b86792d0a37c0bf4bd83f13b44cec26f9b06`.

## Initial winning acceptance

- At least 65% wins against both Daveey and the incumbent in balanced, fresh
  hosted panels; each 95% confidence interval has a lower bound above 50%.
- Strong performance across the fixed diverse pool, with no unexplained seat
  imbalance or regression against key controls.
- Clean hosted execution and verified policy/serving parity.
- Reproducible model lineage, qualified GPU throughput, and a frozen versioned
  bundle with a documented promotion decision.

Do not tune on the confirmation panel. Record broader opponents and additional
confirmation requirements before claiming stronger general dominance.
