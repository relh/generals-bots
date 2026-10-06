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
Givemeanode job `job-i8tp6` is building its native CUDA image on the way to one
H100 from source `8e41477`, bounded
30 min GPU / 60 min build, zero restarts/resume none and queue TTL 60 min.
No new measurement or strength result exists. Known 8,192/H256 buffers account
for 172.503 GiB; the 205.584 GiB device-used reading includes caches and is not
an intrinsic requirement of the 2.21 MiB policy.
Learning improvement and hosted winning qualification remain open; see
[current-state.md](current-state.md).

## Experiment decisions

Retain baseline `f4ef5616…` with log-gap scale 0. The latest 8M-step exploration
candidate regressed against both its initialization and the unchanged baseline
on matched development games and was rejected. The stale launcher binding is
resolved; repeating that intervention or scaling its training is unjustified.

The matched public-defense warmstart experiment is deferred until memory/throughput
sizing selects geometry. Separate
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

The submitted H100 probe compares 2,048/H256 (43.126 GiB known buffers) with
4,096/H128 (44.904 GiB), using minibatch 8,192, replay 0.5 and 3,145,728 steps
per probe (six epochs, two warmup). Require measured fit and ≥30K end-to-end SPS;
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
