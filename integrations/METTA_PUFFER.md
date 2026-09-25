# Generals on native PufferLib

## Throughput gate for long GPU runs

Long Generals training experiments require at least **30,000 sustained Puffer agent steps per second** on the proposed one-GPU setup. Measure end-to-end training after JAX compilation, including rollouts, observation transfer, teacher labels, and optimizer updates. Record wall time, completed steps, B200/B300 model, GPU utilization, number of vector environments and games per JAX batch, and per-trainer and aggregate SPS. A short GPU profiling job can run below the gate to identify bottlenecks. Keep dependent long jobs held and do not submit another long run until the end-to-end gate passes. Do not use CPU-only Slurm jobs or metta4.

The four-trainer B300 configuration below sustained only about 3,200 aggregate SPS despite about 90% GPU utilization and 256 concurrent games. It is retained as a reproducible baseline and must be profiled and improved before another long experiment. Preserve its checkpoints. After clearing the throughput gate, train for tens of millions of steps and evaluate on held-out 10×10 classic maps against mixed scripted opponents; the target is at least 0.60 performance.

The 2026-09-23 B300 rollout profile isolated the bottleneck: the original Python list and teacher-target construction peaked near 8,000 SPS even at 1,024 games per JAX batch. Vectorized observation construction plus array transport reached 31,031 SPS at 1,024 games **including native numeric encoding**, but this excludes policy inference and optimization. The bounded 1,048,576-step Puffer smoke, job 7092, produced zero steps after about three minutes with GPU utilization near 0%; it was canceled, and its detached container was stopped. This does not pass the training throughput gate. Profile native trainer startup and the first rollout before submitting another long run.

The 2026-09-24 B300 probes now clear the end-to-end gate. Each trainer used 1,024 classic-map games per JAX batch, 4,096 Puffer agents, four vector threads and buffers, a 32-step horizon, and 8,192-sample minibatches. The rates below use completed steps from epochs 2–32 divided by elapsed training time, including checkpoints and optimizer updates. These are training SPS, not rollout-only estimates.

| Classic-map policy and job | Steps per policy | Checkpoint-inclusive SPS | Validation performance, seeds 901 / 902 |
| --- | ---: | ---: | ---: |
| Memoryless MLP, 8368 | 4.19M | 38,302 | 0.231 / 0.233 |
| Spatial recurrent, 8420 | 4.19M | 34,417 | 0.185 / 0.183 |
| Memoryless spatial, 8453 | 4.19M | 37,265 | 0.214 / 0.212 |
| Memoryless MLP with 50% student actions, 8475 | 4.19M | 41,903 | 0.271 / 0.268 |
| Two parallel mixed-rollout MLPs, 8493 | 4.19M each | 36,601 + 37,570 = 74,171 aggregate | Throughput probe only |

The two-trainer probe used one B300 and averaged 55.1% active-interval GPU utilization, versus roughly 27% for a single memoryless MLP. The scripted teacher scored 0.754 / 0.743 on the same validation seeds. These small policy probes do **not** establish the 0.60 target. Select models using validation seeds; reserve seeds 1001–1005 for one final held-out evaluation after choosing a credible trained policy. The Metta native bridge now discards disabled PPO cotangents instead of multiplying nonfinite values by zero; see [Metta PR #24777](https://app.graphite.dev/github/pr/Metta-AI/metta/24777).

The guarded mixed-rollout job 8522 completed two independent 31,457,280-step policies on one B300. Checkpoint-inclusive epochs 2–240 measured 38,647 and 38,882 SPS, for 77,529 aggregate SPS; active GPU samples averaged 59.4%. Its final checkpoints are archived under `/home/metta/relh-generals-puffer/mlp-dagger-two-30m-8522` on metta0, with hashes `fe1dc29400ec2938fa2965180a138e583a0f96ef42d8ae2c015df023e25e9836` and `291f11ea1adce794158446c3ac608c9dfff964102fbe8e72a427afaaaf97fb70`. Validation job 8523 measured 0.313 / 0.312 for policy 0 and 0.315 / 0.316 for policy 1 on seeds 901 / 902. This meets the throughput and training-duration gates, but **fails the 0.60 performance target**. Leave held-out seeds 1001–1005 untouched and diagnose policy learning before another long run.

The next bounded experiments tested student-only MLP rollouts (8580: 0.185 / 0.185), tied spatial Fabric kernels (8639: 0.222 / 0.216), and tied local-action Fabric kernels (8686: 0.263 / 0.254). Each completed 4.19M steps on one B300 and is archived on metta0. Public Harvester route features added seven channels computed solely from the visible observation: distances from the general, distances to a visible goal, a goal mask, and four improving-direction masks. The 21-channel tied local-action probe 8731 improved validation to 0.334 / 0.330 at 4.19M steps. A single trainer measured 28,550 SPS, below the gate for a one-trainer long run. The proposed **two-trainer** setup, job 8769, completed 4.19M steps each at 28,209 and 28,352 checkpoint-inclusive SPS, or **56,561 aggregate SPS on one B300**, with 72.5% mean warmed GPU utilization. That setup clears the one-GPU gate; its evidence is archived at `/home/metta/relh-generals-puffer/two-goalaction-probe-8769`. Guarded long job 8794 used the same two-trainer setup for 31.46M steps per policy, with validation job 8795 dependent on successful completion. [`generals_b300_goalaction_two_30m.sbatch`](generals_b300_goalaction_two_30m.sbatch) stops on nonfinite gradients, low aggregate SPS, or low GPU use after warmup; [`generals_b300_goalaction_validation.sbatch`](generals_b300_goalaction_validation.sbatch) evaluates seeds 901 and 902.

Job 8794 completed **31,457,280 steps for each of two independent policies** on one B300, with 1,024 games per JAX batch, 4,096 Puffer agents per trainer, four vector threads and buffers, a 32-step horizon, and 8,192-sample minibatches. Checkpoint-inclusive epochs 2–240 measured 26,561 and 30,030 SPS, **56,591 aggregate SPS**; GPU utilization averaged 68.2% after the first 90 seconds, including the tail when only one trainer remained. Validation job 8795 scored 0.8122 and 0.8292 for policies 0 and 1 across seeds 901/902, so policy 1 was selected before the held-out run. GPU held-out job 8892 scored **0.8302492** across 20,480 individual 10×10 classic games against mixed scripted opponents on untouched seeds 1001–1005, above the 0.60 target. The five seed scores were 0.827637, 0.831543, 0.839844, 0.826050, and 0.826172. [`verify_goalaction_heldout.py`](verify_goalaction_heldout.py) passed against the durable metta0 archive, checking complete training, source and evaluation builds, classic map and opponent settings, selection, seeds, and checkpoint hash. The selected final checkpoint SHA-256 is `18489fe680cca211ace244ab0f9d17023a3d003b1ca7bc4eef3d5376cbeba489`; the training, validation, held-out, resume-state archives and `heldout-proof.json` are at `/home/metta/relh-generals-puffer/goalaction-two-30m-8794` on metta0. The original B300 artifacts remain in `/tmp/relh-generals-gpu`.

The array transport path depends on [Metta PR #24777](https://app.graphite.dev/github/pr/Metta-AI/metta/24777). The bounded benchmark and full Puffer smoke launchers are [`generals_b300_rollout_bench.sbatch`](generals_b300_rollout_bench.sbatch) and [`generals_b300_throughput_smoke.sbatch`](generals_b300_throughput_smoke.sbatch).

`integrations.metta_puffer:GeneralsPufferEnvironment` is a one-seat numeric
environment for Metta's native PufferLib 5 runner. It trains against
`ExpanderAgent` by default on 10×10 maps with fog of war, a 300-turn horizon, legal action
masks, and terminal win/loss reward plus bounded economic shaping. It uses the
plain ruleset; it does not use the 18–21 tile competition preset.

Put this checkout and `packages/metta-training/src` from Metta on `PYTHONPATH`.
The runtime needs JAX, NumPy, Pydantic, and Msgpack. The native Puffer build
also needs the CUDA and compiler tools described in Metta's
`packages/metta-training/README.md`.

Build configuration:

```json
{
  "environment": "metta_generals",
  "python_environment": {
    "factory": "integrations.metta_puffer:GeneralsPufferEnvironment",
    "options": {"board_size": 10, "horizon": 300},
    "spec": {"observation_size": 1400, "action_sizes": [801]},
    "source_modules": ["generals", "integrations.puffer_codec"]
  }
}
```

The action index is eight directional move planes (four full and four split)
followed by pass. Observations contain the 14 public channels from
`Observation.as_tensor()`. The environment alternates the learner's seat from
the episode seed and regenerates maps on reset. Native Puffer checkpoints are
specific to this observation and action contract.

Run `python -m metta_training.cli build --config build.json --output build`,
then train with `python -m metta_training.cli train --config run.json --build
build --output run`. Set `train.gamma` to `0.99` in the run overrides to match
the shaping reward's discount. Keep training and held-out evaluation seeds
distinct. An evaluation reports win as performance 1, draw as 0.5, and loss
as 0.

For a separate opponent evaluation build, set `options.opponent` to `hunter`,
`harvester`, or `random`. Set it to `mixed` to choose Random, Expander, or Hunter per episode.
`options.shaping_weight` sets the potential reward scale (default 0.2). The
observation and action dimensions remain the same, so a
checkpoint can be evaluated against those opponents with a matching build
manifest. Keep the original training source and build for the standard
held-out Expander evaluation.

For a curriculum run, `options.teacher` can name a scripted agent and
`options.imitation_weight` can reward matching its non-pass action. This is
disabled by default and does not alter the public observation or action
contract. Keep the held-out win/performance metric separate from this training
reward.

For direct imitation loss, set `options.teacher` to `harvester` and
`options.supervise_teacher` to `true`. Set `python_environment.spec.teacher` to
`true` and configure a matching Fabric policy with `fabric.observation_size`
1400, `fabric.action_sizes` `[801]`, and `fabric.teacher` `{}`. Training
observations carry a one-hot legal scripted action; evaluation observations
carry zero-weight targets. This uses the trainer's teacher cross-entropy loss
and requires a separate Fabric build and checkpoint from the native MinGRU
policy above.

For a two-head policy, set `options.factorized_actions` to `true`, use
`action_sizes` `[401, 2]` in both the environment spec and Fabric configuration,
and build a new checkpoint. The first head selects a source cell and direction
or pass; the second selects full or half army. Teacher targets supervise each
head independently, leaving the split head unweighted when passing. This keeps
the complete move set while reducing the policy output size.

`integrations.puffer_policy.NativePufferPolicy.from_run(run_directory)` loads a
completed native checkpoint for one-game CPU inference. Call `reset()` before
each game, then `act(observation, key, deterministic=True)` on each public
observation. This loader supports the pinned Puffer revision and its default
linear encoder, MinGRU network, and linear decoder; it rejects incompatible
checkpoint dimensions. The native evaluator remains the source of held-out
performance results.

For GPU training, use
`integrations.metta_puffer:BatchedGeneralsPufferEnvironment` with
`options.parallel_games` set to a positive number such as 16. Set
`python_environment.spec.agents` to the same number, and set
`fabric.platform` to `cuda`. One environment process advances all of its games
with a single `jax.vmap` call on the GPU. Set `vec.total_agents` to a multiple
of `parallel_games * vec.num_buffers`; the number of environment processes is
`vec.total_agents / parallel_games`. This path needs a CUDA JAX installation
and a Slurm GPU allocation. The launcher enables both CUDA and CPU JAX
backends because Fabric traces part of the graph on CPU; the environment
requires an actual CUDA device. During training, a finished game starts a new
map on that seat and ends its old agent life. Held-out evaluation keeps the
finished seat absorbing until the batch ends. Its episode score is the mean
outcome across the batched games, so multiply the evaluator's episode count
by `parallel_games` to obtain the underlying number of games.

The checked-in B300 pilot uses
[`gpu-batch-build.json`](configs/gpu-batch-build.json),
[`gpu-batch-pilot.json`](configs/gpu-batch-pilot.json),
[`Dockerfile.b300`](Dockerfile.b300), and
[`generals_b300_train.sbatch`](generals_b300_train.sbatch). The
[`run_puffer_gpu.sh`](run_puffer_gpu.sh) launcher puts JAX's CUDA 13 libraries
first in the library path and checks that CUDA is the default JAX device while
the CPU backend remains available for Fabric graph tracing. Stage the source,
native build, and CUDA JAX runtime together on the GPU node; Metta verifies
their source fingerprints when a run starts.

The launcher stores JAX compilations under `/work/jax-compile-cache`. Full
evaluation starts a separate native process for each held-out seed, so later
seeds can reuse compiled kernels. The cache is node-local and can be overridden
with `JAX_COMPILATION_CACHE_DIR`.

The GPU evaluation job uses five held-out seeds. The full
[`gpu-batch-eval.json`](configs/gpu-batch-eval.json) config runs 16 batched
episodes per seed, or 1,280 underlying games at `parallel_games=16`. The
[`gpu-batch-eval-short.json`](configs/gpu-batch-eval-short.json) config runs four
episodes per seed, or 320 games. After staging this checkout as `source-gpu`
under the GPU workspace, evaluate a completed run with:

```bash
RUN_NAME=gpu-batch-30m EVAL_CONFIG=gpu-batch-eval.json \
  sbatch --export=ALL \
  integrations/generals_b300_eval.sbatch
```

Set `CHECKPOINT_PATH` to a checkpoint path inside the container to evaluate an
in-progress run. Give each evaluation its own `OUTPUT_NAME`:

```bash
RUN_NAME=gpu-batch-30m \
  CHECKPOINT_PATH=/work/gpu-batch-30m/checkpoints/metta_generals/gpu-batch-30m/0000000010240000.bin \
  OUTPUT_NAME=gpu-batch-30m-eval-10m EVAL_CONFIG=gpu-batch-eval-short.json \
  sbatch --export=ALL \
  integrations/generals_b300_eval.sbatch
```

The job and its `srun` step both request a GPU. Its environment adapter checks
that JAX placed game state on CUDA. Use `evaluation.json` from the output
directory for model-only held-out performance; training `perf` includes mixed
teacher actions.

The recycling fallback script reads the first run's full `evaluation.json` and
starts a second 31,457,280-step GPU run only when mean held-out performance is
below 0.60. Stage this entire checkout under `source-recycle` in the GPU
workspace before submitting it; the build, training, and evaluation JSON
configs under `integrations/configs/` are required. The script checks those
files before launching its GPU step.

While a long run writes checkpoints on node-local storage, run
[`archive_puffer_checkpoints.sh`](archive_puffer_checkpoints.sh) on the Slurm
submit host with the running job ID, node-local run directory, and a durable
destination directory. It polls every five minutes by default, uses GPU-bearing
steps inside the existing allocation to copy new checkpoints, verifies each
checkpoint's SHA-256 after transfer, and exits when the job stops. Pass `0` as
the fourth argument for a single archive pass.

The optional wider follow-up changes `features_per_site` from 8 to 32 in
[`gpu-batch-wide-build.json`](configs/gpu-batch-wide-build.json). It keeps the
same 31,457,280 training steps and held-out evaluation protocol. Stage this
checkout under `source-wide` on the B300 node together with the tested
`metta_training` and `fabric` source directories, then submit
[`generals_b300_wide_fallback.sbatch`](generals_b300_wide_fallback.sbatch) with
`--dependency=afterok:<recycling-job-id> --export=ALL`. The job reads the
recycling run's five-seed evaluation and starts training only if its mean
performance is below 0.60. Both the job and its training step request one GPU.

For the capacity-tested wider run, stage the 16-game wider build and
`gpu-batch-30m.json` as `build-gpu-wide16x16-6021` and
`gpu-batch-four-long-base.json` under `/tmp/relh-generals-gpu` on B300. Submit
[`generals_b300_four_trainers.sbatch`](generals_b300_four_trainers.sbatch) from
the mettabox. It starts four independent 31,457,280-step Puffer trainers on
one GPU. The bounded pilot completed 65,536 steps for each trainer while the
GPU stayed near 90–96% utilization in a steady four-trainer window. Launch
[`archive_four_puffer_trainers.sh`](archive_four_puffer_trainers.sh) on the
mettabox with the Slurm job ID and a durable archive directory; it checks the
hash of each copied checkpoint. The training allocation waits 150 seconds
after all four runs complete so the archiver can copy final checkpoints.

After training, use [`generals_b300_parallel_eval.sbatch`](generals_b300_parallel_eval.sbatch)
with `EVAL_RUN_JOB_ID=<training-job-id>`, `EVAL_INDEXES=0,1,2,3`,
`EVAL_SEEDS=901,902,903,904,905`, and `EVAL_EPISODES=8` for validation.
Select one policy using its validation summary, then rerun with that single
index, `EVAL_SEEDS=1001,1002,1003,1004,1005`, and `EVAL_EPISODES=16` for
held-out performance. Set a distinct `EVAL_PREFIX` for each evaluation. The
optional `EVAL_CHECKPOINT_STEP` selects a numbered checkpoint instead of the
run's final checkpoint. The launcher evaluates four seeds or policies
concurrently on one GPU by default.
Set `EVAL_PARALLELISM=8` to fill the 64 CPU allocation with eight concurrent
evaluators; each then receives eight CPUs. The launcher
writes a summary weighted by the number of completed games. It needs the
node-local training runs and matching build; restore them from the mettabox
archives if the B300 workspace has been recycled.

For the named classic map distribution, build
[`gpu-batch-classic10-build.json`](configs/gpu-batch-classic10-build.json)
from a separate `source-classic10` checkout. It keeps the same 10×10 policy
dimensions and mixed scripted opponents, but uses the classic scenario's
minimum general distance (8), castle values (20–40), and 800-turn limit.
Do not overwrite the source tree pinned by an active training run. Pass
`EVAL_BUILD_NAME=<classic-build-directory>`, `EVAL_SOURCE_NAME=source-classic10`,
and `EVAL_ALLOW_TRANSFER=true` to the parallel evaluator. Metta checks the
training and evaluation model identities and records the source build in each
result. Keep validation seeds 901–905 separate from the final held-out seeds
1001–1005.
The staged evaluation and Sentinel fallback source trees use Metta's merged
vectorized numeric encoder from commit `53baa67cf1`; the active trainer keeps
its original pinned source. Rebuild the evaluation binary after staging that
file so the environment fingerprint records the encoder used.
The classic launcher validates each of the four policies at 20.48M, 25.6M,
and final 31.46M steps, then tests the best validation checkpoint on the
held-out seeds. Every candidate therefore has at least 20M training steps.
Run `python integrations/verify_classic_heldout.py /tmp/relh-generals-gpu
<training-job-id>` against the retained workspace before reporting success.
The verifier recomputes the selection and held-out score from raw per-seed
results and checks the checkpoint hash, training record, map options, and
opponent mix. The classic Slurm launcher also runs this audit and writes a
proof JSON before it exits. A valid score below 0.60 permits the bounded
Sentinel smoke; invalid evidence stops the dependency chain.
Run `archive_puffer_evaluations.sh <classic-job-id> <durable-directory>
/tmp/relh-generals-gpu <pilot-job-id> <training-job-id>` on the mettabox while
the classic job is pending. It preserves the pilot's GPU utilization CSV and
validation summary alongside the classic raw results and proof, checking each
copy against the B300 source hash.
Run `archive_puffer_pilot.sh <pilot-job-id> <durable-directory>` as well when
the pilot may fail before the classic job starts. It refreshes a SHA-checked
copy of the live GPU samples every two seconds inside the pilot allocation.
The parallel evaluator also keeps a separate GPU CSV under each evaluation
prefix, so the classic watcher preserves utilization for every validation
checkpoint and the held-out pass.

If the Harvester-supervised run misses the classic held-out target, use
[`gpu-batch-sentinel-classic10-build.json`](configs/gpu-batch-sentinel-classic10-build.json)
for a new run. It uses the stronger observation-only Sentinel as the teacher
and trains on the same 800-turn classic 10×10 map distribution used for the
held-out evaluation. The teacher is called only on the player's public
observation. Its late curriculum continues to mix some teacher actions into
rollouts while the learner acts independently on most turns. This is a new
training build and requires its own checkpoints; the active Harvester run is
unchanged.

[`gpu-batch-sentinel-transfer-build.json`](configs/gpu-batch-sentinel-transfer-build.json)
keeps the active run's Fabric model and teacher schedule exactly, changing
only the Python environment to the Sentinel teacher and classic maps. Metta's
checkpoint initialization can therefore verify and load an active-run
checkpoint with `initialize.allow_environment_transfer=true`, provided its
model hash and state size match the new build. The new run starts its own
learner clock unless `restore_learner` is explicitly set. Its source run and
checkpoint hash remain recorded in the training lineage.

[`generals_b300_sentinel_smoke.sbatch`](generals_b300_sentinel_smoke.sbatch)
is a bounded follow-on after classic held-out evaluation. It exits immediately
if the selected policy scores at least 0.60. Otherwise it verifies the selected
checkpoint, builds the Sentinel transfer environment, and trains four
65,536-step policies on one GPU. Its utilization audit must pass before a
long Sentinel continuation is scheduled.

After inspecting the smoke throughput, submit
[`generals_b300_sentinel_long.sbatch`](generals_b300_sentinel_long.sbatch)
with `TRAIN_JOB_ID` and `SMOKE_JOB_ID` if the classic score is below 0.60 and
the smoke's median active GPU utilization is at least 50%. The launcher
rechecks both gates and the source checkpoint before training four transferred
policies for 31,457,280 additional steps each on one B300. It is not in the
prequeued dependency chain. Start the mettabox checkpoint watcher as
`archive_four_puffer_trainers.sh <sentinel-job-id> <durable-directory>
/tmp/relh-generals-gpu sentinel-long` when the job starts.
