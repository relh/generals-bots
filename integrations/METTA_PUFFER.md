# Generals on native PufferLib

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
requires an actual CUDA device. An episode score is the mean outcome across the
batched games, so multiply the evaluator's episode count by `parallel_games`
to obtain the underlying number of games.

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
