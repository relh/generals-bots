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
    "source_modules": ["generals"]
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
