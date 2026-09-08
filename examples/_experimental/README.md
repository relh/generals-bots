# Experimental Examples

This folder contains experimental code and advanced examples that are not yet part of the main API.

## Contents

### `benchmark_performance.py`
Performance benchmark for measuring throughput with vectorized environments.

```bash
python benchmark_performance.py [num_envs] [num_steps] [iterations]

# Examples
python benchmark_performance.py 256 100 5      # 256 envs, 100 steps, 5 iterations
python benchmark_performance.py 1024 500 3     # 1024 envs, 500 steps, 3 iterations
python benchmark_performance.py 4096 100 2     # 4096 envs, 100 steps, 2 iterations
```

Outputs throughput statistics (steps/s) and helps identify performance bottlenecks.

### `ppo/`
Experimental PPO (Proximal Policy Optimization) training implementation. Work in progress.

The raw-game trainer runs a 4x4 policy against a random opponent. This is a
small training baseline; it does not use the competition's maps or modifiers.
From the repository root:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[dev,train]'
# Optional NVIDIA GPU support (requires a compatible driver):
uv pip install --python .venv/bin/python 'jax[cuda12]'

# Short check, including model saving:
JAX_PLATFORMS=cpu .venv/bin/python examples/_experimental/ppo/train.py 8 \
  --steps 32 --iterations 2 --save-every 1 --output .cache/runs/smoke/model.eqx

# Training with periodic model saves:
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -u \
  examples/_experimental/ppo/train.py 64 --iterations 1000 \
  --save-every 10 --output .cache/runs/ppo/model.eqx
```

The trainer reports loss, reward, completed episodes, wins, and steps per second
every ten iterations. The output file holds the latest saved model weights;
optimizer state is not saved, so it is not a resumable training checkpoint.
Use `JAX_PLATFORMS=cpu` to select CPU explicitly. Regression checks are available
with `.venv/bin/python -m pytest examples/_experimental/ppo/test_train.py -q`.

`ppo/train2.py` uses an older environment API and has not been updated for the
explicit reset pool. Use `ppo/train.py` for the commands above.

### Evaluating the saved policy

From the repository root, run a paired comparison against the training random
policy, RandomAgent, Expander, Hunter, and Harvester:

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu .venv/bin/python -u \
  -m examples._experimental.ppo.evaluate \
  --model .cache/runs/ppo/model.eqx --output .cache/runs/evaluation
```

The default evaluates both trained weights and their exact pre-training
initialization across 14,720 games. It covers every pair of starting locations
on an empty 4x4 board and 64 connected terrain maps, using fresh random streams,
both spawn assignments, and both player IDs. Games end on general capture or
a 500-turn draw; evaluation does not train or reset completed games.

Results include individual games in `games.csv`, boards, model metadata, and
`summary.json` with win/loss/draw counts and confidence intervals clustered by
board. The terrain suite is a small-board stress test, not competition play.
The current network's value head requires 4x4 input and its actions do not
include building castles.

For a supported NVIDIA GPU, change the environment prefix to
`env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false`.
Clearing `LD_LIBRARY_PATH` for this process prevents a shell's older CUDA
libraries from overriding the installed JAX libraries.

Run evaluator checks with
`.venv/bin/python -m pytest examples/_experimental/ppo/test_evaluate.py -q`.

### `visualize_policy.py`
Visualization tools for trained policies. Work in progress.
