# Measured PPO performance audit

On September 8, 2026, the original experimental 4×4 PPO rollout was compared with
the same rollout step inside `jax.lax.scan`. Both paths were measured in the same
process, from identical state/network/random-key inputs, with one warmup and five
timed repeats. The standalone tool does not alter the trainer or checkpoints.

## Reproduce

Install the repository's `train` extra first, then run from its root:

```bash
env -u LD_LIBRARY_PATH .venv/bin/python scripts/profile_agent.py \
  --device cpu --num-envs 64 --steps 64 --repeats 5 \
  --output .cache/runs/profiling/cpu-64x64.json

env -u LD_LIBRARY_PATH .venv/bin/python scripts/profile_agent.py \
  --device gpu --num-envs 64 --steps 64 --repeats 5 \
  --trace .cache/runs/profiling/gpu-trace \
  --output .cache/runs/profiling/gpu-64x64.json
```

GPU selection requires CUDA and fails if unavailable. Unsetting `LD_LIBRARY_PATH`
avoids this machine's incompatible system CUDA libraries overriding the virtual
environment's packages. The tool disables JAX GPU preallocation unless explicitly
configured otherwise. Optional profiler traces are captured **after** timing, so
trace overhead does not enter steady-state performance results.

JSON reports include all repeat times, first-call times, hardware, CPU affinity,
runtime/package versions, environment flags, seed, configuration, git revision,
dirty status, relevant source hashes, equivalence checks, and available device
memory statistics. Preserve the JSON with any result; rerun both alternatives
together after changing a workload or runtime. These examples profile the original
fixed-size trainer, not the new strategic agent or the new training pipeline.

## Same-command evidence

Hardware: AMD Ryzen Threadripper 1920X (24 logical CPUs), NVIDIA RTX 2080 Ti.
Python 3.12.11, JAX/JAXlib 0.11.1, Equinox 0.13.8, Optax 0.2.8, NumPy 2.5.3.
Seed 42, 64 parallel games, 64 turns per rollout, 4×4 empty maps, original
randomly initialized network and random opponent. Initial armies are multiplied
by ten; half the games start at turn 498 to exercise the 500-turn reset boundary.
This is a deliberately reproducible microbenchmark, not a trained-policy score.

All entries below are synchronized median wall times. "Iteration" includes
rollout, bootstrap/GAE, and a single full-batch Adam update, with checkpointing and
logging excluded. Both iteration alternatives use the same compiled batch
preparation function, so their difference isolates the rollout implementation.

| Device / stage | Python rollout baseline | Scan alternative | Baseline / scan |
| --- | ---: | ---: | ---: |
| CPU, rollout and stacking | 209.21 ms | 254.43 ms | 0.82× |
| CPU, iteration | 425.57 ms | 391.93 ms | 1.09× |
| GPU, rollout and stacking | 93.69 ms | 20.87 ms | 4.49× |
| GPU, iteration | 84.82 ms | 28.50 ms | 2.98× |

GPU rollout throughput increased from 43,721 to 196,272 environment steps/second;
GPU iteration throughput increased from 48,290 to 143,703 environment steps/second.
These are **prototype measurements**, not a claim that the existing production
trainer has already become faster. The scan wraps exactly the existing
`rollout_step`; there is no change to reward semantics or sampled actions.

CPU rollout scan was slower (Python repeats 200–235 ms; scan 244–272 ms). CPU
iteration ranges overlap (Python 400–453 ms; scan 380–441 ms), and the apparent
small iteration improvement conflicts with the isolated rollout result. Do not
claim a CPU speedup or replace CPU collection on this evidence. Other development
processes were active on the host; longer, isolated repeated runs are required
for a CPU acceptance decision. GPU timing ranges were separated: rollout
83–110 ms versus 20–22 ms; iteration 82–103 ms versus 27–30 ms.

Separate stage timings aid attribution; **do not sum them** because fusion,
dispatch, synchronization and run-to-run noise differ across stages.

| Stage | CPU median | GPU median |
| --- | ---: | ---: |
| One complete rollout step | 3.82 ms | 0.95 ms |
| Observation, mask, and batched inference | 1.94 ms | 1.00 ms |
| Bootstrap and GAE | 1.93 ms | 2.10 ms |
| Optimization of 4,096 samples | 186.33 ms | 5.94 ms |

First invocation includes tracing, compilation, autotuning, and execution. It is
reported separately from warmed execution, but **is not an isolated compiler
timer**. Original rollout-step first calls took 2.92 s CPU / 5.95 s GPU; scan
first calls took 3.01 s CPU / 5.97 s GPU; optimizer first calls took 1.06 s CPU /
2.22 s GPU. Some first calls reuse compiled subfunctions, so first-call times
must not be added or compared as total cold startup. CUDA autotuning emitted
delay-kernel timing warnings during initial compilation; warmed host wall-clock
results are reported here, not those autotuner timer estimates.

## Correctness and instrumentation

The tool compares every final-state leaf, final PRNG key, and every collected
observation, mask, action, log probability, value, reward, done flag and game-info
leaf. Integer and boolean leaves matched exactly on both devices. Floating
leaves matched exactly on GPU and within 1.49e-8 absolute error on CPU (acceptance
tolerances 1e-5 absolute and relative). Each rollout exercised 36 episode ends.
The tool writes its report and exits unsuccessfully if equivalence fails.

One captured GPU trace contains 85,834 host events for the Python iteration and
7,288 for scan; host callback launches fall from 78 to 3. CUDA graph launches
remain similar (132 versus 133), and GPU events increase (6,654 to 7,434). This
supports **reduced Python/host dispatch overhead**, not a claim that scan removed
most device kernels. Counts include profiler instrumentation and are only
comparative evidence. Trace durations are inflated by instrumentation and must
not replace untraced benchmark timings.

Trace artifacts are under `.cache/runs/profiling/gpu-trace/plugins/profile/`:
`titan.trace.json.gz` for a Chrome/Perfetto-compatible trace viewer and
`titan.xplane.pb` for an XPlane-compatible profiler. The trace annotations
`python` and `scan` delimit one warmed iteration of each alternative.

## Audit recommendations and acceptance gates

1. Use a compiled scan collector for GPU training, then rerun this protocol on
   the actual new network, reward components and opponent mixture. The original
   4×4 result cannot establish throughput at 18–21-cell competition sizes.
2. Preserve explicit `(states, key)` carry and return all trajectory arrays.
   Compare state, randomness, transitions and reset boundaries before accepting
   a collector change. Synchronize returned leaves when timing asynchronous JAX.
3. Benchmark CPU and GPU separately. Tiny maps do not amortize dispatch well;
   optimization already dominates a substantial part of CPU iteration cost.
   Avoid increasing network width or full-batch size without a measured budget.
4. Instrument true terminal outcomes, timeouts, raw reward components, value
   loss, policy loss, entropy, approximate KL, clipping fraction and invalid
   action/pass rates. Throughput without gameplay competence or valid learning
   targets is not a useful training success criterion.
5. The original trainer creates candidate reset maps and states every step,
   including steps with no completed games. A cached map pool may help larger
   maps, but changing RNG or map distributions requires separate correctness
   and learning-quality checks; no speed claim is made without measurement.
6. The original random opponent truncates its valid-action list at 100, and the
   policy represents pass once per board cell. Those choices distort action
   distributions when scaling maps. They must be redesigned rather than treated
   as harmless performance details. The fixed-size value head and 4×4 reset
   generator are additional limits on generalization.
7. Full-batch optimizer timing is not a minibatch PPO epoch benchmark. Future
   comparisons must include the same rollout budget, optimization epochs,
   minibatch sizes, opponent distribution, seeds and held-out evaluation.

No simulator or original trainer optimization was merged by this first audit. The
deliverables are reproducible tooling, measured prototype evidence, and the
device-specific decision gate above.

## Spatial PPO optimizer follow-up

The replacement 8×8 trainer has a different workload: 32 environments × 64
steps, four PPO epochs and 256-sample minibatches, yielding 32 optimizer updates
per iteration. Its throughput must not be compared directly with the original
4×4 single-update benchmark above.

Source inspection identified a host synchronization after every minibatch:
seven scalar diagnostics were converted to Python floats before the next update
could start. The new [compiled driver](../../generals/training/optimization.py)
keeps minibatch gathering, shuffling, finite checks and KL stopping on the device.
It returns buffered diagnostics once per iteration. The
[training entry point](../../generals/training/train.py) now selects this driver
on GPU and retains the reference Python loop on CPU, whose throughput has not
been measured for this change.

The prototype was measured before integration with this exact command:

```bash
env -u LD_LIBRARY_PATH .venv/bin/python scripts/profile_optimizer.py \
  --device gpu \
  --checkpoint .cache/runs/spatial-terminal/final_checkpoint.pkl \
  --with-rollout --repeats 7 \
  --trace .cache/runs/profiling/optimizer-gpu-trace \
  --output .cache/runs/profiling/optimizer-gpu.json
```

The [benchmark tool](../../scripts/profile_optimizer.py) collects one real rollout
from the frozen terminal-only pilot checkpoint, then compares both implementations
from identical network, optimizer, batch and random-key inputs. Both implementations
are warmed first; timed order alternates across seven paired repeats, synchronizing
all outputs. It saves an exact input fixture beside the JSON report. Use
`--fixture FILE` for later optimizer-only comparisons without regenerating data.

| GPU workload | Reference Python | Compiled | Ratio |
| --- | ---: | ---: | ---: |
| Optimizer phase, median | 650.76 ms | 37.23 ms | 17.48× |
| Optimizer phase, repeat range | 497.23–839.70 ms | 34.68–48.98 ms | |
| Rollout + GAE + optimizer, median | 921.09 ms | 321.21 ms | 2.87× |
| Rollout + GAE + optimizer, repeat range | 737.84–1,123.51 ms | 245.54–387.54 ms | |
| Environment steps per iteration second | 2,223 | 6,376 | 2.87× |

Hardware and package versions match the earlier audit. The GPU was reserved for
this benchmark; CPU correctness tests were active concurrently, so host contention
and the broad repeat ranges limit extrapolation. These are matched measurements
on one frozen pilot workload, not a promised speedup for every model or map size.
Iteration timings include rollout, GAE, optimizer and diagnostic means; they
exclude pool refresh, episode JSON logging and checkpoint writes. First optimizer
invocations took 4.65 s for the reference and 3.20 s for the compiled driver,
including compilation and execution. The trained checkpoint and benchmark fixture
are unchanged by profiling.

Both drivers performed exactly 32 updates without KL stopping. Final RNG and
integer optimizer state matched exactly; floating parameters and optimizer state
differed by at most 2.98e-8, and diagnostic means agreed within the recorded
tolerances. A separate traced iteration reduced host callback launches from
3,184 to 21, while CUDA graph launches increased from 100 to 129. This supports
the host-overhead explanation; traced timings are excluded from the table.

The compiled driver preserves a split only when an epoch is entered, applies
each update before checking that update's reported KL, stops before any further
update on nonfinite diagnostics, and preserves partial final minibatches. Only
executed finite diagnostic rows enter the same float64 NumPy means; inactive
buffer slots never dilute reported statistics.

[Optimizer regressions](../../tests/test_compiled_optimizer.py) cover those
control-flow cases, real PPO parameter/optimizer parity, strict KL comparison,
failures on the first and later updates, and exact checkpoint continuation.
The full compiled trainer is also exercised on CPU test inputs, including
partial minibatches and uninterrupted-versus-resumed equality. Together with the
training foundation regressions, 22 focused tests passed after integration.

Checkpoints and startup/resume events now record `optimizer_implementation`
(`compiled-v1` or `python-v1`), and resume events record the previous driver.
Exact continuation is tested within the same implementation and backend;
switching between drivers can introduce the small floating differences measured
above and is not promised to reproduce a historical trajectory bit for bit.
