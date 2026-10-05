# Policy runbook

Use [current-state.md](current-state.md) for the selected baseline and
[roadmap.md](roadmap.md) for unfinished work. Repository `AGENTS.md` is the
compute policy. Historical experiments remain in [Git history](https://github.com/relh/generals-bots/blob/106ac6af647d8a2148f9ebd1d43409b73edd4ddb/integrations/COWORLD_CLASSIC_STATUS.md); the commands below are the supported operational path.

## Game and policy contract

The normal hosted Classic target uses `coworld_classic=True` in the Puffer
wrapper and `coworld_classic_rules=True` in `GeneralsEnv`. Dimensions are sampled
independently in 18–21, padded to 21 for inference; games cap at 2,000 turns.
The wrapper currently uses mountain density 0.24–0.26, minimum general distance
17, castle range `(9, 11)`, and castle army range `(40, 51)`. Interpret range
endpoints from the implementation. Smaller curricula are explicit experiments
and cannot supply the final hosted qualification.

The local Softmax server uses this same pinned engine and shared
`CLASSIC_MAP_OPTIONS`, including the 2,000-turn cap. Short local smoke matches
may explicitly reduce the cap; those runs are execution checks.

The default `GeneralsEnv()` and `mode="competition"` configure different games.
Check the instantiated environment, engine hash, observation channels, legal
masks, action encoding, sampling settings, and termination behavior. Never
infer checkpoint compatibility from board size alone.

For potential shaping, set `shaping_gamma` explicitly to the learner's effective
`train.gamma`. Record reward scales, terminal objective, shaping weights, and
any teacher inputs. Evaluation scores must use game outcomes.

## Runtime and existing entry points

An editable install supplies the simulator and optional local dependencies:

```bash
pip install -e '.[dev,train,softmax]'
```

Native training also requires the matching Metta `metta_training` package,
pinned Puffer trainer, Fabric build, CUDA/JAX runtime, model source, and matching
build manifests. The existing pipeline has several artifact-specific adapters;
these are not interchangeable general launchers.

| Entry point | Scope |
| --- | --- |
| `integrations.metta_puffer` | One-seat and batched JAX environments |
| `integrations.spatial_selfplay` | Frozen and population policy opponents |
| `integrations.launch_spatial_selfplay_training` | Native launcher, transfer and sampler guards |
| `integrations.portable_classic_pilot` | Immutable-input pilot phases: `smoke`, `build`, `sampling_gate`, `train`, `evaluate` |
| `integrations.classic_learner_continuation` | Checked learner/optimizer continuation and constrained geometry migration |
| `integrations.audit_spatial_hosted_replays` | Public-action legality and serving comparison against hosted replay panels |

The portable pilot assumes `/work/input`, `/work/out`, and retained parent
artifacts under `/recovery` or an explicit continuation manifest. It is designed
for the prepared GPU container. Running its phases on an ordinary editable
checkout does not reconstruct those inputs. Use the operational facade below for the supported commands.

## Operational commands

Use the policy facade from the repository root:

```bash
python -m integrations.policy status
python -m integrations.policy validate --build BUILD_CONFIG_JSON --run RUN_CONFIG_JSON
python -m integrations.policy preflight --build BUILD_DIRECTORY --config RUN_CONFIG_JSON --output NEW_DIRECTORY
```

`status` reports the recorded baseline. `validate` checks the effective build/run
contract; its `--build` argument is a JSON configuration file. `preflight` takes
an existing native build directory and exercises the real CPU launcher using
its run configuration, writing evidence to a new output directory. Native
preflight requires the supported Metta/Puffer container runtime and dependencies.

The packaged GPU container exposes the existing bounded pilot phases:

```bash
python -m integrations.policy pilot smoke
python -m integrations.policy pilot build
python -m integrations.policy pilot preflight
python -m integrations.policy pilot sampling_gate
python -m integrations.policy pilot train
python -m integrations.policy pilot evaluate
```

These operate on the container's prepared `/work/input`, `/work/out`, and
recovery artifacts. They do not stage missing inputs or provide arbitrary
profiles for ordinary local checkouts.

`train` and `resume` forward the complete native launcher arguments and perform
its bootstrap. Use their supported container environment, with the audited
initialization/continuation configuration. `evaluate`, `export`, and `compare`
forward the existing module arguments; inspect each command's `--help` for
required artifacts and outputs. `promotion --help` describes the evidence
report used to assess hosted qualification.

The production factory `integrations/generals_fabric.py` has been restored to
its canonical source SHA-256:
`5221cd60c85a7a056717d27eb630b1e975442e6b50ce65c99a9f35b876f03474`.
Source alignment is an implementation repair; new GPU throughput, learning,
and hosted results still require execution.

## Prepare and qualify a run

1. Verify the baseline policy, optimizer, source, and build hashes. Preserve the
   originals and write the proposed effective configuration in a new run folder.
2. Exercise the real launcher in CPU preflight, including its pinned-trainer,
   transfer, sampling, geometry, and discount checks. A successful image build
   alone did not catch the most recent launch failure.
3. Verify training and serving compute the same acting probabilities, legal
   masks, priors, half moves, and temperature schedules. Inference and PPO must
   agree on the sampler used to collect the rollout.
4. Submit one finite B200/B300 Slurm qualification job using
   `--nice=2147483645`. Record `scontrol show job -o JOB_ID` readback with
   Nice, Priority, and finite TimeLimit. Follow all restrictions in `AGENTS.md`.
5. Measure completed native Puffer steps over a steady interval after compilation
   and warmup, including rollout, transfer, inference, and optimizer updates.
   Require at least **30,000 SPS on the proposed GPU configuration** before
   extending training. Keep any long dependent job held until that measurement.
6. Retain hardware identity, GPU utilization, environments and agents per trainer,
   horizon, minibatch, replay ratio, timing bounds, completed steps, per-trainer
   and aggregate SPS, and opponent counts by seat. A crashed trainer's steps
   cannot establish a sustained gate. Release failed or idle allocations promptly.

Read native dashboards or metrics while a trainer runs; the subprocess summary
JSON may show zero until exit. A bounded profiling run can diagnose sub-30K SPS.
For recurrent runs, pass the known nonfinite-gradient region before extending.

## Strength evaluation and promotion

Freeze the evaluation opponent versions, map distribution, seeds, seats, and
sampler. Keep development and fresh confirmation maps separate. Paired baseline
and candidate comparisons should share initial maps and report W/L/D by
opponent and seat, paired deltas, uncertainty, and failures.

For sampler changes, compare against the actual changed initialization and the
unchanged qualified parent. A win over a weakened initialization alone does
not establish stronger play. Checkpoint selection must precede final held-out
and hosted panels.

Export a frozen serving bundle with a content manifest and exact sampler.
Verify native/serving parity and wire legality, then collect fresh balanced
hosted matches against Daveey and the incumbent. Initial acceptance: at least
65% wins against each, with a 95% confidence lower bound above 50%, preservation
against the broad pool, and clean execution. Record the decision and exact
bundle identity before promoting the selected policy on an eligible account.

## Serving the frozen policy

Export and serve the supported spatial bundle with its exact weights, model
metadata, and sampling settings. The neural player exclusively uses
`SpatialPlayerPolicy`; it needs no private Metta training runtime or factory
source. Build `integrations/softmax/Dockerfile.neural` with the exported bundle
as its named `policy` build context. See the
[local serving commands](../../integrations/softmax/README.md#frozen-neural-policy).
A local serving pass does not supply new training or hosted strength evidence.

## Evidence and documentation updates

Keep a bounded current-state page: selected policy, latest qualified measurements,
remaining failures, and next experiment. Put full run outputs and receipts in
versioned artifact directories or durable storage; link them from the state page.
Record paths as unverified until their existence and hashes have been checked.

Preserve checkpoints, optimizer state, and raw experiment outcomes. Protected Codex session history and trajectory databases must never
be removed or rewritten as cleanup. Current user authorization and repository instructions govern the active
overhaul. Git history retains superseded documentation and experiment recipes.
