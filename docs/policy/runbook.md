# Classic policy runbook

[Current state](current-state.md) names the selected policy and live decisions;
[the manifest](../../integrations/policy_baseline.json) binds exact assets and
receipts. Repository `AGENTS.md` governs compute. Keep full experiment output
in verified artifact directories; Git history retains superseded procedures.

## Game and asset contract

Training and evaluation must instantiate `GeneralsEnv` with
`coworld_classic_rules=True` and pinned `CLASSIC_MAP_OPTIONS`. The official
Softmax Classic engine hash is recorded in the manifest. Map width and height
are independently sampled from 18–21, inference pads to 21×21, and games cap
at 2,000 turns. Position curricula are explicit training inputs; held-out
first episodes start from the official map distribution. The default
`GeneralsEnv()` and `mode="competition"` are different games.

Verify the engine hash, source, model and ABI identities, observation channels,
legal mask, action encoding, sampler and termination before comparing policies.
For potential shaping, explicitly set environment `shaping_gamma` equal to the
learner's effective `train.gamma`; record both and the reward configuration.
Score evaluation by game outcome.

Native training uses the supported pinned Metta/Puffer CUDA image, Fabric build
and verified current asset. `generals-native-spatial-asset-v1` binds `asset.json`,
`policy.bin` and optional `policy.bin.learner` to source/model/ABI hashes,
sampler, seed and provenance. Initialization with `restore_learner: false`
starts a fresh optimizer from exact policy weights; `true` requires matching
learner bytes and effective game/codec/reward settings. Actual completed PPO
writes `training.json`. Publish only that authentic run; do not synthesize
training records. Portable NumPy serving needs no private Metta runtime.

An editable local install is available with `pip install -e '.[dev,train,softmax]'`.
Inspect entry-point help and configuration before launch:

```bash
python -m integrations.policy status
python -m integrations.policy validate --build BUILD_CONFIG_JSON --run RUN_CONFIG_JSON
python -m integrations.policy preflight --build BUILD_DIRECTORY --config RUN_CONFIG_JSON --output NEW_DIRECTORY
```

`preflight` exercises the real CPU launcher in a supported native container.
Run it once per sealed configuration; actual GPU initialization still checks
its asset. AppleDouble sidecars are metadata and never justify bypassing asset
identity. Seal source revision and inputs; provider contexts use files-only GNU
tar with mode 644 so rootless builders can read the Dockerfile. Verify tar
contents before upload. A native image build alone does not prove launch.

## GPU qualification and training

Use a bounded provider GPU job with hardware, finite duration, cost cap and zero
restarts, or one finite B200/B300 Slurm job. Every `sbatch`, `srun` or `salloc`
must use `--nice=2147483645`; verify `scontrol show job -o JOB_ID` reports
Nice `2147483645`, Priority 1 and finite TimeLimit. Do not raise scheduling
priority or alter another user's job. For Slurm, the supported runner uses
single-step Enroot under the allocated GPU cgroup with per-job directories in
`/var/tmp`; verify available space, inodes, GPU UUID and native CUDA identity.
Use current `AGENTS.md` for detailed Slurm constraints.

Before **each new long training setup**, run a short GPU probe through the
same rollout, transfer, inference and optimizer path. Measure completed Puffer
agent steps divided by wall time over a steady interval **after compilation
warmup**. Require ≥30,000 end-to-end SPS on the proposed configuration;
GPU utilization alone does not qualify. Record hardware, interval and step
count, utilization, environment count, horizon, minibatch, replay ratio,
per-process and aggregate SPS, and opponent counts by seat. All opponent types
must appear on both seats. Record illegal actions, nonfinite/clipped rewards
and gradient failures; crashed steps do not qualify. For recurrent policies,
probe through the prior ~2.6-million-step nonfinite-gradient region. Stop idle
or failed probes promptly, preserving checkpoints and verified output.

The subprocess JSON monitor can remain at zero until exit; read the native
Puffer dashboard or metrics to assess live progress. Allow bounded cold-start
and final publication time, but investigate sustained low SPS before long
training. Change parallelism, batch or transfer only with measured end-to-end
evidence. Keep dependent long jobs held until the exact treatment passes its
throughput gate.

Publish a completed run and export its exact sampler as a portable bundle:

```bash
python -m integrations.policy publish --build BUILD_DIRECTORY/build.json \
  --training RUN_DIRECTORY/training.json --checkpoint COMPLETED_CHECKPOINT \
  --sha256 CHECKPOINT_SHA256 --sampler SAMPLER_JSON \
  --factory-source integrations/generals_fabric.py --output NEW_NATIVE_ASSET_DIRECTORY
python -m integrations.policy export --asset NEW_NATIVE_ASSET_DIRECTORY/asset.json \
  --manifest-sha256 ASSET_MANIFEST_SHA256 \
  --factory-source integrations/generals_fabric.py --output NEW_PORTABLE_BUNDLE_DIRECTORY
```

Publication verifies the completed PPO checkpoint and authentic optimizer
clock. Export preserves policy bytes, model/ABI, sampler and training seeds.
Preserve completed checkpoints and learner bytes if publication fails; rerun
publication from the existing run.

## Development and hosted acceptance

Freeze opponent versions, map distribution, sampler, seeds and seats before each
comparison. Pair source, exact matched control and treatment on initial Classic
states. Report W/L/D by opponent and seat, paired signed-score delta, uncertainty
and failures. Confirm a promising development result on independent maps before
hosting. Select and freeze the checkpoint before held-out or hosted evaluation.
For a sampler edit, compare both the changed initialization and unchanged
qualified parent. Verify native/serving acting probabilities, masks, priors,
half moves and temperature schedules, plus wire legality.

Serve the frozen bundle through `SpatialPlayerPolicy`. Build
`integrations/softmax/Dockerfile.neural` with the exported bundle as its named
`policy` context; see the [serving README](../../integrations/softmax/README.md#frozen-neural-policy).
Before registering, read the immutable AMD64 Docker config ID and inspect the
bundle inside that exact image. Match its manifest, asset, policy and weights
hashes to the exported bundle. Require SDK image status `ready` and
`client_hash` equal to that config ID; retain the registry digest and the
policy-registration request/response linking image ID to returned immutable
policy version. A promotion report's caller-supplied identity fields alone do
not prove this link.

The supported panel workflow is `integrations.hosted_policy`:

```bash
python -m integrations.hosted_policy submit --dry-run \
  --policy POLICY_VERSION_ID \
  --opponent incumbent=e53e30be-0b23-4d62-b944-4dd249a483fe \
  --opponent Daveey=76b0a083-f0a4-4ec7-9811-038349266633 \
  --games-per-opponent 256 --key UNIQUE_CONFIRMATION_KEY \
  --checkpoint-sha256 CHECKPOINT_SHA256 --source-commit SOURCE_COMMIT \
  --image-digest sha256:IMAGE_SHA256 --output NEW_PANEL_DIRECTORY
# Submit the identical preserved intent without --dry-run.
python -m integrations.hosted_policy status --output NEW_PANEL_DIRECTORY
python -m integrations.hosted_policy collect --output NEW_PANEL_DIRECTORY
python -m integrations.audit_hosted_panel_replays --panel NEW_PANEL_DIRECTORY
python -m integrations.policy promotion --summary NEW_PANEL_DIRECTORY/summary.json \
  --panel NEW_PANEL_DIRECTORY --opponents incumbent Daveey \
  --checkpoint-sha256 CHECKPOINT_SHA256 --image-digest sha256:IMAGE_SHA256 \
  --source-commit SOURCE_COMMIT --output NEW_PANEL_DIRECTORY/promotion.json
```

Confirm the frozen opponent IDs before submitting. Run a small hosted runtime
smoke first, then a fresh balanced panel of 256 games per opponent (128 per
seat). The workflow splits requests at the provider's 100-episode limit and
preserves exact request hashes and idempotency keys. If the 300-undispatched-
episode account limit returns HTTP 429, wait for accepted batches and rerun the
same intent. `status` polls only batches with preserved receipts; `collect`
counts batches without receipts as pending and lists `unsubmitted_requests`.
Neither invents provider receipts. Do not create new games because a poll deadline elapsed. Collect
the complete panel, audit SHA-bound replays, and require zero unexplained
illegal actions, timeouts, forfeits or incomplete requests.

Promotion needs ≥65% wins **against each** incumbent and Daveey, each Wilson
95% lower bound above 50%, broad Classic pool preservation, full identity
chain and clean execution. `promotion` recomputes the preserved panel and does
not itself change the champion. Promote only the qualified frozen version on an
eligible account. Authentication uses `SOFTMAX_TOKEN` or the SDK saved user
token; never print tokens or signed asset URLs.

If league placement is needed before the complete acceptance review, explicitly
use `coworld submit NAME:vN --league LEAGUE_ID --auto-champion never --no-open-browser`.
The SDK defaults to `--auto-champion always`; uploading a policy or running the
private panel does not require that preliminary league submission.

Keep selected state and latest verified decisions in [current state](current-state.md),
exact hashes in the manifest, and raw evidence in artifact directories. Never
delete or rewrite protected Codex session histories or trajectory databases.

Completed experiment runners live in Git history and their sealed job inputs.
The defense distillation trial and completed first-contact diagnostic are no
longer supported runtime commands; their retained results remain evidence.
