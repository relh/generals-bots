# Policy runbook

Use [current-state.md](current-state.md) for the selected baseline and
[roadmap.md](roadmap.md) for unfinished work. Repository `AGENTS.md` is the
compute policy. Historical experiments remain in [Git history](https://github.com/relh/generals-bots/blob/106ac6af647d8a2148f9ebd1d43409b73edd4ddb/integrations/COWORLD_CLASSIC_STATUS.md); the commands below are the supported operational path.

## Game and policy contract

The current Puffer wrapper always instantiates `GeneralsEnv` with
`coworld_classic_rules=True` and pinned `CLASSIC_MAP_OPTIONS`. Dimensions are sampled
independently in 18–21, padded to 21 for inference; games cap at 2,000 turns.
The wrapper currently uses mountain density 0.24–0.26, minimum general distance
17, castle range `(9, 11)`, and castle army range `(40, 51)`. Interpret range
endpoints from the implementation. Position curricula are explicit training
inputs; held-out evaluation starts from the official initial map distribution.

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

Native training requires the supported Metta/Puffer CUDA container, pinned
trainer and native bridge, compiled Fabric build and verified current assets.
Local CPU inference and hosted serving use the portable NumPy policy without
private Metta dependencies.

| Entry point | Scope |
| --- | --- |
| `integrations.metta_puffer` | Batched device-resident Classic games |
| `integrations.spatial_selfplay` | Frozen and population policy opponents |
| `integrations.launch_spatial_selfplay_training` | Native launcher, transfer and sampler guards |
| `integrations.policy_trial` | Matched control/warmstart phases with explicit inputs and audits |
| `integrations.native_spatial_asset` | Current native policy/learner identity and provenance |
| `integrations.publish_policy_asset` | Publish actual completed PPO checkpoints as native assets |
| `integrations.audit_spatial_hosted_replays` | Public-action legality and serving comparison against hosted replay panels |

The matched trial defaults to immutable inputs at `/work/input` and results at
`/work/out`; `--input` and `--output` select explicit alternate directories.
Inputs include `source/`, `source-manifest.json`, `build-config.json`,
`config.json`, `assets/cold/asset.json`, `bundles/cold/`, frozen opponent bundles,
`curriculum/manifest.json`, separate `defense/train/` and `defense/heldout/`
manifests, and `leader-root/` replay fixtures. The supported CUDA container
supplies Metta/Puffer. An editable checkout does not reconstruct these inputs.

## GPU container backend and storage

The supported Slurm runner uses direct Enroot in a single allocated GPU step.
It creates one container from the verified immutable image, then executes the
sealed `integrations/slurm_s3_job.py --enroot-step` within the Slurm GPU cgroup.
The allocated-step runner SHA must match the input source receipt. Container
GPU identity is checked against the host step's assigned UUID.

Set `scratch_parent=/var/tmp`. A fresh job root with mode `0700` owns Enroot
DATA, TEMP, CACHE, RUNTIME and CONFIG directories, together with workload caches
and results. Storage paths are derived by the runner; personal Enroot config
and shared site storage are not used. The launch schema rejects
`enroot_storage_paths`, `mount_recovery` and `retained_container`.

Current capacity declarations are:

| Stage | Free bytes | Available inodes |
| --- | --- | --- |
| Startup | 32 GiB | 60,000 |
| Image unpack | 15 GiB | 40,000 |
| Each workload phase | 8 GiB | 20,000 |

Verify real filesystem availability and installed Enroot hooks before submission.
The runner records actual byte/inode gauges on failures. Job 35949 failed
before downloads because `/tmp` had only 13,170 available inodes. The corrected
owned `/var/tmp` backend passed actual allocated host/container GPU ownership,
UUID and 46-state native CUDA parity in 35956. That job then failed before
sampling games on a missing explicit balanced-seat option; warm training and
broad strength evaluation remain incomplete. See current-state for evidence.

## Current native asset boundary

`generals-native-spatial-asset-v1` stores `asset.json`, `policy.bin` and optional
`policy.bin.learner`. The manifest binds the current fabric, source/model/ABI
hashes, complete parameter allocation, policy/learner hashes, explicit structured
sampler, unique training seeds and provenance. Learner-bearing assets also
require current `learner_configuration` and `training_contract`. Policy-only
assets set both to null. Optimizer epoch, actual environment steps and learning
rate come from the authentic `METTAL01` payload; opaque ancestor metadata is
never used to reconstruct runtime settings.

Initialization in a run JSON is exactly:

```json
{
  "initialize": {
    "asset": "/work/input/assets/cold/asset.json",
    "manifest_sha256": "FULL_ASSET_MANIFEST_SHA256",
    "restore_learner": false
  }
}
```

`restore_learner: false` copies policy weights into a fresh optimizer.
`restore_learner: true` requires the asset's actual learner bytes, matching
seed/overrides and effective game/codec/reward contract. Distribution changes
are separate explicit settings; they cannot alter the restored objective.
There are no historical run/checkpoint parser fallbacks. `training.json` is
written by actual PPO execution, and used to publish completed runs; source
cleanup and supervised updates do not manufacture PPO training records.

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

The prepared GPU container executes the matched qualification workflow:

```bash
python -m integrations.policy trial smoke
python -m integrations.policy trial build
python -m integrations.policy trial preflight
python -m integrations.policy trial distill
python -m integrations.policy trial control
python -m integrations.policy trial warm
python -m integrations.policy trial evaluate
```

Run these phases in order inside the prepared GPU allocation. Both PPO arms
start fresh optimizers (`restore_learner=False`) and use the same sampler,
opponents, position curriculum, reward settings, seed, and **8,388,608 RL steps
per arm**. `distill` records 256 supervised updates on separate public-defense
training/held-out data; the warm arm initializes from those recorded policy
weights. Supervised updates add no RL environment steps. Each arm runs its own
512-game sampler gate; warm training repeats CPU preflight after distillation.
The intended final phase compares source, distilled, control, and warm bundles
on the same 4,096-game evaluation seeds and writes paired comparisons. This is a
policy-weight intervention with matched fresh optimizers, not learner resume.

`train` and `resume` forward the complete native launcher arguments and perform
its bootstrap. Use the supported container environment and explicit native
asset initialization. `evaluate`, `export`, and `compare`
forward the existing module arguments; inspect each command's `--help` for
required artifacts and outputs. `promotion --help` describes the evidence
report used to assess hosted qualification.

The cleaned production factory is pinned to SHA-256
`48767fb4ee333ae0b1a02ae644fbdf6f52f7f6df6c90c97ab3fc3888ba0c0d8a`.
It supports the current sixteen-plane F32/G32 graph with five priors and radius
1.01 or 2.01. See current-state for canonical model/ABI identities and the
one-time byte-preserving source equivalence proof. GPU parity and training
must still pass in the current CUDA runtime.

Publish an actual completed PPO run as a native asset, then export it:

```bash
python -m integrations.policy publish --build BUILD_DIRECTORY/build.json \
  --training RUN_DIRECTORY/training.json --checkpoint COMPLETED_CHECKPOINT \
  --sha256 CHECKPOINT_SHA256 --sampler SAMPLER_JSON \
  --factory-source integrations/generals_fabric.py --output NEW_NATIVE_ASSET_DIRECTORY
python -m integrations.policy export --asset NEW_NATIVE_ASSET_DIRECTORY/asset.json \
  --manifest-sha256 ASSET_MANIFEST_SHA256 \
  --factory-source integrations/generals_fabric.py --output NEW_PORTABLE_BUNDLE_DIRECTORY
```

Publication verifies the real completed run, checkpoint and authentic optimizer
clock, and adds only actual new RL steps to provenance. Export verifies the
realized model and ABI, preserves policy bytes/sampler/seeds and creates the
portable `generals-spatial-policy-v1` manifest. Its file map is exactly
`asset.json`, `policy.bin`, `weights.npz`; the serving asset has no learner.
Existing outputs must be preserved; these operations author new directories.

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

## Hosted panels

Register the frozen AMD64 image using the installed Coworld SDK `upload-policy`
command, then retain its immutable policy version and registry digest.
Before scheduling the panel, verify the frozen identity chain:

1. Record `docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}}' IMAGE`.
   Use that immutable config ID for readback, rather than the mutable tag:

   ```bash
   docker run --rm --platform linux/amd64 --read-only --network none \
     --entrypoint python IMAGE_CONFIG_ID -c 'import hashlib,json; from pathlib import Path; from integrations.spatial_policy_bundle import SpatialPlayerPolicy; p=Path("/app/policy"); policy=SpatialPlayerPolicy(p); print(json.dumps({"bundle_manifest_sha256":hashlib.sha256((p/"spatial-policy.json").read_bytes()).hexdigest(),"files":{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ("asset.json","policy.bin","weights.npz")},"policy_sha256":policy.asset.metadata["policy_sha256"]}))' \
     > image-bundle-readback.json
   ```

   Compare every reported hash with the selected exported bundle. Successful
   loading validates the current `asset.json` bundle and sampler; it does not
   establish wire parity or strength. Keep the build's frozen source revision.
2. Read the server image's ready metadata through SDK `get_image(IMAGE_ID)`.
   Require `status=ready`, `client_hash=IMAGE_CONFIG_ID`, and retain the returned
   immutable `image_digest`. The SDK uses Docker's config ID as `client_hash`;
   that ID covers the image's rootfs layers. The registry digest can differ.
3. Retain the authenticated registration request with `container_image_id` and
   its response containing the immutable policy version ID. Use that returned
   version in the panel. Also verify a server policy-to-image association if
   the API exposes it. The known baseline's current policy lookup returns
   `container_image_id=null`; it does **not** independently prove this link.
   The retained request/response records the original registration transaction.

The promotion report checks panel identities and outcomes; its caller-supplied
checkpoint/image/source fields do not prove this image-to-policy chain.
Build and registration stay with Docker/SDK; the supported match workflow is
`integrations.hosted_policy`. It uses the current Observatory HTTP contract,
without the SDK's outdated typed pagination parser.

```bash
python -m integrations.hosted_policy submit --dry-run \
  --policy POLICY_VERSION_ID --opponent incumbent=OPPONENT_VERSION_ID \
  --games-per-opponent 2 --key UNIQUE_PANEL_KEY \
  --checkpoint-sha256 CHECKPOINT_SHA256 --source-commit SOURCE_COMMIT \
  --image-digest sha256:IMAGE_SHA256 --output NEW_PANEL_DIRECTORY
# Repeat the same command without --dry-run to submit the preserved intent.
python -m integrations.hosted_policy status --output NEW_PANEL_DIRECTORY
python -m integrations.hosted_policy collect --output NEW_PANEL_DIRECTORY
```

Each opponent gets equal games in both seats; the explicit count includes both
seats. Add repeated `--opponent NAME=VERSION_ID` arguments for a larger fixed
pool. Dry run writes exact request bodies and hashes without authentication or
external writes. Submission reads owned requests first, preserves receipts, and
uses stable idempotency keys; repeating the identical intent reuses requests.
Collection verifies preserved payload hashes and frozen episode rosters, then
writes `summary.json` for the promotion report, including incomplete/failing
requests and recorded episode costs. It performs no champion change.

After the two-game runtime smoke passes, prepare fresh confirmation against
both frozen opponent versions. This schedules 256 games per opponent, 128 in
each seat (512 games total):

```bash
python -m integrations.hosted_policy submit --dry-run \
  --policy POLICY_VERSION_ID \
  --opponent incumbent=e53e30be-0b23-4d62-b944-4dd249a483fe \
  --opponent Daveey=76b0a083-f0a4-4ec7-9811-038349266633 \
  --games-per-opponent 256 --key UNIQUE_CONFIRMATION_KEY \
  --checkpoint-sha256 CHECKPOINT_SHA256 --source-commit SOURCE_COMMIT \
  --image-digest sha256:IMAGE_SHA256 --output CONFIRMATION_PANEL_DIRECTORY
# Submit the identical preserved intent without --dry-run, then status/collect.
```

These IDs are the recorded frozen opponents; confirm their identities before
submission. Preserve the complete panel even when an early result looks strong.

Operational promotion requires the current hosted summary and its complete
preserved panel directory, plus explicit frozen identities:

```bash
python -m integrations.policy promotion --summary PANEL_DIRECTORY/summary.json \
  --panel PANEL_DIRECTORY --opponents incumbent Daveey \
  --checkpoint-sha256 CHECKPOINT_SHA256 --image-digest sha256:IMAGE_SHA256 \
  --source-commit SOURCE_COMMIT --output PANEL_DIRECTORY/promotion.json
```

It recomputes the summary from hashed payloads and retained request receipts and
states. Pending or incomplete panels cannot qualify, even when completed games
already exceed statistical thresholds. The independent `strength_report`
function evaluates counts alone and does not establish artifact readiness.

Authentication uses `SOFTMAX_TOKEN` or the SDK's canonical saved **user** token;
install the Coworld/Softmax SDK in the execution environment for saved-token
loading. Tokens and signed asset URLs are never printed. Read existing request
IDs until terminal; do not treat a polling deadline as permission for new games.

## Evidence and documentation updates

Keep a bounded current-state page: selected policy, latest qualified measurements,
remaining failures, and next experiment. Put full run outputs and receipts in
versioned artifact directories or durable storage; link them from the state page.
Record paths as unverified until their existence and hashes have been checked.

Preserve checkpoints, optimizer state, and raw experiment outcomes. Protected Codex session history and trajectory databases must never
be removed or rewritten as cleanup. Current user authorization and repository instructions govern the active
overhaul. Git history retains superseded documentation and experiment recipes.
