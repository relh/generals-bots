#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-selfplay-preflight
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-selfplay-preflight-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
printf "%s  %s\n" 1316ac4bc5018af20edd02377cf508a5e9a2dc18faa9e65ccf8847b188ec1a12 \
    "$disk/metta_puffer-selfplay-v1.py" | sha256sum -c -
IFS=, read -r -a allocated <<< "$SLURM_STEP_GPUS"
mapfile -t usage < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
test "${#allocated[@]}" -eq 1
test "${#usage[@]}" -eq 1
IFS=, read -r memory utilization <<< "${usage[0]}"
memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
(( memory < 2048 && utilization < 20 ))
gpu=${allocated[0]}
docker run --rm -i --name "relh-classic-selfplay-preflight-$SLURM_JOB_ID" \
    --gpus "device=$gpu" --cpus=4 --ulimit core=0:0 \
    --user "$(id -u):$(id -g)" -v "$w:/work" -v "$disk:/recovery" \
    -v "$disk/metta_puffer-selfplay-v1.py:/work/source-spatial2/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-spatial2/integrations/puffer_codec.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/recovery -e XDG_CACHE_HOME=/recovery/cache-selfplay \
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource \
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b python3 - <<"PY"
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext
from integrations.metta_puffer import BatchedGeneralsSelfPlayPufferEnvironment
from integrations.puffer_codec import decode_action

manifest=json.loads(Path("/recovery/spatial-hint2-mixed-build-27340/build.json").read_text())
options=manifest["config"]["python_environment"]["options"].copy()
options["parallel_games"]=8
env=BatchedGeneralsSelfPlayPufferEnvironment(
    context=EnvironmentContext(seed=701,index=0,mode="train",output=Path("/work")), **options,
)
assert env.spec.agents == 16 and env.spec.observation_size == 14*21*21
values,masks=env.reset_device("701:0:0")
assert values.shape == (16,14*21*21) and masks.shape == (16,1767)
assert jnp.all(masks[:,1764])
assert not np.array_equal(np.asarray(values[0]),np.asarray(values[1]))
actions=jnp.tile(jnp.asarray([1764,0],dtype=jnp.int32),(16,1))
next_values,next_masks,rewards,dones,episode_done=env.step_device(actions)
assert next_values.shape == values.shape and next_masks.shape == masks.shape
assert rewards.shape == (16,) and dones.shape == (16,) and episode_done is False
assert np.isfinite(np.asarray(rewards)).all()
assert np.allclose(np.asarray(rewards).reshape(8,2).sum(axis=1),0,atol=1e-5)
assert jnp.all(next_masks[:,1764])
for _ in range(12):
    legal=np.asarray(next_masks[:2,:1764])
    if legal.any(axis=1).all():
        break
    next_values,next_masks,_,_,_=env.step_device(actions)
else:
    raise AssertionError("Both policy seats never gained a legal non-pass move")
chosen=np.array([np.flatnonzero(legal[0])[0],np.flatnonzero(legal[1])[0]])
joint=np.array(actions)
joint[0,0]=chosen[0]
joint[1,0]=chosen[1]
state=jax.tree.map(lambda field: field[0],env.states)
game_actions=jnp.stack([decode_action(int(chosen[0]),21,0),decode_action(int(chosen[1]),21,0)])
direct_timestep,direct_state=env.base.env.step(state,game_actions,env.base.pool)
assert not bool(direct_timestep.terminated | direct_timestep.truncated)
env.step_device(jnp.asarray(joint))
actual_state=jax.tree.map(lambda field: field[0],env.states)
assert all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(
    jax.tree.leaves(actual_state),jax.tree.leaves(direct_state),strict=True
))
env.close()
print("selfplay_two_seat_device_preflight=ok",jax.devices("cuda")[0],flush=True)
PY
'
