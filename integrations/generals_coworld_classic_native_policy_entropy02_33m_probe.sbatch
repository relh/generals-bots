#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-policy-entropy02-33m-probe
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/native-policy-entropy02-33m-probe-%j.log
set -euo pipefail

# Mandatory user policy: verify lowest scheduling priority before workload startup.
slurm_task_priority_record=$(scontrol show job -o "${SLURM_JOB_ID:?}")
[[ " $slurm_task_priority_record " == *" Nice=2147483645 "* ]] || { printf 'Lowest Slurm Nice verification failed\n' >&2; exit 1; }
slurm_task_priority=unknown
for slurm_task_field in $slurm_task_priority_record; do
  case "$slurm_task_field" in Priority=*) slurm_task_priority=${slurm_task_field#Priority=} ;; esac
done
[[ "$slurm_task_priority" =~ ^[0-9]+$ ]] || exit 1
printf 'SLURM_PRIORITY_READBACK JobId=%s Nice=2147483645 Priority=%s\n' "$SLURM_JOB_ID" "$slurm_task_priority" >&2
unset slurm_task_priority_record slurm_task_field


srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
disk=/var/tmp/relh-generals-recovery
python3 - "$disk" <<"PY"
import base64
from pathlib import Path
import sys
root = Path(sys.argv[1])
path = root / "train_native_hint_initialized-entropy-20260927.py"
data = base64.b64decode("IiIiQm91bmRlZCBuYXRpdmUgUHVmZmVyNSBwcm9iZSBmcm9tIGFuIGV4cGxpY2l0IHVudHJhaW5lZCBwdWJsaWMtaGludCBhcnRpZmFjdC4iIiIKCmltcG9ydCBhcmdwYXJzZQppbXBvcnQgaGFzaGxpYgppbXBvcnQganNvbgppbXBvcnQgbWF0aAppbXBvcnQgc3RydWN0CmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAoKZnJvbSBtZXR0YV90cmFpbmluZy5wdWZmZXIgaW1wb3J0IE5hdGl2ZVBvbGljeUluaXRpYWxpemF0aW9uLCBSdW5Db25maWcsIHRyYWluX3B1ZmZlcgoKZnJvbSBpbnRlZ3JhdGlvbnMubmF0aXZlX2hpbnRfaW5pdGlhbGl6ZXIgaW1wb3J0IGV4cG9ydAoKCmRlZiBtYWluKCk6CiAgICBwYXJzZXIgPSBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcigpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXRlbXBsYXRlIiwgdHlwZT1QYXRoLCByZXF1aXJlZD1UcnVlKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1idWlsZCIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tb3V0cHV0IiwgdHlwZT1QYXRoLCByZXF1aXJlZD1UcnVlKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1sb2dpdC1zY2FsZSIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9MjQuMCkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tc2VlZCIsIHR5cGU9aW50LCBkZWZhdWx0PTEzNDYpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLWVudHJvcHktY29lZiIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9MC4wKQogICAgYXJncyA9IHBhcnNlci5wYXJzZV9hcmdzKCkKICAgIGlmIG5vdCBtYXRoLmlzZmluaXRlKGFyZ3MuZW50cm9weV9jb2VmKSBvciBhcmdzLmVudHJvcHlfY29lZiA8IDA6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiRW50cm9weSBjb2VmZmljaWVudCBtdXN0IGJlIGZpbml0ZSBhbmQgbm9ubmVnYXRpdmUiKQogICAgYXNzZXJ0IG5vdCAoYXJncy5vdXRwdXQgLyAicnVuIikuZXhpc3RzKCkKICAgIF8sIGluaXRpYWxpemVyID0gZXhwb3J0KGFyZ3MuYnVpbGQgLyAiYnVpbGQuanNvbiIsIGFyZ3Mub3V0cHV0IC8gImluaXRpYWxpemVyIiwgc2NhbGU9YXJncy5sb2dpdF9zY2FsZSkKICAgIGFydGlmYWN0ID0gYXJncy5vdXRwdXQgLyAiaW5pdGlhbGl6ZXIvaW5pdGlhbGl6ZXIuanNvbiIKICAgIHJlY29yZCA9IGpzb24ubG9hZHMoYXJncy50ZW1wbGF0ZS5yZWFkX3RleHQoKSkKICAgIHJlY29yZFsidG90YWxfdGltZXN0ZXBzIl0gPSAzM181NTRfNDMyCiAgICByZWNvcmRbInNlZWQiXSA9IGFyZ3Muc2VlZAogICAgcmVjb3JkWyJpbml0aWFsaXplIl0gPSBOb25lCiAgICByZWNvcmRbIm5hdGl2ZV9wb2xpY3lfaW5pdGlhbGl6ZXIiXSA9IGRpY3QoCiAgICAgICAgbWFuaWZlc3Q9c3RyKGFydGlmYWN0KSwgc2hhMjU2PWhhc2hsaWIuc2hhMjU2KGFydGlmYWN0LnJlYWRfYnl0ZXMoKSkuaGV4ZGlnZXN0KCkKICAgICkKICAgIHJlY29yZFsib3ZlcnJpZGVzIl0udXBkYXRlKHsKICAgICAgICAidmVjLnRvdGFsX2FnZW50cyI6IDY1NTM2LCAidmVjLm51bV9idWZmZXJzIjogMSwgInZlYy5udW1fdGhyZWFkcyI6IDEsCiAgICAgICAgImJhc2UuY3VkYWdyYXBocyI6IC0xLCAiYmFzZS5jaGVja3BvaW50X2ludGVydmFsIjogMTYsCiAgICAgICAgInBvbGljeS5oaWRkZW5fc2l6ZSI6IDUxMiwgInBvbGljeS5udW1fbGF5ZXJzIjogMSwKICAgICAgICAidHJhaW4ubGVhcm5pbmdfcmF0ZSI6IC4wMDMsICJ0cmFpbi5lbnRfY29lZiI6IGFyZ3MuZW50cm9weV9jb2VmLCAidHJhaW4ucmVwbGF5X3JhdGlvIjogMS4sCiAgICAgICAgInRyYWluLmhvcml6b24iOiAxNiwgInRyYWluLm1pbmliYXRjaF9zaXplIjogNTI0Mjg4LAogICAgfSkKICAgIGFzc2VydCByZWNvcmRbIm92ZXJyaWRlcyJdWyJ0cmFpbi5nYW1tYSJdID09IC45OTkKICAgIGNvbmZpZyA9IFJ1bkNvbmZpZy5tb2RlbF92YWxpZGF0ZShyZWNvcmQpCiAgICBhc3NlcnQgaXNpbnN0YW5jZShjb25maWcubmF0aXZlX3BvbGljeV9pbml0aWFsaXplciwgTmF0aXZlUG9saWN5SW5pdGlhbGl6YXRpb24pCiAgICAoYXJncy5vdXRwdXQgLyAiY29uZmlnLmpzb24iKS53cml0ZV90ZXh0KGNvbmZpZy5tb2RlbF9kdW1wX2pzb24oaW5kZW50PTIpICsgIlxuIikKICAgIHJlc3VsdCA9IHRyYWluX3B1ZmZlcihhcmdzLmJ1aWxkLCBhcmdzLm91dHB1dCAvICJydW4iLCBjb25maWcpCiAgICBhc3NlcnQgcmVzdWx0LnRyYWluZWRfdGltZXN0ZXBzID09IDMzXzU1NF80MzIKICAgIGNvcGllZF9zaGEgPSBoYXNobGliLnNoYTI1NigoYXJncy5vdXRwdXQgLyAicnVuL2luaXRpYWwtcG9saWN5LmJpbiIpLnJlYWRfYnl0ZXMoKSkuaGV4ZGlnZXN0KCkKICAgIGFzc2VydCBjb3BpZWRfc2hhID09IGluaXRpYWxpemVyWyJwb2xpY3lfc2hhMjU2Il0KICAgIGNoZWNrcG9pbnQgPSBhcmdzLm91dHB1dCAvICJydW4iIC8gcmVzdWx0LmZpbmFsX2NoZWNrcG9pbnQKICAgIGRhdGEgPSBjaGVja3BvaW50LnJlYWRfYnl0ZXMoKQogICAgYXNzZXJ0IGxlbihkYXRhKSA9PSBpbml0aWFsaXplclsicGFyYW1ldGVyX2NvdW50Il0gKiA0CiAgICBhc3NlcnQgYWxsKG1hdGguaXNmaW5pdGUodmFsdWVbMF0pIGZvciB2YWx1ZSBpbiBzdHJ1Y3QuaXRlcl91bnBhY2soIjxmIiwgZGF0YSkpCiAgICBwcmludCgiTkFUSVZFX1BBUkFNRVRFUlNfRklOSVRFIiwgbGVuKGRhdGEpIC8vIDQsIGhhc2hsaWIuc2hhMjU2KGRhdGEpLmhleGRpZ2VzdCgpLCBmbHVzaD1UcnVlKQoKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICBtYWluKCkK")
if path.exists():
    assert path.read_bytes() == data
else:
    with path.open("xb") as handle:
        handle.write(data)
path = root / "launch_native_hint_entropy_probe-20260927.py"
data = base64.b64decode("IiIiTG9hZCB2ZXJpZmllZCBydW4gY29udHJvbCB3aGlsZSBwcmVzZXJ2aW5nIGFyY2hpdmVkIGVudmlyb25tZW50IGZpbmdlcnByaW50cy4iIiIKCmltcG9ydCBoYXNobGliCmltcG9ydCBpbXBvcnRsaWIudXRpbAppbXBvcnQgcnVucHkKaW1wb3J0IHN5cwpmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKCnNvdXJjZSA9IFBhdGgoX19maWxlX18pLndpdGhfbmFtZSgicHVmZmVyLW5hdGl2ZS1pbml0aWFsaXplci0yMDI2MDkyNy5weSIpCmFzc2VydCBoYXNobGliLnNoYTI1Nihzb3VyY2UucmVhZF9ieXRlcygpKS5oZXhkaWdlc3QoKSA9PSAoCiAgICAiMDk1OGQ4NTI4NTE0NDY0N2FlYTRkMzJmYzYwZjA4ZTM5NjVkODJkNGNhZDgzZjdiMTg0YTU4OTQ5Y2Q2M2VjNCIKKQpzcGVjID0gaW1wb3J0bGliLnV0aWwuc3BlY19mcm9tX2ZpbGVfbG9jYXRpb24oIm1ldHRhX3RyYWluaW5nLnB1ZmZlciIsIHNvdXJjZSkKbW9kdWxlID0gaW1wb3J0bGliLnV0aWwubW9kdWxlX2Zyb21fc3BlYyhzcGVjKQpzeXMubW9kdWxlc1tzcGVjLm5hbWVdID0gbW9kdWxlCnNwZWMubG9hZGVyLmV4ZWNfbW9kdWxlKG1vZHVsZSkKcnVubmVyID0gUGF0aChfX2ZpbGVfXykud2l0aF9uYW1lKCJ0cmFpbl9uYXRpdmVfaGludF9pbml0aWFsaXplZC1lbnRyb3B5LTIwMjYwOTI3LnB5IikKYXNzZXJ0IGhhc2hsaWIuc2hhMjU2KHJ1bm5lci5yZWFkX2J5dGVzKCkpLmhleGRpZ2VzdCgpID09ICgKICAgICJlZTU3ODYzZTJmYmYxMTVlM2VhZDFiYjE5OTQ3MjRkNGE1NjA0ODRiMjg5MjllYzgxMGNkZjAwODEwODNkMWEyIgopCnJ1bnB5LnJ1bl9wYXRoKHN0cihydW5uZXIpLCBydW5fbmFtZT0iX19tYWluX18iKQo=")
if path.exists():
    assert path.read_bytes() == data
else:
    with path.open("xb") as handle:
        handle.write(data)
PY
out=$disk/native-policy-entropy02-33m-probe-$job
mkdir -p "$out"
printf "%s  %s\n" fbf8e4a12d96da6bf2fb32e63b10916b7d3699c360b5d33ad2b1a3493a9c3509 \
    "$disk/launch_puffer_calibrated_transfer.py" | sha256sum -c -
printf "%s  %s\n" 9f59b9047fa642c8dd4815f268d60cadba67fb691480ee82dd9d68f6de806626 \
    "$disk/run_puffer_calibrated_transfer.sh" | sha256sum -c -
printf "%s  %s\n" 5a2142491aad55cf4b2961f0c21572c80a80b76760d7ba6a4eccacf194b07371 \
    "$disk/monitor_coworld_steady_interval_v4.py" | sha256sum -c -
printf "%s  %s\n" 37550f9dea166c0a7f8e6b165a2e52f12b41883f2b7599ac9f57aa5b12ca0eda \
    "$disk/puffer-calibrated-transfer.py" | sha256sum -c -
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" a7291d2a2f25166941b07309789865822e85ccbef721edbb199bc8fb3520e3ba \
    "$disk/metta_puffer-reward-scaled-20260927.py" | sha256sum -c -
printf "%s  %s\n" 020dbafc66861496295787330361b07e0492c892805594105fdbb050dcbb37f5 \
    "$disk/puffer_codec-calibrated-v1.py" | sha256sum -c -
printf "%s  %s\n" 0958d85285144647aea4d32fc60f08e3965d82d4cad83f7b184a58949cd63ec4 "$disk/puffer-native-initializer-20260927.py" | sha256sum -c -
printf "%s  %s\n" 88efc61a2140af3a444ac53cd0d67ce457fb845792bd496204320ff55eb653ae "$disk/native_hint_initializer-importable-20260927.py" | sha256sum -c -
printf "%s  %s\n" ee57863e2fbf115e3ead1bb1994724d4a560484b28929ec810cdf0081083d1a2 "$disk/train_native_hint_initialized-entropy-20260927.py" | sha256sum -c -
printf "%s  %s\n" d599e4e251879737f622d83edc5edf012456de67c04b49582364799b2452502f "$disk/launch_native_hint_entropy_probe-20260927.py" | sha256sum -c -
python3 - "$disk/native-policy-reward-scaled-build-26260/build/build.json" <<"PY"
import hashlib, json, sys
from pathlib import Path
manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text())
assert hashlib.sha256(manifest_path.with_name("puffer").read_bytes()).hexdigest() == manifest["binary_sha256"]
assert manifest["model_state_words"] == 0
assert manifest["model_sha256"] == ""
config = manifest["config"]
assert config["python_environment"]["spec"]["agents"] == 65536
assert not config["python_environment"]["spec"]["teacher"]
options = config["python_environment"]["options"]
assert options["opponent"] == "strong_mixed"
assert options["teacher"] == "expander_harvester"
assert not options["supervise_teacher"] and options["imitation_weight"] == 0.0
assert options["expander_hint_features"] and options["prior_hint_features"]
assert config["fabric"] is None
assert options["move_hint_scale"] == .375 and options["split_hint_scale"] == .125
assert options["shaping_gamma"] == .999
assert options["balance_opponent_sides"]
assert options["reward_scale"] == .5
assert options["army_shaping_weight"] == .5
assert options["land_shaping_weight"] == .3
assert options["castle_shaping_weight"] == 0.
assert manifest["environment_sha256"] == "73e9271702b1b2d9d548f73ee7c3bb9e9148a3e4c26b4e91fabba15cf16e229b"
PY
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
mapfile -t rows <<< "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}
memory=${memory//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
(( memory < 2048 && utilization < 20 ))
test -n "$gpu"
echo "allocated_gpus=$SLURM_STEP_GPUS training_gpu=$gpu"
name=relh-classic-native-policy-entropy02-33m-probe-$job
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$disk/native-policy-entropy02-33m-probe-gpu-$job.csv" 2>&1 &
sampler=$!
ln -s "$out/run" "$disk/native-policy-entropy02-33m-probe-pilot-$job-0"
timeout -k 30s 25m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v "$out:/recovery-output" \
    -v "$disk:/recovery" \
    -v "$disk/native_hint_initializer-importable-20260927.py:/work/source-normalized-16245/integrations/native_hint_initializer.py:ro" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$disk/metta_puffer-reward-scaled-20260927.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e XDG_CACHE_HOME=/recovery-output/cache -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 -e JAX_COMPILATION_CACHE_DIR=/cache \
    relh-generals-b300:20260923b python /recovery/launch_native_hint_entropy_probe-20260927.py \
    --template /work/normalized-clean-pilot-16788/config.json \
    --build /recovery/native-policy-reward-scaled-build-26260/build \
    --entropy-coef .02 --output /recovery-output > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval_v4.py" \
        "$disk" "$job" "$((now-start))" native-policy-entropy02-33m-probe 600 1048576 300000; then
        echo "Device bridge smoke stopped by startup or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$out/run/completed.json" || status=1
exit "$status"
'
