#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-policy-feedforward134m-quality-v2
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:35:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/native-policy-feedforward134m-quality-v2-%j.log
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


python3 - /var/tmp/relh-generals-recovery <<'STAGEPY'
import base64,sys
from pathlib import Path
path=Path(sys.argv[1])/"native_puffer_policy-feedforward-20260927.py"
data=base64.b64decode("IiIiRnJvemVuIGZsb2F0MzIgTWluR1JVIGluZmVyZW5jZSBmb3IgdGhlIHBpbm5lZCBkZWZhdWx0IFB1ZmZlcjUgYXJjaGl0ZWN0dXJlLgoKTWF0Y2hlcyBlbmNvZGVyL2RlY29kZXIgYW5kIG1pbmdydV9nYXRlIGluIHVwc3RyZWFtIDZmZmE1YjEwIHNyYy9hbGdvLmN1LgpUaGlzIGRlbGliZXJhdGVseSBhY2NlcHRzIG9ubHkgdGhlIHZlcmlmaWVkIENsYXNzaWMgbmF0aXZlIGJ1aWxkIGNvbnRyYWN0LgoiIiIKCmltcG9ydCBoYXNobGliCmltcG9ydCBqc29uCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAoKaW1wb3J0IGpheAppbXBvcnQgamF4Lm51bXB5IGFzIGpucAppbXBvcnQgbnVtcHkgYXMgbnAKCgpjbGFzcyBOYXRpdmVQdWZmZXJQb2xpY3k6CiAgICBkZWYgX19pbml0X18oc2VsZiwgYnVpbGQ6IFBhdGgsIHRyYWluaW5nOiBQYXRoLCBjaGVja3BvaW50OiBQYXRoLCBzaGEyNTY6IHN0cik6CiAgICAgICAgbWFuaWZlc3QgPSBqc29uLmxvYWRzKGJ1aWxkLnJlYWRfdGV4dCgpKQogICAgICAgIHJlY29yZCA9IGpzb24ubG9hZHModHJhaW5pbmcucmVhZF90ZXh0KCkpCiAgICAgICAgaWYgbWFuaWZlc3QgIT0gcmVjb3JkWyJidWlsZCJdOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJUcmFpbmluZy9idWlsZCBtYW5pZmVzdHMgZGlmZmVyIikKICAgICAgICBjb25maWcgPSBtYW5pZmVzdFsiY29uZmlnIl0KICAgICAgICBzcGVjID0gY29uZmlnWyJweXRob25fZW52aXJvbm1lbnQiXVsic3BlYyJdCiAgICAgICAgaWYgKG1hbmlmZXN0WyJyZXZpc2lvbiJdICE9ICI2ZmZhNWIxMGRiYmJlNGQxZTgyODgzNjdjN2Q5ZDNhY2QzYmFkNGEyIgogICAgICAgICAgICAgICAgb3IgY29uZmlnWyJmYWJyaWMiXSBpcyBub3QgTm9uZSBvciBjb25maWdbInByZWNpc2lvbiJdICE9ICJmbG9hdDMyIgogICAgICAgICAgICAgICAgb3Igc3BlY1sib2JzZXJ2YXRpb25fc2l6ZSJdICE9IDYxNzQgb3Igc3BlY1siYWN0aW9uX3NpemVzIl0gIT0gWzE3NjUsIDJdCiAgICAgICAgICAgICAgICBvciBzcGVjWyJ0ZWFjaGVyIl0gb3Igc3BlY1sicm91dGluZyJdIG9yIHNwZWNbInJlcGxheV9tZXRhZGF0YV9zaXplIl0pOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJVbnN1cHBvcnRlZCBuYXRpdmUgcG9saWN5IGNvbnRyYWN0IikKICAgICAgICBzZWxmLmhpZGRlbiA9IGludChyZWNvcmRbImNvbmZpZyJdWyJvdmVycmlkZXMiXVsicG9saWN5LmhpZGRlbl9zaXplIl0pCiAgICAgICAgc2VsZi5sYXllcnMgPSBpbnQocmVjb3JkWyJjb25maWciXVsib3ZlcnJpZGVzIl1bInBvbGljeS5udW1fbGF5ZXJzIl0pCiAgICAgICAgaWYgc2VsZi5oaWRkZW4gPD0gMCBvciBzZWxmLmxheWVycyA8IDA6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIkludmFsaWQgcmVjdXJyZW50IGRpbWVuc2lvbnMiKQogICAgICAgIGRhdGEgPSBjaGVja3BvaW50LnJlYWRfYnl0ZXMoKQogICAgICAgIGlmIGhhc2hsaWIuc2hhMjU2KGRhdGEpLmhleGRpZ2VzdCgpICE9IHNoYTI1NjoKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiQ2hlY2twb2ludCBTSEEyNTYgbWlzbWF0Y2giKQogICAgICAgIG9mZnNldCA9IDAKCiAgICAgICAgZGVmIHRha2Uoc2hhcGUpOgogICAgICAgICAgICBub25sb2NhbCBvZmZzZXQKICAgICAgICAgICAgIyBQdWZmZXIgQWxsb2NhdG9yIGFsaWducyB0aGUgc3RhcnQgb2YgZWFjaCByZWdpc3RlcmVkIHRlbnNvciB0bzE2IGJ5dGVzLgogICAgICAgICAgICBvZmZzZXQgPSAob2Zmc2V0ICsgMTUpICYgfjE1CiAgICAgICAgICAgIGNvdW50ID0gaW50KG5wLnByb2Qoc2hhcGUpKQogICAgICAgICAgICBhcnJheSA9IG5wLmZyb21idWZmZXIoZGF0YSwgZHR5cGU9IjxmNCIsIGNvdW50PWNvdW50LCBvZmZzZXQ9b2Zmc2V0KS5yZXNoYXBlKHNoYXBlKQogICAgICAgICAgICBvZmZzZXQgKz0gY291bnQgKiA0CiAgICAgICAgICAgIGlmIG5vdCBucC5pc2Zpbml0ZShhcnJheSkuYWxsKCk6CiAgICAgICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJOb25maW5pdGUgY2hlY2twb2ludCBwYXJhbWV0ZXJzIikKICAgICAgICAgICAgcmV0dXJuIGpucC5hc2FycmF5KGFycmF5KQoKICAgICAgICBzZWxmLmVuY29kZXIgPSB0YWtlKChzZWxmLmhpZGRlbiwgNjE3NCkpCiAgICAgICAgc2VsZi5kZWNvZGVyID0gdGFrZSgoMTc2OCwgc2VsZi5oaWRkZW4pKQogICAgICAgIHNlbGYucmVjdXJyZW50ID0gdHVwbGUodGFrZSgoMyAqIHNlbGYuaGlkZGVuLCBzZWxmLmhpZGRlbikpIGZvciBfIGluIHJhbmdlKHNlbGYubGF5ZXJzKSkKICAgICAgICBpZiBvZmZzZXQgIT0gbGVuKGRhdGEpOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJVbmV4cGVjdGVkIGNoZWNrcG9pbnQgbGVuZ3RoIikKICAgICAgICBzZWxmLmZvcndhcmQgPSBqYXguaml0KHNlbGYuX2ZvcndhcmQpCgogICAgZGVmIGluaXRpYWxfc3RhdGUoc2VsZiwgbGFuZXMpOgogICAgICAgIHJldHVybiBqbnAuemVyb3MoKHNlbGYubGF5ZXJzLCBsYW5lcywgc2VsZi5oaWRkZW4pLCBkdHlwZT1qbnAuZmxvYXQzMikKCiAgICBkZWYgX2ZvcndhcmQoc2VsZiwgb2JzZXJ2YXRpb25zLCBzdGF0ZSk6CiAgICAgICAgeCA9IGpucC5tYXRtdWwob2JzZXJ2YXRpb25zLCBzZWxmLmVuY29kZXIuVCwgcHJlY2lzaW9uPWpheC5sYXguUHJlY2lzaW9uLkhJR0hFU1QpCiAgICAgICAgc3RhdGVzID0gW10KICAgICAgICBmb3IgbGF5ZXIsIHdlaWdodCBpbiBlbnVtZXJhdGUoc2VsZi5yZWN1cnJlbnQpOgogICAgICAgICAgICBjb21iaW5lZCA9IGpucC5tYXRtdWwoeCwgd2VpZ2h0LlQsIHByZWNpc2lvbj1qYXgubGF4LlByZWNpc2lvbi5ISUdIRVNUKQogICAgICAgICAgICBoaWRkZW4sIGdhdGUsIHByb2plY3Rpb24gPSBqbnAuc3BsaXQoY29tYmluZWQsIDMsIGF4aXM9LTEpCiAgICAgICAgICAgIHogPSBqYXgubm4uc2lnbW9pZChnYXRlKQogICAgICAgICAgICBjYW5kaWRhdGUgPSBqbnAud2hlcmUoaGlkZGVuID49IDAsIGhpZGRlbiArIC41LCBqYXgubm4uc2lnbW9pZChoaWRkZW4pKQogICAgICAgICAgICBkaWZmZXJlbmNlID0gY2FuZGlkYXRlIC0gc3RhdGVbbGF5ZXJdCiAgICAgICAgICAgICMgTWF0Y2ggbmF0aXZlIGxlcnAncyBzdGFibGUgYnJhbmNoZXMuCiAgICAgICAgICAgIG5leHRfc3RhdGUgPSBqbnAud2hlcmUoeiA8IC41LCBzdGF0ZVtsYXllcl0gKyB6ICogZGlmZmVyZW5jZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjYW5kaWRhdGUgLSBkaWZmZXJlbmNlICogKDEgLSB6KSkKICAgICAgICAgICAgcyA9IGpheC5ubi5zaWdtb2lkKHByb2plY3Rpb24pCiAgICAgICAgICAgIHggPSBzICogbmV4dF9zdGF0ZSArICgxIC0gcykgKiB4CiAgICAgICAgICAgIHN0YXRlcy5hcHBlbmQobmV4dF9zdGF0ZSkKICAgICAgICByZXR1cm4gam5wLm1hdG11bCh4LCBzZWxmLmRlY29kZXIuVCwgcHJlY2lzaW9uPWpheC5sYXguUHJlY2lzaW9uLkhJR0hFU1QpLCAoCiAgICAgICAgICAgIGpucC5zdGFjayhzdGF0ZXMpIGlmIHNlbGYubGF5ZXJzIGVsc2Ugc3RhdGUKICAgICAgICApCgogICAgZGVmIGFjdGlvbnMoc2VsZiwgb2JzZXJ2YXRpb25zLCBtYXNrcywgc3RhdGUsIGtleT1Ob25lKToKICAgICAgICBpZiBvYnNlcnZhdGlvbnMuc2hhcGUgIT0gKHN0YXRlLnNoYXBlWzFdLCA2MTc0KSBvciBtYXNrcy5zaGFwZSAhPSAoc3RhdGUuc2hhcGVbMV0sIDE3NjcpOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJVbmV4cGVjdGVkIG9ic2VydmF0aW9uL21hc2sgZGltZW5zaW9ucyIpCiAgICAgICAgb3V0cHV0LCBzdGF0ZSA9IHNlbGYuZm9yd2FyZChqbnAuYXNhcnJheShvYnNlcnZhdGlvbnMsIGR0eXBlPWpucC5mbG9hdDMyKSwgc3RhdGUpCiAgICAgICAgbG9naXRzID0gam5wLndoZXJlKGpucC5hc2FycmF5KG1hc2tzLCBkdHlwZT1ib29sKSwgb3V0cHV0WzosIDoxNzY3XSwgLWpucC5pbmYpCiAgICAgICAgaWYga2V5IGlzIE5vbmU6CiAgICAgICAgICAgIGFjdGlvbnMgPSBqbnAuc3RhY2soKGpucC5hcmdtYXgobG9naXRzWzosIDoxNzY1XSwgYXhpcz0tMSksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGpucC5hcmdtYXgobG9naXRzWzosIDE3NjU6XSwgYXhpcz0tMSkpLCBheGlzPS0xKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIG1vdmVfa2V5LCBzcGxpdF9rZXkgPSBqYXgucmFuZG9tLnNwbGl0KGtleSkKICAgICAgICAgICAgYWN0aW9ucyA9IGpucC5zdGFjaygoamF4LnJhbmRvbS5jYXRlZ29yaWNhbChtb3ZlX2tleSwgbG9naXRzWzosIDoxNzY1XSksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGpheC5yYW5kb20uY2F0ZWdvcmljYWwoc3BsaXRfa2V5LCBsb2dpdHNbOiwgMTc2NTpdKSksIGF4aXM9LTEpCiAgICAgICAgcmV0dXJuIGFjdGlvbnMsIHN0YXRlCg==")
if path.exists():
    assert path.read_bytes()==data
else:
    with path.open("xb") as f:
        f.write(data)
STAGEPY
python3 - /var/tmp/relh-generals-recovery <<'STAGEPY'
import base64,sys
from pathlib import Path
path=Path(sys.argv[1])/"launch_native_feedforward_evaluation-v2-20260927.py"
data=base64.b64decode("IiIiTG9hZCB2ZXJpZmllZCBjb250cm9sIGFuZCBmcm96ZW4gZXZhbHVhdG9yIGZvciB6ZXJvLWxheWVyIGFuZCByZWN1cnJlbnQgbmF0aXZlIHBvbGljaWVzLiIiIgoKaW1wb3J0IGhhc2hsaWIKaW1wb3J0IGltcG9ydGxpYi51dGlsCmltcG9ydCBydW5weQppbXBvcnQgc3lzCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAoKcm9vdCA9IFBhdGgoX19maWxlX18pLnBhcmVudApzb3VyY2UgPSByb290IC8gInB1ZmZlci1uYXRpdmUtZmVlZGZvcndhcmQtMjAyNjA5MjcucHkiCmFzc2VydCAoCiAgICBoYXNobGliLnNoYTI1Nihzb3VyY2UucmVhZF9ieXRlcygpKS5oZXhkaWdlc3QoKQogICAgPT0gIjI1ZWM2ZDFiMGY3YzA4MDEyYTU4NjgxY2U0NGI1NzIxMTQzZjZkYTEwYzc4YjZjZjUwZTlkYWMwMWY3ZDljY2YiCikKc3BlYyA9IGltcG9ydGxpYi51dGlsLnNwZWNfZnJvbV9maWxlX2xvY2F0aW9uKCJtZXR0YV90cmFpbmluZy5wdWZmZXIiLCBzb3VyY2UpCm1vZHVsZSA9IGltcG9ydGxpYi51dGlsLm1vZHVsZV9mcm9tX3NwZWMoc3BlYykKc3lzLm1vZHVsZXNbc3BlYy5uYW1lXSA9IG1vZHVsZQpzcGVjLmxvYWRlci5leGVjX21vZHVsZShtb2R1bGUpCmV2YWx1YXRvciA9IHJvb3QgLyAiZXZhbHVhdGVfY293b3JsZF9mcm96ZW5fZ3JlZWR5LXJld2FyZC0yMDI2MDkyNy5weSIKYXNzZXJ0ICgKICAgIGhhc2hsaWIuc2hhMjU2KGV2YWx1YXRvci5yZWFkX2J5dGVzKCkpLmhleGRpZ2VzdCgpCiAgICA9PSAiZDliZjcxMmY1MTU1NmNkYTQxM2E5MmFlYzM1NmFhN2NmZTMzNDZkYTZmZThkYzFlNmU0M2I0OGQ4NTg2MmIyOCIKKQpydW5weS5ydW5fcGF0aChzdHIoZXZhbHVhdG9yKSwgcnVuX25hbWU9Il9fbWFpbl9fIikK")
if path.exists():
    assert path.read_bytes()==data
else:
    with path.open("xb") as f:
        f.write(data)
STAGEPY

srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
disk=/var/tmp/relh-generals-recovery
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" 17171d409be5d618b85ac6578f43576876e96f1e86ade540bb9f2a07c0f661a2 \
    "$disk/metta_puffer-terminal-boundary-fixed-20260927.py" | sha256sum -c -
printf "%s  %s\n" 020dbafc66861496295787330361b07e0492c892805594105fdbb050dcbb37f5 \
    "$disk/puffer_codec-calibrated-v1.py" | sha256sum -c -
disk=/var/tmp/relh-generals-recovery
out=$disk/native-policy-feedforward134m-quality-v2-$job
mkdir -p "$out"
printf "%s  %s\n" d9bf712f51556cda413a92aec356aa7cfe3346da6fe8dc1e6e43b48d85862b28 \
    "$disk/evaluate_coworld_frozen_greedy-reward-20260927.py" | sha256sum -c -
printf "%s  %s\n" 0d6128a0413a77ab167cc34080fc35106ce1ee312d0c44c5e67c159dce69a814 \
    "$disk/launch_native_feedforward_evaluation-v2-20260927.py" | sha256sum -c -
printf "%s  %s\n" 37550f9dea166c0a7f8e6b165a2e52f12b41883f2b7599ac9f57aa5b12ca0eda \
    "$disk/puffer-calibrated-transfer.py" | sha256sum -c -
printf "%s  %s\n" e5691723d137e61a44db53c6dd20eac8050b8e7be732c3045e4fe207c0238519 "$disk/native_puffer_policy-feedforward-20260927.py" | sha256sum -c -
printf "%s  %s\n" 25ec6d1b0f7c08012a58681ce44b5721143f6da10c78b6cf50e9dac01f7d9ccf "$disk/puffer-native-feedforward-20260927.py" | sha256sum -c -
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
ready=0
for attempt in {1..15}; do
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
printf "GPU_QUERY attempt=%s step_gpus=%s rows=%s\n" "$attempt" "$SLURM_STEP_GPUS" "$usage"
mapfile -t rows <<< "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}
memory=${memory//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
echo "gpu_idle_check=$attempt gpu=$gpu memory_mb=$memory utilization=$utilization step_gpus=$SLURM_STEP_GPUS"
if (( memory < 2048 && utilization < 20 )); then
    ready=1
    break
fi
sleep 2
done
(( ready == 1 ))
test -n "$gpu"
name=relh-classic-native-policy-feedforward134m-quality-v2-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
for tag in baseline candidate; do
    run=native-policy-reward-scaled-33m-probe-26286
    checkpoint=0000000033554432.bin
    build=native-policy-reward-scaled-build-26260
    if [[ "$tag" == candidate ]]; then
        run=native-policy-feedforward-state-fixed-134m-probe-26604
        build=native-policy-feedforward-state-fixed-33m-probe-v2-26484
        checkpoint=0000000134217728.bin
    fi
    test -f "$disk/$run/run/completed.json"
    relative_checkpoint="checkpoints/metta_generals/run/$checkpoint"
    digest=$(python3 - "$disk/$run/run" "$relative_checkpoint" <<"PY"
import hashlib,json,sys
from pathlib import Path
run=Path(sys.argv[1])
checkpoint=run/sys.argv[2]
identity=json.loads(Path(str(checkpoint)+".learner.json").read_text())
policy_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert policy_sha == identity["policy_sha256"]
assert hashlib.sha256(Path(str(checkpoint)+".learner").read_bytes()).hexdigest() == identity["state_sha256"]
assert hashlib.sha256((run/"training.json").read_bytes()).hexdigest() == identity["run_sha256"]
print(policy_sha)
PY
    )
    printf "%s  %s\n" "$digest" "$disk/$run/run/$relative_checkpoint" | sha256sum -c -
    for opponent in expander_harvester sentinel strong_mixed; do
timeout -k 30s 5m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v "$out:/recovery-output" \
    -v "$disk:/recovery" \
    -v "$disk/native_puffer_policy-feedforward-20260927.py:/work/source-normalized-16245/integrations/native_puffer_policy.py:ro" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$disk/metta_puffer-terminal-boundary-fixed-20260927.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e XDG_CACHE_HOME=/recovery-output/cache -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache \
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b python /recovery/launch_native_feedforward_evaluation-v2-20260927.py \
    --native --build "/recovery/$build/build/build.json" \
    --run "/recovery/$run/run" --checkpoint "/recovery/$run/run/$relative_checkpoint" \
    --sha256 "$digest" --reward-diagnostics --games 1024 --pool-size 1024 --seed 1386 --opponent "$opponent" \
    --output "/recovery-output/$tag-$opponent" > "$out/$tag-$opponent.log" 2>&1
        cat "$out/$tag-$opponent/evaluation.json"
    done
done

'
