"""Guard one Puffer trainer using completed steps over wall time.

The console's instantaneous SPS can dip after a checkpoint even when the
end-to-end rate over a sustained interval remains above the training gate.
"""

from __future__ import annotations

import re
import statistics
import sys
import time
from pathlib import Path

STEPS_PER_EPOCH = 4096 * 32
MIN_SPS = 30_000


def completed_epoch_times(history: str) -> dict[int, float]:
    result = {}
    for block in history.split("╭"):
        epoch = re.search(r"Epoch\s+(\d+)", block)
        uptime = re.search(r"Uptime\s+(?:(\d+)m\s+)?(\d+)s\s+(\d+)ms", block)
        if epoch and uptime:
            result[int(epoch[1])] = (
                int(uptime[1] or 0) * 60 + int(uptime[2]) + int(uptime[3]) / 1000
            )
    return result


def interval_sps(times: dict[int, float], span: int, steps_per_epoch: int = STEPS_PER_EPOCH) -> float | None:
    if not times:
        return None
    last = max(times)
    first = last - span
    if first not in times or times[last] <= times[first]:
        return None
    return span * steps_per_epoch / (times[last] - times[first])


def main() -> None:
    workspace, job, startup_seconds, prefix = (
        Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
    )
    startup_limit = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    steps_per_epoch = int(sys.argv[6]) if len(sys.argv) > 6 else STEPS_PER_EPOCH
    if steps_per_epoch <= 0:
        raise SystemExit("Steps per epoch must be positive")
    run = workspace / f"{prefix}-pilot-{job}-0"
    log = run / "console.log"
    history = log.read_text(errors="replace") if log.exists() else ""
    if "NonFiniteGradsError" in history or "FloatingPointError" in history:
        raise SystemExit("Trainer produced nonfinite gradients")
    times = completed_epoch_times(history)
    epoch = max(times, default=0)
    completed = (run / "completed.json").exists()
    samples = []
    gpu_path = workspace / f"{prefix}-gpu-{job}.csv"
    if gpu_path.exists():
        for line in gpu_path.read_text().splitlines():
            field = line.split(",", 1)[0].strip()
            if field.isdigit():
                samples.append(int(field))
    gpu_mean = statistics.mean(samples[-60:]) if len(samples) >= 60 else -1
    sps_16 = interval_sps(times, 16, steps_per_epoch)
    sps_20 = interval_sps(times, 20, steps_per_epoch)
    print(
        f"epoch={epoch} sps_16={sps_16} sps_20={sps_20} "
        f"gpu_mean_60s={gpu_mean:.1f} completed={completed}", flush=True,
    )
    if times and not completed and time.time() - log.stat().st_mtime > 90:
        raise SystemExit("No console progress for 90 seconds")
    if not completed and startup_seconds >= startup_limit and epoch == 0:
        raise SystemExit(f"No completed training epoch after {startup_limit} seconds")
    if (
        not completed and epoch >= 30
        and sps_16 is not None and sps_20 is not None
        and sps_16 < MIN_SPS and sps_20 < MIN_SPS
    ):
        raise SystemExit("Sustained end-to-end training SPS below 30,000")


if __name__ == "__main__":
    main()
