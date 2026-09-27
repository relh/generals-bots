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
        uptime = re.search(r"Uptime\s+((?:\d+(?:ms|[dhms])\s*)+)", block)
        if epoch and uptime:
            factors = {"d": 86400, "h": 3600, "m": 60, "s": 1, "ms": 0.001}
            result[int(epoch[1])] = sum(
                int(value) * factors[unit]
                for value, unit in re.findall(r"(\d+)(ms|[dhms])", uptime[1])
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
    min_sps = int(sys.argv[7]) if len(sys.argv) > 7 else MIN_SPS
    short_span = int(sys.argv[8]) if len(sys.argv) > 8 else 16
    long_span = int(sys.argv[9]) if len(sys.argv) > 9 else 20
    gate_epoch = int(sys.argv[10]) if len(sys.argv) > 10 else 30
    if min_sps <= 0:
        raise SystemExit("Minimum SPS must be positive")
    if steps_per_epoch <= 0:
        raise SystemExit("Steps per epoch must be positive")
    if not 0 < short_span < long_span < gate_epoch:
        raise SystemExit("Require 0 < short_span < long_span < gate_epoch")
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
    sps_short = interval_sps(times, short_span, steps_per_epoch)
    sps_long = interval_sps(times, long_span, steps_per_epoch)
    print(
        f"epoch={epoch} sps_{short_span}={sps_short} sps_{long_span}={sps_long} "
        f"gpu_mean_60s={gpu_mean:.1f} completed={completed}", flush=True,
    )
    if times and not completed and time.time() - log.stat().st_mtime > 90:
        raise SystemExit("No console progress for 90 seconds")
    if not completed and startup_seconds >= startup_limit and epoch == 0:
        raise SystemExit(f"No completed training epoch after {startup_limit} seconds")
    if (
        not completed and epoch >= gate_epoch
        and sps_short is not None and sps_long is not None
        and sps_short < min_sps and sps_long < min_sps
    ):
        raise SystemExit(f"Sustained end-to-end training SPS below {min_sps:,}")


if __name__ == "__main__":
    main()
