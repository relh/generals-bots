"""Guard a two-trainer B300 run using completed Puffer epochs and GPU samples."""

import re
import statistics
import sys
from pathlib import Path


def main() -> None:
    workspace, job, startup_seconds = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    prefix = sys.argv[4] if len(sys.argv) > 4 else "sparse-long"
    rates = []
    epochs = []
    completed = []
    for index in (0, 1):
        run = workspace / f"{prefix}-pilot-{job}-{index}"
        log = run / "console.log"
        history = log.read_text() if log.exists() else ""
        if "NonFiniteGradsError" in history or "FloatingPointError" in history:
            raise SystemExit(f"Trainer {index} produced nonfinite gradients")
        rows = []
        for block in history.split("╭"):
            epoch = re.search(r"Epoch\s+(\d+)", block)
            sps = re.search(r"SPS\s+([\d.]+)([KM]?)", block)
            if epoch and sps:
                rows.append((int(epoch[1]), float(sps[1]) * {"": 1, "K": 1000, "M": 1_000_000}[sps[2]]))
        epochs.append(rows[-1][0] if rows else 0)
        warm = [rate for epoch, rate in rows if epoch >= 3 and epoch % 64]
        rates.append(statistics.median(warm[-5:]) if len(warm) >= 5 else 0)
        completed.append((run / "completed.json").exists())
    samples = []
    for line in (workspace / f"{prefix}-gpu-{job}.csv").read_text().splitlines():
        field = line.split(",", 1)[0].strip()
        if field.isdigit():
            samples.append(int(field))
    gpu_mean = statistics.mean(samples[-60:]) if len(samples) >= 60 else -1
    aggregate = sum(rates)
    print(f"epochs={epochs} aggregate_sps={aggregate:.0f} gpu_mean_60s={gpu_mean:.1f} completed={completed}", flush=True)
    if not any(completed) and startup_seconds >= 300 and min(epochs) == 0:
        raise SystemExit("No completed training epoch after 300 seconds")
    if not any(completed) and min(epochs) >= 10 and aggregate < 30_000:
        raise SystemExit("Aggregate training SPS below 30,000 after warmup")
    # Utilization is diagnostic, while completed steps per wall-clock second
    # decide whether this allocation is productive. The two-trainer setup can
    # exceed the 30K SPS gate at roughly 25-30% sampled GPU utilization.


if __name__ == "__main__":
    main()
