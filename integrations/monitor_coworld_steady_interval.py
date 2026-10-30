"""Guard one Puffer trainer using completed steps over wall time.

The console's instantaneous SPS can dip after a checkpoint even when the
end-to-end rate over a sustained interval remains above the training gate.
"""

from __future__ import annotations

import re

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


def interval_sps(times: dict[int, float], span: int, steps_per_epoch: int) -> float | None:
    if not times:
        return None
    last = max(times)
    first = last - span
    if first not in times or times[last] <= times[first]:
        return None
    return span * steps_per_epoch / (times[last] - times[first])
