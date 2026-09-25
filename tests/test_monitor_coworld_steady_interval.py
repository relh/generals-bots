from integrations.monitor_coworld_steady_interval import completed_epoch_times, interval_sps


def test_checkpoint_dip_does_not_hide_sustained_slowdown():
    history = ""
    uptime = 50.0
    for epoch in range(41):
        if epoch:
            uptime += 8.31 if epoch == 32 else 4.08
        minutes, seconds = divmod(int(uptime), 60)
        milliseconds = round((uptime % 1) * 1000)
        history += (
            f"╭\n│ Epoch {epoch} │\n"
            f"│ Uptime {minutes}m {seconds}s {milliseconds}ms │\n"
        )
    times = completed_epoch_times(history)
    assert interval_sps({k: v for k, v in times.items() if k <= 36}, 16) > 30_000
    assert interval_sps({k: v for k, v in times.items() if k <= 36}, 20) > 30_000

    slow = {epoch: (uptime + (epoch - 40) * 5.0 if epoch > 40 else value)
            for epoch, value in times.items()}
    slow.update({epoch: times[40] + (epoch - 40) * 5.0 for epoch in range(41, 62)})
    assert interval_sps(slow, 16) < 30_000
    assert interval_sps(slow, 20) < 30_000
