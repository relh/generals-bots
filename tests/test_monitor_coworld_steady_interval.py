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
    assert interval_sps({k: v for k, v in times.items() if k <= 36}, 16, 4096 * 32) > 30_000
    assert interval_sps({k: v for k, v in times.items() if k <= 36}, 20, 4096 * 32) > 30_000

    slow = {epoch: (uptime + (epoch - 40) * 5.0 if epoch > 40 else value)
            for epoch, value in times.items()}
    slow.update({epoch: times[40] + (epoch - 40) * 5.0 for epoch in range(41, 62)})
    assert interval_sps(slow, 16, 4096 * 32) < 30_000
    assert interval_sps(slow, 20, 4096 * 32) < 30_000


def test_steady_interval_crosses_hour_boundary():
    history = ""
    for epoch in range(21):
        total_seconds = 3590 + 2 * epoch
        hours, remaining = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remaining, 60)
        hours_text = f"{hours}h " if hours else ""
        history += f"╭\n│ Epoch {epoch} │\n│ Uptime {hours_text}{minutes}m {seconds}s 125ms │\n"
    times = completed_epoch_times(history)
    assert times[0] == 3590.125
    assert times[20] == 3630.125
    assert interval_sps(times, 20, 8192 * 32) == 131072


def test_native_dashboard_day_hour_format_without_milliseconds():
    history = "╭\n│ Epoch 0 │\n│ Uptime 0ms │\n"
    history += "╭\n│ Epoch 12 │\n│ Uptime 59m 58s 500ms │\n"
    history += "╭\n│ Epoch 32 │\n│ Uptime 0d 1h 0m 40s │\n"
    times = completed_epoch_times(history)
    assert times == {0: 0.0, 12: 3598.5, 32: 3640.0}
    assert interval_sps(times, 20, 8192 * 32) > 120000


def test_full_fast_interval_can_contain_slow_rolling_window():
    times = dict(enumerate([100, 112, 113, 114, 150, 151, 152, 153], 1))
    assert interval_sps(times, 6, 524288) > 30000
    windows = [interval_sps({epoch: seconds for epoch, seconds in times.items() if epoch <= end},
                            2, 524288) for end in range(4, 9)]
    assert any(rate < 30000 for rate in windows)
