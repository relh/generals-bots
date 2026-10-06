from integrations.monitor_coworld_steady_interval import completed_epoch_times, interval_sps, steady_sps


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
    assert steady_sps({k: v for k, v in times.items() if k <= 36}) > 30_000

    slow = {epoch: (uptime + (epoch - 40) * 5.0 if epoch > 40 else value)
            for epoch, value in times.items()}
    slow.update({epoch: times[40] + (epoch - 40) * 5.0 for epoch in range(41, 62)})
    assert interval_sps(slow, 16) < 30_000
    assert interval_sps(slow, 20) < 30_000
    assert steady_sps(slow) < 30_000


def test_steady_interval_excludes_startup_compilation():
    assert steady_sps({0: 0.0, 1: 120.0}) is None
    assert steady_sps({0: 0.0, 1: 120.0, 2: 124.0, 3: 128.0}) == 32768


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


def test_h100_transient_dip_does_not_stop_qualified_training():
    # Completed epoch timings from job-tmpv7; the two-epoch guard stopped it.
    times = {
        1: 106.335,
        2: 122.9,
        3: 139.939,
        4: 155.341,
        5: 170.793,
        6: 186.2,
        7: 201.517,
        8: 216.626,
        9: 232.182,
        10: 248.345,
        11: 264.312,
        12: 280.323,
        13: 295.949,
        14: 311.096,
        15: 326.014,
        16: 341.177,
        17: 356.214,
        18: 371.125,
        19: 386.051,
        20: 400.967,
        21: 415.981,
        22: 431.152,
        23: 446.154,
        24: 461.348,
        25: 478.802,
        26: 495.945,
        27: 513.331,
        28: 529.313,
        29: 545.194,
        30: 560.932,
        31: 576.34,
        32: 591.768,
        33: 610.277,
        34: 629.286,
    }
    assert interval_sps(times, 2, 4096 * 128) < 30_000
    assert steady_sps(times, 4096 * 128) > 30_000
