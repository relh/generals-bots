import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from integrations import portable_classic_pilot as pilot
from integrations.monitor_coworld_steady_interval import interval_sps


def write_config(root, games=2048, horizon=256, minibatch=8192):
    (root / 'config.json').write_text(json.dumps({'overrides': {
        'vec.total_agents': games, 'train.horizon': horizon,
        'train.minibatch_size': minibatch, 'base.checkpoint_interval': 8,
    }}))


def test_smaller_rollout_cannot_inherit_parent_sps():
    with TemporaryDirectory() as directory, patch.object(pilot, 'OUT', Path(directory)):
        write_config(pilot.OUT)
        geometry = pilot.training_geometry()
        # Same observed epoch timings: parent dimensions would falsely pass 30k.
        times = {10: 100, 12: 140}
        assert interval_sps(times, 2, geometry['steps_per_epoch']) == 26214.4
        assert interval_sps(times, 2, 8192 * 256) > 30000
        write_config(pilot.OUT, games=8192)
        assert pilot.training_geometry()['steps_per_epoch'] == 2097152


@pytest.mark.parametrize('games,horizon,minibatch', [
    (True, 256, 8192), (0, 256, 8192), (2048, -1, 8192),
    (2048, 256, 3), (2048, 256, 0), (2048.0, 256, 8192),
])
def test_invalid_or_partial_rollout_rejected(games, horizon, minibatch):
    with TemporaryDirectory() as directory, patch.object(pilot, 'OUT', Path(directory)):
        write_config(pilot.OUT, games, horizon, minibatch)
        with pytest.raises(ValueError):
            pilot.training_geometry()


def test_smaller_rollout_midpoint_cannot_accept_parent_sized_gap():
    with TemporaryDirectory() as directory, patch.object(pilot, 'OUT', Path(directory)):
        write_config(pilot.OUT)
        checkpoints = pilot.OUT / 'run/checkpoints/metta_generals/run'
        checkpoints.mkdir(parents=True)
        start, steps = 2499805184, 33554432
        # 4M away is permitted by the old parent grid, but exceeds the actual
        # smaller rollout's half-save interval (2M).
        candidate = start + steps // 2 + 4194304
        (checkpoints / f'{candidate:016d}.bin').write_bytes(b'fixture')
        with pytest.raises(FileNotFoundError, match='half a save interval'):
            pilot.midpoint_checkpoint_steps(start, steps)
