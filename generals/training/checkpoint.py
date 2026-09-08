"""Atomic complete snapshots for trusted local training artifacts only."""
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def save_checkpoint(path, snapshot):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    snapshot = jax.device_get(snapshot)
    with temporary.open("wb") as stream:
        pickle.dump({"format_version": 1, "snapshot": snapshot}, stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path):
    """Never load a pickle from an untrusted source."""
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if payload["format_version"] != 1:
        raise ValueError("unsupported checkpoint format")
    return jax.tree.map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, payload["snapshot"])
