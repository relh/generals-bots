"""Explicit-memory, Classic-only native siege opponent.

This is an opponent candidate, not enabled in the training population by default.
It follows the cee053c Python siege strategy except that otherwise equal frontier
choices use row-major order. Only public wire grids and public board dimensions
are accepted. The caller must clear memory on every episode reset, including
resets into nonzero-turn curriculum positions.
"""

import ctypes
import subprocess
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).with_name("native") / "classic_siege.cpp"


def compile_library(output, *, compiler="c++"):
    """Build inside the execution image; never copy a host library into it."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [compiler, "-std=c++17", "-O3", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", "-pthread",
         str(SOURCE), "-o", str(output)],
        check=True, timeout=120,
    )
    return output


class ClassicSiegeBatch:
    """Pure row-independent decisions; workers includes the calling thread.

    Default serial execution preserves the qualified training configuration.
    Additional workers require an explicit CPU budget and GPU throughput trial.
    """

    def __init__(self, library, *, workers=1):
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
            raise ValueError("Native siege workers must be an integer from 1 to 8")
        self.workers = workers
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        pointer = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.library.classic_siege_batch.argtypes = [ctypes.c_int] + [pointer] * 6
        self.library.classic_siege_batch.restype = ctypes.c_int
        if workers > 1:
            self.library.classic_siege_batch_parallel.argtypes = [ctypes.c_int] + [pointer] * 6 + [ctypes.c_int]
            self.library.classic_siege_batch_parallel.restype = ctypes.c_int

    @staticmethod
    def initial_memory(count):
        return np.full((count, 3), -1, np.int32)

    def __call__(self, dimensions, turns, grids, memory):
        """Return actions (B,5), memory (B,3); all inputs are immutable int32.

        grids is (B,3,21,21) containing wire type, owner and army planes;
        dimensions is (B,2), turns (B,), memory (B,3). Padding is ignored.
        Memory stores city/general/spearhead as row*21+column, or -1.
        """
        inputs = tuple(np.asarray(x) for x in (dimensions, turns, grids, memory))
        dimensions, turns, grids, memory = inputs
        if dimensions.ndim != 2 or dimensions.shape[1:] != (2,):
            raise ValueError("Dimensions must have shape (batch, 2)")
        count = len(dimensions)
        if count > 1_000_000:
            raise ValueError("Opponent batch exceeds bounded size")
        if (turns.shape != (count,) or grids.shape != (count, 3, 21, 21)
                or memory.shape != (count, 3)):
            raise ValueError("Opponent input shapes differ")
        if any(x.dtype != np.int32 for x in inputs):
            raise ValueError("Opponent inputs must be int32")
        if np.any((dimensions < 1) | (dimensions > 21)) or np.any((turns < 0) | (turns > 2000)):
            raise ValueError("Invalid Classic dimensions or turn")
        if (np.any((grids[:, 0] < 0) | (grids[:, 0] > 5))
                or np.any((grids[:, 1] < 0) | (grids[:, 1] > 2))
                or np.any((grids[:, 2] < 0) | (grids[:, 2] > 1_000_000))):
            raise ValueError("Invalid public wire grids")
        actions = np.empty((count, 5), np.int32)
        next_memory = np.empty_like(memory)
        arguments = (count, *(np.ascontiguousarray(x) for x in inputs), actions, next_memory)
        rc = (self.library.classic_siege_batch(*arguments) if self.workers == 1
              else self.library.classic_siege_batch_parallel(*arguments, self.workers))
        if rc:
            raise ValueError(f"Native siege input rejected with code {rc}")
        return actions, next_memory


def padded_device_actions(opponent, observations, memory):
    """One batched host callback, public grids only, explicit returned memory.

    The training observation is padded to 21x21; this variant deliberately uses
    that geometry for edge ranking. It must not be labeled the exact hosted bot.
    JAX can reorder or repeat this pure callback because no agent state lives
    inside Python or the native library.
    """
    import jax
    import jax.numpy as jnp

    shape = observations.armies.shape
    if len(shape) != 3 or shape[1:] != (21, 21) or memory.shape != (shape[0], 3):
        raise ValueError("Padded siege requires batched 21x21 public observations")
    kinds = jnp.ones(shape, jnp.int32)
    for name, value in (("fog_cells", 0), ("structures_in_fog", 5), ("mountains", 2),
                        ("castles", 3), ("generals", 4)):
        kinds = jnp.where(getattr(observations, name), value, kinds)
    owners = jnp.where(observations.opponent_cells, 2, observations.owned_cells.astype(jnp.int32))
    grids = jnp.stack((kinds, owners, observations.armies), axis=1).astype(jnp.int32)
    actions, after = jax.pure_callback(
        opponent,
        (jax.ShapeDtypeStruct((shape[0], 5), jnp.int32),
         jax.ShapeDtypeStruct((shape[0], 3), jnp.int32)),
        jnp.full((shape[0], 2), 21, jnp.int32), observations.timestep.astype(jnp.int32), grids, memory,
    )
    indices = (actions[:, 4] * 4 + actions[:, 3]) * 441 + actions[:, 1] * 21 + actions[:, 2]
    return jnp.where(actions[:, 0] != 0, 3528, indices).astype(jnp.int32), after


def reset_completed_memory(memory, done):
    """Reset by episode completion, never by turn==0 (curriculum may start late)."""
    import jax.numpy as jnp

    return jnp.where(done[:, None], jnp.int32(-1), memory)
