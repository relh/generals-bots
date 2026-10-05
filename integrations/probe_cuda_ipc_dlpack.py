"""Check whether a separate JAX worker can update a shared CUDA buffer.

Run inside the pinned B300 image. This exercises buffer ownership and CUDA
IPC only; it does not measure Puffer5 training or implement its environment.
"""

from functools import partial
import multiprocessing as mp

import jax
import jax.numpy as jnp
import torch


def worker(input_queue, output_queue) -> None:
    tensor = input_queue.get(timeout=60)
    source_ptr = tensor.data_ptr()
    imported = jax.dlpack.from_dlpack(tensor)

    @partial(jax.jit, donate_argnums=(0,))
    def add_one(value):
        return value + jnp.float32(1)

    result = add_one(imported)
    result.block_until_ready()
    output_queue.put((imported.is_deleted(), result.unsafe_buffer_pointer() == source_ptr))
    output_queue.get(timeout=60)


def main() -> None:
    mp.set_start_method("spawn")
    tensor = torch.zeros((4096, 6174), dtype=torch.float32, device="cuda")
    input_queue, output_queue = mp.Queue(), mp.Queue()
    process = mp.Process(target=worker, args=(input_queue, output_queue))
    process.start()
    input_queue.put(tensor)
    donated, aliased = output_queue.get(timeout=90)
    torch.cuda.synchronize()
    changed = bool(torch.all(tensor == 1).item())
    output_queue.put("done")
    process.join(timeout=30)
    print({
        "donated": donated,
        "aliased": aliased,
        "parent_sees_write": changed,
        "worker_exitcode": process.exitcode,
        "elements": tensor.numel(),
    }, flush=True)
    assert donated and aliased and changed and process.exitcode == 0


if __name__ == "__main__":
    main()
