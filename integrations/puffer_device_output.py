"""Keep DLPack output owners alive through one ordered native copy fence."""
import hashlib
from pathlib import Path

BRIDGE_SHA256 = 'f08c651974c8e9df34d712308203df42a155264f0dc7a8b10cc90304b1025320'


def install_output_fence(source: Path):
    path = source/'src/metta_device_environment.cuh'
    text = path.read_text()
    if hashlib.sha256(path.read_bytes()).hexdigest() != BRIDGE_SHA256:
        raise ValueError('Pinned device bridge changed; review DLPack ownership before patching')
    text = text.replace('static void metta_env_copy(', 'static PyObject *metta_env_copy(', 1)
    text = text.replace('  assert(cudaStreamSynchronize(metta_environment_stream) == cudaSuccess);\n  Py_DECREF(capsule);\n', '', 1)
    text = text.replace('  Py_DECREF(method);\n}', '  Py_DECREF(method);\n  return capsule; // Caller retains the exported owner until all copies finish.\n}', 1)
    start = text.index('static void metta_env_output(')
    end = text.index('\nEnv *puf_vec_create', start)
    text = text[:start] + '''static void metta_env_output(PyObject *result) {
  assert(PyTuple_Check(result) && PyTuple_GET_SIZE(result) == 4);
  // Export establishes producer-to-consumer stream dependencies. Keep every
  // capsule and the caller-owned result tuple alive until the final copy fence.
  PyObject *owners[4] = {
    metta_env_copy(metta_env_obs, PyTuple_GET_ITEM(result, 0),
                   (long)METTA_AGENTS * OBS_SIZE, 2, 32),
    metta_env_copy(metta_env_masks, PyTuple_GET_ITEM(result, 1),
                   (long)METTA_AGENTS * METTA_MASK_SIZE, 1, 8),
    metta_env_copy(metta_env_rewards, PyTuple_GET_ITEM(result, 2), METTA_AGENTS, 2, 32),
    metta_env_copy(metta_env_terminals, PyTuple_GET_ITEM(result, 3), METTA_AGENTS, 2, 32)
  };
  assert(cudaStreamSynchronize(metta_environment_stream) == cudaSuccess);
  for (PyObject *owner : owners) Py_DECREF(owner);
}
''' + text[end:]
    path.write_text(text)
