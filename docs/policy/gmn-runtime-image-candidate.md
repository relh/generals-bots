# Bounded GMN image fallback for the repaired Product probe

This is an isolated image candidate for a **new** context if `job-24kw9` also
fails at registry push. It does not change that submitted job or its source,
asset, opponent pool, training settings, or receipts.

The staged `nvidia/cuda:13.0.2-devel-ubuntu24.04` amd64 base has 3,971,526,496
compressed layer bytes. The matching runtime image has 1,647,845,811 bytes:
2,323,680,685 fewer compressed bytes before the extra development packages,
identical pip dependencies, and context layers. The five pinned CUDA library
development packages total about 535 MB of published `.deb` downloads, leaving
roughly 1.8 GB estimated savings before dependencies and layer compression.
The context files are about 165 MB. These counts come from the pinned
Docker registry manifests (devel `sha256:0eee3094…`, runtime
`sha256:121b2512…`) and [NVIDIA's CUDA package index](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/);
actual registry storage and build scratch savings must be measured. A fresh
builder can still run out of space during wheel installation.

`integrations/Dockerfile.gmn-runtime` keeps the pinned Ubuntu 24.04/CUDA 13.0.2
runtime base, Python 3.12, `requirements.lock`, JAX 0.11.0 CUDA 13 wheels,
`LD_LIBRARY_PATH`, framework source, sealed input, and job command. The already
locked `nvidia-cuda-nvcc`, `nvidia-cuda-runtime`, `nvidia-cuda-cccl`, and
`nvidia-nvvm` wheels place `nvcc`, core CUDA headers, CCCL, and libdevice under
`site-packages/nvidia/cu13`. The Puffer source also includes cuBLAS, cuSOLVER,
cuRAND, and NVML headers, so the candidate pins only those CUDA 13.0
development packages plus cudart headers from NVIDIA's Ubuntu repository.
It sets `CUDA_HOME` to the wheel toolkit, adds the repository headers to the
native compiler, and links the matching runtime-image CUDA libraries and
NVML stub. Image build asserts each header, link target, and compiler path;
native build and GPU execution remain mandatory gates.

The Puffer compiler changes from the devel image's CUDA 13.0.2 `nvcc` to the
locked wheel's CUDA 13.4.92 `nvcc`. This is a real native-build change even
though the JAX and runtime requirements remain pinned. Before any training,
verify `nvcc --version`, CUDA headers, NCCL, driver library, a complete native
Puffer build, the exact zero-head source/serving parity, and the ordinary
4,194,304-step throughput and activation gates. Do not reuse a qualification
marker from the devel image.

The local host has only 3.4 GiB free, so a full image pull/build would risk
disk exhaustion. Static checks completed: registry manifests and pinned wheel
contents were inspected; BuildKit parsed the candidate and resolved the amd64
base digest (its local lint warning concerns this arm64 development host).
The fallback image itself has **not** been built or run. If used, copy this
Dockerfile into a fresh GMN context, keep the exact
sealed probe input and source revision, reseal the new context, and submit only
after the current job is terminal and its failure is reviewed.
