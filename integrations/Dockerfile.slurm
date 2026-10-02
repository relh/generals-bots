FROM nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev git curl ca-certificates \
    build-essential clang ccache libomp-dev libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/lib/x86_64-linux-gnu/libomp.so.5 /usr/lib/x86_64-linux-gnu/libomp5.so \
    && ln -sf /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
RUN python3 -m venv /opt/generals \
    && /opt/generals/bin/pip install --no-cache-dir \
       'jax[cuda13]==0.11.0' 'numpy==2.5.3' 'pydantic==2.13.5' 'msgpack==1.1.2'

# Pure source only: no copied virtualenv, bytecode, binary wheel or CUDA cache.
COPY source /opt/generals-source
ENV PATH=/opt/generals/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH=/opt/generals-source
RUN python -m pip check && python -c \
    'import numpy, pydantic, msgpack, importlib.metadata as m; assert m.version("jax") == "0.11.0"; print("IMAGE_DEPENDENCIES_OK")'
# JAX CPU execution is audited separately in this exact image: default Apple
# x86 translation lacks AVX, so use an AVX-capable CPU or explicit QEMU TCG.
