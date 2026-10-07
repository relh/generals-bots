"""Expose the verified spatial weight layout to the native matrix optimizer.

Fabric stores scalar sharing groups as (groups, 1). Those storage dimensions
do not describe a convolution or dense projection. This opt-in adapter changes
optimizer tensor metadata only; checkpoint order and policy algebra stay fixed.
"""

import numpy as np


def logical_optimizer_shapes(model, buffers, *, context_matrix=False):
    matrices = {}

    def register(name, indices, shape):
        indices = np.asarray(indices)
        start = int(indices.min())
        if not np.array_equal(indices.reshape(-1), np.arange(start, start + indices.size)):
            raise ValueError(f"{name} storage does not match its logical tensor axes")
        if np.prod(shape) != indices.size or start in matrices:
            raise ValueError(f"{name} has an invalid or repeated optimizer block")
        matrices[start] = (name, tuple(shape), indices.size)

    register("input", model.input_kernel.T, model.input_kernel.T.shape)
    register("action", model.action_kernel.T, model.action_kernel.T.shape)
    register("product_local", model.product_local_kernel.T, model.product_local_kernel.T.shape)
    register("product_global", model.product_global_kernel, model.product_global_kernel.shape)
    register("product_action", model.product_action_kernel.T, model.product_action_kernel.T.shape)
    register("global", model.global_kernel, model.global_kernel.shape)
    register("readout", model.readout_kernel, model.readout_kernel.shape)
    if context_matrix:
        from integrations.spatial_muon_context import geometry, validate_model_gather
        gather = validate_model_gather(model)
        offset, shape, _, _, _ = geometry(model.context_kernel.shape[0] // 2 + .01)
        if offset in matrices:
            raise ValueError("Convolution optimizer block overlaps another matrix")
        matrices[offset] = ("context", shape, len(gather))

    shapes, report = [], []
    offset = 0
    for parameter in sorted(buffers.parameters, key=lambda parameter: parameter.offset):
        if parameter.offset != offset:
            raise ValueError("Optimizer parameter blocks do not cover checkpoint storage")
        entry = matrices.pop(offset, None)
        if entry:
            name, shape, size = entry
            if size != parameter.size:
                raise ValueError(f"{name} crosses its native parameter block")
        else:
            # The stencil's sharing-group order is not a rectangular tensor.
            # Keep it and the atom coefficients as vectors. Reconstructing a
            # convolution optimizer would require an explicit permutation.
            if len(parameter.shape) != 2 or parameter.shape[1] != 1:
                raise ValueError("Unmapped optimizer block is not a scalar sharing group")
            name, shape = "scalar_groups", (parameter.size,)
        shapes.append(shape)
        report.append(dict(name=name, offset=offset, size=parameter.size, shape=list(shape)))
        offset += parameter.size
        if parameter.padding:
            shapes.append((parameter.padding,))
            offset += parameter.padding
    if matrices or offset != buffers.parameter_words:
        raise ValueError("Logical optimizer layout leaves unmatched checkpoint parameters")
    return shapes, report
